//! Daily sufficient features for the 21-session, seven-channel Logsig-RNN.
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};

use rayon::prelude::*;

use crate::artifact_io::StoragePathExt;
use crate::calendar::TradingCalendar;
use crate::config::EngineConfig;
use crate::core::DatasetId;
use crate::data::{ColumnData, DataCatalog, MarketDataLoader, Table};
use crate::derive::bar::{derive_stock_minute_bars_selected, minute_index};
use crate::derive::storage::write_derived_parquet;
use crate::error::{err, Result};
use crate::path_logsignature::PathLogSignature;
use crate::progress::ProgressBar;

pub const WIDTH: usize = 728;
const SLOTS: usize = 48;
const TIME_SCALE: f64 = 1.0 / (21 * SLOTS - 1) as f64;
type Day = BTreeMap<String, Vec<[f64; 5]>>;

pub struct LogsigRnnRequest {
    pub start_date: i32,
    pub end_date: i32,
    pub date_batch_size: usize,
    pub threads: usize,
    pub overwrite: bool,
}

pub fn output_path(root: &Path, date: i32) -> PathBuf {
    root.join("derived/stock/logsigrnn_week")
        .join((date / 10000).to_string())
        .join(format!("{date}.parquet"))
}

fn read_day(loader: &MarketDataLoader, date: i32) -> Result<Day> {
    let columns = [
        "ts_code",
        "trade_time",
        "open",
        "high",
        "low",
        "close",
        "vol",
    ]
    .map(str::to_owned);
    let mut tables = loader.load_minute_by_date(DatasetId::StockMinute1m, &columns, &[date])?;
    let table = tables
        .remove(&date)
        .ok_or_else(|| err(format!("missing Logsig-RNN minute date {date}")))?;
    let mut keys = HashSet::new();
    for (code, time) in table
        .required_utf8("ts_code")?
        .iter()
        .zip(table.required_utf8("trade_time")?)
    {
        if let (Some(code), Some(time)) = (code, time) {
            if let Some(minute) = minute_index(time) {
                if !keys.insert((code, minute)) {
                    return Err(err(format!("duplicate minute for {code} on {date}")));
                }
            }
        }
    }
    let adjustment =
        loader.load_daily_by_dates(DatasetId::StockAdjFactor, &["adj_factor".into()], &[date])?;
    if adjustment.len == 0 {
        return Err(err(format!("missing Logsig-RNN adjustments on {date}")));
    }
    let mut adjustment_keys = HashSet::new();
    for code in adjustment.required_utf8("ts_code")? {
        let code = code
            .as_deref()
            .filter(|v| !v.is_empty())
            .ok_or_else(|| err("null/empty adjustment stock code"))?;
        if !adjustment_keys.insert(code) {
            return Err(err(format!("duplicate adjustment for {code} on {date}")));
        }
    }
    let factors: HashMap<_, _> = adjustment
        .required_utf8("ts_code")?
        .iter()
        .zip(adjustment.required_f64_cast("adj_factor")?)
        .filter_map(|(code, value)| Some((code.as_ref()?.clone(), value?)))
        .collect();
    let selected = ["open", "high", "low", "close", "volume", "minute_count"].map(str::to_owned);
    let bars = derive_stock_minute_bars_selected(&table, date, 5, &selected)?;
    let mut days = Day::new();
    for bar in bars {
        let Some(&factor) = factors.get(&bar.ts_code) else {
            continue;
        };
        if bar.minute_count != 5
            || !factor.is_finite()
            || factor <= 0.0
            || bar.volume < 0.0
            || ![bar.open, bar.high, bar.low, bar.close, bar.volume]
                .iter()
                .all(|v| v.is_finite())
            || bar.low <= 0.0
            || bar.high < bar.open.max(bar.close)
            || bar.low > bar.open.min(bar.close)
        {
            continue;
        }
        let range = bar.high - bar.low;
        let body = if range > 0.0 {
            (bar.close - bar.open) / range
        } else {
            0.0
        };
        let upper = (bar.high - bar.open.max(bar.close)) * factor;
        let lower = (bar.open.min(bar.close) - bar.low) * factor;
        days.entry(bar.ts_code).or_default().push([
            (bar.close * factor).ln(),
            body,
            upper,
            lower,
            bar.volume,
        ]);
    }
    days.retain(|_, bars| bars.len() == SLOTS);
    Ok(days)
}

fn moments(bars: &[[f64; 5]], axis: usize) -> (f64, f64) {
    let mut mean = 0.0;
    let mut m2 = 0.0;
    for (i, bar) in bars.iter().enumerate() {
        let delta = bar[axis] - mean;
        mean += delta / (i + 1) as f64;
        m2 += delta * (bar[axis] - mean);
    }
    (mean, m2.max(0.0))
}

fn increment(
    previous: &[f64; 5],
    current: &[f64; 5],
    upper_scale: f64,
    lower_scale: f64,
    volume_scale: f64,
) -> [f64; 14] {
    // Global lead-lag: first move all six lead coordinates, then lagged price.
    [
        TIME_SCALE,
        current[0] - previous[0],
        current[1] - previous[1],
        (current[2] - previous[2]) / upper_scale,
        (current[3] - previous[3]) / lower_scale,
        current[4] / volume_scale,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        current[0] - previous[0],
    ]
}

pub fn day_features(
    kernel: &PathLogSignature,
    bars: &[[f64; 5]],
    previous: Option<&[f64; 5]>,
) -> Result<(Vec<f64>, Vec<f64>, [f64; 5])> {
    if bars.len() != SLOTS {
        return Err(err("Logsig-RNN day must contain 48 complete bars"));
    }
    let (upper_mean, upper_m2) = moments(bars, 2);
    let (lower_mean, lower_m2) = moments(bars, 3);
    let volume = bars.iter().map(|bar| bar[4]).sum::<f64>();
    let positive = |value: f64| if value > 0.0 { value } else { 1.0 };
    let upper_scale = positive((upper_m2 / SLOTS as f64).sqrt());
    let lower_scale = positive((lower_m2 / SLOTS as f64).sqrt());
    let volume_scale = positive(volume);
    let increments: Vec<_> = bars
        .windows(2)
        .flat_map(|pair| increment(&pair[0], &pair[1], upper_scale, lower_scale, volume_scale))
        .collect();
    let intra_signature = kernel.signature(&increments)?;
    let intra = kernel.project(&intra_signature);
    let bridge = if let Some(previous) = previous {
        let first = kernel.signature(&increment(
            previous,
            &bars[0],
            upper_scale,
            lower_scale,
            volume_scale,
        ))?;
        kernel.project(&kernel.concatenate(&first, &intra_signature))
    } else {
        vec![f64::NAN; WIDTH]
    };
    if !intra.iter().all(|v| (*v as f32).is_finite())
        || (previous.is_some() && !bridge.iter().all(|v| (*v as f32).is_finite()))
    {
        return Err(err("nonfinite Logsig-RNN coefficients"));
    }
    Ok((
        intra,
        bridge,
        [upper_mean, upper_m2, lower_mean, lower_m2, volume],
    ))
}

fn one_date(
    config: &EngineConfig,
    kernel: &PathLogSignature,
    date: i32,
    previous_date: Option<i32>,
    overwrite: bool,
) -> Result<Option<usize>> {
    let path = output_path(&config.data_root, date);
    if path.storage_exists() && !overwrite {
        return Ok(None);
    }
    let loader = MarketDataLoader::new(DataCatalog::new(config.data_root.clone()));
    let day = read_day(&loader, date)?;
    let catalog = DataCatalog::new(config.data_root.clone());
    let previous = previous_date
        .filter(|&date| {
            catalog
                .minute_file(DatasetId::StockMinute1m, date)
                .is_some()
        })
        .map(|date| read_day(&loader, date))
        .transpose()?;
    let mut codes = Vec::with_capacity(day.len());
    let mut features = Vec::with_capacity(day.len());
    for (code, bars) in &day {
        let last = previous
            .as_ref()
            .and_then(|day| day.get(code))
            .and_then(|bars| bars.last());
        features.push(day_features(kernel, bars, last)?);
        codes.push(code.clone());
    }
    let count = codes.len();
    let mut columns = BTreeMap::from([
        (
            "trade_date".into(),
            ColumnData::I32(vec![Some(date); count]),
        ),
        (
            "ts_code".into(),
            ColumnData::Utf8(codes.into_iter().map(Some).collect()),
        ),
        (
            "definition_version".into(),
            ColumnData::I32(vec![Some(1); count]),
        ),
    ]);
    for i in 0..WIDTH {
        for (name, bridge) in [("intra", false), ("bridge", true)] {
            columns.insert(
                format!("{name}_{:04}", i + 1),
                ColumnData::F32(
                    features
                        .iter()
                        .map(|(a, b, _)| {
                            let value = if bridge { b[i] } else { a[i] } as f32;
                            value.is_finite().then_some(value)
                        })
                        .collect(),
                ),
            );
        }
    }
    for (i, name) in [
        "upper_mean",
        "upper_m2",
        "lower_mean",
        "lower_m2",
        "volume_sum",
    ]
    .iter()
    .enumerate()
    {
        columns.insert(
            name.to_string(),
            ColumnData::F64(
                features
                    .iter()
                    .map(|(_, _, stats)| Some(stats[i]))
                    .collect(),
            ),
        );
    }
    let table = Table::new(columns)?;
    let temp = path.with_extension(format!("{}.tmp", std::process::id()));
    let result =
        write_derived_parquet(&temp, &table).and_then(|_| crate::artifact_io::rename(&temp, &path));
    if result.is_err() {
        let _ = crate::artifact_io::remove_file(&temp);
    }
    result?;
    Ok(Some(count))
}

pub fn derive(config: &EngineConfig, request: &LogsigRnnRequest) -> Result<()> {
    if request.start_date > request.end_date || request.date_batch_size == 0 || request.threads == 0
    {
        return Err(err("invalid Logsig-RNN dates, date batch size or threads"));
    }
    let calendar = TradingCalendar::load(&config.data_root, &config.stock_calendar_exchange)?;
    let dates = calendar.open_dates_between(request.start_date, request.end_date);
    let kernel = PathLogSignature::new(7, 4)?;
    if kernel.log_width() != WIDTH {
        return Err(err("Logsig-RNN Lyndon width mismatch"));
    }
    let pool = crate::resources::thread_pool(Some(request.threads))?;
    let progress = ProgressBar::new("derive-logsigrnn", dates.len(), true);
    for batch in dates.chunks(request.date_batch_size) {
        let results = pool.install(|| {
            batch
                .par_iter()
                .map(|&date| {
                    let previous = calendar.warmup_start(date, 1);
                    one_date(
                        config,
                        &kernel,
                        date,
                        (previous < date).then_some(previous),
                        request.overwrite,
                    )
                    .map(|rows| (date, rows))
                })
                .collect::<Vec<_>>()
        });
        for result in results {
            let (date, rows) = result?;
            progress.tick(match rows {
                Some(n) => format!("date={date} rows={n}"),
                None => format!("date={date} skipped existing"),
            });
        }
    }
    progress.finish();
    Ok(())
}

//! Strict 40-session return, turnover and volatility inputs for analyst GAT.
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::path::{Path, PathBuf};
use std::time::Instant;

use rayon::prelude::*;

use crate::artifact_io::StoragePathExt;
use crate::calendar::TradingCalendar;
use crate::config::EngineConfig;
use crate::core::DatasetId;
use crate::data::parquet_io::parquet_column_names;
use crate::data::{ColumnData, DataCatalog, MarketDataLoader, Table};
use crate::derive::storage::write_derived_parquet;
use crate::error::{err, Result};

pub const VERSION: &str = "analyst_gat_inputs_v1";
const LOOKBACK: usize = 40;

#[derive(Clone, Debug)]
pub struct AnalystGatInputsRequest {
    pub start_date: i32,
    pub end_date: i32,
    pub date_batch_size: usize,
    pub threads: Option<usize>,
}

fn manifest_text() -> String {
    format!(
        "{{\"version\":\"{VERSION}\",\"lookback_days\":40,\
        \"return\":\"adjusted_close_t_over_t_minus_40_minus_one\",\
        \"adjustment\":\"close_times_adj_factor\",\
        \"turnover\":\"mean_turnover_rate_f_percent_last_40\",\
        \"volatility\":\"std_simple_adjusted_close_returns_40_ddof0\",\
        \"missing\":\"strict_41_closes_and_last_40_turnover\"}}\n"
    )
}

#[derive(Clone, Copy)]
struct Observation {
    close: Option<f64>,
    turnover: Option<f64>,
}

#[derive(Default)]
struct StockWindow {
    observations: VecDeque<Observation>,
}

impl StockWindow {
    fn advance(&mut self, observation: Observation) {
        self.observations.push_back(observation);
        if self.observations.len() > LOOKBACK + 1 {
            self.observations.pop_front();
        }
    }

    fn values(&self) -> Option<[f64; 3]> {
        if self.observations.len() != LOOKBACK + 1 {
            return None;
        }
        let first = self.observations.front()?.close?;
        let mut previous = first;
        let mut turnover_mean = 0.0;
        let mut return_mean = 0.0;
        let mut return_m2 = 0.0;
        // The anchor day's turnover is deliberately not inspected.
        for (index, observation) in self.observations.iter().skip(1).enumerate() {
            let close = observation.close?;
            let rate = observation.turnover?;
            let daily_return = close / previous - 1.0;
            if !daily_return.is_finite() {
                return None;
            }
            let count = (index + 1) as f64;
            turnover_mean += (rate - turnover_mean) / count;
            // Welford moments avoid subtracting two nearly equal squared sums.
            let delta = daily_return - return_mean;
            return_mean += delta / count;
            return_m2 += delta * (daily_return - return_mean);
            previous = close;
        }
        let cumulative_return = previous / first - 1.0;
        if !return_m2.is_finite() {
            return None;
        }
        let volatility = (return_m2.max(0.0) / LOOKBACK as f64).sqrt();
        let values = [cumulative_return, turnover_mean, volatility];
        values
            .iter()
            .all(|value| value.is_finite())
            .then_some(values)
    }
}

type NumericDays = BTreeMap<i32, BTreeMap<String, Option<f64>>>;

fn numeric_days(table: &Table, column: &str, dataset: DatasetId) -> Result<NumericDays> {
    let dataset = dataset.as_str();
    let dates = table.required_i32_date_cast("trade_date")?;
    let symbols = table.required_utf8("ts_code")?;
    let values = table.required_f64_cast(column)?;
    let mut output: NumericDays = BTreeMap::new();
    for index in 0..table.len {
        let date = dates[index].ok_or_else(|| err(format!("null {dataset} trade_date")))?;
        let symbol = symbols[index]
            .as_deref()
            .filter(|symbol| !symbol.trim().is_empty())
            .ok_or_else(|| err(format!("null/empty {dataset} ts_code on {date}")))?;
        if output
            .entry(date)
            .or_default()
            .insert(symbol.to_string(), values[index])
            .is_some()
        {
            return Err(err(format!(
                "duplicate {dataset} row: date={date} ts_code={symbol}"
            )));
        }
    }
    Ok(output)
}

fn validate_source(path: &Path, columns: &[String], checked: &mut BTreeSet<PathBuf>) -> Result<()> {
    if !checked.contains(path) {
        let available = parquet_column_names(path)?;
        for column in columns {
            if !available.contains(column) {
                return Err(err(format!(
                    "missing required analyst GAT source column {column}: {}",
                    path.display()
                )));
            }
        }
        checked.insert(path.to_path_buf());
    }
    Ok(())
}

fn load_days(
    loader: &MarketDataLoader,
    catalog: &DataCatalog,
    dataset: DatasetId,
    column: &str,
    dates: &[i32],
    checked: &mut BTreeSet<PathBuf>,
) -> Result<NumericDays> {
    let dataset_name = dataset.as_str();
    let columns = vec!["trade_date".into(), "ts_code".into(), column.into()];
    for &date in dates {
        if let Some(path) = catalog.daily_date_file(dataset, date) {
            validate_source(&path, &columns, checked)?;
        } else {
            let files = catalog.daily_year_files(dataset, date, date);
            if files.is_empty() {
                return Err(err(format!(
                    "missing analyst GAT source {dataset_name} on {date}"
                )));
            }
            for path in files {
                validate_source(&path, &columns, checked)?;
            }
        }
    }
    let table = loader.load_daily_by_dates(dataset, &[column.to_string()], dates)?;
    let output = numeric_days(&table, column, dataset)?;
    for date in dates {
        if !output.contains_key(date) {
            return Err(err(format!(
                "missing/empty analyst GAT source {dataset_name} on {date}"
            )));
        }
    }
    Ok(output)
}

fn output_table(
    date: i32,
    prices: &BTreeMap<String, Option<f64>>,
    states: &BTreeMap<String, StockWindow>,
) -> Result<Table> {
    let mut returns = Vec::with_capacity(prices.len());
    let mut turnover = Vec::with_capacity(prices.len());
    let mut volatility = Vec::with_capacity(prices.len());
    for symbol in prices.keys() {
        let values = states.get(symbol).and_then(StockWindow::values);
        returns.push(values.map(|values| values[0]));
        turnover.push(values.map(|values| values[1]));
        volatility.push(values.map(|values| values[2]));
    }
    Table::new(BTreeMap::from([
        (
            "trade_date".into(),
            ColumnData::I32(vec![Some(date); prices.len()]),
        ),
        (
            "ts_code".into(),
            ColumnData::Utf8(prices.keys().cloned().map(Some).collect()),
        ),
        ("return_40d".into(), ColumnData::F64(returns)),
        ("turnover_mean_40d".into(), ColumnData::F64(turnover)),
        ("volatility_40d".into(), ColumnData::F64(volatility)),
    ]))
}

pub fn derive_analyst_gat_inputs(
    config: &EngineConfig,
    request: &AnalystGatInputsRequest,
) -> Result<()> {
    if request.start_date > request.end_date || request.date_batch_size == 0 {
        return Err(err(
            "invalid analyst GAT inputs date range or date-batch-size",
        ));
    }
    let root = config.data_root.join("derived/stock/analyst_gat_inputs");
    let absolute = if root.is_absolute() {
        root.clone()
    } else {
        std::env::current_dir()?.join(&root)
    };
    let spelling = absolute
        .to_string_lossy()
        .replace('\\', "/")
        .to_ascii_lowercase();
    if spelling.starts_with("c:") || spelling.starts_with("//?/c:") {
        return Err(err("analyst GAT input output on C: is prohibited"));
    }
    let _guard = crate::artifact_io::lock(&root.with_extension("lock"))?;
    let manifest = root.join("manifest.json");
    let expected = manifest_text();
    let has_manifest = manifest.storage_exists();
    if has_manifest {
        if crate::artifact_io::read_to_string(&manifest)?.trim() != expected.trim() {
            return Err(err(
                "analyst GAT input identity changed; use a clean derived directory",
            ));
        }
    } else if root.storage_exists() && crate::artifact_io::read_dir(&root)?.next().is_some() {
        return Err(err(
            "analyst GAT input directory is nonempty without an identity manifest",
        ));
    }
    let calendar = TradingCalendar::load(&config.data_root, &config.stock_calendar_exchange)?;
    let targets = calendar.open_dates_between(request.start_date, request.end_date);
    let first = *targets
        .first()
        .ok_or_else(|| err("no target trading dates for analyst GAT inputs"))?;
    let warmup = calendar
        .open_dates_between(calendar.warmup_start(first, LOOKBACK), first)
        .into_iter()
        .filter(|date| *date < first)
        .collect::<Vec<_>>();
    let first_source = warmup.first().copied().unwrap_or(first);
    let catalog = DataCatalog::new(config.data_root.clone());
    let loader = MarketDataLoader::new(catalog.clone());
    let pool = crate::resources::thread_pool(request.threads)?;
    if !has_manifest {
        crate::artifact_io::create_dir_all(&root)?;
        let temporary = manifest.with_extension(format!("{}.tmp", std::process::id()));
        crate::artifact_io::write(&temporary, expected)?;
        crate::artifact_io::rename(temporary, &manifest)?;
    }
    let started = Instant::now();
    let mut checked = BTreeSet::new();
    let mut states: BTreeMap<String, StockWindow> = BTreeMap::new();
    let mut processed = 0;
    for (batch_index, batch) in warmup
        .chunks(request.date_batch_size)
        .chain(targets.chunks(request.date_batch_size))
        .enumerate()
    {
        let batch_started = Instant::now();
        let mut price_days = load_days(
            &loader,
            &catalog,
            DatasetId::StockDailyPv,
            "close",
            batch,
            &mut checked,
        )?;
        let mut adjustment_days = load_days(
            &loader,
            &catalog,
            DatasetId::StockAdjFactor,
            "adj_factor",
            batch,
            &mut checked,
        )?;
        // The oldest close is an anchor only. Its turnover is never part of a
        // complete target window, so even its whole basic file is unnecessary.
        let basic_dates = batch
            .iter()
            .copied()
            .filter(|date| *date > first_source)
            .collect::<Vec<_>>();
        let mut turnover_days = if basic_dates.is_empty() {
            NumericDays::new()
        } else {
            load_days(
                &loader,
                &catalog,
                DatasetId::StockDailyBasic,
                "turnover_rate_f",
                &basic_dates,
                &mut checked,
            )?
        };
        println!(
            "derive-analyst-gat-inputs batch={} first={} last={} dates={} warmup={} batch_load_ms={:.1}",
            batch_index + 1,
            batch[0],
            batch[batch.len() - 1],
            batch.len(),
            batch[0] < first,
            batch_started.elapsed().as_secs_f64() * 1000.0
        );
        let mut outputs = Vec::new();
        for &date in batch {
            let prices = price_days
                .remove(&date)
                .ok_or_else(|| err("missing loaded price day"))?;
            let adjustments = adjustment_days
                .remove(&date)
                .ok_or_else(|| err("missing loaded adjustment day"))?;
            let turnovers = if date == first_source {
                BTreeMap::new()
            } else {
                turnover_days
                    .remove(&date)
                    .ok_or_else(|| err("missing loaded turnover day"))?
            };
            for symbol in prices.keys() {
                states.entry(symbol.clone()).or_default();
            }
            // Every known stock advances on EVERY calendar session, even if its
            // price row is absent. A gap cannot compress the historical window.
            for (symbol, state) in &mut states {
                let close = prices
                    .get(symbol)
                    .copied()
                    .flatten()
                    .filter(|value| value.is_finite() && *value > 0.0)
                    .and_then(|value| {
                        let adjustment = adjustments.get(symbol).copied().flatten()?;
                        if !adjustment.is_finite() || adjustment <= 0.0 {
                            return None;
                        }
                        let adjusted = value * adjustment;
                        (adjusted.is_finite() && adjusted > 0.0).then_some(adjusted)
                    });
                let turnover = prices
                    .contains_key(symbol)
                    .then(|| turnovers.get(symbol).copied().flatten())
                    .flatten()
                    .filter(|value| value.is_finite() && *value >= 0.0);
                state.advance(Observation { close, turnover });
            }
            if date >= first {
                outputs.push((date, output_table(date, &prices, &states)?));
            }
        }
        let results = pool.install(|| {
            outputs
                .par_iter()
                .map(|(date, table)| -> Result<usize> {
                    let path = root
                        .join((date / 10000).to_string())
                        .join(format!("{date}.parquet"));
                    write_derived_parquet(&path, table)?;
                    Ok(table.len)
                })
                .collect::<Vec<_>>()
        });
        for ((date, _), result) in outputs.iter().zip(results) {
            let rows = result?;
            processed += 1;
            println!(
                "derive-analyst-gat-inputs [{processed}/{}] date={date} rows={rows} elapsed_s={:.1}",
                targets.len(),
                started.elapsed().as_secs_f64()
            );
        }
    }
    Ok(())
}

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::path::{Path, PathBuf};

use crate::calendar::TradingCalendar;
use crate::config::EngineConfig;
use crate::core::{AssetClass, DatasetId};
use crate::data::{ColumnData, DataCatalog, MarketDataLoader, Table};
use crate::derive::bar::minute_index;
use crate::derive::request::DeriveLogsigRequest;
use crate::derive::storage::write_derived_parquet;
use crate::error::{err, Result};
use crate::logsig_signature::logsig_signature_batch_in_pool;
use crate::progress::ProgressBar;

pub const LOOKBACK: usize = 20;
pub const BAR_SIZE: usize = 5;
pub const ORDER: usize = 10;
pub const WIDTH: usize = 226;
const SLOTS: usize = 240 / BAR_SIZE;
type DayVolume = BTreeMap<String, Vec<f64>>;

#[derive(Clone, Debug, Default)]
pub struct DeriveLogsigReport {
    pub output_files: Vec<PathBuf>,
    pub processed_dates: usize,
    pub missing_input_dates: Vec<i32>,
    pub skipped_existing_dates: Vec<i32>,
    pub total_rows: usize,
}

pub fn derived_logsig_path(root: &Path, date: i32) -> PathBuf {
    root.join("derived/stock/logsig_v")
        .join((date / 10000).to_string())
        .join(format!("{date}.parquet"))
}

/// Chronological summation matches derive-bar, without retaining full OHLCV rows.
fn volume_day(table: &Table) -> Result<DayVolume> {
    let codes = table.required_utf8("ts_code")?;
    let times = table.required_utf8("trade_time")?;
    let volumes = table.required_f64_cast("vol")?;
    let mut ids = HashMap::new();
    let mut symbols = Vec::new();
    let mut rows = Vec::with_capacity(table.len);
    for i in 0..table.len {
        let (Some(code), Some(time), Some(volume)) = (&codes[i], &times[i], volumes[i]) else {
            continue;
        };
        let Some(minute) = minute_index(time) else {
            continue;
        };
        if !volume.is_finite() {
            continue;
        }
        let id = *ids.entry(code.as_str()).or_insert_with(|| {
            symbols.push(code.as_str());
            symbols.len() - 1
        });
        rows.push((id, minute, volume));
    }
    rows.sort_by_key(|(id, minute, _)| (*id, *minute));
    let mut values = vec![vec![f64::NAN; SLOTS]; symbols.len()];
    for (id, minute, volume) in rows {
        let slot = &mut values[id][minute / BAR_SIZE];
        if slot.is_nan() {
            *slot = volume;
        } else {
            *slot += volume;
        }
    }
    Ok(symbols
        .into_iter()
        .zip(values)
        .filter(|(_, values)| values.iter().all(|v| v.is_finite()))
        .map(|(code, values)| (code.to_owned(), values))
        .collect())
}

#[derive(Default)]
struct VolumeWindow {
    days: VecDeque<DayVolume>,
}

impl VolumeWindow {
    fn push(&mut self, day: DayVolume) {
        if self.days.len() == LOOKBACK {
            self.days.pop_front();
        }
        self.days.push_back(day);
    }

    fn matrix(&self) -> (Vec<String>, Vec<f64>) {
        if self.days.len() != LOOKBACK {
            return (Vec::new(), Vec::new());
        }
        let codes: Vec<_> = self.days[0]
            .keys()
            .filter(|code| self.days.iter().all(|day| day.contains_key(*code)))
            .cloned()
            .collect();
        let mut values = Vec::with_capacity(codes.len() * LOOKBACK * SLOTS);
        for code in &codes {
            for day in &self.days {
                values.extend_from_slice(&day[code]);
            }
        }
        (codes, values)
    }
}

fn feature_table(date: i32, codes: Vec<String>, features: Vec<f32>) -> Result<Table> {
    let count = codes.len();
    if features.len() != count * WIDTH {
        return Err(err("invalid logsignature output shape"));
    }
    let mut columns = BTreeMap::new();
    columns.insert(
        "trade_date".into(),
        ColumnData::I32(vec![Some(date); count]),
    );
    columns.insert(
        "ts_code".into(),
        ColumnData::Utf8(codes.into_iter().map(Some).collect()),
    );
    for column in 0..WIDTH {
        columns.insert(
            format!("logsig_{:04}", column + 1),
            ColumnData::F32(
                (0..count)
                    .map(|row| Some(features[row * WIDTH + column]))
                    .collect(),
            ),
        );
    }
    Table::new(columns)
}

pub fn derive_logsig(
    config: &EngineConfig,
    request: &DeriveLogsigRequest,
) -> Result<DeriveLogsigReport> {
    if request.asset_class != AssetClass::Stock
        || request.start_date > request.end_date
        || request.threads == 0
    {
        return Err(err(
            "derive-logsig requires stock, start-date <= end-date and threads > 0",
        ));
    }
    let calendar = TradingCalendar::load(&config.data_root, &config.stock_calendar_exchange)?;
    let targets = calendar.open_dates_between(request.start_date, request.end_date);
    let mut report = DeriveLogsigReport::default();
    let Some(first) = targets.first() else {
        return Ok(report);
    };
    let dates = calendar.open_dates_between(
        calendar.warmup_start(*first, LOOKBACK - 1),
        request.end_date,
    );
    let loader = MarketDataLoader::new(DataCatalog::new(config.data_root.clone()));
    let columns = ["ts_code", "trade_time", "vol"].map(str::to_owned).to_vec();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(request.threads)
        .build()?;
    let mut window = VolumeWindow::default();
    let progress = ProgressBar::new("derive-logsig", targets.len(), true);
    for date in dates {
        let mut tables = loader.load_minute_by_date(DatasetId::StockMinute1m, &columns, &[date])?;
        let day = match tables.remove(&date) {
            Some(table) => volume_day(&table)?,
            None => {
                report.missing_input_dates.push(date);
                DayVolume::new()
            }
        };
        window.push(day);
        if date < *first {
            continue;
        }
        let path = derived_logsig_path(&config.data_root, date);
        if !request.overwrite && path.exists() {
            report.skipped_existing_dates.push(date);
            progress.tick(format!("date={date} skipped existing"));
            continue;
        }
        let (codes, volumes) = window.matrix();
        let count = codes.len();
        let features =
            logsig_signature_batch_in_pool(&volumes, count, LOOKBACK * SLOTS, ORDER, &pool)?;
        drop(volumes);
        let table = feature_table(date, codes, features)?;
        // Replace a completed date atomically; a failed write must not truncate existing data.
        let temp = path.with_extension(format!("{}.tmp", std::process::id()));
        let result: Result<()> = (|| {
            write_derived_parquet(&temp, &table)?;
            std::fs::rename(&temp, &path)?;
            Ok(())
        })();
        if result.is_err() {
            let _ = std::fs::remove_file(&temp);
        }
        result?;
        report.output_files.push(path);
        report.processed_dates += 1;
        report.total_rows += count;
        progress.tick(format!("date={date} rows={count}"));
    }
    progress.finish();
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logsig_window_is_bounded_aligned_and_missing_days_are_not_skipped() {
        let mut window = VolumeWindow::default();
        for _ in 0..LOOKBACK {
            window.push(BTreeMap::from([
                ("B".into(), vec![2.; SLOTS]),
                ("A".into(), vec![1.; SLOTS]),
            ]));
        }
        let (codes, matrix) = window.matrix();
        assert_eq!(codes, vec!["A", "B"]);
        assert_eq!(matrix.len(), 2 * LOOKBACK * SLOTS);
        assert!(matrix[..LOOKBACK * SLOTS].iter().all(|v| *v == 1.));
        window.push(DayVolume::new());
        assert_eq!(window.days.len(), LOOKBACK);
        assert!(window.matrix().0.is_empty());
        for _ in 0..LOOKBACK {
            window.push(BTreeMap::from([("B".into(), vec![2.; SLOTS])]));
        }
        assert_eq!(window.matrix().0, vec!["B"]);
    }

    #[test]
    fn logsig_projection_aggregation_matches_bar() {
        let mut codes = Vec::new();
        let mut times = Vec::new();
        let mut volumes = Vec::new();
        for minute in (0..240).rev() {
            let clock = if minute < 120 {
                571 + minute
            } else {
                781 + minute - 120
            };
            codes.push(Some("A".into()));
            times.push(Some(format!("{:02}:{:02}:00", clock / 60, clock % 60)));
            volumes.push(Some(minute as f64 + 1.));
        }
        codes.extend([Some("A".into()), Some("A".into()), None]);
        times.extend([
            Some("09:31:00".into()),
            Some("09:30:00".into()),
            Some("09:31:00".into()),
        ]);
        volumes.extend([Some(7.), Some(99999.), Some(99999.)]);
        let table = Table::new(BTreeMap::from([
            ("ts_code".into(), ColumnData::Utf8(codes)),
            ("trade_time".into(), ColumnData::Utf8(times)),
            ("vol".into(), ColumnData::F64(volumes)),
        ]))
        .unwrap();
        let day = volume_day(&table).unwrap();
        let bars = crate::derive::bar::derive_stock_minute_bars_selected(
            &table,
            20260105,
            BAR_SIZE,
            &["volume".into()],
        )
        .unwrap();
        assert_eq!(day["A"], bars.iter().map(|r| r.volume).collect::<Vec<_>>());
        assert_eq!(day["A"][0], 22.);
    }

    #[test]
    fn logsig_schema_and_path_are_fixed_even_for_empty_dates() {
        assert_eq!(
            feature_table(20260105, vec![], vec![])
                .unwrap()
                .columns
                .len(),
            WIDTH + 2
        );
        assert_eq!(
            derived_logsig_path(Path::new("data"), 20260105),
            PathBuf::from("data/derived/stock/logsig_v/2026/20260105.parquet")
        );
        assert_eq!(
            crate::logsig_signature::signature_width(ORDER).unwrap(),
            WIDTH
        );
    }
}

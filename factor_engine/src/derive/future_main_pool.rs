use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use crate::artifact_io::StoragePathExt;
use crate::calendar::TradingCalendar;
use crate::config::EngineConfig;
use crate::core::DatasetId;
use crate::data::{parquet_io::read_parquet, ColumnData, DataCatalog, MarketDataLoader, Table};
use crate::derive::storage::write_derived_parquet;
use crate::error::{err, Result};
use crate::factor::future::minute_input::{product, FutureMinuteInput, FutureMinuteOptions};

pub(crate) const DIRECTORY: &str = "future_data/basic/mapping_pool";
pub const POOL_ID: &str = "liquidity_001";
const POOL_README: &str = include_str!("../../docs/FUTURE_MAPPING_POOL_LIQUIDITY_001.md");
const LOOKBACK_DAYS: usize = 240;
const MIN_AVG_VOLUME: f64 = 10_000.0;
const TOP_PRODUCTS: usize = 50;

#[derive(Debug, Default)]
pub struct FutureMainPoolReport {
    pub files_written: usize,
    pub rows_written: usize,
    pub missing_target_dates: Vec<i32>,
    pub members_per_year: BTreeMap<i32, usize>,
}

pub fn derive(
    config: &EngineConfig,
    start_date: i32,
    end_date: i32,
    batch_size: usize,
) -> Result<FutureMainPoolReport> {
    if batch_size == 0 || start_date > end_date {
        return Err(err(
            "derive-future-main-pool requires a positive batch size and start <= end",
        ));
    }
    let calendar = TradingCalendar::load(&config.data_root, &config.future_calendar_exchange)?;
    let requested = calendar.open_dates_between(start_date, end_date);
    let loader = MarketDataLoader::new(DataCatalog::new(config.data_root.clone()));
    let output = config.data_root.join(DIRECTORY).join(POOL_ID);
    let mut report = FutureMainPoolReport::default();

    for target in requested.chunk_by(|left, right| left / 10_000 == right / 10_000) {
        let year = target[0] / 10_000;
        let cutoff = calendar
            .previous_open(year * 10_000 + 101)
            .filter(|date| date / 10_000 == year - 1)
            .ok_or_else(|| err(format!("missing previous year-end trading day for {year}")))?;
        let start = calendar.warmup_start(cutoff, LOOKBACK_DAYS - 1);
        let history = calendar.open_dates_between(start, cutoff);
        if history.len() != LOOKBACK_DAYS {
            return Err(err(format!(
                "need {LOOKBACK_DAYS} calendar sessions before {year}"
            )));
        }
        let mut dates = history.clone();
        dates.extend_from_slice(target);
        let mapping =
            FutureMinuteInput::load(FutureMinuteOptions::default(), &config.data_root, &dates)?;
        let members = choose_members(&loader, &mapping, &history, batch_size)?;
        report.members_per_year.insert(year, members.len());
        let covered: BTreeSet<i32> = target
            .iter()
            .copied()
            .filter(|&date| mapping.selection(date).is_some())
            .collect();
        report.missing_target_dates.extend(
            target
                .iter()
                .copied()
                .filter(|date| !covered.contains(date)),
        );
        if covered.is_empty() {
            continue;
        }
        let table = pool_rows(&mapping, &covered, &members)?;
        write_year(&output.join(format!("{year}.parquet")), &table, &covered)?;
        report.files_written += 1;
        report.rows_written += table.len;
    }
    if report.files_written > 0 {
        crate::artifact_io::write(output.join("README.md"), POOL_README)?;
    }
    Ok(report)
}

fn choose_members(
    loader: &MarketDataLoader,
    mapping: &FutureMinuteInput,
    history: &[i32],
    batch_size: usize,
) -> Result<BTreeSet<String>> {
    let mut totals = BTreeMap::<String, (usize, f64)>::new();
    let columns = ["vol".to_owned()];
    for batch in history.chunks(batch_size) {
        let daily = loader.load_daily_by_dates(DatasetId::FutureDaily, &columns, batch)?;
        if daily.len == 0 {
            continue;
        }
        let dates = daily.required_i32_date_cast("trade_date")?;
        let codes = daily.required_utf8("ts_code")?;
        let volumes = daily.required_f64_cast("vol")?;
        let mut seen = BTreeSet::new();
        for ((date, code), volume) in dates.iter().zip(codes).zip(volumes) {
            let (Some(date), Some(code)) = (*date, code) else {
                continue;
            };
            let Some(key) = product(code) else {
                continue;
            };
            if mapping.selection(date).and_then(|day| day.get(&key)) != Some(code) {
                continue;
            }
            if !seen.insert((date, key.clone())) {
                return Err(err(format!(
                    "duplicate main contract daily volume for {code} on {date}"
                )));
            }
            if let Some(volume) = volume.filter(|v| v.is_finite() && *v >= 0.0) {
                let (count, total) = totals.entry(key).or_default();
                *count += 1;
                *total += volume;
            }
        }
    }
    let mut ranked = totals
        .into_iter()
        // A complete main-contract sequence also establishes 240 sessions of listing history.
        .filter(|(_, (count, total))| {
            *count == LOOKBACK_DAYS
                && total.is_finite()
                && *total / LOOKBACK_DAYS as f64 >= MIN_AVG_VOLUME
        })
        .collect::<Vec<_>>();
    ranked.sort_unstable_by(|(left, (_, left_vol)), (right, (_, right_vol))| {
        right_vol.total_cmp(left_vol).then_with(|| left.cmp(right))
    });
    Ok(ranked
        .into_iter()
        .take(TOP_PRODUCTS)
        .map(|(code, _)| code)
        .collect())
}

fn pool_rows(
    mapping: &FutureMinuteInput,
    dates: &BTreeSet<i32>,
    members: &BTreeSet<String>,
) -> Result<Table> {
    let mut keys = Vec::new();
    let mut days = Vec::new();
    let mut contracts = Vec::new();
    for &date in dates {
        for (key, contract) in mapping.selection(date).into_iter().flatten() {
            if members.contains(key) {
                keys.push(Some(key.clone()));
                days.push(Some(date.to_string()));
                contracts.push(Some(contract.clone()));
            }
        }
    }
    let mut table = Table::empty();
    table.insert("ts_code", ColumnData::Utf8(keys))?;
    table.insert("trade_date", ColumnData::Utf8(days))?;
    table.insert("mapping_ts_code", ColumnData::Utf8(contracts))?;
    Ok(table)
}

fn write_year(path: &Path, updates: &Table, replaced: &BTreeSet<i32>) -> Result<()> {
    let _guard = crate::artifact_io::lock(path)?;
    if !path.storage_exists() {
        return write_derived_parquet(path, updates);
    }
    let old = read_parquet(path, None)?;
    let retained = old
        .required_i32_date_cast("trade_date")?
        .iter()
        .enumerate()
        .filter_map(|(index, date)| date.filter(|day| !replaced.contains(day)).map(|_| index))
        .collect::<Vec<_>>();
    let mut merged = old.take(&retained)?;
    merged.append(updates)?;
    let dates = merged.required_i32_date_cast("trade_date")?;
    let codes = merged.required_utf8("ts_code")?;
    let mut order = (0..merged.len).collect::<Vec<_>>();
    order.sort_unstable_by(|&left, &right| {
        dates[left]
            .cmp(&dates[right])
            .then_with(|| codes[left].cmp(&codes[right]))
    });
    write_derived_parquet(path, &merged.take(&order)?)
}

use std::collections::{BTreeMap, BTreeSet};

use crate::artifact_io::StoragePathExt;
use crate::calendar::TradingCalendar;
use crate::config::EngineConfig;
use crate::core::DatasetId;
use crate::data::{
    parquet_io::{read_parquet, write_parquet},
    ColumnData, DataCatalog, MarketDataLoader, Table,
};
use crate::error::{err, Result};
use crate::factor::future::minute_input::product;

pub(crate) const DIRECTORY: &str = "future_data/basic/mapping_oi";

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct Contract {
    maturity: i32,
    code: String,
}

type DayInterest = BTreeMap<String, BTreeMap<Contract, f64>>;
type Selection = BTreeMap<String, (Contract, f64)>;

pub fn derive(
    config: &EngineConfig,
    start_date: i32,
    end_date: i32,
    batch_size: usize,
    include_next_day: bool,
) -> Result<usize> {
    if batch_size == 0 || start_date > end_date {
        return Err(err(
            "derive-future-main-oi requires a positive batch size and start <= end",
        ));
    }
    let calendar = TradingCalendar::load(&config.data_root, &config.future_calendar_exchange)?;
    let mut dates = calendar.open_dates_between(0, end_date);
    if include_next_day {
        dates.extend(calendar.first_open_on_or_after(end_date + 1));
    }
    let loader = MarketDataLoader::new(DataCatalog::new(config.data_root.clone()));
    let mut state = BTreeMap::new();
    let mut previous = None;
    let mut written = 0;
    let mut year = 0;
    let mut pending = Table::empty();
    // Replay from the first local snapshot. Only the requested dates are written.
    for batch in dates.chunks(batch_size) {
        let table = loader.load_daily_by_dates(DatasetId::FutureDaily, &["oi".into()], batch)?;
        let mut daily = collect_interest(&table)?;
        for &date in batch {
            let today = daily.remove(&date).unwrap_or_default();
            let first = state.is_empty() && !today.is_empty();
            let next = if first {
                select(&mut state, today.clone())
            } else {
                Selection::new()
            };
            if date >= start_date {
                let (source, mapping) = match (&previous, first) {
                    (_, true) => (date, Some(&next)),
                    (Some((source, selected)), false) => (*source, Some(selected)),
                    (None, false) => (date, None),
                };
                if date / 10_000 != year {
                    if pending.len > 0 {
                        write_year(config, year, &pending)?;
                        written += 1;
                        pending = Table::empty();
                    }
                    year = date / 10_000;
                }
                pending.append(&day_table(date, source, mapping)?)?;
            }
            previous = Some((
                date,
                if first {
                    next
                } else {
                    select(&mut state, today)
                },
            ));
        }
    }
    if pending.len > 0 {
        write_year(config, year, &pending)?;
        written += 1;
    }
    Ok(written)
}

fn collect_interest(table: &Table) -> Result<BTreeMap<i32, DayInterest>> {
    let mut daily = BTreeMap::<i32, DayInterest>::new();
    if table.len == 0 {
        return Ok(daily);
    }
    let dates = table.required_i32_date_cast("trade_date")?;
    let codes = table.required_utf8("ts_code")?;
    let interest = table.required_f64_cast("oi")?;
    for idx in 0..table.len {
        let (Some(date), Some(code), Some(oi)) = (dates[idx], &codes[idx], interest[idx]) else {
            continue;
        };
        let Some(key) = product(code).filter(|_| oi.is_finite() && oi >= 0.0) else {
            continue;
        };
        let symbol = code.split_once('.').expect("validated contract").0;
        let digits = symbol.trim_start_matches(|ch: char| ch.is_ascii_alphabetic());
        let number = digits.parse::<i32>()?;
        let year = date / 10_000;
        let cycle = if digits.len() == 3 { 10 } else { 100 };
        let contract_year =
            year - cycle / 2 + (number / 100 - (year - cycle / 2)).rem_euclid(cycle);
        let contract = Contract {
            maturity: contract_year * 100 + number % 100,
            code: code.clone(),
        };
        let score = daily
            .entry(date)
            .or_default()
            .entry(key)
            .or_default()
            .entry(contract)
            .or_insert(oi);
        *score = score.max(oi);
    }
    Ok(daily)
}

fn select(state: &mut BTreeMap<String, Contract>, today: DayInterest) -> Selection {
    let mut selected = Selection::new();
    for (key, contracts) in today {
        let old = state.get(&key);
        let old_oi = old.and_then(|code| contracts.get(code)).copied();
        let best = contracts
            .iter()
            .filter(|(candidate, oi)| {
                old.is_none_or(|old| {
                    candidate.maturity > old.maturity
                        && old_oi
                            .is_none_or(|current| **oi > current && **oi * 10.0 >= current * 11.0)
                })
            })
            .max_by(|(left, left_oi), (right, right_oi)| {
                left_oi.total_cmp(right_oi).then_with(|| right.cmp(left))
            });
        if let Some((contract, &oi)) = best {
            state.insert(key.clone(), contract.clone());
            selected.insert(key, (contract.clone(), oi));
        } else if let (Some(contract), Some(oi)) = (old, old_oi) {
            selected.insert(key, (contract.clone(), oi));
        }
    }
    selected
}

fn day_table(date: i32, source: i32, mapping: Option<&Selection>) -> Result<Table> {
    let mut keys = Vec::new();
    let mut codes = Vec::new();
    let mut interest = Vec::new();
    for (key, (contract, oi)) in mapping.into_iter().flatten() {
        keys.push(Some(key.clone()));
        codes.push(Some(contract.code.clone()));
        interest.push(Some(*oi));
    }
    // Keep a date marker so an empty recalculation overrides old daily data.
    if keys.is_empty() {
        keys.push(None);
        codes.push(None);
        interest.push(None);
    }
    let len = keys.len();
    let mut table = Table::empty();
    table.insert("ts_code", ColumnData::Utf8(keys))?;
    table.insert("trade_date", ColumnData::I32(vec![Some(date); len]))?;
    table.insert("mapping_ts_code", ColumnData::Utf8(codes))?;
    table.insert(
        "source_trade_date",
        ColumnData::I32(vec![Some(source); len]),
    )?;
    table.insert("oi", ColumnData::F64(interest))?;
    Ok(table)
}

fn write_year(config: &EngineConfig, year: i32, pending: &Table) -> Result<()> {
    let path = config
        .data_root
        .join(DIRECTORY)
        .join(format!("{year}.parquet"));
    let _guard = crate::artifact_io::lock(&path)?;
    if !path.storage_exists() {
        return write_parquet(&path, pending);
    }
    let old = read_parquet(&path, None)?;
    let replaced = pending
        .required_i32("trade_date")?
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let retained = old
        .required_i32("trade_date")?
        .iter()
        .enumerate()
        .filter_map(|(idx, date)| (!replaced.contains(date)).then_some(idx))
        .collect::<Vec<_>>();
    let mut merged = old.take(&retained)?;
    merged.append(pending)?;
    let dates = merged.required_i32("trade_date")?;
    let codes = merged.required_utf8("ts_code")?;
    let mut order = (0..merged.len).collect::<Vec<_>>();
    order.sort_unstable_by(|&a, &b| {
        dates[a]
            .cmp(&dates[b])
            .then_with(|| codes[a].cmp(&codes[b]))
    });
    write_parquet(&path, &merged.take(&order)?)
}

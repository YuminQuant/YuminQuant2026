//! Complete canonical event input -> daily decayed event-count features.
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::path::Path;
use std::time::Instant;

use rayon::prelude::*;
use serde_json::{json, Value};

use crate::artifact_io::StoragePathExt;
use crate::calendar::TradingCalendar;
use crate::config::EngineConfig;
use crate::core::DatasetId;
use crate::data::parquet_io::{parquet_column_names, read_parquet};
use crate::data::{ColumnData, DataCatalog, MarketDataLoader, Table};
use crate::derive::storage::write_derived_parquet;
use crate::error::{err, Result};
use crate::factor::common::financial::add_months;

pub const SOURCE_VERSION: &str = "stock_events_source_v1";
pub const VERSION: &str = "stock_events_v1";
pub const EVENT_TYPES: [&str; 28] = [
    "earnings_profit",
    "equity_incentive",
    "major_contract",
    "fiscal_subsidy",
    "shareholder_increase",
    "litigation_win",
    "negative_rumor_clarification",
    "dishonest_execution_removed",
    "issuer_rating_upgrade",
    "earnings_loss_warning",
    "enforcement_subject",
    "shareholder_reduction_plan",
    "equity_freeze",
    "regulatory_rectification",
    "litigation_defendant",
    "dishonest_execution",
    "issuer_rating_downgrade",
    "exchange_inquiry",
    "abnormal_auditor_change",
    "seasoned_offering_termination",
    "survey_securities_asset_management",
    "survey_securities_proprietary",
    "survey_fund",
    "survey_insurance",
    "survey_investment",
    "survey_foreign",
    "survey_securities",
    "survey_other",
];

#[derive(Clone, Debug)]
pub struct StockEventsRequest {
    pub start_date: i32,
    pub end_date: i32,
    pub date_batch_size: usize,
    pub threads: Option<usize>,
}

struct SourceManifest {
    start: i32,
    end: i32,
    revision: String,
}

fn date_value(value: &Value, key: &str) -> Result<i32> {
    let date = value
        .get(key)
        .and_then(Value::as_i64)
        .and_then(|date| i32::try_from(date).ok())
        .filter(|date| *date >= 10_000_101 && *date <= 99_991_231)
        .ok_or_else(|| {
            err(format!(
                "event manifest {key} must be an eight-digit date integer"
            ))
        })?;
    // Verify the calendar date without accepting arbitrary YYYYMMDD-shaped integers.
    let month = date / 100 % 100;
    let day = date % 100;
    let year = date / 10000;
    let leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
    let days = match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            if leap {
                29
            } else {
                28
            }
        }
        _ => 0,
    };
    if day < 1 || day > days {
        return Err(err(format!(
            "invalid event manifest calendar date {key}={date}"
        )));
    }
    Ok(date)
}

fn source_manifest(path: &Path) -> Result<SourceManifest> {
    if !path.storage_exists() {
        return Err(err(format!(
            "missing complete 28-category stock event source manifest: {}",
            path.display()
        )));
    }
    let value: Value = serde_json::from_str(&crate::artifact_io::read_to_string(path)?)?;
    let object = value
        .as_object()
        .ok_or_else(|| err("stock event source manifest must be a JSON object"))?;
    let allowed = [
        "version",
        "event_types",
        "coverage_start",
        "coverage_end",
        "complete",
        "availability",
        "universe",
        "source_revision",
    ];
    if object.keys().any(|key| !allowed.contains(&key.as_str())) {
        return Err(err("unknown stock event source manifest field"));
    }
    if value.get("version") != Some(&json!(SOURCE_VERSION))
        || value.get("event_types") != Some(&json!(EVENT_TYPES))
        || value.get("complete") != Some(&json!(true))
        || value.get("availability") != Some(&json!("explicit_first_available_trading_day"))
        || value.get("universe") != Some(&json!("all_a_shares"))
    {
        return Err(err("stock event source must explicitly declare complete, ordered 28-category all-A-share coverage"));
    }
    let start = date_value(&value, "coverage_start")?;
    let end = date_value(&value, "coverage_end")?;
    let revision = value
        .get("source_revision")
        .and_then(Value::as_str)
        .filter(|revision| !revision.trim().is_empty())
        .ok_or_else(|| err("stock event source_revision must be a nonempty string"))?
        .to_string();
    if start > end {
        return Err(err(
            "stock event source coverage_start exceeds coverage_end",
        ));
    }
    Ok(SourceManifest {
        start,
        end,
        revision,
    })
}

fn derived_manifest(source: &SourceManifest) -> Value {
    let columns = EVENT_TYPES
        .iter()
        .map(|event| format!("event_{event}"))
        .collect::<Vec<_>>();
    json!({
        "version": VERSION, "event_types": EVENT_TYPES, "columns": columns,
        "window_months": 3, "half_life_trading_days": 20,
        "availability": "explicit_first_available_trading_day",
        "source_version": SOURCE_VERSION, "source_universe": "all_a_shares",
        "source_complete": true, "source_revision": source.revision,
        "source_coverage_start": source.start, "source_coverage_end": source.end,
        "row_universe": "daily_pv_union_active_event_stocks",
        "missing_neighbor": "zero_only_under_complete_event_coverage"
    })
}

fn persist_identity(root: &Path, expected: &Value) -> Result<()> {
    let path = root.join("manifest.json");
    if path.storage_exists() {
        let previous: Value = serde_json::from_str(&crate::artifact_io::read_to_string(&path)?)?;
        let old_start = date_value(&previous, "source_coverage_start")?;
        let old_end = date_value(&previous, "source_coverage_end")?;
        let new_start = date_value(expected, "source_coverage_start")?;
        let new_end = date_value(expected, "source_coverage_end")?;
        let mut old_semantics = previous;
        let mut new_semantics = expected.clone();
        for value in [&mut old_semantics, &mut new_semantics] {
            let fields = value
                .as_object_mut()
                .ok_or_else(|| err("derived event identity must be a JSON object"))?;
            fields.remove("source_coverage_start");
            fields.remove("source_coverage_end");
        }
        if old_semantics != new_semantics || new_start > old_start || new_end < old_end {
            return Err(err("stock event identity/revision changed or complete coverage shrank; rebuild in a clean directory"));
        }
    } else if root.storage_exists() && crate::artifact_io::read_dir(root)?.next().is_some() {
        return Err(err(
            "stock event derived directory is nonempty without an identity manifest",
        ));
    }
    crate::artifact_io::create_dir_all(root)?;
    let temporary = path.with_extension(format!("{}.tmp", std::process::id()));
    crate::artifact_io::write(&temporary, serde_json::to_vec_pretty(expected)?)?;
    crate::artifact_io::rename(temporary, path)?;
    Ok(())
}

#[derive(Clone)]
struct Event {
    date: i32,
    position: usize,
    stock: String,
    kind: usize,
}

struct YearEvents {
    events: Vec<Event>,
    cursor: usize,
}

type SeenEvents = BTreeMap<(String, String), (i32, String, usize)>;

fn required_text(values: &[Option<String>], index: usize, name: &str) -> Result<String> {
    values[index]
        .as_deref()
        .filter(|value| !value.trim().is_empty())
        .map(str::to_string)
        .ok_or_else(|| err(format!("null/empty canonical stock event {name}")))
}

fn load_event_year(
    path: &Path,
    year: i32,
    positions: &BTreeMap<i32, usize>,
    lower: i32,
    upper: i32,
    seen: &mut SeenEvents,
) -> Result<YearEvents> {
    let columns = [
        "event_id",
        "ts_code",
        "available_date",
        "event_type",
        "source",
    ]
    .iter()
    .map(|name| name.to_string())
    .collect::<Vec<_>>();
    let available_columns = parquet_column_names(path)?;
    if columns
        .iter()
        .any(|column| !available_columns.contains(column))
    {
        return Err(err(format!(
            "incomplete canonical stock event schema: {}",
            path.display()
        )));
    }
    let table = read_parquet(path, Some(&columns))?;
    let ids = table.required_utf8("event_id")?;
    let stocks = table.required_utf8("ts_code")?;
    let dates = table.required_i32_date_cast("available_date")?;
    let kinds = table.required_utf8("event_type")?;
    let sources = table.required_utf8("source")?;
    let mut events = Vec::new();
    for index in 0..table.len {
        let date = dates[index].ok_or_else(|| err("null canonical stock event available_date"))?;
        if date / 10000 != year {
            return Err(err(format!(
                "stock event partition {year} contains available_date {date}"
            )));
        }
        if date <= lower || date > upper {
            continue;
        }
        let position = *positions.get(&date).ok_or_else(|| {
            err(format!(
                "stock event available_date is not a known trading session: {date}"
            ))
        })?;
        let id = required_text(ids, index, "event_id")?;
        let stock = required_text(stocks, index, "ts_code")?;
        let source = required_text(sources, index, "source")?;
        let kind_text = required_text(kinds, index, "event_type")?;
        let kind = EVENT_TYPES
            .iter()
            .position(|value| *value == kind_text)
            .ok_or_else(|| err(format!("unknown canonical stock event type: {kind_text}")))?;
        let signature = (date, stock.clone(), kind);
        if let Some(previous) = seen.get(&(source.clone(), id.clone())) {
            if previous != &signature {
                return Err(err(format!(
                    "conflicting stock event ID: source={source} event_id={id}"
                )));
            }
            continue;
        }
        seen.insert((source, id), signature);
        events.push(Event {
            date,
            position,
            stock,
            kind,
        });
    }
    events.sort_by(|left, right| {
        (left.date, &left.stock, left.kind).cmp(&(right.date, &right.stock, right.kind))
    });
    Ok(YearEvents { events, cursor: 0 })
}

#[derive(Default)]
struct EventState {
    values: [f64; 28],
    counts: [usize; 28],
}

fn decay(age: usize) -> f64 {
    2.0_f64.powf(-(age as f64) / 20.0)
}

fn daily_prices(
    loader: &MarketDataLoader,
    catalog: &DataCatalog,
    dates: &[i32],
) -> Result<BTreeMap<i32, BTreeSet<String>>> {
    for &date in dates {
        if catalog
            .daily_date_file(DatasetId::StockDailyPv, date)
            .is_none()
            && catalog
                .daily_year_files(DatasetId::StockDailyPv, date, date)
                .is_empty()
        {
            return Err(err(format!(
                "missing daily PV universe for event feature date {date}"
            )));
        }
    }
    let table = loader.load_daily_by_dates(DatasetId::StockDailyPv, &[], dates)?;
    let stock_dates = table.required_i32_date_cast("trade_date")?;
    let symbols = table.required_utf8("ts_code")?;
    let mut output: BTreeMap<i32, BTreeSet<String>> = BTreeMap::new();
    for index in 0..table.len {
        let date = stock_dates[index].ok_or_else(|| err("null daily PV date"))?;
        let stock = required_text(symbols, index, "daily PV ts_code")?;
        if !output.entry(date).or_default().insert(stock) {
            return Err(err(format!("duplicate daily PV stock/date: {date}")));
        }
    }
    if dates.iter().any(|date| !output.contains_key(date)) {
        return Err(err(
            "missing/empty daily PV partition for event feature output",
        ));
    }
    Ok(output)
}

fn feature_table(
    date: i32,
    prices: &BTreeSet<String>,
    states: &BTreeMap<String, EventState>,
) -> Result<Table> {
    let symbols = prices
        .iter()
        .cloned()
        .chain(states.keys().cloned())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let mut columns = BTreeMap::from([
        (
            "trade_date".into(),
            ColumnData::I32(vec![Some(date); symbols.len()]),
        ),
        (
            "ts_code".into(),
            ColumnData::Utf8(symbols.iter().cloned().map(Some).collect()),
        ),
        (
            "is_daily_pv".into(),
            ColumnData::Bool(
                symbols
                    .iter()
                    .map(|stock| Some(prices.contains(stock)))
                    .collect(),
            ),
        ),
    ]);
    for (kind, name) in EVENT_TYPES.iter().enumerate() {
        columns.insert(
            format!("event_{name}"),
            ColumnData::F64(
                symbols
                    .iter()
                    .map(|stock| {
                        Some(
                            states
                                .get(stock)
                                .map(|state| state.values[kind])
                                .unwrap_or(0.0),
                        )
                    })
                    .collect(),
            ),
        );
    }
    Table::new(columns)
}

pub fn derive_stock_events(config: &EngineConfig, request: &StockEventsRequest) -> Result<()> {
    if request.start_date > request.end_date || request.date_batch_size == 0 {
        return Err(err("invalid stock event date range or date-batch-size"));
    }
    let source_root = config.data_root.join("chn_stock_data/stock_events");
    let source = source_manifest(&source_root.join("manifest.json"))?;
    let calendar = TradingCalendar::load(&config.data_root, &config.stock_calendar_exchange)?;
    let dates = calendar.open_dates_between(request.start_date, request.end_date);
    let first = *dates
        .first()
        .ok_or_else(|| err("no target trading sessions for stock events"))?;
    let last = *dates.last().unwrap();
    let lower = add_months(first, -3);
    if source.start > lower || source.end < last {
        return Err(err("canonical stock event complete coverage must include the full initial three-month window and all targets"));
    }
    for year in lower / 10000..=last / 10000 {
        let path = source_root.join(format!("{year}.parquet"));
        if !path.storage_exists() {
            return Err(err(format!(
                "missing canonical stock event year {year}: {}",
                path.display()
            )));
        }
    }
    let positions = calendar
        .open_dates_between(i32::MIN, i32::MAX)
        .into_iter()
        .enumerate()
        .map(|(position, date)| (date, position))
        .collect::<BTreeMap<_, _>>();
    let root = config.data_root.join("derived/stock/stock_events");
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
        return Err(err("stock event derived output on C: is prohibited"));
    }
    let _guard = crate::artifact_io::lock(&root.with_extension("lock"))?;
    persist_identity(&root, &derived_manifest(&source))?;
    let catalog = DataCatalog::new(config.data_root.clone());
    let loader = MarketDataLoader::new(catalog.clone());
    let pool = crate::resources::thread_pool(request.threads)?;
    let mut years: BTreeMap<i32, YearEvents> = BTreeMap::new();
    let mut seen = SeenEvents::new();
    let mut active: VecDeque<Event> = VecDeque::new();
    let mut states: BTreeMap<String, EventState> = BTreeMap::new();
    let mut previous_position = None;
    let started = Instant::now();
    let mut processed = 0;
    for batch in dates.chunks(request.date_batch_size) {
        let prices = daily_prices(&loader, &catalog, batch)?;
        let mut outputs = Vec::new();
        for &date in batch {
            let position = positions[&date];
            if let Some(previous) = previous_position {
                let multiplier = decay(position - previous);
                for state in states.values_mut() {
                    for value in &mut state.values {
                        *value *= multiplier;
                    }
                }
            }
            previous_position = Some(position);
            let cutoff = add_months(date, -3);
            while active.front().is_some_and(|event| event.date <= cutoff) {
                let event = active.pop_front().unwrap();
                let state = states
                    .get_mut(&event.stock)
                    .ok_or_else(|| err("event expiry state missing"))?;
                state.counts[event.kind] -= 1;
                state.values[event.kind] = if state.counts[event.kind] == 0 {
                    0.0
                } else {
                    (state.values[event.kind] - decay(position - event.position)).max(0.0)
                };
            }
            states.retain(|_, state| state.counts.iter().any(|count| *count != 0));
            years.retain(|year, _| *year >= cutoff / 10000);
            for year in cutoff / 10000..=date / 10000 {
                if !years.contains_key(&year) {
                    let path = source_root.join(format!("{year}.parquet"));
                    let events = load_event_year(&path, year, &positions, lower, last, &mut seen)?;
                    println!(
                        "derive-stock-events load_year={year} unique_events={} elapsed_s={:.1}",
                        events.events.len(),
                        started.elapsed().as_secs_f64()
                    );
                    years.insert(year, events);
                }
                let rows = years.get_mut(&year).unwrap();
                while rows.cursor < rows.events.len() && rows.events[rows.cursor].date <= date {
                    let event = rows.events[rows.cursor].clone();
                    rows.cursor += 1;
                    if event.date <= cutoff {
                        continue;
                    }
                    let state = states.entry(event.stock.clone()).or_default();
                    state.values[event.kind] += decay(position - event.position);
                    state.counts[event.kind] += 1;
                    active.push_back(event);
                }
            }
            outputs.push((date, feature_table(date, &prices[&date], &states)?));
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
                "derive-stock-events [{processed}/{}] date={date} rows={rows} elapsed_s={:.1}",
                dates.len(),
                started.elapsed().as_secs_f64()
            );
        }
    }
    Ok(())
}

//! DFQ-HIST main-business industry concepts with report-prescribed availability.
use crate::artifact_io::StoragePathExt;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::time::Instant;

use rayon::prelude::*;

use crate::calendar::TradingCalendar;
use crate::config::EngineConfig;
use crate::data::parquet_io::read_parquet;
use crate::data::{ColumnData, Table};
use crate::derive::storage::write_derived_parquet;
use crate::error::{err, Result};

pub const VERSION: &str = "hist_mainbz_i_v1";
const EXCLUDED: &[&str] = &[
    "合计",
    "总计",
    "其他",
    "其它",
    "其他业务",
    "其它业务",
    "其他行业",
    "其它行业",
    "行业合计",
    "主营业务合计",
    "营业收入合计",
    "未分配",
    "抵销",
    "抵消",
    "分部间抵销",
    "分部间抵消",
];

fn manifest_text() -> String {
    let excluded = EXCLUDED
        .iter()
        .map(|item| format!("\"{item}\""))
        .collect::<Vec<_>>()
        .join(",");
    format!(
        "{{\"version\":\"{VERSION}\",\"source_type\":\"I\",\
        \"availability\":\"annual_next_may01_interim_sep01\",\
        \"membership\":\"latest_visible_period_per_stock\",\
        \"normalization\":\"remove_whitespace_exact_names\",\
        \"excluded_items\":[{excluded}]}}\n"
    )
}

#[derive(Clone, Debug)]
pub struct HistConceptRequest {
    pub start_date: i32,
    pub end_date: i32,
    pub date_batch_size: usize,
}

#[derive(Clone)]
struct Disclosure {
    available: i32,
    period: i32,
    stock: String,
    concepts: BTreeSet<String>,
}

fn events(path: &Path, end: i32) -> Result<Vec<Disclosure>> {
    let columns = ["ts_code", "end_date", "bz_type", "bz_item"]
        .map(str::to_string)
        .to_vec();
    let table = read_parquet(path, Some(&columns))?;
    let stocks = table.required_utf8("ts_code")?;
    let periods = table.required_i32_date_cast("end_date")?;
    let kinds = table.required_utf8("bz_type")?;
    let items = table.required_utf8("bz_item")?;
    let mut groups: BTreeMap<(i32, i32, String), BTreeSet<String>> = BTreeMap::new();
    for i in 0..table.len {
        if kinds[i].as_deref().map(str::trim) != Some("I") {
            continue;
        }
        let period = periods[i].ok_or_else(|| err("mainbz industry row has no end_date"))?;
        let year = period / 10000;
        let available = match period % 10000 {
            1231 => (year + 1) * 10000 + 501,
            630 => year * 10000 + 901,
            _ => continue,
        };
        if available > end {
            continue;
        }
        let stock = stocks[i]
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .ok_or_else(|| err("mainbz industry row has no ts_code"))?;
        // Create an empty update too: an all-generic new disclosure clears the
        // preceding concepts rather than silently retaining that stale list.
        let group = groups
            .entry((available, period, stock.to_string()))
            .or_default();
        if let Some(item) = &items[i] {
            let item: String = item.chars().filter(|c| !c.is_whitespace()).collect();
            if !item.is_empty() && !EXCLUDED.contains(&item.as_str()) {
                group.insert(format!("I:{item}"));
            }
        }
    }
    Ok(groups
        .into_iter()
        .map(|((available, period, stock), concepts)| Disclosure {
            available,
            period,
            stock,
            concepts,
        })
        .collect())
}

fn snapshot(date: i32, state: &BTreeMap<String, Disclosure>) -> Result<Table> {
    let mut stocks = Vec::new();
    let mut concepts = Vec::new();
    let mut periods = Vec::new();
    let mut available = Vec::new();
    for event in state.values() {
        for concept in &event.concepts {
            stocks.push(Some(event.stock.clone()));
            concepts.push(Some(concept.clone()));
            periods.push(Some(event.period));
            available.push(Some(event.available));
        }
    }
    Table::new(BTreeMap::from([
        (
            "trade_date".into(),
            ColumnData::I32(vec![Some(date); stocks.len()]),
        ),
        ("ts_code".into(), ColumnData::Utf8(stocks)),
        ("concept_id".into(), ColumnData::Utf8(concepts)),
        ("end_date".into(), ColumnData::I32(periods)),
        ("available_date".into(), ColumnData::I32(available)),
    ]))
}

pub fn derive_hist_concepts(config: &EngineConfig, request: &HistConceptRequest) -> Result<()> {
    if request.start_date > request.end_date || request.date_batch_size == 0 {
        return Err(err("invalid HIST concept date range or date-batch-size"));
    }
    let calendar = TradingCalendar::load(&config.data_root, &config.stock_calendar_exchange)?;
    let dates = calendar.open_dates_between(request.start_date, request.end_date);
    if dates.is_empty() {
        return Err(err("no trading dates for HIST concepts"));
    }
    let source = config.data_root.join("chn_stock_data/mainbz");
    let mut files = BTreeMap::new();
    for entry in crate::artifact_io::read_dir(&source)? {
        let path = entry?.path();
        if path.extension().is_some_and(|v| v == "parquet") {
            if let Some(year) = path
                .file_stem()
                .and_then(|v| v.to_str())
                .and_then(|v| v.parse::<i32>().ok())
            {
                files.insert(year, path);
            }
        }
    }
    let first_year = *files
        .keys()
        .next()
        .ok_or_else(|| err("no mainbz source files"))?;
    // Before September, this year's interim results cannot be visible.
    let end = *dates.last().unwrap();
    let last_year = end / 10000 - if end % 10000 < 901 { 1 } else { 0 };
    let mut disclosures = Vec::new();
    let started = Instant::now();
    for year in first_year..=last_year {
        let path = files
            .get(&year)
            .ok_or_else(|| err(format!("missing mainbz year {year}")))?;
        disclosures.extend(events(path, end)?);
        println!(
            "derive-hist-concepts load_year={year} disclosures={} elapsed_s={:.1}",
            disclosures.len(),
            started.elapsed().as_secs_f64()
        );
    }
    disclosures
        .sort_by(|a, b| (a.available, &a.stock, a.period).cmp(&(b.available, &b.stock, b.period)));
    let root = config.data_root.join("derived/stock/hist_concepts");
    let manifest = root.join("manifest.json");
    if manifest.storage_exists() {
        if crate::artifact_io::read_to_string(&manifest)?.trim() != manifest_text().trim() {
            return Err(err(
                "HIST concept version changed; use a clean derived directory",
            ));
        }
    }
    let workers = std::thread::available_parallelism()
        .map_or(1, |v| v.get())
        .min(4);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .build()?;
    let mut state = BTreeMap::new();
    let mut cursor = 0;
    let mut processed = 0;
    for batch in dates.chunks(request.date_batch_size) {
        let mut outputs = Vec::new();
        for &date in batch {
            while cursor < disclosures.len() && disclosures[cursor].available <= date {
                let event = &disclosures[cursor];
                state.insert(event.stock.clone(), event.clone());
                cursor += 1;
            }
            let table = snapshot(date, &state)?;
            if table.len == 0 {
                return Err(err(format!(
                    "no visible HIST concepts on {date}; restore earlier mainbz history"
                )));
            }
            outputs.push((date, table));
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
                "derive-hist-concepts [{processed}/{}] date={date} rows={rows} elapsed_s={:.1}",
                dates.len(),
                started.elapsed().as_secs_f64()
            );
        }
    }
    let temporary = manifest.with_extension(format!("{}.tmp", std::process::id()));
    crate::artifact_io::write(&temporary, manifest_text())?;
    crate::artifact_io::rename(temporary, manifest)?;
    Ok(())
}

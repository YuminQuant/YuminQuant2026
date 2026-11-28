//! DFQ-HIST main-business industry concepts with report-prescribed availability.
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use arrow_array::{Int32Array, RecordBatch};

use crate::calendar::TradingCalendar;
use crate::config::EngineConfig;
use crate::data::parquet_io::{table_record_batch, PreparedProjection};
use crate::data::read_ahead::{configure_threads, visit_tables, write_files};
use crate::data::{ColumnData, Table};
use crate::derive::storage::{check_derived_manifest, write_derived_daily_batch, yearly_files};
use crate::error::{err, Result};
use crate::progress::ProgressBar;

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
    pub threads: Option<usize>,
}

#[derive(Clone)]
struct Disclosure {
    available: i32,
    period: i32,
    stock: String,
    concepts: BTreeSet<String>,
}

fn events(table: Table, partition_year: i32, end: i32) -> Result<Vec<Disclosure>> {
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
        if year != partition_year {
            return Err(err(format!(
                "mainbz report period {period} is outside partition {partition_year}"
            )));
        }
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
    let files = yearly_files(&source)?;
    let first_year = *files.keys().next().unwrap();
    // Before September, this year's interim results cannot be visible.
    let end = *dates.last().unwrap();
    let last_year = end / 10000 - if end % 10000 < 901 { 1 } else { 0 };
    let years = (first_year..=last_year).collect::<Vec<_>>();
    let paths = years
        .iter()
        .map(|year| {
            files
                .get(year)
                .cloned()
                .ok_or_else(|| err(format!("missing mainbz year {year}")))
        })
        .collect::<Result<Vec<_>>>()?;
    if paths.is_empty() {
        return Err(err(format!(
            "no visible HIST concepts on {}; restore earlier mainbz history",
            dates[0]
        )));
    }
    let root = config.data_root.join("derived/stock/hist_concepts");
    let manifest = root.join("manifest.json");
    let expected = manifest_text();
    let has_manifest = check_derived_manifest(&manifest, &expected)?;
    configure_threads(crate::resources::compute_threads(request.threads)?);
    let progress = ProgressBar::new("derive-hist-concepts", dates.len(), true);
    let started = Instant::now();
    let columns = ["ts_code", "end_date", "bz_type", "bz_item"].map(str::to_owned);
    let mut state = BTreeMap::new();
    let mut date_cursor = 0;
    let mut template = None;
    visit_tables(
        &paths,
        |index, input| PreparedProjection::new(input, &paths[index], &columns, &BTreeSet::new()),
        |index, table| events(table, years[index], end),
        |index, events| {
            let year = years[index];
            progress.update(format!(
                "load_year={year} disclosures={} elapsed_s={:.1}",
                events.len(),
                started.elapsed().as_secs_f64()
            ));
            let mut events = events.into_iter().peekable();
            // Annual Y is visible in May Y+1, before any period in the next file.
            let next_file_available = (year + 1) * 10000 + 901;
            let year_end = date_cursor
                + dates[date_cursor..].partition_point(|date| *date < next_file_available);
            for batch in dates[date_cursor..year_end].chunks(request.date_batch_size) {
                let mut outputs = Vec::with_capacity(batch.len());
                for &date in batch {
                    while events.peek().is_some_and(|event| event.available <= date) {
                        let event = events.next().unwrap();
                        state.insert(event.stock.clone(), event);
                        template = None;
                    }
                    if template.is_none() {
                        let table = snapshot(date, &state)?;
                        if table.len == 0 {
                            return Err(err(format!("no visible HIST concepts on {date}; restore earlier mainbz history")));
                        }
                        template = Some(table_record_batch(&table)?);
                    }
                    // Cloning a batch shares the immutable concept columns.
                    outputs.push((date, template.as_ref().unwrap().clone()));
                }
                let outcomes = write_files(&root, outputs, |(date, batch)| {
                    write_snapshot(&root, date, batch)
                })?;
                for (date, rows) in outcomes {
                    progress.tick(format!(
                        "date={date} rows={rows} elapsed_s={:.1}",
                        started.elapsed().as_secs_f64()
                    ));
                }
            }
            for event in events {
                state.insert(event.stock.clone(), event);
                template = None;
            }
            date_cursor = year_end;
            Ok(())
        },
    )?;
    if !has_manifest {
        crate::artifact_io::write(manifest, expected)?;
    }
    progress.finish();
    Ok(())
}

fn write_snapshot(root: &Path, date: i32, template: RecordBatch) -> Result<(i32, usize)> {
    let rows = template.num_rows();
    let date_pos = template.schema().index_of("trade_date")?;
    let mut columns = template.columns().to_vec();
    columns[date_pos] = Arc::new(Int32Array::from_value(date, rows));
    let batch = RecordBatch::try_new(template.schema(), columns)?;
    write_derived_daily_batch(root, date, &batch)?;
    Ok((date, rows))
}

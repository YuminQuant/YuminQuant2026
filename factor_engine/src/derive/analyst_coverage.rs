//! Quarter-end analyst memberships, independent of industry and fund sources.
use std::collections::BTreeMap;
use std::time::Instant;

use rayon::prelude::*;

use crate::artifact_io::StoragePathExt;
use crate::calendar::TradingCalendar;
use crate::config::EngineConfig;
use crate::derive::stock_relations::{
    analyst_events, membership_table, yearly_files, AnalystEvent, Memberships,
};
use crate::derive::storage::write_derived_parquet;
use crate::error::{err, Result};
use crate::factor::common::financial::add_months;

pub const VERSION: &str = "analyst_coverage_v1";

#[derive(Clone, Debug)]
pub struct AnalystCoverageRequest {
    pub start_date: i32,
    pub end_date: i32,
    pub date_batch_size: usize,
    pub window_months: i32,
    pub threads: Option<usize>,
}

fn manifest_text(window_months: i32) -> String {
    format!(
        "{{\"version\":\"{VERSION}\",\"relations\":[\"analyst\"],\
        \"window_months\":{window_months},\"snapshot_frequency\":\"quarterly_end\",\
        \"snapshot_anchor\":\"last_open_session\",\
        \"availability\":\"report_date_lte_snapshot\",\
        \"analyst_identity\":\"institution_split_author\"}}\n"
    )
}

fn quarter(date: i32) -> (i32, i32) {
    (date / 10000, ((date / 100) % 100 - 1) / 3)
}

fn snapshot_dates(calendar: &TradingCalendar, first: i32, last: i32) -> Result<Vec<i32>> {
    // Inspect the entire calendar: a later quarter confirms a completed boundary.
    // The requested range's final open session is never itself sufficient proof.
    let dates = calendar.open_dates_between(i32::MIN, i32::MAX);
    let completed = dates
        .windows(2)
        .filter_map(|pair| (quarter(pair[0]) != quarter(pair[1])).then_some(pair[0]))
        .collect::<Vec<_>>();
    let previous = completed
        .iter()
        .rev()
        .copied()
        .find(|date| *date < first)
        .ok_or_else(|| {
            err("analyst coverage requires a completed quarter before the first request date")
        })?;
    let mut selected = vec![previous];
    selected.extend(
        completed
            .into_iter()
            .filter(|date| *date >= first && *date <= last),
    );
    Ok(selected)
}

pub fn derive_analyst_coverage(
    config: &EngineConfig,
    request: &AnalystCoverageRequest,
) -> Result<()> {
    if request.start_date > request.end_date
        || request.date_batch_size == 0
        || request.window_months <= 0
    {
        return Err(err(
            "invalid analyst coverage date range, date-batch-size or window-months",
        ));
    }
    let calendar = TradingCalendar::load(&config.data_root, &config.stock_calendar_exchange)?;
    let dates = calendar.open_dates_between(request.start_date, request.end_date);
    let first = dates
        .first()
        .copied()
        .ok_or_else(|| err("no trading dates for analyst coverage"))?;
    let snapshots = snapshot_dates(&calendar, first, request.end_date)?;
    let earliest = snapshots[0];
    let earliest_month = earliest / 10000 * 12 + (earliest / 100) % 100 - 1;
    if request.window_months > earliest_month - 12 {
        return Err(err("analyst window extends before year 1"));
    }
    let first_year = add_months(earliest, -request.window_months) / 10000;
    let last_year = snapshots[snapshots.len() - 1] / 10000;
    let files = yearly_files(&config.data_root.join("chn_stock_data/analyst_report"))?;
    for year in first_year..=last_year {
        if !files.contains_key(&year) {
            return Err(err(format!("missing analyst report year {year}")));
        }
    }
    let root = config.data_root.join("derived/stock/analyst_coverage");
    let _guard = crate::artifact_io::lock(&root.with_extension("lock"))?;
    let manifest = root.join("manifest.json");
    let expected = manifest_text(request.window_months);
    let has_manifest = manifest.storage_exists();
    if has_manifest {
        if crate::artifact_io::read_to_string(&manifest)?.trim() != expected.trim() {
            return Err(err(
                "analyst coverage identity changed; use a clean derived directory",
            ));
        }
    } else if root.storage_exists() && crate::artifact_io::read_dir(&root)?.next().is_some() {
        return Err(err(
            "analyst coverage directory is nonempty but has no identity manifest",
        ));
    }
    let pool = crate::resources::thread_pool(request.threads)?;
    if !has_manifest {
        // Persist identity before the first snapshot, so an interrupted run cannot
        // later mix different windows into an unversioned partial dataset.
        crate::artifact_io::create_dir_all(&root)?;
        let temporary = manifest.with_extension(format!("{}.tmp", std::process::id()));
        crate::artifact_io::write(&temporary, expected)?;
        crate::artifact_io::rename(temporary, &manifest)?;
    }
    let started = Instant::now();
    let mut loaded: BTreeMap<i32, Vec<AnalystEvent>> = BTreeMap::new();
    let mut processed = 0;
    for batch in snapshots.chunks(request.date_batch_size) {
        let mut outputs = Vec::new();
        for &date in batch {
            let cutoff = add_months(date, -request.window_months);
            let cutoff_year = cutoff / 10000;
            loaded.retain(|year, _| *year >= cutoff_year);
            for year in cutoff_year..=date / 10000 {
                if !loaded.contains_key(&year) {
                    let events = analyst_events(&files[&year])?;
                    println!(
                        "derive-analyst-coverage load_year={year} events={} elapsed_s={:.1}",
                        events.len(),
                        started.elapsed().as_secs_f64()
                    );
                    loaded.insert(year, events);
                }
            }
            let mut rows = Memberships::new();
            for events in loaded.values() {
                for event in events
                    .iter()
                    .filter(|event| event.date > cutoff && event.date <= date)
                {
                    rows.insert(("analyst".into(), event.group.clone(), event.stock.clone()));
                }
            }
            // Empty coverage is an explicit empty snapshot, never stale carry-forward.
            outputs.push((date, membership_table(date, rows)?));
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
                "derive-analyst-coverage [{processed}/{}] snapshot={date} membership_rows={rows} elapsed_s={:.1}",
                snapshots.len(),
                started.elapsed().as_secs_f64()
            );
        }
    }
    Ok(())
}

//! Announcement-date stock/group memberships. Pairwise adjacency is never materialized.
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use rayon::prelude::*;

use crate::calendar::TradingCalendar;
use crate::config::EngineConfig;
use crate::data::parquet_io::read_parquet;
use crate::data::{ColumnData, Table};
use crate::derive::storage::write_derived_parquet;
use crate::error::{err, Result};
use crate::factor::common::cross_section::{ClassificationLevel, ClassificationMap};
use crate::factor::common::financial::add_months;

pub const VERSION: &str = "stock_relations_v1";

#[derive(Clone, Debug)]
pub struct StockRelationsRequest {
    pub start_date: i32,
    pub end_date: i32,
    pub date_batch_size: usize,
}

#[derive(Clone, Debug)]
struct FundEvent {
    fund: String,
    announced: i32,
    period: i32,
    stocks: Vec<String>,
}

type FundState = BTreeMap<String, FundEvent>;
type Memberships = BTreeSet<(String, String, String)>;

fn projected(path: &Path, columns: &[&str]) -> Result<Table> {
    if !path.exists() {
        return Err(err(format!("missing relation source: {}", path.display())));
    }
    read_parquet(
        path,
        Some(&columns.iter().map(|s| s.to_string()).collect::<Vec<_>>()),
    )
}

fn yearly_files(root: &Path) -> Result<BTreeMap<i32, PathBuf>> {
    if !root.exists() {
        return Err(err(format!(
            "missing relation source directory: {}",
            root.display()
        )));
    }
    let mut files = BTreeMap::new();
    for entry in std::fs::read_dir(root)? {
        let path = entry?.path();
        if path.extension().is_some_and(|s| s == "parquet") {
            if let Some(year) = path
                .file_stem()
                .and_then(|s| s.to_str())
                .and_then(|s| s.parse::<i32>().ok())
            {
                files.insert(year, path);
            }
        }
    }
    if files.is_empty() {
        return Err(err(format!(
            "no annual relation sources under {}; restore raw data first",
            root.display()
        )));
    }
    Ok(files)
}

fn text(values: &[Option<String>], row: usize) -> Result<String> {
    values[row]
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .ok_or_else(|| err(format!("missing relation key at row {row}")))
}

fn stock_code(raw: &str) -> String {
    if raw.len() == 6 && raw.bytes().all(|b| b.is_ascii_digit()) {
        let suffix = if raw.starts_with('6') {
            "SH"
        } else if raw.starts_with(['0', '3']) {
            "SZ"
        } else {
            "BJ"
        };
        format!("{raw}.{suffix}")
    } else {
        raw.to_uppercase()
    }
}

fn fund_events(path: &Path) -> Result<Vec<FundEvent>> {
    let table = projected(path, &["ts_code", "ann_date", "end_date", "symbol", "mkv"])?;
    let funds = table.required_utf8("ts_code")?;
    let symbols = table.required_utf8("symbol")?;
    let announced = table.required_i32_date_cast("ann_date")?;
    let periods = table.required_i32_date_cast("end_date")?;
    let values = table.required_f64_cast("mkv")?;
    let mut groups: BTreeMap<(i32, String, i32), BTreeMap<String, f64>> = BTreeMap::new();
    for i in 0..table.len {
        let ann = announced[i].ok_or_else(|| err("fund announcement date is missing"))?;
        let period = periods[i].ok_or_else(|| err("fund report period is missing"))?;
        if ann < period {
            return Err(err("fund announcement precedes report period"));
        }
        let fund = text(funds, i)?;
        let group = groups.entry((ann, fund, period)).or_default();
        let Some(value) = values[i].filter(|v| v.is_finite() && *v > 0.0) else {
            continue;
        };
        let stock = stock_code(&text(symbols, i)?);
        if let Some(previous) = group.insert(stock, value) {
            if previous != value {
                return Err(err("conflicting duplicate fund holding"));
            }
        }
    }
    Ok(groups
        .into_iter()
        .map(|((announced, fund, period), holdings)| {
            let mut holdings = holdings.into_iter().collect::<Vec<_>>();
            holdings.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            FundEvent {
                fund,
                announced,
                period,
                stocks: holdings.into_iter().take(10).map(|(s, _)| s).collect(),
            }
        })
        .collect())
}

fn apply_fund(state: &mut FundState, event: &FundEvent) {
    if state
        .get(&event.fund)
        .is_none_or(|old| (event.period, event.announced) >= (old.period, old.announced))
    {
        state.insert(event.fund.clone(), event.clone());
    }
}

#[derive(Clone)]
struct AnalystEvent {
    date: i32,
    group: String,
    stock: String,
}

fn analyst_events(path: &Path) -> Result<Vec<AnalystEvent>> {
    let table = projected(path, &["ts_code", "report_date", "org_name", "author_name"])?;
    let stocks = table.required_utf8("ts_code")?;
    let dates = table.required_i32_date_cast("report_date")?;
    let orgs = table.required_utf8("org_name")?;
    let authors = table.required_utf8("author_name")?;
    let mut unique = BTreeSet::new();
    for i in 0..table.len {
        let Some(date) = dates[i] else { continue };
        let Some(org) = orgs[i].as_deref().map(str::trim).filter(|s| !s.is_empty()) else {
            continue;
        };
        let Some(author) = authors[i].as_deref() else {
            continue;
        };
        let stock = text(stocks, i)?;
        for name in author.split([',', '，', '、', ';', '；', '/', '|']) {
            let name = name.trim();
            if name.is_empty() || ["nan", "None", "--", "佚名"].contains(&name) {
                continue;
            }
            // Length prefix avoids collisions between institution and analyst names.
            unique.insert((date, format!("{}:{org}{name}", org.len()), stock.clone()));
        }
    }
    Ok(unique
        .into_iter()
        .map(|(date, group, stock)| AnalystEvent { date, group, stock })
        .collect())
}

fn membership_table(date: i32, rows: Memberships) -> Result<Table> {
    let mut relations = Vec::new();
    let mut groups = Vec::new();
    let mut stocks = Vec::new();
    for (relation, group, stock) in rows {
        relations.push(Some(relation));
        groups.push(Some(group));
        stocks.push(Some(stock));
    }
    Table::new(BTreeMap::from([
        (
            "trade_date".into(),
            ColumnData::I32(vec![Some(date); stocks.len()]),
        ),
        ("relation".into(), ColumnData::Utf8(relations)),
        ("group_id".into(), ColumnData::Utf8(groups)),
        ("ts_code".into(), ColumnData::Utf8(stocks)),
    ]))
}

pub fn derive_stock_relations(
    config: &EngineConfig,
    request: &StockRelationsRequest,
) -> Result<()> {
    if request.start_date > request.end_date || request.date_batch_size == 0 {
        return Err(err("invalid relation date range or date-batch-size"));
    }
    let fund_root = config.data_root.join("fund_data/portfolio");
    let fund_files = yearly_files(&fund_root)?;
    let analyst_files = yearly_files(&config.data_root.join("chn_stock_data/analyst_report"))?;
    let industry = projected(
        &config.stock_ci_classification_path,
        &["ts_code", "l1_code", "in_date", "out_date"],
    )?;
    let industry_map = ClassificationMap::from_table(&industry, ClassificationLevel::Sector)?;
    let industry_stocks = industry
        .required_utf8("ts_code")?
        .iter()
        .filter_map(|value| value.clone())
        .collect::<BTreeSet<_>>();
    let calendar = TradingCalendar::load(&config.data_root, &config.stock_calendar_exchange)?;
    let dates = calendar.open_dates_between(request.start_date, request.end_date);
    if dates.is_empty() {
        return Err(err("no trading dates in relation request"));
    }
    let first_year = dates[0] / 10000;
    let last_year = dates[dates.len() - 1] / 10000;
    // Missing annual partitions must not silently turn a relation into an empty graph.
    for year in first_year..=last_year {
        if !fund_files.contains_key(&year) {
            return Err(err(format!(
                "missing fund portfolio announcement-year {year}"
            )));
        }
    }
    let analyst_first_year = add_months(dates[0], -6) / 10000;
    for year in analyst_first_year..=last_year {
        if !analyst_files.contains_key(&year) {
            return Err(err(format!("missing analyst report year {year}")));
        }
    }
    let mut state = FundState::new();
    for (_, path) in fund_files.range(..first_year) {
        for event in fund_events(path)? {
            apply_fund(&mut state, &event);
        }
    }
    let root = config.data_root.join("derived/stock/relations");
    let workers = std::thread::available_parallelism().map_or(1, |n| n.get());
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(request.date_batch_size.min(workers))
        .build()?;
    let mut processed = 0;
    for year in first_year..=last_year {
        let events = fund_events(&fund_files[&year])?;
        let mut cursor = 0;
        let mut reports = Vec::new();
        for y in (year - 1).max(analyst_first_year)..=year {
            reports.extend(analyst_events(&analyst_files[&y])?);
        }
        let year_dates = dates
            .iter()
            .copied()
            .filter(|d| d / 10000 == year)
            .collect::<Vec<_>>();
        for batch in year_dates.chunks(request.date_batch_size) {
            let mut outputs = Vec::new();
            for &date in batch {
                while cursor < events.len() && events[cursor].announced <= date {
                    apply_fund(&mut state, &events[cursor]);
                    cursor += 1;
                }
                let mut rows = Memberships::new();
                for stock in &industry_stocks {
                    if let Some(group) = industry_map.group_for(date, stock) {
                        rows.insert(("industry".into(), group.to_string(), stock.clone()));
                    }
                }
                // Carry the latest announced period until a newer disclosure supersedes it.
                for event in state.values() {
                    for stock in &event.stocks {
                        rows.insert(("fund".into(), event.fund.clone(), stock.clone()));
                    }
                }
                let cutoff = add_months(date, -6);
                for event in reports.iter().filter(|r| r.date > cutoff && r.date <= date) {
                    rows.insert(("analyst".into(), event.group.clone(), event.stock.clone()));
                }
                if !rows.iter().any(|r| r.0 == "fund") {
                    return Err(err(format!(
                        "no visible fund disclosures on {date}; restore earlier portfolio history"
                    )));
                }
                outputs.push((date, membership_table(date, rows)?));
            }
            let outcomes = pool.install(|| {
                outputs
                    .par_iter()
                    .map(|(date, table)| -> Result<_> {
                        let path = root.join(year.to_string()).join(format!("{date}.parquet"));
                        write_derived_parquet(&path, table)?;
                        Ok((*date, table.len))
                    })
                    .collect::<Vec<_>>()
            });
            for outcome in outcomes {
                let (date, rows) = outcome?;
                processed += 1;
                println!(
                    "derive-stock-relations [{processed}/{}] date={date} membership_rows={rows}",
                    dates.len()
                );
            }
        }
    }
    std::fs::create_dir_all(&root)?;
    std::fs::write(root.join("manifest.json"), format!(
        "{{\"version\":\"{VERSION}\",\"relations\":[\"industry\",\"fund\",\"analyst\"],\"industry\":\"CI_L1\",\"fund_top_n\":10,\"fund_period\":\"latest_visible_per_fund\",\"analyst_window_months\":6,\"visibility\":\"announcement_or_report_date\",\"share_classes\":\"fund_code\"}}\n"
    ))?;
    Ok(())
}

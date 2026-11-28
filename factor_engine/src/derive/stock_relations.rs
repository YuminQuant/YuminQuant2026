//! Announcement-date stock/group memberships. Pairwise adjacency is never materialized.
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use arrow_array::{builder::StringBuilder, Array, ArrayRef, Int32Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};

use crate::calendar::TradingCalendar;
use crate::config::EngineConfig;
use crate::data::parquet_io::{read_parquet, PreparedProjection};
use crate::data::read_ahead::{configure_threads, visit_tables, write_files};
use crate::data::{ColumnData, Table};
use crate::derive::storage::{check_derived_manifest, write_derived_daily_batch, yearly_files};
use crate::error::{err, Result};
use crate::factor::common::cross_section::{ClassificationLevel, ClassificationMap};
use crate::factor::common::financial::add_months;
use crate::progress::ProgressBar;

pub const VERSION: &str = "stock_relations_v1";
const FUND_COLUMNS: [&str; 5] = ["ts_code", "ann_date", "end_date", "symbol", "mkv"];
const ANALYST_COLUMNS: [&str; 4] = ["ts_code", "report_date", "org_name", "author_name"];

#[derive(Clone, Debug)]
pub struct StockRelationsRequest {
    pub start_date: i32,
    pub end_date: i32,
    pub date_batch_size: usize,
    pub threads: Option<usize>,
}

#[derive(Clone, Debug)]
struct FundEvent {
    fund: String,
    announced: i32,
    period: i32,
    stocks: Vec<String>,
}

type FundState = BTreeMap<String, FundEvent>;
pub(super) type Memberships = BTreeSet<(String, String, String)>;

fn projected(path: &Path, columns: &[&str]) -> Result<Table> {
    read_parquet(
        path,
        Some(&columns.iter().map(|s| s.to_string()).collect::<Vec<_>>()),
    )
}

fn text(values: &[Option<String>], row: usize) -> Result<&str> {
    values[row]
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
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

fn fund_events(table: Table, year: i32) -> Result<Vec<FundEvent>> {
    let funds = table.required_utf8("ts_code")?;
    let symbols = table.required_utf8("symbol")?;
    let announced = table.required_i32_date_cast("ann_date")?;
    let periods = table.required_i32_date_cast("end_date")?;
    let values = table.required_f64_cast("mkv")?;
    let mut groups: BTreeMap<(i32, &str, i32), HashMap<String, f64>> = BTreeMap::new();
    for i in 0..table.len {
        let ann = announced[i].ok_or_else(|| err("fund announcement date is missing"))?;
        let period = periods[i].ok_or_else(|| err("fund report period is missing"))?;
        if ann < period {
            return Err(err("fund announcement precedes report period"));
        }
        if ann / 10000 != year {
            return Err(err(format!(
                "fund announcement {ann} is outside partition {year}"
            )));
        }
        let fund = text(funds, i)?;
        let group = groups.entry((ann, fund, period)).or_default();
        let Some(value) = values[i].filter(|v| v.is_finite() && *v > 0.0) else {
            continue;
        };
        let stock = stock_code(text(symbols, i)?);
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
            if holdings.len() > 10 {
                holdings.select_nth_unstable_by(10, |a, b| {
                    b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0))
                });
                holdings.truncate(10);
            }
            let mut stocks: Vec<_> = holdings.into_iter().map(|(stock, _)| stock).collect();
            stocks.sort_unstable();
            FundEvent {
                fund: fund.to_owned(),
                announced,
                period,
                stocks,
            }
        })
        .collect())
}

fn apply_fund(state: &mut FundState, event: FundEvent) {
    if state
        .get(&event.fund)
        .is_none_or(|old| (event.period, event.announced) >= (old.period, old.announced))
    {
        state.insert(event.fund.clone(), event);
    }
}

#[derive(Clone)]
pub(super) struct AnalystEvent {
    pub(super) date: i32,
    pub(super) group: String,
    pub(super) stock: String,
}

pub(super) fn analyst_events(path: &Path) -> Result<Vec<AnalystEvent>> {
    parse_analyst_events(projected(path, &ANALYST_COLUMNS)?)
}

fn parse_analyst_events(table: Table) -> Result<Vec<AnalystEvent>> {
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
            unique.insert((date, format!("{}:{org}{name}", org.len()), stock));
        }
    }
    Ok(unique
        .into_iter()
        .map(|(date, group, stock)| AnalystEvent {
            date,
            group,
            stock: stock.to_owned(),
        })
        .collect())
}

pub(super) fn membership_table(date: i32, rows: Memberships) -> Result<Table> {
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

#[derive(Default)]
struct AnalystWindow {
    pending: VecDeque<AnalystEvent>,
    active: VecDeque<AnalystEvent>,
    memberships: BTreeMap<String, BTreeMap<String, usize>>,
}

impl AnalystWindow {
    fn advance(&mut self, date: i32) {
        let cutoff = add_months(date, -6);
        while self
            .active
            .front()
            .is_some_and(|event| event.date <= cutoff)
        {
            let event = self.active.pop_front().unwrap();
            let stocks = self.memberships.get_mut(&event.group).unwrap();
            let count = stocks.get_mut(&event.stock).unwrap();
            *count -= 1;
            if *count == 0 {
                stocks.remove(&event.stock);
            }
            if stocks.is_empty() {
                self.memberships.remove(&event.group);
            }
        }
        while self.pending.front().is_some_and(|event| event.date <= date) {
            let event = self.pending.pop_front().unwrap();
            if event.date > cutoff {
                *self
                    .memberships
                    .entry(event.group.clone())
                    .or_default()
                    .entry(event.stock.clone())
                    .or_default() += 1;
                self.active.push_back(event);
            }
        }
    }
}

fn daily_memberships(
    date: i32,
    analysts: &AnalystWindow,
    funds: &FundState,
    industry: &ClassificationMap,
    industry_stocks: &BTreeSet<String>,
) -> Result<RecordBatch> {
    let (mut relations, mut groups, mut stocks) = (
        StringBuilder::new(),
        StringBuilder::new(),
        StringBuilder::new(),
    );
    {
        let mut push = |relation, group, stock| {
            relations.append_value(relation);
            groups.append_value(group);
            stocks.append_value(stock);
        };
        for (group, covered) in &analysts.memberships {
            for stock in covered.keys() {
                push("analyst", group.as_str(), stock.as_str());
            }
        }
        let mut has_funds = false;
        for (fund, event) in funds {
            for stock in &event.stocks {
                push("fund", fund.as_str(), stock.as_str());
                has_funds = true;
            }
        }
        if !has_funds {
            return Err(err(format!(
                "no visible fund disclosures on {date}; restore earlier portfolio history"
            )));
        }
        let mut members = industry_stocks
            .iter()
            .filter_map(|stock| industry.group_for(date, stock).map(|group| (group, stock)))
            .collect::<Vec<_>>();
        members.sort_unstable();
        for (group, stock) in members {
            push("industry", group, stock.as_str());
        }
    }
    let stocks = stocks.finish();
    let fields = [
        ("trade_date", DataType::Int32),
        ("ts_code", DataType::Utf8),
        ("group_id", DataType::Utf8),
        ("relation", DataType::Utf8),
    ];
    let columns: Vec<ArrayRef> = vec![
        Arc::new(Int32Array::from_value(date, stocks.len())),
        Arc::new(stocks),
        Arc::new(groups.finish()),
        Arc::new(relations.finish()),
    ];
    let schema = Schema::new(fields.map(|(name, ty)| Field::new(name, ty, true)).to_vec());
    Ok(RecordBatch::try_new(Arc::new(schema), columns)?)
}

enum AnnualEvents {
    Funds(Vec<FundEvent>),
    Analysts(Vec<AnalystEvent>),
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
    let root = config.data_root.join("derived/stock/relations");
    let manifest = root.join("manifest.json");
    let expected = format!(
        "{{\"version\":\"{VERSION}\",\"relations\":[\"industry\",\"fund\",\"analyst\"],\"industry\":\"CI_L1\",\"fund_top_n\":10,\"fund_period\":\"latest_visible_per_fund\",\"analyst_window_months\":6,\"visibility\":\"announcement_or_report_date\",\"share_classes\":\"fund_code\"}}\n"
    );
    let has_manifest = check_derived_manifest(&manifest, &expected)?;
    configure_threads(crate::resources::compute_threads(request.threads)?);
    let progress = ProgressBar::new("derive-stock-relations", dates.len(), true);
    let started = Instant::now();
    let (mut sources, mut paths) = (Vec::new(), Vec::new());
    for (&year, path) in fund_files.range(..first_year) {
        sources.push((year, true));
        paths.push(path.clone());
    }
    for year in analyst_first_year..=last_year {
        sources.push((year, false));
        paths.push(analyst_files[&year].clone());
        if year >= first_year {
            sources.push((year, true));
            paths.push(fund_files[&year].clone());
        }
    }
    let mut funds = FundState::new();
    let mut analysts = AnalystWindow::default();
    let mut date_cursor = 0;
    let fund_columns = FUND_COLUMNS.map(str::to_owned);
    let analyst_columns = ANALYST_COLUMNS.map(str::to_owned);
    visit_tables(
        &paths,
        |index, input| {
            let columns: &[String] = if sources[index].1 {
                &fund_columns
            } else {
                &analyst_columns
            };
            PreparedProjection::new(input, &paths[index], columns, &BTreeSet::new())
        },
        |index, table| {
            let (year, is_fund) = sources[index];
            if is_fund {
                Ok(AnnualEvents::Funds(fund_events(table, year)?))
            } else {
                let events = parse_analyst_events(table)?;
                if events.iter().any(|event| event.date / 10000 != year) {
                    return Err(err(format!(
                        "analyst report date is outside partition {year}"
                    )));
                }
                Ok(AnnualEvents::Analysts(events))
            }
        },
        |index, events| {
            let year = sources[index].0;
            let mut events = match events {
                AnnualEvents::Analysts(events) => {
                    analysts.pending.extend(events);
                    return Ok(());
                }
                AnnualEvents::Funds(events) => events.into_iter().peekable(),
            };
            let year_end =
                date_cursor + dates[date_cursor..].partition_point(|date| date / 10000 <= year);
            for batch in dates[date_cursor..year_end].chunks(request.date_batch_size) {
                let mut outputs = Vec::with_capacity(batch.len());
                for &date in batch {
                    while events.peek().is_some_and(|event| event.announced <= date) {
                        apply_fund(&mut funds, events.next().unwrap());
                    }
                    analysts.advance(date);
                    outputs.push((
                        date,
                        daily_memberships(
                            date,
                            &analysts,
                            &funds,
                            &industry_map,
                            &industry_stocks,
                        )?,
                    ));
                }
                let outcomes = write_files(&root, outputs, |(date, batch)| {
                    write_derived_daily_batch(&root, date, &batch)?;
                    Ok((date, batch.num_rows()))
                })?;
                for (date, rows) in outcomes {
                    progress.tick(format!(
                        "date={date} membership_rows={rows} elapsed_s={:.1}",
                        started.elapsed().as_secs_f64()
                    ));
                }
            }
            // Announcements on non-trading days at year-end still take effect next year.
            for event in events {
                apply_fund(&mut funds, event);
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

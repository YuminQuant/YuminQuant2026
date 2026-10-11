use std::any::Any;
use std::collections::{BTreeMap, HashMap, HashSet};

use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::{DataPool, SourceRowsReader};
use crate::error::{err, Result};
use crate::factor::common::financial::{add_days, add_months};
use crate::factor::common::{ClassificationLevel, ClassificationMap};
use crate::factor::{Factor, FactorUpdatePolicy};
use crate::operators::cs_neutralize_regression;

const INITIAL_LOOKBACK: usize = 32;
const HALF_LIFE: f64 = 90.0;
const PENALTY_DAYS: f64 = 180.0;

pub struct StockDailyCCafr;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyCCafr)
}

impl Factor for StockDailyCCafr {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "c_cafr".into(),
            aliases: vec!["C_CAFR".into()],
            name: "Corrected Consistent Analyst Forecast Revisions".into(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: "0.1.0".into(),
            tags: ["CICC", "analyst", "forecast_revision", "monthly_update", "daily",
                "neutralize", "barra", "size", "sector"]
                .into_iter().map(str::to_string).collect(),
            description: "Month-end C_CAFR approximation with institutions as analysts. Use absolute annual Q4 forecast year equal to the anchor calendar year, not consensus FY labels. Deduplicate stock/date/org/author/title/year (last finite source row wins), average distinct same-day reports by institution, then subtract its strictly previous valid dated forecast for the same stock/year. New revisions must be in (anchor minus six calendar months, anchor]; the strictly previous prediction must be in (anchor minus twelve calendar months, anchor]; retain all revision events including zeros/negative forecasts, no percentage scaling or clipping. Mean weights: 0.5^(calendar age/90) times institution's deduplicated non-BJ A-share report count in (anchor minus twelve months, anchor], across all forecast years and without np filtering. Divide weighted mean by UNWEIGHTED population standard deviation, >=2 events, zero deviation null. Multiply by max(0,1-age_latest_NONZERO_revision/180). SW L1 + Barra SIZE neutralize at month-end, without fill, winsorization or extra zscore; hold FINAL values by stock code daily. Compact report/revision index built once per date-batch over shared annual sources; no PV anchor or disk intermediates.".into(),
            dependencies: vec![
                DataRequest::new(DatasetId::StockAnalystReport,
                    &["np", "report_date", "quarter", "org_name", "author_name", "report_title"]),
                DataRequest::new(DatasetId::StockBarraDaily, &["SIZE"]),
                DataRequest::new(DatasetId::StockSwClassification, &["l1_code"]),
            ],
            intraday_raw_dependencies: vec![],
            lookback: Lookback { trading_days: INITIAL_LOOKBACK },
        }
    }

    fn update_policy(&self) -> FactorUpdatePolicy {
        FactorUpdatePolicy::FinancialEventSnapshot
    }

    fn initial_compute_state(&self, _: &[String]) -> Box<dyn Any + Send> {
        Box::<State>::default()
    }

    fn compute(&self, context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        compute(context, data, &mut State::default())
    }

    fn compute_many_stateful(
        &self,
        ids: &[String],
        context: &FactorContext,
        data: &DataPool,
        state: &mut (dyn Any + Send),
    ) -> Result<Vec<FactorSeries>> {
        if !ids.iter().any(|id| id == "c_cafr") {
            return Ok(vec![]);
        }
        let state = state
            .downcast_mut::<State>()
            .ok_or_else(|| err("c_cafr state type mismatch"))?;
        Ok(vec![compute(context, data, state)?])
    }
}

#[derive(Default)]
struct State {
    last_anchor: Option<i32>,
    last_target: Option<i32>,
    final_values: HashMap<String, Option<f64>>,
}

#[derive(Clone, Copy, Debug)]
struct Revision {
    org: usize,
    previous_date: i32,
    date: i32,
    delta: f64,
}

struct ReportCounts {
    dates: Vec<i32>,
    prefix: Vec<usize>,
}

impl ReportCounts {
    fn new(counts: BTreeMap<i32, usize>) -> Self {
        let mut dates = Vec::with_capacity(counts.len());
        let mut prefix = Vec::with_capacity(counts.len() + 1);
        prefix.push(0);
        for (date, count) in counts {
            dates.push(date);
            prefix.push(prefix.last().unwrap() + count);
        }
        Self { dates, prefix }
    }

    fn between(&self, lower: i32, upper: i32) -> usize {
        let start = self.dates.partition_point(|date| *date <= lower);
        let end = self.dates.partition_point(|date| *date <= upper);
        self.prefix[end] - self.prefix[start]
    }
}

#[derive(Default)]
struct ReportIndex {
    counts: Vec<ReportCounts>,
    revisions: BTreeMap<i32, HashMap<String, Vec<Revision>>>,
}

fn standard_stock(code: &str) -> bool {
    let b = code.as_bytes();
    b.len() == 9 && b[..6].iter().all(u8::is_ascii_digit) && matches!(&b[6..], b".SH" | b".SZ")
}

fn text(value: Option<&str>) -> Option<&str> {
    value.map(str::trim).filter(|value| !value.is_empty())
}

fn annual_year(value: &str) -> Option<i32> {
    let value = value.trim().strip_suffix("Q4")?;
    (value.len() == 4 && value.bytes().all(|b| b.is_ascii_digit()))
        .then(|| value.parse().ok())
        .flatten()
}

fn report_index(
    reader: SourceRowsReader<'_>,
    codes: &[String],
    first_anchor: i32,
    end_date: i32,
) -> Result<ReportIndex> {
    let lower_counts = add_months(first_anchor, -12);
    let lower_forecasts = add_months(first_anchor, -12);
    let legal: HashSet<_> = codes
        .iter()
        .map(String::as_str)
        .filter(|code| standard_stock(code))
        .collect();
    let stocks = reader.required_utf8("ts_code")?;
    let dates = reader.required_i32_date_cast("report_date")?;
    let orgs = reader.required_utf8("org_name")?;
    let authors = reader.required_utf8("author_name")?;
    let titles = reader.required_utf8("report_title")?;
    let years = reader.required_utf8("quarter")?;
    let profits = reader.required_f64_cast("np")?;
    let mut publications = HashSet::new();
    let mut counts: BTreeMap<&str, BTreeMap<i32, usize>> = BTreeMap::new();
    let mut predictions = BTreeMap::new();
    for row in 0..reader.len {
        let (Some(code), Some(date), Some(org)) = (
            stocks[row].as_deref().filter(|code| legal.contains(*code)),
            dates[row],
            text(orgs[row].as_deref()),
        ) else {
            continue;
        };
        if date <= lower_counts || date > end_date {
            continue;
        }
        let author = text(authors[row].as_deref());
        let title = text(titles[row].as_deref());
        if publications.insert((code, date, org, author, title)) {
            *counts.entry(org).or_default().entry(date).or_default() += 1;
        }
        let (Some(year), Some(profit)) = (
            years[row].as_deref().and_then(annual_year),
            profits[row].filter(|value| value.is_finite()),
        ) else {
            continue;
        };
        if date > lower_forecasts && year >= first_anchor / 10000 && year <= end_date / 10000 {
            predictions.insert((code, org, year, date, author, title), profit);
        }
    }
    drop(publications);
    let org_ids: HashMap<_, _> = counts
        .keys()
        .enumerate()
        .map(|(i, org)| (*org, i))
        .collect();
    let mut daily = BTreeMap::<(&str, &str, i32, i32), (f64, usize)>::new();
    for ((code, org, year, date, _, _), profit) in predictions {
        let (mean, count) = daily.entry((code, org, year, date)).or_default();
        *count += 1;
        *mean += (profit - *mean) / *count as f64;
    }
    let mut index = ReportIndex::default();
    let mut previous = None;
    for ((code, org, year, date), (profit, _)) in daily {
        if !profit.is_finite() {
            continue;
        }
        let key = (code, org, year);
        if let Some((previous_key, previous_date, previous_profit)) = previous {
            if previous_key == key {
                let delta: f64 = profit - previous_profit;
                if delta.is_finite() {
                    index
                        .revisions
                        .entry(year)
                        .or_default()
                        .entry(code.to_owned())
                        .or_default()
                        .push(Revision {
                            org: org_ids[org],
                            previous_date,
                            date,
                            delta,
                        });
                }
            }
        }
        previous = Some((key, date, profit));
    }
    for stocks in index.revisions.values_mut() {
        for events in stocks.values_mut() {
            events.sort_by_key(|event| (event.date, event.previous_date, event.org));
        }
    }
    index.counts = counts.into_values().map(ReportCounts::new).collect();
    Ok(index)
}

fn time_weights(anchor: i32) -> BTreeMap<i32, (f64, f64)> {
    // Six calendar months are at most 184 days; walk dates once, including leap days.
    let mut date = anchor;
    (0..=190)
        .map(|age| {
            let weight = 0.5_f64.powf(age as f64 / HALF_LIFE);
            let penalty = (1.0 - age as f64 / PENALTY_DAYS).max(0.0);
            let result = (date, (weight, penalty));
            date = add_days(date, -1);
            result
        })
        .collect()
}

fn raw_value(
    events: &[Revision],
    anchor: i32,
    org_counts: &[usize],
    time: &BTreeMap<i32, (f64, f64)>,
) -> Option<f64> {
    let lower = add_months(anchor, -6);
    let pair_lower = add_months(anchor, -12);
    let start = events.partition_point(|event| event.date <= lower);
    let end = events.partition_point(|event| event.date <= anchor);
    let (mut n, mut mean, mut m2, mut weight_sum, mut weighted_mean) = (0usize, 0.0, 0.0, 0.0, 0.0);
    let mut latest_nonzero = None;
    for event in &events[start..end] {
        if event.previous_date <= pair_lower {
            continue;
        }
        let &(decay, _) = time.get(&event.date)?;
        let weight = decay * *org_counts.get(event.org)? as f64;
        if !weight.is_finite() || weight <= 0.0 {
            return None;
        }
        n += 1;
        let difference = event.delta - mean;
        mean += difference / n as f64;
        m2 += difference * (event.delta - mean);
        weight_sum += weight;
        weighted_mean += weight / weight_sum * (event.delta - weighted_mean);
        if event.delta != 0.0 {
            latest_nonzero = Some(event.date);
        }
    }
    if n < 2 || !m2.is_finite() || m2 <= 0.0 {
        return None;
    }
    let std = (m2 / n as f64).sqrt();
    let penalty = time.get(&latest_nonzero?)?.1;
    let value = penalty * weighted_mean / std;
    value.is_finite().then_some(value)
}

fn compute(context: &FactorContext, data: &DataPool, state: &mut State) -> Result<FactorSeries> {
    let panel = data.stock_universe_panel()?;
    let calendar = data.trading_calendar()?;
    let n = panel.instruments().len();
    let mut values = vec![None; panel.shape_len()];
    let mut index = None;
    let mut size = None;
    let mut sectors = None;
    for (day, &date) in panel.dates().iter().enumerate() {
        if !panel.is_target_date(date) {
            continue;
        }
        if state.last_target.is_some_and(|previous| date < previous) {
            return Err(err("c_cafr provider state cannot move backwards"));
        }
        if let Some(anchor) = calendar.month_end_on_or_before(date) {
            if state.last_anchor != Some(anchor) {
                let anchor_day = panel
                    .dates()
                    .binary_search(&anchor)
                    .map_err(|_| err("c_cafr prior month-end missing from warmup panel"))?;
                if index.is_none() {
                    index = Some(report_index(
                        data.analyst_report_reader()?,
                        panel.instruments(),
                        anchor,
                        context.end_date,
                    )?);
                    size = Some(
                        panel.column_from_table(data.daily(DatasetId::StockBarraDaily)?, "SIZE")?,
                    );
                    sectors = Some(ClassificationMap::from_table(
                        data.daily(DatasetId::StockSwClassification)?,
                        ClassificationLevel::Sector,
                    )?);
                }
                let index = index.as_ref().unwrap();
                let counts: Vec<_> = index
                    .counts
                    .iter()
                    .map(|counts| counts.between(add_months(anchor, -12), anchor))
                    .collect();
                let weights = time_weights(anchor);
                let stocks = index.revisions.get(&(anchor / 10000));
                let mut groups = sectors
                    .as_ref()
                    .unwrap()
                    .groups_for(anchor, panel.instruments());
                let raw: Vec<_> = panel
                    .instruments()
                    .iter()
                    .enumerate()
                    .map(|(i, code)| {
                        if !standard_stock(code) || !panel.is_present_offset(anchor_day * n + i) {
                            groups[i] = None;
                            return None;
                        }
                        let events = stocks?.get(code)?;
                        raw_value(events, anchor, &counts, &weights)
                    })
                    .collect();
                let exposure =
                    &size.as_ref().unwrap().values()[anchor_day * n..(anchor_day + 1) * n];
                let residual = cs_neutralize_regression(&raw, &[exposure], Some(&groups), None);
                state.final_values = panel.instruments().iter().cloned().zip(residual).collect();
                state.last_anchor = Some(anchor);
            }
        }
        for (i, code) in panel.instruments().iter().enumerate() {
            if panel.is_present_offset(day * n + i) {
                values[day * n + i] = state.final_values.get(code).copied().flatten();
            }
        }
        state.last_target = Some(date);
    }
    Ok(panel
        .column_from_values(values)?
        .to_factor_series(StockDailyCCafr.spec()))
}

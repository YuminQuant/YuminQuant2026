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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calendar::TradingCalendar;
    use crate::core::FactorRowKey;
    use crate::data::{ColumnData, Table};
    use std::sync::Arc;

    fn strings(values: impl IntoIterator<Item = impl AsRef<str>>) -> ColumnData {
        ColumnData::Utf8(
            values
                .into_iter()
                .map(|v| Some(v.as_ref().to_owned()))
                .collect(),
        )
    }

    fn reports(rows: &[(&str, i32, &str, &str, &str, Option<f64>)]) -> Table {
        Table::new(BTreeMap::from([
            ("ts_code".into(), strings(rows.iter().map(|r| r.0))),
            (
                "report_date".into(),
                ColumnData::I32(rows.iter().map(|r| Some(r.1)).collect()),
            ),
            ("org_name".into(), strings(rows.iter().map(|r| r.2))),
            ("quarter".into(), strings(rows.iter().map(|r| r.3))),
            ("report_title".into(), strings(rows.iter().map(|r| r.4))),
            ("author_name".into(), strings(rows.iter().map(|_| "author"))),
            (
                "np".into(),
                ColumnData::F64(rows.iter().map(|r| r.5).collect()),
            ),
        ]))
        .unwrap()
    }

    fn close(a: Option<f64>, b: Option<f64>) {
        match (a, b) {
            (None, None) => (),
            (Some(a), Some(b)) => assert!((a - b).abs() < 1e-10, "{a} != {b}"),
            _ => panic!("{a:?} != {b:?}"),
        }
    }

    #[test]
    fn c_cafr_weighted_mean_unweighted_population_std_and_nonzero_penalty() {
        let anchor = 20250630;
        let events = [
            Revision {
                org: 0,
                previous_date: 20250101,
                date: add_days(anchor, -90),
                delta: 2.0,
            },
            Revision {
                org: 1,
                previous_date: 20250101,
                date: anchor,
                delta: 0.0,
            },
        ];
        // Weights are 0.5*2 and 1*1; std([2,0])=1. Zero revision does not reset age.
        close(
            raw_value(&events, anchor, &[2, 1], &time_weights(anchor)),
            Some(0.5),
        );
        let scaled = events.map(|e| Revision {
            delta: e.delta * 10000.0,
            ..e
        });
        close(
            raw_value(&scaled, anchor, &[2, 1], &time_weights(anchor)),
            Some(0.5),
        );
        assert_eq!(
            raw_value(&events[..1], anchor, &[2, 1], &time_weights(anchor)),
            None
        );
        let constant = events.map(|e| Revision { delta: 1.0, ..e });
        assert_eq!(
            raw_value(&constant, anchor, &[2, 1], &time_weights(anchor)),
            None
        );
        let old = [
            Revision {
                previous_date: 20240701,
                date: add_days(anchor, -181),
                ..events[0]
            },
            events[1],
        ];
        close(
            raw_value(&old, anchor, &[2, 1], &time_weights(anchor)),
            Some(0.0),
        );
    }

    #[test]
    fn c_cafr_new_six_month_previous_twelve_month_boundaries() {
        let anchor = 20250228;
        let events = [
            Revision {
                org: 0,
                previous_date: 20240229,
                date: 20250115,
                delta: -2.0,
            },
            Revision {
                org: 0,
                previous_date: 20240301,
                date: anchor,
                delta: 4.0,
            },
        ];
        assert!(raw_value(&events, anchor, &[1], &time_weights(anchor)).is_some());
        let expired_pair = [
            Revision {
                previous_date: 20240228,
                ..events[0]
            },
            events[1],
        ];
        assert_eq!(
            raw_value(&expired_pair, anchor, &[1], &time_weights(anchor)),
            None
        );
        let expired_new = [
            Revision {
                date: add_months(anchor, -6),
                ..events[0]
            },
            events[1],
        ];
        assert_eq!(
            raw_value(&expired_new, anchor, &[1], &time_weights(anchor)),
            None
        );
        let future_new = [
            events[0],
            Revision {
                date: 20250301,
                ..events[1]
            },
        ];
        assert_eq!(
            raw_value(&future_new, anchor, &[1], &time_weights(anchor)),
            None
        );
    }

    #[test]
    fn c_cafr_report_dedup_daily_average_year_pairing_and_authority_counts() {
        let code = "000001.SZ";
        let table = reports(&[
            (code, 20240401, "a", "2025Q4", "old", Some(-100.0)),
            (code, 20250110, "a", "2025Q4", "new1", Some(-90.0)),
            (code, 20250110, "a", "2025Q4", "new1", Some(-80.0)),
            (code, 20250110, "a", "2025Q4", "new2", Some(-60.0)),
            (code, 20250110, "a", "2026Q4", "new1", Some(1000.0)),
            (code, 20250111, "a", "2025Q4", "zero", Some(0.0)),
            (code, 20250112, "a", "2025Q4", "equal", Some(0.0)),
            (code, 20250113, "a", "2025Q4", "missing", None),
            (code, 20250114, "b", "2025Q4", "only", Some(200.0)),
            (code, 20250210, "a", "2025Q4", "future", Some(1e6)),
            ("830001.BJ", 20250110, "a", "2025Q4", "bj", Some(10.0)),
            ("A00001.SZ", 20250110, "a", "2025Q4", "invalid", Some(10.0)),
            (code, 20250110, " ", "2025Q4", "noorg", Some(10.0)),
            (code, 20240131, "a", "2025Q4", "boundary", Some(-200.0)),
        ]);
        let codes = [code.to_owned(), "830001.BJ".into(), "A00001.SZ".into()];
        let index = report_index((&table).into(), &codes, 20250131, 20250228).unwrap();
        let events = &index.revisions[&2025][code];
        assert_eq!(events.len(), 4);
        assert_eq!(
            (events[0].previous_date, events[0].date, events[0].delta),
            (20240401, 20250110, 30.0)
        );
        assert_eq!(events[1].delta, 70.0);
        assert_eq!(events[2].delta, 0.0);
        assert_eq!(index.counts[0].between(20240131, 20250131), 6);
        assert_eq!(index.counts[1].between(20240131, 20250131), 1);
        assert!(!index.revisions.contains_key(&2026));
        let cold = report_index((&table).into(), &codes, 20250131, 20250131).unwrap();
        let counts: Vec<_> = index
            .counts
            .iter()
            .map(|c| c.between(20240131, 20250131))
            .collect();
        let cold_counts: Vec<_> = cold
            .counts
            .iter()
            .map(|c| c.between(20240131, 20250131))
            .collect();
        close(
            raw_value(events, 20250131, &counts, &time_weights(20250131)),
            raw_value(
                &cold.revisions[&2025][code],
                20250131,
                &cold_counts,
                &time_weights(20250131),
            ),
        );
    }

    const CODES: [&str; 8] = [
        "000001.SZ",
        "000002.SZ",
        "000003.SZ",
        "000004.SZ",
        "000005.SZ",
        "000006.SZ",
        "000007.SZ",
        "830001.BJ",
    ];

    #[test]
    fn c_cafr_uses_anchor_calendar_year_not_consensus_fiscal_labels() {
        let code = "000001.SZ";
        let table = reports(&[
            (code, 20241001, "a", "2024Q4", "old", Some(100.0)),
            (code, 20241201, "a", "2024Q4", "new", Some(110.0)),
            (code, 20241001, "a", "2025Q4", "old", Some(200.0)),
            (code, 20250115, "a", "2025Q4", "new", Some(250.0)),
        ]);
        let index = report_index((&table).into(), &[code.into()], 20241231, 20250131).unwrap();
        assert_eq!(index.revisions[&2024][code][0].delta, 10.0);
        assert_eq!(index.revisions[&2025][code][0].delta, 50.0);
        assert_eq!(index.counts[0].between(20231231, 20241231), 2);
        assert_eq!(index.counts[0].between(20240131, 20250131), 3);
    }
    const DATES: [i32; 5] = [20250131, 20250203, 20250214, 20250228, 20250303];

    fn fixture(target: &[i32], reverse: bool) -> (FactorContext, DataPool) {
        let context = FactorContext {
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            start_date: target[0],
            end_date: *target.last().unwrap(),
            load_start_date: DATES[0],
            load_dates: DATES.to_vec(),
            target_dates: target.to_vec(),
        };
        let basic = Table::new(BTreeMap::from([
            ("ts_code".into(), strings(CODES)),
            ("list_date".into(), ColumnData::I32(vec![Some(20200101); 8])),
            ("delist_date".into(), ColumnData::I32(vec![None; 8])),
        ]))
        .unwrap();
        let classification = Table::new(BTreeMap::from([
            ("ts_code".into(), strings(CODES)),
            ("in_date".into(), ColumnData::I32(vec![Some(20200101); 8])),
            ("out_date".into(), ColumnData::I32(vec![None; 8])),
            (
                "l1_code".into(),
                strings(["a", "a", "a", "b", "b", "b", "b", "a"]),
            ),
        ]))
        .unwrap();
        let barra = Table::new(BTreeMap::from([
            ("ts_code".into(), strings(DATES.iter().flat_map(|_| CODES))),
            (
                "trade_date".into(),
                ColumnData::I32(DATES.iter().flat_map(|&d| vec![Some(d); 8]).collect()),
            ),
            (
                "SIZE".into(),
                ColumnData::F64(
                    DATES
                        .iter()
                        .flat_map(|&d| {
                            (0..8).map(move |i| {
                                if i == 6 {
                                    None
                                } else {
                                    Some(
                                        (i + 1) as f64
                                            + if d == 20250214 { (i * i) as f64 } else { 0.0 },
                                    )
                                }
                            })
                        })
                        .collect(),
                ),
            ),
        ]))
        .unwrap();
        let mut rows = Vec::new();
        for (i, code) in CODES.iter().enumerate() {
            if i == 5 {
                continue;
            }
            let x = i as f64;
            rows.extend([
                (*code, 20240401, "a", "2025Q4", "old_a", Some(100.0)),
                (*code, 20240501, "b", "2025Q4", "old_b", Some(150.0)),
                (*code, 20250115, "a", "2025Q4", "new_a", Some(102.0 + x * x)),
                (*code, 20250120, "b", "2025Q4", "new_b", Some(148.0 + x)),
                (*code, 20250210, "a", "2025Q4", "feb_a", Some(108.0 + x)),
                (*code, 20250220, "b", "2025Q4", "feb_b", Some(147.0 - x * x)),
            ]);
        }
        let mut tables = HashMap::from([
            (DatasetId::StockBasic, basic),
            (DatasetId::StockSwClassification, classification),
            (DatasetId::StockBarraDaily, barra),
            (DatasetId::StockAnalystReport, reports(&rows)),
        ]);
        if reverse {
            for table in tables.values_mut() {
                *table = table
                    .take(&(0..table.len).rev().collect::<Vec<_>>())
                    .unwrap();
            }
        }
        let mut pool = DataPool::from_daily_tables_for_test(tables, &context).unwrap();
        pool.set_trading_calendar(Arc::new(TradingCalendar::from_open_dates(DATES.to_vec())));
        (context, pool)
    }

    fn value(series: &FactorSeries, date: i32, code: &str) -> Option<f64> {
        series
            .values
            .iter()
            .find_map(|row| match &row.key {
                FactorRowKey::Daily {
                    trade_date,
                    ts_code,
                } if *trade_date == date && ts_code == code => Some(row.value),
                _ => None,
            })
            .unwrap()
    }

    #[test]
    fn c_cafr_monthly_final_freeze_stock_alignment_batches_and_cold_start() {
        let target = &DATES[1..];
        let (context, pool) = fixture(target, false);
        assert!(pool.stock_universe_panel().is_ok());
        assert!(pool.daily(DatasetId::StockDailyPv).is_err());
        let full = compute(&context, &pool, &mut State::default()).unwrap();
        assert!(full.values.iter().all(|r| match r.key {
            FactorRowKey::Daily { trade_date, .. } => target.contains(&trade_date),
            _ => false,
        }));
        assert!(full
            .values
            .iter()
            .any(|r| r.value.is_some_and(|v| v.abs() > 1e-8)));
        for code in CODES {
            assert_eq!(value(&full, 20250203, code), value(&full, 20250214, code));
            assert_eq!(value(&full, 20250228, code), value(&full, 20250303, code));
        }
        for code in &CODES[5..] {
            assert_eq!(value(&full, 20250203, code), None);
        }
        assert!(CODES[..5]
            .iter()
            .any(|code| value(&full, 20250203, code) != value(&full, 20250228, code)));
        let mut state = State::default();
        for dates in [&DATES[1..2], &DATES[2..]] {
            let (context, pool) = fixture(dates, true);
            let batch = compute(&context, &pool, &mut state).unwrap();
            let cold = compute(&context, &pool, &mut State::default()).unwrap();
            for &date in dates {
                for code in CODES {
                    close(value(&batch, date, code), value(&full, date, code));
                    close(value(&cold, date, code), value(&full, date, code));
                }
            }
        }
        assert_eq!(state.final_values.len(), CODES.len());
        let spec = StockDailyCCafr.spec();
        assert_eq!(spec.id, "c_cafr");
        assert!(spec.aliases.contains(&"C_CAFR".into()));
        for tag in ["CICC", "analyst", "monthly_update"] {
            assert!(spec.tags.contains(&tag.into()));
        }
        assert!(!spec.tags.contains(&"fundamental".into()));
        assert_eq!(spec.dependencies.len(), 3);
        assert!(spec
            .dependencies
            .iter()
            .any(|r| r.dataset == DatasetId::StockAnalystReport));
        let mut boxed = StockDailyCCafr.initial_compute_state(&[]);
        assert!(StockDailyCCafr
            .compute_many_stateful(&["pafr".into()], &context, &pool, &mut *boxed)
            .unwrap()
            .is_empty());
        let result = StockDailyCCafr
            .compute_many_stateful(&["c_cafr".into()], &context, &pool, &mut *boxed)
            .unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].spec.id, "c_cafr");
    }
}

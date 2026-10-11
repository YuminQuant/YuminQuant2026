use std::any::Any;
use std::collections::{HashMap, HashSet};

use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;

use crate::error::{err, Result};
use crate::factor::common::financial::{add_days, add_months};
use crate::factor::common::stock_daily_ops::{is_bj_stock, neutralize_and_fill_missing};
use crate::factor::common::{ClassificationLevel, ClassificationMap, ReportTypePreference};
use crate::factor::{Factor, FactorUpdatePolicy};
use crate::operators::cross_sectional::cs_neutralize_regression;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Output {
    Cov,
    Anncov,
    Firstcov,
}

impl Output {
    fn id(self) -> &'static str {
        match self {
            Self::Cov => "cov",
            Self::Anncov => "anncov",
            Self::Firstcov => "firstcov",
        }
    }
}

pub struct AnalystCoverage(pub Output);

impl Factor for AnalystCoverage {
    fn spec(&self) -> FactorSpec {
        let mut dependencies = vec![
            DataRequest::new(
                DatasetId::StockAnalystReport,
                if self.0 == Output::Firstcov {
                    &["report_date", "org_name", "rating"]
                } else {
                    &["report_date", "org_name", "author_name", "report_title"]
                },
            ),
            DataRequest::new(DatasetId::StockBarraDaily, &["SIZE"]),
            DataRequest::new(DatasetId::StockSwClassification, &["l1_code"]),
        ];
        let mut tags = vec![
            "deprecated",
            "DFZQ",
            "analyst",
            "coverage",
            "monthly_update",
            "daily",
            "neutralize",
            "barra",
            "size",
            "sector",
        ];
        if self.0 == Output::Anncov {
            tags.extend(["fundamental", "financial", "pit", "earnings_announcement"]);
            for dataset in [
                DatasetId::StockIncome,
                DatasetId::StockBalanceSheet,
                DatasetId::StockCashFlow,
            ] {
                dependencies.push(DataRequest::financial_quarters(dataset, &[], 4));
            }
        }
        FactorSpec {
            id: self.0.id().into(), aliases: vec![self.0.id().to_ascii_uppercase()],
            name: match self.0 { Output::Cov => "Analyst Report Coverage", Output::Anncov => "Post-Announcement Analyst Coverage", Output::Firstcov => "Analyst First Coverage" }.into(),
            asset_class: AssetClass::Stock, frequency: Frequency::Daily, version: "0.1.0".into(),
            tags: tags.into_iter().map(str::to_string).collect(),
            description: format!("Month-end square root of coverage event count in (anchor minus six calendar months, anchor]. {} SW L1 + Barra SIZE neutralization at anchor only; freeze final values by stock code until next month end. Stock universe panel, excludes BJ, no winsorization or zscore. Only requested outputs computed.",
                match self.0 {
                    Output::Cov => "Deduplicate stock/date/org/author/title across forecast years, without earnings or rating filters. No reports means raw zero.",
                    Output::Anncov => "Deduplicate stock/date/org/author/title across forecast years, without earnings or rating filters. Only regular consolidated PIT disclosures (f_ann_date else ann_date), union of income/balance/cashflow dates within the same six-month window, including visible revisions. Count each report once in (announcement, announcement+7 calendar days], capped at anchor. Announcements without reports mean raw zero; no announcements mean missing, structurally imputed via zero residual only with valid SIZE and a fitted industry. Missing exposures or failed regression stay null.",
                    Output::Firstcov => "Per stock/institution, count the first report date inside the window once; add one first-rating event only if the earliest valid rating date in the window is later. Same-day reports/forecast rows are one institution-day; any valid rating wins. No one-year re-coverage rule and no lifetime-first-history requirement. Null, blank and no-rating sentinels are unrated. Trim institution names; no speculative alias merging; missing institution excluded. No events means raw zero.",
                }),
            dependencies, intraday_raw_dependencies: vec![], lookback: Lookback { trading_days: 32 },
        }
    }
    fn compute_provider_key(&self) -> String {
        "dfzq_analyst_coverage".into()
    }
    fn update_policy(&self) -> FactorUpdatePolicy {
        FactorUpdatePolicy::FinancialEventStateDailyFast
    }
    fn initial_compute_state(&self, _: &[String]) -> Box<dyn Any + Send> {
        Box::new(State::default())
    }
    fn compute(&self, _: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        compute(&[self.0.id().into()], data, &mut State::default())?
            .pop()
            .ok_or_else(|| err("missing analyst coverage output"))
    }
    fn compute_many(
        &self,
        ids: &[String],
        _: &FactorContext,
        data: &DataPool,
    ) -> Result<Vec<FactorSeries>> {
        compute(ids, data, &mut State::default())
    }
    fn compute_many_stateful(
        &self,
        ids: &[String],
        _: &FactorContext,
        data: &DataPool,
        state: &mut (dyn Any + Send),
    ) -> Result<Vec<FactorSeries>> {
        compute(
            ids,
            data,
            state
                .downcast_mut::<State>()
                .ok_or_else(|| err("analyst coverage state mismatch"))?,
        )
    }
}

#[derive(Default)]
struct MonthlySnapshot {
    anchor: Option<i32>,
    values: HashMap<String, Option<f64>>,
}

#[derive(Default)]
struct State {
    outputs: HashMap<Output, MonthlySnapshot>,
}

type ReportDates<'a> = HashMap<&'a str, Vec<i32>>;

#[derive(Default)]
struct InstitutionHistory {
    dates: Vec<i32>,
    rated_dates: Vec<i32>,
}

type FirstCoverage<'a> = HashMap<&'a str, HashMap<&'a str, InstitutionHistory>>;

#[derive(Default)]
struct CoverageIndex<'a> {
    dates: ReportDates<'a>,
    first: FirstCoverage<'a>,
}

fn valid_rating(rating: Option<&str>) -> bool {
    let Some(rating) = rating.map(str::trim).filter(|s| !s.is_empty()) else {
        return false;
    };
    ![
        "\u{65e0}",
        "\u{672a}\u{8bc4}\u{7ea7}",
        "\u{65e0}\u{8bc4}\u{7ea7}",
        "\u{6682}\u{65e0}\u{8bc4}\u{7ea7}",
        "\u{4e0d}\u{8bc4}\u{7ea7}",
        "na",
        "n/a",
        "none",
        "null",
        "nan",
        "not rated",
        "unrated",
        "nr",
        "-",
        "--",
    ]
    .iter()
    .any(|sentinel| rating.eq_ignore_ascii_case(sentinel))
}

fn report_index<'a>(table: impl Into<crate::data::SourceRowsReader<'a>>, need_counts: bool, need_first: bool) -> Result<CoverageIndex<'a>> {
    let table = table.into();
    let codes = table.required_utf8("ts_code")?;
    let dates = table.required_i32_date_cast("report_date")?;
    let orgs = table.required_utf8("org_name")?;
    let authors = need_counts
        .then(|| table.required_utf8("author_name"))
        .transpose()?;
    let titles = need_counts
        .then(|| table.required_utf8("report_title"))
        .transpose()?;
    let ratings = need_first
        .then(|| table.required_utf8("rating"))
        .transpose()?;
    let mut unique = HashSet::with_capacity(if need_counts { table.len } else { 0 });
    let mut result = CoverageIndex::default();
    for i in 0..table.len {
        let (Some(code), Some(date)) = (codes[i].as_deref(), dates[i]) else {
            continue;
        };
        if is_bj_stock(code) {
            continue;
        }
        // A report's FY0/FY1/FY2 rows are one publication; np/rating may be absent.
        if need_counts
            && unique.insert((
                code,
                date,
                orgs[i].as_deref(),
                authors.as_ref().unwrap()[i].as_deref(),
                titles.as_ref().unwrap()[i].as_deref(),
            ))
        {
            result.dates.entry(code).or_default().push(date);
        }
        if let Some(ref ratings) = ratings {
            if let Some(org) = orgs[i].as_deref().map(str::trim).filter(|s| !s.is_empty()) {
                let history = result
                    .first
                    .entry(code)
                    .or_default()
                    .entry(org)
                    .or_default();
                history.dates.push(date);
                if valid_rating(ratings[i].as_deref()) {
                    history.rated_dates.push(date);
                }
            }
        }
    }
    for dates in result.dates.values_mut() {
        dates.sort_unstable();
    }
    for orgs in result.first.values_mut() {
        for history in orgs.values_mut() {
            history.dates.sort_unstable();
            history.dates.dedup();
            history.rated_dates.sort_unstable();
            history.rated_dates.dedup();
        }
    }
    Ok(result)
}

fn firstcov_raw(
    institutions: Option<&HashMap<&str, InstitutionHistory>>,
    lower: i32,
    anchor: i32,
) -> f64 {
    let mut count = 0_usize;
    if let Some(institutions) = institutions {
        for history in institutions.values() {
            let start = history.dates.partition_point(|date| *date <= lower);
            let Some(&first) = history.dates.get(start).filter(|&&date| date <= anchor) else {
                continue;
            };
            count += 1;
            let rated_start = history.rated_dates.partition_point(|date| *date <= lower);
            if history
                .rated_dates
                .get(rated_start)
                .is_some_and(|&date| date > first && date <= anchor)
            {
                count += 1;
            }
        }
    }
    (count as f64).sqrt()
}

fn count_between(dates: &[i32], lower: i32, upper: i32) -> usize {
    if lower >= upper {
        return 0;
    }
    dates.partition_point(|d| *d <= upper) - dates.partition_point(|d| *d <= lower)
}

fn anncov_raw(reports: &[i32], announcements: &[i32], lower: i32, anchor: i32) -> Option<f64> {
    let start = announcements.partition_point(|d| *d <= lower);
    let end = announcements.partition_point(|d| *d <= anchor);
    let announcements = &announcements[start..end];
    if announcements.is_empty() {
        return None;
    }
    let mut count = 0;
    let mut counted_until = lower;
    // Sorted announcement windows form a union: overlapping windows never double-count reports.
    for &date in announcements {
        let upper = add_days(date, 7).min(anchor);
        count += count_between(reports, date.max(counted_until), upper);
        counted_until = counted_until.max(upper);
    }
    Some((count as f64).sqrt())
}

fn compute(ids: &[String], data: &DataPool, state: &mut State) -> Result<Vec<FactorSeries>> {
    let outputs: Vec<_> = [Output::Cov, Output::Anncov, Output::Firstcov]
        .into_iter()
        .filter(|output| ids.iter().any(|id| id == output.id()))
        .collect();
    if outputs.is_empty() {
        return Ok(vec![]);
    }
    let panel = data.stock_universe_panel()?;
    let calendar = data.trading_calendar()?;
    let size = panel.column_from_table(data.daily(DatasetId::StockBarraDaily)?, "SIZE")?;
    let sectors = ClassificationMap::from_table(
        data.daily(DatasetId::StockSwClassification)?,
        ClassificationLevel::Sector,
    )?;
    let readers = if outputs.contains(&Output::Anncov) {
        [
            DatasetId::StockIncome,
            DatasetId::StockBalanceSheet,
            DatasetId::StockCashFlow,
        ]
        .into_iter()
        .map(|dataset| data.financial_reader(dataset, ReportTypePreference::consolidated()))
        .collect::<Result<Vec<_>>>()?
    } else {
        vec![]
    };
    let mut reports = None;
    let n = panel.instruments().len();
    let mut columns = vec![vec![None; panel.shape_len()]; outputs.len()];
    for (day, &date) in panel.dates().iter().enumerate() {
        if !panel.is_target_date(date) {
            continue;
        }
        let Some(anchor) = calendar.month_end_on_or_before(date) else {
            continue;
        };
        let pending: Vec<_> = outputs
            .iter()
            .copied()
            .filter(|output| {
                state
                    .outputs
                    .get(output)
                    .is_none_or(|snapshot| snapshot.anchor != Some(anchor))
            })
            .collect();
        if !pending.is_empty() {
            if reports.is_none() {
                reports = Some(report_index(
                    data.analyst_report_reader()?,
                    outputs.iter().any(|output| *output != Output::Firstcov),
                    outputs.contains(&Output::Firstcov),
                )?);
            }
            let reports = reports.as_ref().unwrap();
            let anchor_day = panel
                .dates()
                .binary_search(&anchor)
                .map_err(|_| err("analyst coverage month-end anchor missing from warmup panel"))?;
            let lower = add_months(anchor, -6);
            let mut groups = sectors.groups_for(anchor, panel.instruments());
            let mut raw = vec![vec![None; n]; pending.len()];
            for (i, code) in panel.instruments().iter().enumerate() {
                if is_bj_stock(code) || !panel.is_present_offset(anchor_day * n + i) {
                    groups[i] = None;
                    continue;
                }
                let dates = reports
                    .dates
                    .get(code.as_str())
                    .map(Vec::as_slice)
                    .unwrap_or(&[]);
                for (j, output) in pending.iter().enumerate() {
                    raw[j][i] = match output {
                        Output::Cov => Some((count_between(dates, lower, anchor) as f64).sqrt()),
                        Output::Firstcov => Some(firstcov_raw(
                            reports.first.get(code.as_str()),
                            lower,
                            anchor,
                        )),
                        Output::Anncov => {
                            let mut announcements: Vec<_> = readers
                                .iter()
                                .flat_map(|reader| {
                                    reader.disclosure_dates_between(code, lower, anchor)
                                })
                                .collect();
                            announcements.sort_unstable();
                            announcements.dedup();
                            anncov_raw(dates, &announcements, lower, anchor)
                        }
                    };
                }
            }
            for (output, raw) in pending.into_iter().zip(raw) {
                let exposure = &size.values()[anchor_day * n..(anchor_day + 1) * n];
                let residual = match output {
                    Output::Cov | Output::Firstcov => {
                        cs_neutralize_regression(&raw, &[exposure], Some(&groups), None)
                    }
                    Output::Anncov => neutralize_and_fill_missing(&raw, exposure, &groups),
                };
                state.outputs.insert(
                    output,
                    MonthlySnapshot {
                        anchor: Some(anchor),
                        values: panel.instruments().iter().cloned().zip(residual).collect(),
                    },
                );
            }
        }
        for (j, output) in outputs.iter().enumerate() {
            if let Some(snapshot) = state.outputs.get(output) {
                for (i, code) in panel.instruments().iter().enumerate() {
                    if panel.is_present_offset(day * n + i) {
                        columns[j][day * n + i] = snapshot.values.get(code).copied().flatten();
                    }
                }
            }
        }
    }
    outputs
        .into_iter()
        .zip(columns)
        .map(|(output, values)| {
            Ok(panel
                .column_from_values(values)?
                .to_factor_series(AnalystCoverage(output).spec()))
        })
        .collect()
}

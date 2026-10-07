use std::any::Any;
use std::collections::{HashMap, HashSet};

use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::{DataPool, Table};
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

fn report_index(table: &Table, need_counts: bool, need_first: bool) -> Result<CoverageIndex<'_>> {
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
                authors.unwrap()[i].as_deref(),
                titles.unwrap()[i].as_deref(),
            ))
        {
            result.dates.entry(code).or_default().push(date);
        }
        if let Some(ratings) = ratings {
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
                    data.daily(DatasetId::StockAnalystReport)?,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calendar::TradingCalendar;
    use crate::core::FactorRowKey;
    use crate::data::ColumnData;
    use std::collections::BTreeMap;
    use std::sync::Arc;

    fn strings(values: &[&str]) -> ColumnData {
        ColumnData::Utf8(values.iter().map(|v| Some((*v).into())).collect())
    }

    fn reports(rows: &[(&str, i32, &str, &str)]) -> Table {
        Table::new(BTreeMap::from([
            (
                "ts_code".into(),
                strings(&rows.iter().map(|r| r.0).collect::<Vec<_>>()),
            ),
            (
                "report_date".into(),
                ColumnData::I32(rows.iter().map(|r| Some(r.1)).collect()),
            ),
            (
                "org_name".into(),
                strings(&rows.iter().map(|r| r.2).collect::<Vec<_>>()),
            ),
            (
                "report_title".into(),
                strings(&rows.iter().map(|r| r.3).collect::<Vec<_>>()),
            ),
            ("author_name".into(), strings(&vec!["author"; rows.len()])),
        ]))
        .unwrap()
    }

    #[test]
    fn analyst_coverage_deduplicates_forecast_rows_not_distinct_reports() {
        let table = reports(&[
            ("000001.SZ", 20250101, "a", "report1"),
            ("000001.SZ", 20250101, "a", "report1"),
            ("000001.SZ", 20250101, "a", "report2"),
            ("000001.SZ", 20250101, "b", "report1"),
            ("000001.SZ", 20250102, "a", "report1"),
            ("830001.BJ", 20250102, "a", "report1"),
        ]);
        let index = report_index(&table, true, false).unwrap().dates;
        assert_eq!(index["000001.SZ"], [20250101, 20250101, 20250101, 20250102]);
        assert!(!index.contains_key("830001.BJ"));
        assert_eq!(count_between(&index["000001.SZ"], 20250101, 20250102), 1);
        // The required columns deliberately do not include forecast earnings or rating.
    }

    #[test]
    fn analyst_coverage_calendar_window_and_announcement_union() {
        assert_eq!(add_months(20250831, -6), 20250228);
        assert_eq!(add_months(20240831, -6), 20240229);
        let dates = [
            20250101, 20250102, 20250102, 20250108, 20250109, 20250112, 20250113,
        ];
        assert_eq!(
            anncov_raw(&dates, &[20250101], 20241231, 20250131),
            Some(3_f64.sqrt())
        );
        assert_eq!(
            anncov_raw(&dates, &[20250101, 20250105], 20241231, 20250131),
            Some(5_f64.sqrt())
        );
        assert_eq!(
            anncov_raw(&dates, &[20250101], 20241231, 20250103),
            Some(2_f64.sqrt())
        );
        assert_eq!(anncov_raw(&dates, &[20250101], 20250101, 20250131), None);
        assert_eq!(anncov_raw(&[], &[20250101], 20241231, 20250131), Some(0.0));
        assert_eq!(anncov_raw(&dates, &[20250201], 20241231, 20250131), None);
        assert_eq!(
            anncov_raw(
                &[20250301, 20250306, 20250307],
                &[20250227],
                20250101,
                20250331
            ),
            Some(2_f64.sqrt())
        );
    }

    const CODES: [&str; 6] = [
        "000001.SZ",
        "000002.SZ",
        "000003.SZ",
        "000004.SZ",
        "000005.SZ",
        "830001.BJ",
    ];
    const DATES: [i32; 5] = [20250131, 20250203, 20250214, 20250228, 20250303];

    fn fixture(target: &[i32], reverse_rows: bool, with_financial: bool) -> DataPool {
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
            ("ts_code".into(), strings(&CODES)),
            ("list_date".into(), ColumnData::I32(vec![Some(20200101); 6])),
            ("delist_date".into(), ColumnData::I32(vec![None; 6])),
        ]))
        .unwrap();
        let classification = Table::new(BTreeMap::from([
            ("ts_code".into(), strings(&CODES)),
            ("in_date".into(), ColumnData::I32(vec![Some(20200101); 6])),
            ("out_date".into(), ColumnData::I32(vec![None; 6])),
            ("l1_code".into(), strings(&["industry"; 6])),
        ]))
        .unwrap();
        let barra = Table::new(BTreeMap::from([
            (
                "ts_code".into(),
                strings(&DATES.iter().flat_map(|_| CODES).collect::<Vec<_>>()),
            ),
            (
                "trade_date".into(),
                ColumnData::I32(DATES.iter().flat_map(|&date| vec![Some(date); 6]).collect()),
            ),
            (
                "SIZE".into(),
                ColumnData::F64(
                    DATES
                        .iter()
                        .flat_map(|&date| {
                            (0..6).map(move |i| {
                                if i == 4 {
                                    None
                                } else {
                                    Some(
                                        (i + 1) as f64
                                            + if date == 20250214 {
                                                (i * i) as f64
                                            } else {
                                                0.0
                                            },
                                    )
                                }
                            })
                        })
                        .collect(),
                ),
            ),
        ]))
        .unwrap();
        let mut reports = reports(&[
            (CODES[0], 20250115, "org", "a"),
            (CODES[0], 20250210, "org", "b"),
            (CODES[1], 20250112, "org", "a"),
            (CODES[1], 20250113, "org", "b"),
            (CODES[1], 20250114, "org", "c"),
            (CODES[1], 20250117, "org", "d"),
            (CODES[3], 20250125, "org", "a"),
            (CODES[3], 20250126, "org", "b"),
            (CODES[4], 20250115, "org", "a"),
            (CODES[5], 20250115, "org", "a"),
        ]);
        reports.columns.insert(
            "rating".into(),
            ColumnData::Utf8(
                [
                    None,
                    Some("Buy"),
                    Some("Buy"),
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                ]
                .map(|v| v.map(str::to_string))
                .to_vec(),
            ),
        );
        let mut tables = HashMap::from([
            (DatasetId::StockBasic, basic),
            (DatasetId::StockSwClassification, classification),
            (DatasetId::StockBarraDaily, barra),
            (DatasetId::StockAnalystReport, reports),
        ]);
        if with_financial {
            let income = Table::new(BTreeMap::from([
                (
                    "ts_code".into(),
                    strings(&[
                        CODES[0], CODES[1], CODES[2], CODES[4], CODES[5], CODES[0], CODES[0],
                    ]),
                ),
                ("ann_date".into(), ColumnData::I32(vec![Some(20250110); 7])),
                (
                    "f_ann_date".into(),
                    ColumnData::I32(vec![
                        None,
                        None,
                        None,
                        None,
                        None,
                        Some(20250205),
                        Some(20260101),
                    ]),
                ),
                ("end_date".into(), ColumnData::I32(vec![Some(20241231); 7])),
                ("report_type".into(), ColumnData::I64(vec![Some(1); 7])),
                ("update_flag".into(), ColumnData::I64(vec![Some(0); 7])),
            ]))
            .unwrap();
            for dataset in [
                DatasetId::StockIncome,
                DatasetId::StockBalanceSheet,
                DatasetId::StockCashFlow,
            ] {
                tables.insert(dataset, income.clone());
            }
        }
        if reverse_rows {
            for table in tables.values_mut() {
                *table = table
                    .take(&(0..table.len).rev().collect::<Vec<_>>())
                    .unwrap();
            }
        }
        let mut pool = DataPool::from_daily_tables_for_test(tables, &context).unwrap();
        pool.set_trading_calendar(Arc::new(TradingCalendar::from_open_dates(DATES.to_vec())));
        pool
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
    fn analyst_coverage_monthly_freeze_pit_fill_and_batch_equivalence() {
        let ids = ["cov".into(), "anncov".into(), "firstcov".into()];
        let target = &DATES[1..];
        let full = compute(&ids, &fixture(target, false, true), &mut State::default()).unwrap();
        for series in &full {
            for code in CODES {
                assert_eq!(value(series, 20250203, code), value(series, 20250214, code));
                assert_eq!(value(series, 20250228, code), value(series, 20250303, code));
            }
            assert_eq!(value(series, 20250203, CODES[4]), None);
            assert_eq!(value(series, 20250203, CODES[5]), None);
        }
        assert_eq!(value(&full[1], 20250203, CODES[3]), Some(0.0));
        assert_ne!(
            value(&full[0], 20250203, CODES[0]),
            value(&full[0], 20250228, CODES[0])
        );
        let mut state = State::default();
        for dates in [&DATES[1..2], &DATES[2..]] {
            let pool = fixture(dates, true, true);
            let batch = compute(&ids, &pool, &mut state).unwrap();
            let cold = compute(&ids, &pool, &mut State::default()).unwrap();
            for j in 0..3 {
                for &date in dates {
                    for code in CODES {
                        assert_eq!(value(&batch[j], date, code), value(&full[j], date, code));
                        assert_eq!(value(&cold[j], date, code), value(&full[j], date, code));
                    }
                }
            }
        }
        let cov_only = compute(
            &ids[..1],
            &fixture(target, true, false),
            &mut State::default(),
        )
        .unwrap();
        assert_eq!(cov_only.len(), 1);
        for code in CODES {
            assert_eq!(
                value(&cov_only[0], 20250203, code),
                value(&full[0], 20250203, code)
            );
        }
        let first_only = compute(
            &ids[2..],
            &fixture(target, true, false),
            &mut State::default(),
        )
        .unwrap();
        assert_eq!(first_only.len(), 1);
        assert_eq!(first_only[0].spec.id, "firstcov");
        for &date in target {
            for code in CODES {
                assert_eq!(
                    value(&first_only[0], date, code),
                    value(&full[2], date, code)
                );
            }
        }
        assert_ne!(
            value(&full[2], 20250203, CODES[0]),
            value(&full[2], 20250228, CODES[0])
        );
    }

    #[test]
    fn analyst_coverage_pit_disclosure_dates_include_versions_not_future() {
        let pool = fixture(&DATES[1..], false, true);
        let reader = pool
            .financial_reader(DatasetId::StockIncome, ReportTypePreference::consolidated())
            .unwrap();
        assert_eq!(
            reader.disclosure_dates_between(CODES[0], 20250101, 20250228),
            [20250110, 20250205]
        );
        assert_eq!(
            reader.disclosure_dates_between(CODES[0], 20250110, 20250131),
            []
        );
    }

    #[test]
    fn analyst_coverage_metadata_and_shared_provider() {
        let cov = AnalystCoverage(Output::Cov);
        let ann = AnalystCoverage(Output::Anncov);
        let first = AnalystCoverage(Output::Firstcov);
        assert_eq!(cov.compute_provider_key(), ann.compute_provider_key());
        assert_eq!(cov.compute_provider_key(), first.compute_provider_key());
        for factor in [&cov, &ann, &first] {
            let spec = factor.spec();
            for tag in ["DFZQ", "analyst"] {
                assert!(spec.tags.contains(&tag.into()));
            }
            assert!(!spec
                .dependencies
                .iter()
                .any(|r| r.dataset == DatasetId::StockDailyPv));
            assert!(spec.tags.contains(&"deprecated".into()));
        }
        assert!(!cov.spec().tags.contains(&"fundamental".into()));
        assert!(ann.spec().tags.contains(&"fundamental".into()));
        assert!(!cov
            .spec()
            .dependencies
            .iter()
            .any(|r| r.dataset == DatasetId::StockIncome));
        assert!(ann
            .spec()
            .dependencies
            .iter()
            .any(|r| r.dataset == DatasetId::StockIncome));
        let first_spec = first.spec();
        assert!(!first_spec.tags.contains(&"fundamental".into()));
        assert!(!first_spec
            .dependencies
            .iter()
            .any(|r| r.dataset == DatasetId::StockIncome));
        assert_eq!(
            first_spec.dependencies[0].columns,
            ["report_date", "org_name", "rating"]
        );
        for factor in [&cov, &ann] {
            assert!(!factor.spec().dependencies[0]
                .columns
                .contains(&"rating".into()));
        }
    }

    #[test]
    fn analyst_coverage_firstcov_window_first_and_first_rating_union() {
        let dates = [
            20230101, 20250110, 20250210, 20250310, 20250310, 20250410, 20250110, 20250210,
            20250310, 20250110, 20250110, 20250701, 20250110,
        ];
        let orgs = [
            "a", " a ", "a", "a", "a", "a", "b", "b", "b", "c", "c", "d", " ",
        ];
        let ratings = [
            Some("Buy"),
            None,
            Some("\u{65e0}"),
            Some("Buy"),
            None,
            Some("Buy"),
            Some("Buy"),
            Some("--"),
            Some("Buy"),
            None,
            Some("Buy"),
            Some("Buy"),
            Some("Buy"),
        ];
        let table = Table::new(BTreeMap::from([
            ("ts_code".into(), strings(&vec![CODES[0]; dates.len()])),
            (
                "report_date".into(),
                ColumnData::I32(dates.map(Some).to_vec()),
            ),
            ("org_name".into(), strings(&orgs)),
            (
                "rating".into(),
                ColumnData::Utf8(ratings.map(|v| v.map(str::to_string)).to_vec()),
            ),
        ]))
        .unwrap();
        // firstcov requires neither forecast values nor report titles/authors.
        let index = report_index(&table, false, true).unwrap();
        assert!(index.dates.is_empty());
        let history = index.first.get(CODES[0]);
        assert_eq!(firstcov_raw(history, 20241231, 20250630), 2.0);
        assert_eq!(firstcov_raw(history, 20241231, 20250228), 3_f64.sqrt());
        assert_eq!(firstcov_raw(history, 20250131, 20250630), 2.0);
        assert_eq!(firstcov_raw(history, 20250630, 20250731), 1.0);
        assert_eq!(firstcov_raw(None, 20241231, 20250630), 0.0);
        let reversed = table
            .take(&(0..table.len).rev().collect::<Vec<_>>())
            .unwrap();
        let reversed = report_index(&reversed, false, true).unwrap();
        assert_eq!(
            firstcov_raw(reversed.first.get(CODES[0]), 20241231, 20250630),
            2.0
        );
    }

    #[test]
    fn analyst_coverage_firstcov_rating_sentinels() {
        for value in [
            None,
            Some(""),
            Some("  "),
            Some("\u{65e0}"),
            Some("\u{672a}\u{8bc4}\u{7ea7}"),
            Some(" n/A "),
            Some("NaN"),
            Some("Not Rated"),
            Some("--"),
        ] {
            assert!(!valid_rating(value), "{value:?}");
        }
        for value in [
            "Buy",
            "Sell",
            "Neutral",
            " \u{4e70}\u{5165} ",
            "\u{4e2d}\u{6027}",
        ] {
            assert!(valid_rating(Some(value)));
        }
    }
}

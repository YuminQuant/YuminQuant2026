use std::any::Any;
use std::collections::HashMap;

use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::{DataPool, Table};
use crate::error::{err, Result};
use crate::factor::common::stock_daily_ops::{
    is_bj_stock, neutralize_and_fill_missing as neutralize_and_fill,
};
use crate::factor::common::{ClassificationLevel, ClassificationMap, ReportTypePreference};
use crate::factor::{Factor, FactorUpdatePolicy};
#[cfg(test)]
use crate::operators::cross_sectional::cs_neutralize_regression;

pub struct StockDailyFom;
pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyFom)
}

#[derive(Default)]
struct State {
    anchor: Option<i32>,
    values: HashMap<String, Option<f64>>,
}

impl Factor for StockDailyFom {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "fom".into(), aliases: vec!["FOM".into(), "FOM Annual 12M".into()],
            name: "Annual Forecast Optimism Monthly".into(),
            asset_class: AssetClass::Stock, frequency: Frequency::Daily, version: "0.1.0".into(),
            tags: ["DFZQ", "analyst", "fundamental", "financial", "pit", "monthly_update", "daily", "neutralize", "barra", "size", "sector", "deprecated"].into_iter().map(str::to_string).collect(),
            description: "Annual FOM by report count: (forecasts below current minus above current)/N, N>=3. Monthly window (anchor minus 12 calendar months, anchor]. Jan-Mar target prior year; April-Dec target current year. Prefer PIT annual n_income_attr_p/10000, otherwise average individual scores of latest-day analyst np forecasts (wan yuan). Deduplicate stock/year/date/org/author/title, last row wins. At month end neutralize valid raw FOM against SW L1 and SIZE; missing raw with valid exposures and a fitted industry gets zero residual, otherwise null. Hold final monthly values by stock code until next month end. No express/forecast or zscore. Excludes BJ. Report_date determines availability.".into(),
            dependencies: vec![
                DataRequest::new(DatasetId::StockAnalystReport, &["np", "report_date", "quarter", "org_name", "author_name", "report_title"]),
                DataRequest::financial_quarters(DatasetId::StockIncome, &["n_income_attr_p"], 12),
                DataRequest::new(DatasetId::StockBarraDaily, &["SIZE"]),
                DataRequest::new(DatasetId::StockSwClassification, &["l1_code"]),
            ],
            intraday_raw_dependencies: vec![], lookback: Lookback { trading_days: 32 },
        }
    }
    fn update_policy(&self) -> FactorUpdatePolicy {
        FactorUpdatePolicy::FinancialEventStateDailyFast
    }
    fn initial_compute_state(&self, _: &[String]) -> Box<dyn Any + Send> {
        Box::new(State::default())
    }
    fn compute(&self, _: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        compute(data, &mut State::default())
    }
    fn compute_many_stateful(
        &self,
        ids: &[String],
        _: &FactorContext,
        data: &DataPool,
        state: &mut (dyn Any + Send),
    ) -> Result<Vec<FactorSeries>> {
        if !ids.iter().any(|id| id == "fom") {
            return Ok(vec![]);
        }
        Ok(vec![compute(
            data,
            state
                .downcast_mut::<State>()
                .ok_or_else(|| err("FOM state mismatch"))?,
        )?])
    }
}

#[derive(Clone, Copy)]
struct Forecast {
    date: i32,
    value: f64,
}
type Reports<'a> = HashMap<(&'a str, i32), Vec<Forecast>>;

fn report_index(table: &Table) -> Result<Reports<'_>> {
    let codes = table.required_utf8("ts_code")?;
    let dates = table.required_i32_date_cast("report_date")?;
    let quarters = table.required_utf8("quarter")?;
    let values = table.required_f64_cast("np")?;
    let orgs = table.required_utf8("org_name")?;
    let authors = table.required_utf8("author_name")?;
    let titles = table.required_utf8("report_title")?;
    let mut unique = HashMap::new();
    for i in 0..table.len {
        let (Some(code), Some(date), Some(quarter)) =
            (codes[i].as_deref(), dates[i], quarters[i].as_deref())
        else {
            continue;
        };
        let Some(year) = quarter
            .strip_suffix("Q4")
            .and_then(|s| s.parse::<i32>().ok())
        else {
            continue;
        };
        // Source has no report ID: preserve distinct reports, collapse repeat rows only.
        unique.insert(
            (
                code,
                year,
                date,
                orgs[i].as_deref(),
                authors[i].as_deref(),
                titles[i].as_deref(),
            ),
            values[i].filter(|v| v.is_finite()),
        );
    }
    let mut reports = Reports::new();
    for ((code, year, date, _, _, _), value) in unique {
        if let Some(value) = value {
            reports
                .entry((code, year))
                .or_default()
                .push(Forecast { date, value });
        }
    }
    for rows in reports.values_mut() {
        rows.sort_unstable_by(|a, b| {
            a.date
                .cmp(&b.date)
                .then_with(|| a.value.total_cmp(&b.value))
        });
    }
    Ok(reports)
}

fn year_for_anchor(anchor: i32) -> i32 {
    anchor / 10000 - i32::from(anchor / 100 % 100 < 4)
}
fn window_start(anchor: i32) -> i32 {
    let md = if anchor % 10000 == 229 {
        228
    } else {
        anchor % 10000
    };
    (anchor / 10000 - 1) * 10000 + md
}
fn score(rows: &[Forecast], anchor: i32, actual: Option<Option<f64>>) -> Option<f64> {
    let start = rows.partition_point(|r| r.date <= window_start(anchor));
    let end = rows.partition_point(|r| r.date <= anchor);
    let rows = &rows[start..end];
    if rows.len() < 3 {
        return None;
    }
    let mut sorted: Vec<_> = rows.iter().map(|r| r.value).collect();
    sorted.sort_unstable_by(f64::total_cmp);
    let rank_score = |base: f64| {
        let k = sorted.partition_point(|v| *v < base);
        let m = sorted.len() - sorted.partition_point(|v| *v <= base);
        (k as f64 - m as f64) / sorted.len() as f64
    };
    if let Some(actual) = actual {
        return actual.filter(|v| v.is_finite()).map(rank_score);
    }
    let latest = rows.last()?.date;
    let last = &rows[rows.partition_point(|r| r.date < latest)..];
    Some(last.iter().map(|r| rank_score(r.value)).sum::<f64>() / last.len() as f64)
}

fn compute(data: &DataPool, state: &mut State) -> Result<FactorSeries> {
    let panel = data.stock_universe_panel()?;
    let calendar = data.trading_calendar()?;
    let size = panel.column_from_table(data.daily(DatasetId::StockBarraDaily)?, "SIZE")?;
    let sectors = ClassificationMap::from_table(
        data.daily(DatasetId::StockSwClassification)?,
        ClassificationLevel::Sector,
    )?;
    let income =
        data.financial_reader(DatasetId::StockIncome, ReportTypePreference::consolidated())?;
    let mut reports = None;
    let n = panel.instruments().len();
    let mut values = vec![None; panel.shape_len()];
    for (day, &date) in panel.dates().iter().enumerate() {
        if !panel.is_target_date(date) {
            continue;
        }
        let Some(anchor) = calendar.month_end_on_or_before(date) else {
            continue;
        };
        if state.anchor != Some(anchor) {
            if reports.is_none() {
                reports = Some(report_index(data.daily(DatasetId::StockAnalystReport)?)?);
            }
            let reports = reports.as_ref().unwrap();
            let anchor_day = panel
                .dates()
                .binary_search(&anchor)
                .map_err(|_| err("FOM month-end anchor missing from warmup panel"))?;
            let year = year_for_anchor(anchor);
            state.values.clear();
            let mut raw = vec![None; n];
            let mut groups = sectors.groups_for(anchor, panel.instruments());
            for (i, code) in panel.instruments().iter().enumerate() {
                if is_bj_stock(code) || !panel.is_present_offset(anchor_day * n + i) {
                    groups[i] = None;
                    continue;
                }
                let actual = income
                    .record_for_end_date(code, anchor, year * 10000 + 1231)
                    .map(|r| {
                        r.column("n_income_attr_p")
                            .filter(|v| v.is_finite())
                            .map(|v| v / 10000.0)
                    });
                let value = reports
                    .get(&(code.as_str(), year))
                    .and_then(|rows| score(rows, anchor, actual));
                raw[i] = value;
            }
            let residual = neutralize_and_fill(
                &raw,
                &size.values()[anchor_day * n..(anchor_day + 1) * n],
                &groups,
            );
            state
                .values
                .extend(panel.instruments().iter().cloned().zip(residual));
            state.anchor = Some(anchor);
        }
        for (i, code) in panel.instruments().iter().enumerate() {
            if panel.is_present_offset(day * n + i) {
                values[day * n + i] = state.values.get(code).copied().flatten();
            }
        }
    }
    Ok(panel
        .column_from_values(values)?
        .to_factor_series(StockDailyFom.spec()))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fom_neutralization_fill_supported_exposures_only() {
        let raw = [Some(1.0), Some(3.0), Some(2.0), None, None, None, None];
        let size = [
            Some(1.0),
            Some(2.0),
            Some(3.0),
            Some(4.0),
            None,
            Some(2.0),
            Some(2.0),
        ];
        let groups = [
            Some("a"),
            Some("a"),
            Some("a"),
            Some("a"),
            Some("a"),
            Some("b"),
            None,
        ]
        .map(|g| g.map(str::to_string));
        let result = neutralize_and_fill(&raw, &size, &groups);
        let original = cs_neutralize_regression(&raw, &[&size], Some(&groups), None);
        assert_eq!(&result[..3], &original[..3]);
        assert_eq!(result[3], Some(0.0));
        assert_eq!(&result[4..], &[None, None, None]);
        let filled = [Some(1.0), Some(3.0), Some(2.0), Some(3.0), None, None, None];
        let explicit = cs_neutralize_regression(&filled, &[&size], Some(&groups), None);
        for i in 0..4 {
            assert!((result[i].unwrap() - explicit[i].unwrap()).abs() < 1e-10);
        }
        assert!(neutralize_and_fill(&[None; 7], &size, &groups)
            .iter()
            .all(Option::is_none));
    }
    #[test]
    fn fom_formula_ties_actual_and_latest_multiple_reports() {
        let rows = [
            Forecast {
                date: 20250101,
                value: 1.0,
            },
            Forecast {
                date: 20250201,
                value: 2.0,
            },
            Forecast {
                date: 20250201,
                value: 4.0,
            },
        ];
        assert_eq!(score(&rows, 20250228, None), Some(1.0 / 3.0));
        assert_eq!(score(&rows, 20250228, Some(Some(3.0))), Some(1.0 / 3.0));
        assert_eq!(score(&rows, 20250228, Some(Some(2.0))), Some(0.0));
        assert_eq!(score(&rows, 20250228, Some(None)), None);
        assert_eq!(score(&rows[..2], 20250228, None), None);
        assert_eq!(score(&rows, 20260101, None), None);
    }
    #[test]
    fn fom_calendar_switch_and_no_false_batch_end() {
        let c = crate::calendar::TradingCalendar::from_open_dates(vec![
            20250331, 20250401, 20250429, 20250506, 20250530, 20250603,
        ]);
        assert_eq!(c.month_end_on_or_before(20250401), Some(20250331));
        assert_eq!(c.month_end_on_or_before(20250428), Some(20250331));
        assert_eq!(c.month_end_on_or_before(20250429), Some(20250429));
        assert_eq!(c.month_end_on_or_before(20250603), Some(20250530));
        assert_eq!(year_for_anchor(20250331), 2024);
        assert_eq!(year_for_anchor(20250429), 2025);
        assert_eq!(window_start(20240229), 20230228);
        let spec = StockDailyFom.spec();
        for tag in ["DFZQ", "analyst", "fundamental"] {
            assert!(spec.tags.contains(&tag.into()));
        }
        assert!(!spec
            .dependencies
            .iter()
            .any(|d| d.dataset == DatasetId::StockDailyPv));
    }
}

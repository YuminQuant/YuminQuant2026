use std::any::Any;

use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::{err, Result};
use crate::factor::common::financial::previous_quarter_end_date;
use crate::factor::common::stock_daily_ops::{is_bj_stock, neutralize_size_sector};
use crate::factor::common::vector::clean;
use crate::factor::common::{
    cached_financial_stock_snapshots, compute_financial_event_snapshot_many, DailyPanel,
    EventDrivenCrossSectionCache, FinancialEventMarker, FinancialEventMarkerBuilder,
    FinancialEventSchedule, FinancialEventTable, FinancialStatementDataset, PanelColumn,
    PitFinancialData, ReportTypePreference,
};
use crate::factor::{Factor, FactorUpdatePolicy};
use crate::operators::cs_pctrank;

const VERSION: &str = "0.1.0";
const FINANCIAL_QUARTERS: usize = 16;
const ROE_CHAIN_COUNT: usize = 13;
const ROE_LAST_COUNT: usize = 12;
const ROE_EPS: f64 = 1e-12;

const INCOME_COLUMN: &str = "n_income_attr_p";
const EQUITY_COLUMN: &str = "total_hldr_eqy_exc_min_int";

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct RoeSnapshot {
    yoy: Option<f64>,
    stb: Option<f64>,
    last: Option<f64>,
    growth: Option<f64>,
}

pub struct StockDailyRoeEnhance;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyRoeEnhance)
}

impl Factor for StockDailyRoeEnhance {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "roe_enhance".to_string(),
            aliases: vec!["ROE Enhance".to_string(), "ROE_ehance".to_string()],
            name: "roe_enhance".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: tags(),
            description: "DWZQ ROE enhancement factor. It builds quarterly ROE from PIT single-quarter attributable net profit over latest shareholder equity. Each ROE component is first neutralized by Barra SIZE and SW sector, then transformed with cross-sectional percentile rank with missing ranks filled to 0.5 for present non-BJ stocks. The components are combined through nested percentile-rank layers without final neutralization.".to_string(),
            dependencies: vec![
                DataRequest::new(DatasetId::StockDailyPv, &["close"]),
                DataRequest::financial_quarters(
                    DatasetId::StockIncome,
                    &[INCOME_COLUMN],
                    FINANCIAL_QUARTERS,
                ),
                DataRequest::financial_quarters(
                    DatasetId::StockBalanceSheet,
                    &[EQUITY_COLUMN],
                    FINANCIAL_QUARTERS,
                ),
                DataRequest::new(DatasetId::StockBarraDaily, &["SIZE"]),
                DataRequest::new(DatasetId::StockSwClassification, &["l1_code"]),
            ],
            intraday_raw_dependencies: Vec::new(),
            lookback: Lookback { trading_days: 0 },
        }
    }

    fn update_policy(&self) -> FactorUpdatePolicy {
        FactorUpdatePolicy::FinancialEventSnapshot
    }

    fn initial_compute_state(&self, _requested_ids: &[String]) -> Box<dyn Any + Send> {
        Box::new(EventDrivenCrossSectionCache::default())
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let panel = data.daily_panel(DatasetId::StockDailyPv)?;
        let income = PitFinancialData::from_table(
            data.daily(DatasetId::StockIncome)?,
            &[INCOME_COLUMN],
            ReportTypePreference::income_single_quarter(),
        )?;
        let balance = PitFinancialData::from_table(
            data.daily(DatasetId::StockBalanceSheet)?,
            &[EQUITY_COLUMN],
            ReportTypePreference::balance_sheet_consolidated(),
        )?;

        let components = roe_component_columns(&panel, &income, &balance)?;
        let r_yoy = neutralize_rank_fill_present_non_bj(&components.yoy, &panel, data)?;
        let r_stb = neutralize_rank_fill_present_non_bj(&components.stb, &panel, data)?;
        let r_last = neutralize_rank_fill_present_non_bj(&components.last, &panel, data)?;
        let r_growth = neutralize_rank_fill_present_non_bj(&components.growth, &panel, data)?;

        let layer1 = pctrank_fill_present_non_bj(&average_pair(&r_growth, &r_last)?, &panel)?;
        let layer2 = pctrank_fill_present_non_bj(&average_pair(&layer1, &r_stb)?, &panel)?;
        let raw = pctrank_fill_present_non_bj(&sum_pair(&layer2, &r_yoy)?, &panel)?;
        Ok(raw.to_factor_series(self.spec()))
    }

    fn compute_many_stateful(
        &self,
        requested_ids: &[String],
        context: &FactorContext,
        data: &DataPool,
        state: &mut (dyn Any + Send),
    ) -> Result<Vec<FactorSeries>> {
        if requested_ids.iter().all(|id| id != "roe_enhance") {
            return Ok(Vec::new());
        }
        let state = state
            .downcast_mut::<EventDrivenCrossSectionCache>()
            .ok_or_else(|| err("roe_enhance received incompatible event cache state"))?;
        let schedule = FinancialEventSchedule::from_tables(&[
            FinancialEventTable::statement(data.daily(DatasetId::StockIncome)?),
            FinancialEventTable::statement(data.daily(DatasetId::StockBalanceSheet)?),
        ])?;
        let specs = [self.spec()];
        compute_financial_event_snapshot_many(
            requested_ids,
            context,
            data,
            state,
            &schedule,
            &specs,
            |_, context, data| self.compute(context, data).map(|series| vec![series]),
        )
    }
}

struct RoeComponentColumns {
    yoy: PanelColumn,
    stb: PanelColumn,
    last: PanelColumn,
    growth: PanelColumn,
}

fn tags() -> Vec<String> {
    [
        "DWZQ",
        "financial",
        "fundamental",
        "pit",
        "roe",
        "profitability",
        "neutralize",
        "barra",
        "size",
        "sector",
        "daily",
    ]
    .iter()
    .map(|value| value.to_string())
    .collect()
}

fn roe_component_columns(
    panel: &DailyPanel,
    income: &PitFinancialData,
    balance: &PitFinancialData,
) -> Result<RoeComponentColumns> {
    let mut yoy = vec![None; panel.shape_len()];
    let mut stb = vec![None; panel.shape_len()];
    let mut last = vec![None; panel.shape_len()];
    let mut growth = vec![None; panel.shape_len()];

    let snapshots = cached_financial_stock_snapshots(
        panel,
        |_, ts_code, offset| is_bj_stock(ts_code) || !panel.is_present_offset(offset),
        |trade_date, ts_code, _| roe_marker(ts_code, trade_date, income, balance),
        |trade_date, ts_code, _| roe_snapshot_for_stock(ts_code, trade_date, income, balance),
    );

    for (offset, snapshot) in snapshots.into_iter().enumerate() {
        let Some(snapshot) = snapshot else {
            continue;
        };
        yoy[offset] = snapshot.yoy;
        stb[offset] = snapshot.stb;
        last[offset] = snapshot.last;
        growth[offset] = snapshot.growth;
    }

    Ok(RoeComponentColumns {
        yoy: panel.column_from_values(yoy)?,
        stb: panel.column_from_values(stb)?,
        last: panel.column_from_values(last)?,
        growth: panel.column_from_values(growth)?,
    })
}

fn roe_marker(
    ts_code: &str,
    trade_date: i32,
    income: &PitFinancialData,
    balance: &PitFinancialData,
) -> Option<FinancialEventMarker> {
    let latest_end = income.latest_quarter_end_date(ts_code, trade_date)?;
    let mut builder = FinancialEventMarkerBuilder::new();
    let mut current = Some(latest_end);
    for _ in 0..ROE_CHAIN_COUNT {
        let Some(end_date) = current else {
            break;
        };
        builder.include_record_for_end_date(
            FinancialStatementDataset::Income,
            income,
            ts_code,
            trade_date,
            end_date,
        );
        builder.include_record_for_end_date(
            FinancialStatementDataset::BalanceSheet,
            balance,
            ts_code,
            trade_date,
            end_date,
        );
        current = previous_quarter_end_date(end_date);
    }
    builder.build()
}

fn roe_snapshot_for_stock(
    ts_code: &str,
    trade_date: i32,
    income: &PitFinancialData,
    balance: &PitFinancialData,
) -> Option<RoeSnapshot> {
    let latest_end = income.latest_quarter_end_date(ts_code, trade_date)?;
    let mut values = Vec::with_capacity(ROE_CHAIN_COUNT);
    let mut current = Some(latest_end);
    for _ in 0..ROE_CHAIN_COUNT {
        let Some(end_date) = current else {
            values.push(None);
            current = None;
            continue;
        };
        let profit = income
            .record_for_end_date(ts_code, trade_date, end_date)
            .and_then(|record| record.column(INCOME_COLUMN));
        let equity = balance
            .record_for_end_date(ts_code, trade_date, end_date)
            .and_then(|record| record.column(EQUITY_COLUMN));
        values.push(roe_quarter(profit, equity));
        current = previous_quarter_end_date(end_date);
    }
    Some(roe_snapshot_from_chain(&values))
}

fn roe_snapshot_from_chain(values: &[Option<f64>]) -> RoeSnapshot {
    RoeSnapshot {
        yoy: binary_diff(values, 0, 4),
        stb: roe_stability(values),
        last: strict_mean(values, 0, ROE_LAST_COUNT),
        growth: roe_growth(values),
    }
}

fn roe_quarter(profit: Option<f64>, equity: Option<f64>) -> Option<f64> {
    match (clean(profit), clean(equity)) {
        (Some(profit), Some(equity)) if equity > ROE_EPS => {
            let value = profit / equity;
            value.is_finite().then_some(value)
        }
        _ => None,
    }
}

fn binary_diff(values: &[Option<f64>], left_idx: usize, right_idx: usize) -> Option<f64> {
    let left = clean(*values.get(left_idx)?)?;
    let right = clean(*values.get(right_idx)?)?;
    let value = left - right;
    value.is_finite().then_some(value)
}

fn roe_growth(values: &[Option<f64>]) -> Option<f64> {
    let current = clean(*values.first()?)?;
    let yoy = clean(*values.get(4)?)?;
    let two_year = clean(*values.get(8)?)?;
    let value = current - 2.0 * yoy + two_year;
    value.is_finite().then_some(value)
}

fn roe_stability(values: &[Option<f64>]) -> Option<f64> {
    let selected = [
        clean(*values.first()?)?,
        clean(*values.get(4)?)?,
        clean(*values.get(8)?)?,
        clean(*values.get(12)?)?,
    ];
    let mean = selected.iter().sum::<f64>() / selected.len() as f64;
    let variance = selected
        .iter()
        .map(|value| {
            let delta = value - mean;
            delta * delta
        })
        .sum::<f64>()
        / (selected.len() as f64 - 1.0);
    let std = variance.sqrt();
    (std > ROE_EPS)
        .then_some(mean / std)
        .filter(|value| value.is_finite())
}

fn strict_mean(values: &[Option<f64>], start: usize, count: usize) -> Option<f64> {
    let mut sum = 0.0;
    for idx in start..start + count {
        sum += clean(*values.get(idx)?)?;
    }
    let value = sum / count as f64;
    value.is_finite().then_some(value)
}

fn neutralize_rank_fill_present_non_bj(
    column: &PanelColumn,
    panel: &DailyPanel,
    data: &DataPool,
) -> Result<PanelColumn> {
    let neutralized = neutralize_size_sector(column, panel, data)?;
    pctrank_fill_present_non_bj(&neutralized, panel)
}

fn pctrank_fill_present_non_bj(column: &PanelColumn, panel: &DailyPanel) -> Result<PanelColumn> {
    let ranked = column.cs(|values| cs_pctrank(values, true))?;
    let instrument_count = panel.instruments().len();
    let values = ranked
        .values()
        .iter()
        .enumerate()
        .map(|(offset, value)| {
            let instrument_idx = offset % instrument_count;
            if panel.is_present_offset(offset) && !is_bj_stock(&panel.instruments()[instrument_idx])
            {
                clean(*value).or(Some(0.5))
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    panel.column_from_values(values)
}

fn average_pair(left: &PanelColumn, right: &PanelColumn) -> Result<PanelColumn> {
    left.zip_binary(right, |left, right| match (clean(left), clean(right)) {
        (Some(left), Some(right)) => {
            let value = 0.5 * (left + right);
            value.is_finite().then_some(value)
        }
        _ => None,
    })
}

fn sum_pair(left: &PanelColumn, right: &PanelColumn) -> Result<PanelColumn> {
    left.zip_binary(right, |left, right| match (clean(left), clean(right)) {
        (Some(left), Some(right)) => {
            let value = left + right;
            value.is_finite().then_some(value)
        }
        _ => None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: Option<f64>, expected: Option<f64>) {
        match (actual, expected) {
            (Some(actual), Some(expected)) => assert!(
                (actual - expected).abs() < 1e-10,
                "expected {expected}, got {actual}"
            ),
            (None, None) => {}
            _ => panic!("expected {:?}, got {:?}", expected, actual),
        }
    }

    #[test]
    fn roe_enhance_roe_quarter_rejects_invalid_equity() {
        assert_close(roe_quarter(Some(12.0), Some(120.0)), Some(0.1));
        assert_eq!(roe_quarter(Some(12.0), Some(0.0)), None);
        assert_eq!(roe_quarter(Some(12.0), Some(-1.0)), None);
    }

    #[test]
    fn roe_enhance_components_follow_report_formulas() {
        let chain = [
            Some(0.15),
            Some(0.10),
            Some(0.11),
            Some(0.12),
            Some(0.10),
            Some(0.09),
            Some(0.08),
            Some(0.07),
            Some(0.06),
            Some(0.05),
            Some(0.04),
            Some(0.03),
            Some(0.02),
        ];
        let snapshot = roe_snapshot_from_chain(&chain);

        assert_close(snapshot.yoy, Some(0.05));
        assert_close(snapshot.last, Some(1.0 / 12.0));
        assert_close(snapshot.growth, Some(0.15 - 2.0 * 0.10 + 0.06));

        let selected = [0.15, 0.10, 0.06, 0.02];
        let mean = selected.iter().sum::<f64>() / selected.len() as f64;
        let std = (selected
            .iter()
            .map(|value| {
                let delta = value - mean;
                delta * delta
            })
            .sum::<f64>()
            / 3.0)
            .sqrt();
        assert_close(snapshot.stb, Some(mean / std));
    }

    #[test]
    fn roe_enhance_requires_strict_samples_for_stability_and_last() {
        let mut chain = [Some(0.1); ROE_CHAIN_COUNT];
        chain[11] = None;
        assert_eq!(roe_snapshot_from_chain(&chain).last, None);
        assert!(roe_snapshot_from_chain(&chain).stb.is_none());

        let mut chain = [Some(0.1); ROE_CHAIN_COUNT];
        chain[5] = None;
        assert!(roe_snapshot_from_chain(&chain).last.is_none());
        assert!(roe_snapshot_from_chain(&chain).stb.is_none());
    }

    #[test]
    fn roe_enhance_spec_has_expected_metadata_and_dependencies() {
        let spec = StockDailyRoeEnhance.spec();
        assert_eq!(spec.id, "roe_enhance");
        assert!(spec.tags.iter().any(|tag| tag == "DWZQ"));
        assert!(spec.tags.iter().any(|tag| tag == "roe"));
        assert_eq!(
            spec.dependencies
                .iter()
                .find(|request| request.dataset == DatasetId::StockIncome)
                .and_then(|request| request.financial_quarters),
            Some(FINANCIAL_QUARTERS)
        );
        assert_eq!(
            spec.dependencies
                .iter()
                .find(|request| request.dataset == DatasetId::StockBalanceSheet)
                .and_then(|request| request.financial_quarters),
            Some(FINANCIAL_QUARTERS)
        );
    }
}

use std::any::Any;
use std::collections::BTreeMap;

use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::{err, Result};
use crate::factor::common::financial::previous_quarter_end_date;
use crate::factor::common::stock_daily_ops::is_bj_stock;
use crate::factor::common::{
    cached_financial_stock_snapshots_for_date, compute_financial_event_snapshot_streaming,
    factor_series_to_panel_column, ClassificationLevel, ClassificationMap, DailyPanel,
    EventDrivenCrossSectionCache, FinancialEventMarker, FinancialEventMarkerBuilder,
    FinancialEventSchedule, FinancialPitReader, FinancialStatementDataset,
    InstrumentAlignedSnapshotCache, PanelColumn, ReportTypePreference,
};
use crate::factor::{Factor, FactorUpdatePolicy};

const VERSION: &str = "0.1.0";
const RAW_ID: &str = "__jones_modified_accrual_raw";
const FINANCIAL_QUARTERS: usize = 2;
const PARAM_COUNT: usize = 3;
const MIN_INDUSTRY_OLS_OBS: usize = 4;
const EPS: f64 = 1e-12;

const OPERATE_PROFIT_COLUMN: &str = "operate_profit";
const REVENUE_COLUMN: &str = "revenue";
const CFO_COLUMN: &str = "n_cashflow_act";
const ASSETS_COLUMN: &str = "total_assets";
const PPE_COLUMN: &str = "fix_assets";

pub struct StockDailyJonesModifiedAccrual;

#[derive(Default)]
struct JonesModifiedAccrualState {
    raw_cache: EventDrivenCrossSectionCache,
    snapshot_cache: InstrumentAlignedSnapshotCache<JonesSnapshot>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct JonesSnapshot {
    y: f64,
    x: [f64; PARAM_COUNT],
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct JonesObservation {
    offset: usize,
    snapshot: JonesSnapshot,
}

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyJonesModifiedAccrual)
}

impl Factor for StockDailyJonesModifiedAccrual {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "jones_modified_accrual".to_string(),
            aliases: vec![
                "Jones Modified Accrual".to_string(),
                "Modified Jones Accrual".to_string(),
            ],
            name: "jones_modified_accrual".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: tags(),
            description: "XYZQ modified Jones accrual factor. It computes PIT single-quarter accruals as operating profit minus operating cashflow, scales by average assets, runs CITIC level-1 industry OLS on 1/average-assets, revenue change/average-assets, and fixed assets/average-assets, excludes firms with negative current operating profit, replays raw residuals between financial events, and finally industry-neutralizes within CITIC level-1 industries.".to_string(),
            dependencies: vec![
                DataRequest::new(DatasetId::StockDailyPv, &["close"]),
                DataRequest::financial_quarters(
                    DatasetId::StockIncome,
                    &[OPERATE_PROFIT_COLUMN, REVENUE_COLUMN],
                    FINANCIAL_QUARTERS,
                ),
                DataRequest::financial_quarters(
                    DatasetId::StockCashFlow,
                    &[CFO_COLUMN],
                    FINANCIAL_QUARTERS,
                ),
                DataRequest::financial_quarters(
                    DatasetId::StockBalanceSheet,
                    &[ASSETS_COLUMN, PPE_COLUMN],
                    FINANCIAL_QUARTERS,
                ),
                DataRequest::new(DatasetId::StockCiClassification, &["l1_code"]),
            ],
            intraday_raw_dependencies: Vec::new(),
            lookback: Lookback { trading_days: 0 },
        }
    }

    fn update_policy(&self) -> FactorUpdatePolicy {
        FactorUpdatePolicy::FinancialEventSnapshot
    }

    fn initial_compute_state(&self, _requested_ids: &[String]) -> Box<dyn Any + Send> {
        Box::new(JonesModifiedAccrualState::default())
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let mut snapshot_cache = InstrumentAlignedSnapshotCache::default();
        self.compute_with_snapshot_cache(data, &mut snapshot_cache)
    }

    fn compute_many_stateful(
        &self,
        requested_ids: &[String],
        context: &FactorContext,
        data: &DataPool,
        state: &mut (dyn Any + Send),
    ) -> Result<Vec<FactorSeries>> {
        if requested_ids
            .iter()
            .all(|id| id != "jones_modified_accrual")
        {
            return Ok(Vec::new());
        }
        let state = state
            .downcast_mut::<JonesModifiedAccrualState>()
            .ok_or_else(|| err("jones_modified_accrual received incompatible state"))?;
        let income = data.financial_reader(
            DatasetId::StockIncome,
            ReportTypePreference::income_single_quarter(),
        )?;
        let cashflow = data.financial_reader(
            DatasetId::StockCashFlow,
            ReportTypePreference::income_single_quarter(),
        )?;
        let balance = data.financial_reader(
            DatasetId::StockBalanceSheet,
            ReportTypePreference::balance_sheet_consolidated(),
        )?;
        let sector_map = ClassificationMap::from_table(
            data.daily(DatasetId::StockCiClassification)?,
            ClassificationLevel::Sector,
        )?;
        let schedule = FinancialEventSchedule::from_pit_readers(&[
            income.clone(),
            cashflow.clone(),
            balance.clone(),
        ]);
        let raw_specs = [raw_spec()];
        let raw_series = compute_financial_event_snapshot_streaming(
            requested_ids,
            context,
            data,
            &mut state.raw_cache,
            &schedule,
            &raw_specs,
            |_, _, data| {
                self.compute_raw_with_prepared_inputs(
                    data,
                    &income,
                    &cashflow,
                    &balance,
                    &sector_map,
                    &mut state.snapshot_cache,
                )
                .map(|series| vec![series])
            },
        )?;
        self.finalize_raw_series(data, raw_series)
            .map(|series| vec![series])
    }
}

impl StockDailyJonesModifiedAccrual {
    fn compute_with_snapshot_cache(
        &self,
        data: &DataPool,
        snapshot_cache: &mut InstrumentAlignedSnapshotCache<JonesSnapshot>,
    ) -> Result<FactorSeries> {
        let income = data.financial_reader(
            DatasetId::StockIncome,
            ReportTypePreference::income_single_quarter(),
        )?;
        let cashflow = data.financial_reader(
            DatasetId::StockCashFlow,
            ReportTypePreference::income_single_quarter(),
        )?;
        let balance = data.financial_reader(
            DatasetId::StockBalanceSheet,
            ReportTypePreference::balance_sheet_consolidated(),
        )?;
        let sector_map = ClassificationMap::from_table(
            data.daily(DatasetId::StockCiClassification)?,
            ClassificationLevel::Sector,
        )?;
        let raw_series = vec![self.compute_raw_with_prepared_inputs(
            data,
            &income,
            &cashflow,
            &balance,
            &sector_map,
            snapshot_cache,
        )?];
        self.finalize_raw_series(data, raw_series)
    }

    fn compute_raw_with_prepared_inputs(
        &self,
        data: &DataPool,
        income: &FinancialPitReader<'_>,
        cashflow: &FinancialPitReader<'_>,
        balance: &FinancialPitReader<'_>,
        sector_map: &ClassificationMap,
        snapshot_cache: &mut InstrumentAlignedSnapshotCache<JonesSnapshot>,
    ) -> Result<FactorSeries> {
        let panel = data.daily_panel(DatasetId::StockDailyPv)?;
        let raw =
            jones_residual_column(panel, income, cashflow, balance, sector_map, snapshot_cache)?;
        Ok(raw.to_factor_series(raw_spec()))
    }

    fn finalize_raw_series(
        &self,
        data: &DataPool,
        raw_series: Vec<FactorSeries>,
    ) -> Result<FactorSeries> {
        let panel = data.daily_panel(DatasetId::StockDailyPv)?;
        let series = raw_series
            .into_iter()
            .find(|series| series.spec.id == RAW_ID)
            .ok_or_else(|| err("missing jones_modified_accrual raw series"))?;
        let raw = factor_series_to_panel_column(panel, &series)?;
        let sector_map = ClassificationMap::from_table(
            data.daily(DatasetId::StockCiClassification)?,
            ClassificationLevel::Sector,
        )?;
        let neutralized = industry_demean(&raw, panel, &sector_map)?;
        Ok(neutralized.to_factor_series(self.spec()))
    }
}

fn jones_residual_column(
    panel: &DailyPanel,
    income: &FinancialPitReader<'_>,
    cashflow: &FinancialPitReader<'_>,
    balance: &FinancialPitReader<'_>,
    sector_map: &ClassificationMap,
    cache: &mut InstrumentAlignedSnapshotCache<JonesSnapshot>,
) -> Result<PanelColumn> {
    let instrument_count = panel.instruments().len();
    let mut values = vec![None; panel.shape_len()];

    for (date_idx, trade_date) in panel.dates().iter().copied().enumerate() {
        if !panel.is_target_date(trade_date) {
            continue;
        }
        let snapshots = cached_financial_stock_snapshots_for_date(
            panel,
            trade_date,
            cache,
            |_, ts_code, offset| {
                is_bj_stock(ts_code)
                    || !panel.is_present_offset(offset)
                    || sector_map.group_for(trade_date, ts_code).is_none()
            },
            |trade_date, ts_code, _| jones_marker(ts_code, trade_date, income, cashflow, balance),
            |trade_date, ts_code, _| jones_snapshot(ts_code, trade_date, income, cashflow, balance),
        );
        let date_offset = date_idx * instrument_count;
        let mut by_sector = BTreeMap::<String, Vec<JonesObservation>>::new();
        for (instrument_idx, ts_code) in panel.instruments().iter().enumerate() {
            let offset = date_offset + instrument_idx;
            if is_bj_stock(ts_code) || !panel.is_present_offset(offset) {
                continue;
            }
            let Some(snapshot) = snapshots[instrument_idx] else {
                continue;
            };
            let Some(group) = sector_map.group_for(trade_date, ts_code) else {
                continue;
            };
            by_sector
                .entry(group.to_string())
                .or_default()
                .push(JonesObservation { offset, snapshot });
        }
        for observations in by_sector.values() {
            for (offset, residual) in ols_residuals(observations) {
                values[offset] = Some(residual);
            }
        }
    }

    panel.column_from_values(values)
}

fn jones_marker(
    ts_code: &str,
    trade_date: i32,
    income: &FinancialPitReader<'_>,
    cashflow: &FinancialPitReader<'_>,
    balance: &FinancialPitReader<'_>,
) -> Option<FinancialEventMarker> {
    let end_t = income.latest_quarter_end_date(ts_code, trade_date)?;
    let end_t1 = previous_quarter_end_date(end_t)?;
    let mut builder = FinancialEventMarkerBuilder::new();
    for end_date in [end_t, end_t1] {
        builder.include_reader_record_for_end_date(
            FinancialStatementDataset::Income,
            income,
            ts_code,
            trade_date,
            end_date,
        );
        builder.include_reader_record_for_end_date(
            FinancialStatementDataset::BalanceSheet,
            balance,
            ts_code,
            trade_date,
            end_date,
        );
    }
    builder.include_reader_record_for_end_date(
        FinancialStatementDataset::CashFlow,
        cashflow,
        ts_code,
        trade_date,
        end_t,
    );
    builder.build()
}

fn jones_snapshot(
    ts_code: &str,
    trade_date: i32,
    income: &FinancialPitReader<'_>,
    cashflow: &FinancialPitReader<'_>,
    balance: &FinancialPitReader<'_>,
) -> Option<JonesSnapshot> {
    let end_t = income.latest_quarter_end_date(ts_code, trade_date)?;
    let end_t1 = previous_quarter_end_date(end_t)?;
    let income_t = income.record_for_end_date(ts_code, trade_date, end_t)?;
    let income_t1 = income.record_for_end_date(ts_code, trade_date, end_t1)?;
    let cashflow_t = cashflow.record_for_end_date(ts_code, trade_date, end_t)?;
    let balance_t = balance.record_for_end_date(ts_code, trade_date, end_t)?;
    let balance_t1 = balance.record_for_end_date(ts_code, trade_date, end_t1)?;

    let operate_profit = clean(income_t.column(OPERATE_PROFIT_COLUMN))?;
    if operate_profit < 0.0 {
        return None;
    }
    let cfo = clean(cashflow_t.column(CFO_COLUMN))?;
    let revenue = clean(income_t.column(REVENUE_COLUMN))?;
    let prev_revenue = clean(income_t1.column(REVENUE_COLUMN))?;
    let assets = clean(balance_t.column(ASSETS_COLUMN)).filter(|value| *value > 0.0)?;
    let prev_assets = clean(balance_t1.column(ASSETS_COLUMN)).filter(|value| *value > 0.0)?;
    let avg_assets = 0.5 * (assets + prev_assets);
    if avg_assets <= EPS {
        return None;
    }
    let accrual = operate_profit - cfo;
    let y = accrual / avg_assets;
    let x = [
        1.0 / avg_assets,
        (revenue - prev_revenue) / avg_assets,
        clean(balance_t.column(PPE_COLUMN)).unwrap_or(0.0) / avg_assets,
    ];
    (y.is_finite() && x.iter().all(|value| value.is_finite())).then_some(JonesSnapshot { y, x })
}

fn ols_residuals(observations: &[JonesObservation]) -> Vec<(usize, f64)> {
    if observations.len() < MIN_INDUSTRY_OLS_OBS {
        return Vec::new();
    }
    let Some(beta) = ols_beta(observations) else {
        return Vec::new();
    };
    observations
        .iter()
        .filter_map(|observation| {
            let fitted = dot(&beta, &observation.snapshot.x);
            let residual = observation.snapshot.y - fitted;
            residual
                .is_finite()
                .then_some((observation.offset, residual))
        })
        .collect()
}

fn ols_beta(observations: &[JonesObservation]) -> Option<[f64; PARAM_COUNT]> {
    let mut xtx = [[0.0; PARAM_COUNT]; PARAM_COUNT];
    let mut xty = [0.0; PARAM_COUNT];
    for observation in observations {
        for i in 0..PARAM_COUNT {
            xty[i] += observation.snapshot.x[i] * observation.snapshot.y;
            for j in 0..PARAM_COUNT {
                xtx[i][j] += observation.snapshot.x[i] * observation.snapshot.x[j];
            }
        }
    }
    solve_linear_system(xtx, xty)
}

fn solve_linear_system(
    mut a: [[f64; PARAM_COUNT]; PARAM_COUNT],
    mut b: [f64; PARAM_COUNT],
) -> Option<[f64; PARAM_COUNT]> {
    for pivot_idx in 0..PARAM_COUNT {
        let mut pivot_row = pivot_idx;
        let mut pivot_abs = a[pivot_idx][pivot_idx].abs();
        for (row_idx, row) in a.iter().enumerate().skip(pivot_idx + 1) {
            let candidate = row[pivot_idx].abs();
            if candidate > pivot_abs {
                pivot_abs = candidate;
                pivot_row = row_idx;
            }
        }
        if pivot_abs <= EPS {
            return None;
        }
        if pivot_row != pivot_idx {
            a.swap(pivot_idx, pivot_row);
            b.swap(pivot_idx, pivot_row);
        }
        let pivot = a[pivot_idx][pivot_idx];
        for col_idx in pivot_idx..PARAM_COUNT {
            a[pivot_idx][col_idx] /= pivot;
        }
        b[pivot_idx] /= pivot;

        for row_idx in 0..PARAM_COUNT {
            if row_idx == pivot_idx {
                continue;
            }
            let factor = a[row_idx][pivot_idx];
            if factor.abs() <= f64::EPSILON {
                continue;
            }
            for col_idx in pivot_idx..PARAM_COUNT {
                a[row_idx][col_idx] -= factor * a[pivot_idx][col_idx];
            }
            b[row_idx] -= factor * b[pivot_idx];
        }
    }
    b.iter().all(|value| value.is_finite()).then_some(b)
}

fn industry_demean(
    raw: &PanelColumn,
    panel: &DailyPanel,
    sector_map: &ClassificationMap,
) -> Result<PanelColumn> {
    let instrument_count = panel.instruments().len();
    let mut output = vec![None; panel.shape_len()];
    for (date_idx, trade_date) in panel.dates().iter().copied().enumerate() {
        let date_offset = date_idx * instrument_count;
        let mut groups = BTreeMap::<String, Vec<usize>>::new();
        for (instrument_idx, ts_code) in panel.instruments().iter().enumerate() {
            let offset = date_offset + instrument_idx;
            if is_bj_stock(ts_code) || !panel.is_present_offset(offset) {
                continue;
            }
            if clean(raw.values()[offset]).is_none() {
                continue;
            }
            let Some(group) = sector_map.group_for(trade_date, ts_code) else {
                continue;
            };
            groups.entry(group.to_string()).or_default().push(offset);
        }
        for offsets in groups.values() {
            let values = offsets
                .iter()
                .filter_map(|offset| clean(raw.values()[*offset]))
                .collect::<Vec<_>>();
            if values.is_empty() {
                continue;
            }
            let mean = values.iter().sum::<f64>() / values.len() as f64;
            for offset in offsets {
                output[*offset] = clean(raw.values()[*offset]).map(|value| value - mean);
            }
        }
    }
    panel.column_from_values(output)
}

fn dot(left: &[f64; PARAM_COUNT], right: &[f64; PARAM_COUNT]) -> f64 {
    (0..PARAM_COUNT).map(|idx| left[idx] * right[idx]).sum()
}

fn clean(value: Option<f64>) -> Option<f64> {
    value.filter(|value| value.is_finite())
}

fn raw_spec() -> FactorSpec {
    FactorSpec {
        id: RAW_ID.to_string(),
        aliases: Vec::new(),
        name: RAW_ID.to_string(),
        asset_class: AssetClass::Stock,
        frequency: Frequency::Daily,
        version: VERSION.to_string(),
        tags: vec!["internal".to_string(), "financial_raw".to_string()],
        description: "Internal jones_modified_accrual industry residual raw series.".to_string(),
        dependencies: Vec::new(),
        intraday_raw_dependencies: Vec::new(),
        lookback: Lookback { trading_days: 0 },
    }
}

fn tags() -> Vec<String> {
    [
        "XYZQ",
        "financial",
        "fundamental",
        "pit",
        "jones",
        "accrual",
        "residual",
        "industry_neutralize",
        "daily",
    ]
    .iter()
    .map(|value| value.to_string())
    .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jones_ols_residuals_fit_simple_three_regressor_model() {
        let observations = vec![
            JonesObservation {
                offset: 0,
                snapshot: JonesSnapshot {
                    y: 1.0,
                    x: [1.0, 0.0, 0.0],
                },
            },
            JonesObservation {
                offset: 1,
                snapshot: JonesSnapshot {
                    y: 2.0,
                    x: [0.0, 1.0, 0.0],
                },
            },
            JonesObservation {
                offset: 2,
                snapshot: JonesSnapshot {
                    y: 3.0,
                    x: [0.0, 0.0, 1.0],
                },
            },
            JonesObservation {
                offset: 3,
                snapshot: JonesSnapshot {
                    y: 6.0,
                    x: [1.0, 1.0, 1.0],
                },
            },
        ];
        let residuals = ols_residuals(&observations);
        assert_eq!(residuals.len(), 4);
        assert!(residuals.iter().all(|(_, residual)| residual.abs() < 1e-10));
    }

    #[test]
    fn spec_has_xyzq_tag() {
        let spec = StockDailyJonesModifiedAccrual.spec();
        assert_eq!(spec.id, "jones_modified_accrual");
        assert!(spec.tags.contains(&"XYZQ".to_string()));
    }
}

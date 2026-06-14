use std::any::Any;
use std::collections::{BTreeMap, BTreeSet};

use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::{err, Result};
use crate::factor::common::financial::previous_quarter_end_date;
use crate::factor::common::stock_daily_ops::{is_bj_stock, neutralize_size_sector};
use crate::factor::common::{
    cached_financial_stock_snapshots_for_date, compute_financial_event_snapshot_streaming_on_panel,
    factor_series_to_panel_column, ClassificationLevel, ClassificationMap, DailyPanel,
    EventDrivenCrossSectionCache, FinancialEventMarker, FinancialEventMarkerBuilder,
    FinancialEventSchedule, FinancialPitReader, FinancialStatementDataset,
    InstrumentAlignedSnapshotCache, PanelColumn, PitFinancialRecordView, ReportTypePreference,
};
use crate::factor::{Factor, FactorUpdatePolicy};
use crate::operators::cs_zscore_by_group;

const VERSION: &str = "0.1.1";
const FACTOR_ID: &str = "fin_quality_stability";
const RAW_ID: &str = "__fin_quality_stability_raw";
const PANEL_WINDOW: usize = 12;
const REQUIRED_QUARTERS: usize = 16;
const REGRESSOR_COUNT: usize = 5;
const MIN_RESIDUALS: usize = 8;
const MIN_INDUSTRY_ROWS: usize = 30;
const EPS: f64 = 1e-12;
const DEMEAN_MAX_ITER: usize = 100;
const DEMEAN_TOL: f64 = 1e-10;

const OPERATE_PROFIT_COLUMN: &str = "operate_profit";
const REVENUE_COLUMN: &str = "revenue";
const OPER_COST_COLUMN: &str = "oper_cost";
const CFO_COLUMN: &str = "n_cashflow_act";
const FNC_CASHFLOW_COLUMN: &str = "n_cash_flows_fnc_act";
const INV_CASHFLOW_COLUMN: &str = "n_cashflow_inv_act";
const ACCOUNTS_RECEIV_COLUMN: &str = "accounts_receiv";
const OTHER_RECEIV_COLUMN: &str = "oth_receiv";
const PREPAYMENT_COLUMN: &str = "prepayment";
const DIV_RECEIV_COLUMN: &str = "div_receiv";
const INT_RECEIV_COLUMN: &str = "int_receiv";
const ACCT_PAYABLE_COLUMN: &str = "acct_payable";
const ADV_RECEIPTS_COLUMN: &str = "adv_receipts";
const CONTRACT_LIAB_COLUMN: &str = "contract_liab";
const INVENTORIES_COLUMN: &str = "inventories";
const FIX_ASSETS_COLUMN: &str = "fix_assets";
const CIP_COLUMN: &str = "cip";
const INTAN_ASSETS_COLUMN: &str = "intan_assets";
const R_AND_D_COLUMN: &str = "r_and_d";
const LT_AMOR_EXP_COLUMN: &str = "lt_amor_exp";
const EQUITY_COLUMN: &str = "total_hldr_eqy_exc_min_int";

pub struct StockDailyFinQualityStability;

#[derive(Default)]
struct FinQualityStabilityState {
    raw_cache: EventDrivenCrossSectionCache,
    snapshot_cache: InstrumentAlignedSnapshotCache<FinQualitySnapshot>,
}

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyFinQualityStability)
}

impl Factor for StockDailyFinQualityStability {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: FACTOR_ID.to_string(),
            aliases: vec![
                "Financial Quality Stability".to_string(),
                "Operating Profit Margin Stability".to_string(),
            ],
            name: FACTOR_ID.to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: tags(),
            description: "DBZQ financial quality stability factor. It builds PIT single-quarter operating-margin quality variables, runs 12-quarter SW level-1 industry panel OLS with iterative two-way fixed-effect demeaning, uses the stock-level residual volatility as raw value, then applies SW industry zscore and Barra SIZE + SW sector neutralization.".to_string(),
            dependencies: vec![
                DataRequest::financial_quarters(
                    DatasetId::StockIncome,
                    &[OPERATE_PROFIT_COLUMN, REVENUE_COLUMN, OPER_COST_COLUMN],
                    REQUIRED_QUARTERS,
                ),
                DataRequest::financial_quarters(
                    DatasetId::StockCashFlow,
                    &[CFO_COLUMN, FNC_CASHFLOW_COLUMN, INV_CASHFLOW_COLUMN],
                    REQUIRED_QUARTERS,
                ),
                DataRequest::financial_quarters(
                    DatasetId::StockBalanceSheet,
                    &[
                        ACCOUNTS_RECEIV_COLUMN,
                        OTHER_RECEIV_COLUMN,
                        PREPAYMENT_COLUMN,
                        DIV_RECEIV_COLUMN,
                        INT_RECEIV_COLUMN,
                        ACCT_PAYABLE_COLUMN,
                        ADV_RECEIPTS_COLUMN,
                        CONTRACT_LIAB_COLUMN,
                        INVENTORIES_COLUMN,
                        FIX_ASSETS_COLUMN,
                        CIP_COLUMN,
                        INTAN_ASSETS_COLUMN,
                        R_AND_D_COLUMN,
                        LT_AMOR_EXP_COLUMN,
                        EQUITY_COLUMN,
                    ],
                    REQUIRED_QUARTERS,
                ),
                DataRequest::new(DatasetId::StockSwClassification, &["l1_code"]),
                DataRequest::new(DatasetId::StockBarraDaily, &["SIZE"]),
            ],
            intraday_raw_dependencies: Vec::new(),
            lookback: Lookback { trading_days: 0 },
        }
    }

    fn update_policy(&self) -> FactorUpdatePolicy {
        FactorUpdatePolicy::FinancialEventSnapshot
    }

    fn initial_compute_state(&self, _requested_ids: &[String]) -> Box<dyn Any + Send> {
        Box::new(FinQualityStabilityState::default())
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
        if requested_ids.iter().all(|id| id != FACTOR_ID) {
            return Ok(Vec::new());
        }
        let state = state
            .downcast_mut::<FinQualityStabilityState>()
            .ok_or_else(|| err("fin_quality_stability received incompatible event cache state"))?;
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
        let schedule = FinancialEventSchedule::from_pit_readers(&[
            income.clone(),
            cashflow.clone(),
            balance.clone(),
        ]);
        let sector_map = ClassificationMap::from_table(
            data.daily(DatasetId::StockSwClassification)?,
            ClassificationLevel::Sector,
        )?;
        let raw_specs = [raw_spec()];
        let raw_cache = &mut state.raw_cache;
        let snapshot_cache = &mut state.snapshot_cache;
        let panel = data.stock_universe_panel()?;
        let raw_series = compute_financial_event_snapshot_streaming_on_panel(
            requested_ids,
            context,
            data,
            panel,
            raw_cache,
            &schedule,
            &raw_specs,
            |_, _, data| {
                self.compute_raw_with_prepared_inputs(
                    data,
                    &income,
                    &cashflow,
                    &balance,
                    &sector_map,
                    snapshot_cache,
                )
                .map(|series| vec![series])
            },
        )?;
        self.finalize_raw_series(data, raw_series)
            .map(|series| vec![series])
    }
}

impl StockDailyFinQualityStability {
    fn compute_with_snapshot_cache(
        &self,
        data: &DataPool,
        snapshot_cache: &mut InstrumentAlignedSnapshotCache<FinQualitySnapshot>,
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
            data.daily(DatasetId::StockSwClassification)?,
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
        snapshot_cache: &mut InstrumentAlignedSnapshotCache<FinQualitySnapshot>,
    ) -> Result<FactorSeries> {
        let panel = data.stock_universe_panel()?;
        let raw = fin_quality_raw_column(
            &panel,
            income,
            cashflow,
            balance,
            sector_map,
            snapshot_cache,
        )?;
        Ok(raw.to_factor_series(raw_spec()))
    }

    fn finalize_raw_series(
        &self,
        data: &DataPool,
        raw_series: Vec<FactorSeries>,
    ) -> Result<FactorSeries> {
        let panel = data.stock_universe_panel()?;
        let series = raw_series
            .into_iter()
            .find(|series| series.spec.id == RAW_ID)
            .ok_or_else(|| err("missing fin_quality_stability raw series"))?;
        let raw = factor_series_to_panel_column(&panel, &series)?;
        let standardized = industry_zscore(&raw, data)?;
        let neutralized = neutralize_size_sector(&standardized, &panel, data)?;
        Ok(neutralized.to_factor_series(self.spec()))
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct FinQualitySnapshot {
    rows: [Option<QuarterQualityRow>; PANEL_WINDOW],
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct QuarterQualityRow {
    y: f64,
    x: [f64; REGRESSOR_COUNT],
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct PanelObservation {
    offset: usize,
    stock_idx: usize,
    quarter_idx: usize,
    row: QuarterQualityRow,
}

fn tags() -> Vec<String> {
    [
        "DBZQ",
        "financial",
        "fundamental",
        "pit",
        "quality",
        "stability",
        "panel_regression",
        "fixed_effects",
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

fn raw_spec() -> FactorSpec {
    FactorSpec {
        id: RAW_ID.to_string(),
        aliases: Vec::new(),
        name: RAW_ID.to_string(),
        asset_class: AssetClass::Stock,
        frequency: Frequency::Daily,
        version: VERSION.to_string(),
        tags: vec!["internal".to_string(), "financial_raw".to_string()],
        description: "Internal fin_quality_stability residual-volatility raw series.".to_string(),
        dependencies: Vec::new(),
        intraday_raw_dependencies: Vec::new(),
        lookback: Lookback { trading_days: 0 },
    }
}

fn fin_quality_raw_column(
    panel: &DailyPanel,
    income: &FinancialPitReader<'_>,
    cashflow: &FinancialPitReader<'_>,
    balance: &FinancialPitReader<'_>,
    sector_map: &ClassificationMap,
    cache: &mut InstrumentAlignedSnapshotCache<FinQualitySnapshot>,
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
            |trade_date, ts_code, _| {
                fin_quality_marker(ts_code, trade_date, income, cashflow, balance)
            },
            |trade_date, ts_code, _| {
                fin_quality_snapshot_for_stock(ts_code, trade_date, income, cashflow, balance)
            },
        );
        let date_offset = date_idx * instrument_count;
        let mut observations_by_sector = BTreeMap::<String, Vec<PanelObservation>>::new();
        for (instrument_idx, ts_code) in panel.instruments().iter().enumerate() {
            let offset = date_offset + instrument_idx;
            if is_bj_stock(ts_code) || !panel.is_present_offset(offset) {
                continue;
            }
            let Some(sector) = sector_map.group_for(trade_date, ts_code) else {
                continue;
            };
            let Some(snapshot) = snapshots[instrument_idx] else {
                continue;
            };
            for (quarter_idx, row) in snapshot.rows.iter().copied().enumerate() {
                let Some(row) = row else {
                    continue;
                };
                observations_by_sector
                    .entry(sector.to_string())
                    .or_default()
                    .push(PanelObservation {
                        offset,
                        stock_idx: instrument_idx,
                        quarter_idx,
                        row,
                    });
            }
        }
        for observations in observations_by_sector.values() {
            for (offset, value) in panel_residual_std_by_stock(observations) {
                values[offset] = Some(value);
            }
        }
    }

    panel.column_from_values(values)
}

fn fin_quality_marker(
    ts_code: &str,
    trade_date: i32,
    income: &FinancialPitReader<'_>,
    cashflow: &FinancialPitReader<'_>,
    balance: &FinancialPitReader<'_>,
) -> Option<FinancialEventMarker> {
    let mut end_date = income.latest_quarter_end_date(ts_code, trade_date)?;
    let mut builder = FinancialEventMarkerBuilder::new();
    for _ in 0..REQUIRED_QUARTERS {
        builder.include_reader_record_for_end_date(
            FinancialStatementDataset::Income,
            income,
            ts_code,
            trade_date,
            end_date,
        );
        builder.include_reader_record_for_end_date(
            FinancialStatementDataset::CashFlow,
            cashflow,
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
        end_date = previous_quarter_end_date(end_date)?;
    }
    builder.build()
}

fn fin_quality_snapshot_for_stock(
    ts_code: &str,
    trade_date: i32,
    income: &FinancialPitReader<'_>,
    cashflow: &FinancialPitReader<'_>,
    balance: &FinancialPitReader<'_>,
) -> Option<FinQualitySnapshot> {
    let anchor = income.latest_quarter_end_date(ts_code, trade_date)?;
    let end_dates = quarter_chain(anchor, REQUIRED_QUARTERS)?;
    let income_records = end_dates
        .iter()
        .map(|end_date| income.record_for_end_date(ts_code, trade_date, *end_date))
        .collect::<Vec<_>>();
    let cashflow_records = end_dates
        .iter()
        .map(|end_date| cashflow.record_for_end_date(ts_code, trade_date, *end_date))
        .collect::<Vec<_>>();
    let balance_records = end_dates
        .iter()
        .map(|end_date| balance.record_for_end_date(ts_code, trade_date, *end_date))
        .collect::<Vec<_>>();

    let mut rows = [None; PANEL_WINDOW];
    for quarter_idx in 0..PANEL_WINDOW {
        rows[quarter_idx] = quality_row_for_quarter(
            income_records[quarter_idx],
            income_records[quarter_idx + 4],
            cashflow_records[quarter_idx],
            balance_records[quarter_idx],
            balance_records[quarter_idx + 1],
        );
    }
    rows.iter()
        .any(Option::is_some)
        .then_some(FinQualitySnapshot { rows })
}

fn quality_row_for_quarter(
    income_t: Option<PitFinancialRecordView<'_>>,
    income_t4: Option<PitFinancialRecordView<'_>>,
    cashflow_t: Option<PitFinancialRecordView<'_>>,
    balance_t: Option<PitFinancialRecordView<'_>>,
    balance_t1: Option<PitFinancialRecordView<'_>>,
) -> Option<QuarterQualityRow> {
    let income_t = income_t?;
    let y = op_margin(income_t)?;
    let x1 = op_margin(income_t4?)?;
    let x2 = accrual_indicator(income_t, balance_t?)?;
    let x3 = inventory_turnover(income_t, balance_t?, balance_t1?)?;
    let x4 = noncurrent_asset_indicator(income_t, balance_t?)?;
    let x5 = cashflow_indicator(cashflow_t?, balance_t?)?;
    let row = QuarterQualityRow {
        y,
        x: [x1, x2, x3, x4, x5],
    };
    row_is_finite(&row).then_some(row)
}

fn op_margin(income: PitFinancialRecordView<'_>) -> Option<f64> {
    safe_ratio(
        income.column(OPERATE_PROFIT_COLUMN),
        income.column(REVENUE_COLUMN),
    )
}

fn accrual_indicator(
    income: PitFinancialRecordView<'_>,
    balance: PitFinancialRecordView<'_>,
) -> Option<f64> {
    let accrual_assets = zero(balance.column(ACCOUNTS_RECEIV_COLUMN))
        + zero(balance.column(OTHER_RECEIV_COLUMN))
        + zero(balance.column(PREPAYMENT_COLUMN))
        - zero(balance.column(DIV_RECEIV_COLUMN))
        - zero(balance.column(INT_RECEIV_COLUMN));
    let accrual_liabilities = zero(balance.column(ACCT_PAYABLE_COLUMN))
        + zero(balance.column(ADV_RECEIPTS_COLUMN))
        + zero(balance.column(CONTRACT_LIAB_COLUMN));
    let gross_profit = zero(income.column(REVENUE_COLUMN)) - zero(income.column(OPER_COST_COLUMN));
    safe_ratio_value(accrual_assets - accrual_liabilities, gross_profit)
}

fn inventory_turnover(
    income: PitFinancialRecordView<'_>,
    balance_t: PitFinancialRecordView<'_>,
    balance_t1: PitFinancialRecordView<'_>,
) -> Option<f64> {
    let numerator = 2.0 * zero(income.column(OPER_COST_COLUMN));
    let denominator =
        zero(balance_t.column(INVENTORIES_COLUMN)) + zero(balance_t1.column(INVENTORIES_COLUMN));
    safe_ratio_value(numerator, denominator)
}

fn noncurrent_asset_indicator(
    income: PitFinancialRecordView<'_>,
    balance: PitFinancialRecordView<'_>,
) -> Option<f64> {
    let noncurrent_assets = zero(balance.column(FIX_ASSETS_COLUMN))
        + zero(balance.column(CIP_COLUMN))
        + zero(balance.column(INTAN_ASSETS_COLUMN))
        + zero(balance.column(R_AND_D_COLUMN))
        + zero(balance.column(LT_AMOR_EXP_COLUMN));
    safe_ratio(
        Some(noncurrent_assets),
        income.column(OPERATE_PROFIT_COLUMN),
    )
}

fn cashflow_indicator(
    cashflow: PitFinancialRecordView<'_>,
    balance: PitFinancialRecordView<'_>,
) -> Option<f64> {
    let cashflow_value = zero(cashflow.column(CFO_COLUMN))
        - zero(cashflow.column(FNC_CASHFLOW_COLUMN))
        - zero(cashflow.column(INV_CASHFLOW_COLUMN));
    safe_ratio_value(
        cashflow_value,
        positive_equity(balance.column(EQUITY_COLUMN))?,
    )
}

fn panel_residual_std_by_stock(observations: &[PanelObservation]) -> Vec<(usize, f64)> {
    if observations.len() < MIN_INDUSTRY_ROWS || observations.len() < REGRESSOR_COUNT + 1 {
        return Vec::new();
    }
    let Some(residuals) = panel_residuals(observations) else {
        return Vec::new();
    };
    residual_std_by_stock(observations, &residuals)
}

fn panel_residuals(observations: &[PanelObservation]) -> Option<Vec<f64>> {
    let stock_groups = compressed_stock_groups(observations);
    let quarter_groups = observations
        .iter()
        .map(|observation| observation.quarter_idx)
        .collect::<Vec<_>>();
    if stock_groups.iter().collect::<BTreeSet<_>>().len() < 2
        || quarter_groups.iter().collect::<BTreeSet<_>>().len() < 2
    {
        return None;
    }
    let y = observations
        .iter()
        .map(|observation| observation.row.y)
        .collect::<Vec<_>>();
    let y_demeaned = two_way_demean(&y, &stock_groups, &quarter_groups);
    let mut x_demeaned = Vec::with_capacity(REGRESSOR_COUNT);
    for regressor_idx in 0..REGRESSOR_COUNT {
        let x = observations
            .iter()
            .map(|observation| observation.row.x[regressor_idx])
            .collect::<Vec<_>>();
        x_demeaned.push(two_way_demean(&x, &stock_groups, &quarter_groups));
    }
    let beta = ols_beta_no_intercept(&y_demeaned, &x_demeaned)?;
    let residuals = (0..observations.len())
        .map(|row_idx| {
            let fitted = (0..REGRESSOR_COUNT)
                .map(|regressor_idx| beta[regressor_idx] * x_demeaned[regressor_idx][row_idx])
                .sum::<f64>();
            y_demeaned[row_idx] - fitted
        })
        .collect::<Vec<_>>();
    residuals
        .iter()
        .all(|value| value.is_finite())
        .then_some(residuals)
}

fn compressed_stock_groups(observations: &[PanelObservation]) -> Vec<usize> {
    let mut lookup = BTreeMap::new();
    let mut next = 0usize;
    observations
        .iter()
        .map(|observation| {
            *lookup.entry(observation.stock_idx).or_insert_with(|| {
                let value = next;
                next += 1;
                value
            })
        })
        .collect()
}

fn two_way_demean(values: &[f64], stock_groups: &[usize], quarter_groups: &[usize]) -> Vec<f64> {
    let mut output = values.to_vec();
    for _ in 0..DEMEAN_MAX_ITER {
        let stock_shift = subtract_group_means(&mut output, stock_groups);
        let quarter_shift = subtract_group_means(&mut output, quarter_groups);
        if stock_shift.max(quarter_shift) < DEMEAN_TOL {
            break;
        }
    }
    output
}

fn subtract_group_means(values: &mut [f64], groups: &[usize]) -> f64 {
    let group_count = groups.iter().copied().max().map(|idx| idx + 1).unwrap_or(0);
    if group_count == 0 {
        return 0.0;
    }
    let mut sums = vec![0.0; group_count];
    let mut counts = vec![0usize; group_count];
    for (value, group) in values.iter().zip(groups.iter().copied()) {
        sums[group] += *value;
        counts[group] += 1;
    }
    let mut max_abs_mean = 0.0f64;
    for (idx, value) in values.iter_mut().enumerate() {
        let group = groups[idx];
        if counts[group] == 0 {
            continue;
        }
        let mean = sums[group] / counts[group] as f64;
        max_abs_mean = max_abs_mean.max(mean.abs());
        *value -= mean;
    }
    max_abs_mean
}

fn ols_beta_no_intercept(y: &[f64], x_by_regressor: &[Vec<f64>]) -> Option<[f64; REGRESSOR_COUNT]> {
    if x_by_regressor.len() != REGRESSOR_COUNT || y.len() < REGRESSOR_COUNT + 1 {
        return None;
    }
    let mut xtx = [[0.0; REGRESSOR_COUNT]; REGRESSOR_COUNT];
    let mut xty = [0.0; REGRESSOR_COUNT];
    for row_idx in 0..y.len() {
        for i in 0..REGRESSOR_COUNT {
            let xi = x_by_regressor[i][row_idx];
            xty[i] += xi * y[row_idx];
            for j in 0..REGRESSOR_COUNT {
                xtx[i][j] += xi * x_by_regressor[j][row_idx];
            }
        }
    }
    solve_linear_system(xtx, xty)
}

fn solve_linear_system(
    mut a: [[f64; REGRESSOR_COUNT]; REGRESSOR_COUNT],
    mut b: [f64; REGRESSOR_COUNT],
) -> Option<[f64; REGRESSOR_COUNT]> {
    for pivot_idx in 0..REGRESSOR_COUNT {
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
        for col_idx in pivot_idx..REGRESSOR_COUNT {
            a[pivot_idx][col_idx] /= pivot;
        }
        b[pivot_idx] /= pivot;
        for row_idx in 0..REGRESSOR_COUNT {
            if row_idx == pivot_idx {
                continue;
            }
            let factor = a[row_idx][pivot_idx];
            if factor.abs() <= f64::EPSILON {
                continue;
            }
            for col_idx in pivot_idx..REGRESSOR_COUNT {
                a[row_idx][col_idx] -= factor * a[pivot_idx][col_idx];
            }
            b[row_idx] -= factor * b[pivot_idx];
        }
    }
    b.iter().all(|value| value.is_finite()).then_some(b)
}

fn residual_std_by_stock(
    observations: &[PanelObservation],
    residuals: &[f64],
) -> Vec<(usize, f64)> {
    let mut by_stock = BTreeMap::<usize, (usize, Vec<f64>)>::new();
    for (observation, residual) in observations.iter().zip(residuals.iter().copied()) {
        if !residual.is_finite() {
            continue;
        }
        by_stock
            .entry(observation.stock_idx)
            .or_insert_with(|| (observation.offset, Vec::new()))
            .1
            .push(residual);
    }
    by_stock
        .into_values()
        .filter_map(|(offset, values)| {
            sample_std_min_periods(&values, MIN_RESIDUALS).map(|std| (offset, std))
        })
        .collect()
}

fn sample_std_min_periods(values: &[f64], min_periods: usize) -> Option<f64> {
    if values.len() < min_periods || values.len() < 2 {
        return None;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / (values.len() - 1) as f64;
    let std = variance.max(0.0).sqrt();
    std.is_finite().then_some(std)
}

fn quarter_chain(anchor: i32, count: usize) -> Option<Vec<i32>> {
    let mut end_date = anchor;
    let mut output = Vec::with_capacity(count);
    for _ in 0..count {
        output.push(end_date);
        end_date = previous_quarter_end_date(end_date)?;
    }
    Some(output)
}

fn industry_zscore(raw: &PanelColumn, data: &DataPool) -> Result<PanelColumn> {
    let sector_map = ClassificationMap::from_table(
        data.daily(DatasetId::StockSwClassification)?,
        ClassificationLevel::Sector,
    )?;
    raw.cs_by_group(
        |trade_date, ts_codes| sector_map.groups_for(trade_date, ts_codes),
        cs_zscore_by_group,
    )
}

fn row_is_finite(row: &QuarterQualityRow) -> bool {
    row.y.is_finite() && row.x.iter().all(|value| value.is_finite())
}

fn safe_ratio(numerator: Option<f64>, denominator: Option<f64>) -> Option<f64> {
    safe_ratio_value(clean(numerator)?, clean(denominator)?)
}

fn safe_ratio_value(numerator: f64, denominator: f64) -> Option<f64> {
    if !numerator.is_finite() || !denominator.is_finite() || denominator.abs() <= EPS {
        return None;
    }
    Some(numerator / denominator)
}

fn positive_equity(value: Option<f64>) -> Option<f64> {
    clean(value).filter(|value| *value > EPS)
}

fn zero(value: Option<f64>) -> f64 {
    clean(value).unwrap_or(0.0)
}

fn clean(value: Option<f64>) -> Option<f64> {
    value.filter(|value| value.is_finite())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(left: f64, right: f64) {
        assert!(
            (left - right).abs() < 1e-9,
            "left={left}, right={right}, diff={}",
            (left - right).abs()
        );
    }

    #[test]
    fn two_way_demean_removes_stock_and_quarter_effects() {
        let stock_groups = vec![0, 0, 1, 1, 2, 2];
        let quarter_groups = vec![0, 1, 0, 1, 0, 1];
        let values = stock_groups
            .iter()
            .zip(quarter_groups.iter())
            .map(|(stock, quarter)| 10.0 * *stock as f64 + 2.0 * *quarter as f64 + 5.0)
            .collect::<Vec<_>>();

        let demeaned = two_way_demean(&values, &stock_groups, &quarter_groups);

        for value in demeaned {
            assert!(value.abs() < 1e-9);
        }
    }

    #[test]
    fn panel_residual_std_requires_min_periods() {
        let observations = (0..7)
            .map(|quarter_idx| PanelObservation {
                offset: 0,
                stock_idx: 0,
                quarter_idx,
                row: QuarterQualityRow {
                    y: quarter_idx as f64,
                    x: [quarter_idx as f64; REGRESSOR_COUNT],
                },
            })
            .collect::<Vec<_>>();
        let residuals = vec![1.0; observations.len()];

        assert!(residual_std_by_stock(&observations, &residuals).is_empty());
    }

    #[test]
    fn sample_std_uses_n_minus_one() {
        let std = sample_std_min_periods(&[1.0, 2.0, 3.0], 2).expect("std");
        assert_close(std, 1.0);
    }

    #[test]
    fn positive_equity_rejects_zero_and_negative_values() {
        assert_eq!(positive_equity(Some(10.0)), Some(10.0));
        assert_eq!(positive_equity(Some(0.0)), None);
        assert_eq!(positive_equity(Some(-1.0)), None);
    }

    #[test]
    fn panel_residuals_recover_residual_volatility_after_fixed_effects() {
        let mut observations = Vec::new();
        for stock_idx in 0..8 {
            for quarter_idx in 0..12 {
                let s = stock_idx as f64 + 1.0;
                let q = quarter_idx as f64 + 1.0;
                let x = [
                    s * q,
                    s * q.powi(2) * 0.01,
                    s.powi(2) * q * 0.02,
                    (s + q).sin(),
                    (s * 0.3 + q * 0.7).cos(),
                ];
                let stock_fe = 3.0 * stock_idx as f64;
                let quarter_fe = -0.2 * quarter_idx as f64;
                let residual = if stock_idx == 0 {
                    quarter_idx as f64 * 0.01
                } else {
                    0.0
                };
                let y = stock_fe + quarter_fe + 0.5 * x[0] - 0.2 * x[1] + 0.1 * x[2] + 0.05 * x[3]
                    - 0.03 * x[4]
                    + residual;
                observations.push(PanelObservation {
                    offset: stock_idx,
                    stock_idx,
                    quarter_idx,
                    row: QuarterQualityRow { y, x },
                });
            }
        }

        let output = panel_residual_std_by_stock(&observations);

        assert!(!output.is_empty());
        assert!(output.iter().all(|(_, value)| value.is_finite()));
    }

    #[test]
    fn metadata_has_expected_id_and_tags() {
        let spec = StockDailyFinQualityStability.spec();

        assert_eq!(spec.id, FACTOR_ID);
        assert_eq!(spec.output_column(), FACTOR_ID);
        assert!(spec.tags.contains(&"DBZQ".to_string()));
        assert!(spec.tags.contains(&"financial".to_string()));
        assert!(spec.tags.contains(&"panel_regression".to_string()));
    }
}

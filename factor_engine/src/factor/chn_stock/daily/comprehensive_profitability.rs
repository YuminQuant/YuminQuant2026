use std::any::Any;

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
    factor_series_to_panel_column, DailyPanel, EventDrivenCrossSectionCache, FinancialEventMarker,
    FinancialEventMarkerBuilder, FinancialEventSchedule, FinancialPitReader,
    FinancialStatementDataset, InstrumentAlignedSnapshotCache, PanelColumn, PitFinancialRecordView,
    ReportTypePreference,
};
use crate::factor::{Factor, FactorUpdatePolicy};
use crate::operators::{cs_regression_residual, cs_zscore};

pub const PROVIDER_KEY: &str = "stock|daily|comprehensive_profitability";
pub const COMPREHENSIVE_PROFITABILITY_ID: &str = "comprehensive_profitability";
pub const STABLE_ROE_ID: &str = "stable_roe";
pub const STABLE_ROIC_ID: &str = "stable_roic";
pub const STABLE_RONOA_ID: &str = "stable_ronoa";
pub const FCFFIC_ID: &str = "fcffic";

const VERSION: &str = "0.2.4";
const HISTORY_WINDOW: usize = 12;
const BALANCE_HISTORY_QUARTERS: usize = HISTORY_WINDOW + 1;
const STABILITY_MIN_PERIODS: usize = 1;
const EPS: f64 = 1e-12;
const ROE_REGRESSION_CLIP: f64 = 3.5;

const RAW_ROE_RESID_ID: &str = "__comprehensive_profitability_roe_resid";
const RAW_ROE_RESID_STABILITY_ID: &str = "__comprehensive_profitability_roe_resid_stability";
const RAW_ROIC_ID: &str = "__comprehensive_profitability_roic";
const RAW_ROIC_STABILITY_ID: &str = "__comprehensive_profitability_roic_stability";
const RAW_RONOA_ID: &str = "__comprehensive_profitability_ronoa";
const RAW_RONOA_STABILITY_ID: &str = "__comprehensive_profitability_ronoa_stability";
const RAW_FCFFIC_ID: &str = "__comprehensive_profitability_fcffic";

const NET_PROFIT_ATTR_P_COLUMN: &str = "n_income_attr_p";
const INCOME_TAX_COLUMN: &str = "income_tax";
const TOTAL_PROFIT_COLUMN: &str = "total_profit";
const INT_EXP_COLUMN: &str = "int_exp";
const OPERATE_PROFIT_COLUMN: &str = "operate_profit";
const FIN_EXP_COLUMN: &str = "fin_exp";
const INVEST_INCOME_COLUMN: &str = "invest_income";
const FV_VALUE_CHG_GAIN_COLUMN: &str = "fv_value_chg_gain";

const CFO_COLUMN: &str = "n_cashflow_act";
const CAPEX_COLUMN: &str = "c_pay_acq_const_fiolta";
const CASH_EQUIVALENTS_END_COLUMN: &str = "c_cash_equ_end_period";

const EQUITY_COLUMN: &str = "total_hldr_eqy_exc_min_int";
const TOTAL_ASSETS_COLUMN: &str = "total_assets";
const SHORT_BORROW_COLUMN: &str = "st_borr";
const NON_CURRENT_LIAB_DUE_1Y_COLUMN: &str = "non_cur_liab_due_1y";
const LONG_BORROW_COLUMN: &str = "lt_borr";
const BOND_PAYABLE_COLUMN: &str = "bond_payable";
const MONEY_CAP_COLUMN: &str = "money_cap";
const TIME_DEPOSITS_COLUMN: &str = "time_deposits";
const TRAD_ASSET_COLUMN: &str = "trad_asset";
const DIV_RECEIV_COLUMN: &str = "div_receiv";
const INT_RECEIV_COLUMN: &str = "int_receiv";
const FA_AVAIL_FOR_SALE_COLUMN: &str = "fa_avail_for_sale";
const HTM_INVEST_COLUMN: &str = "htm_invest";
const LT_EQT_INVEST_COLUMN: &str = "lt_eqt_invest";
const INVEST_REAL_ESTATE_COLUMN: &str = "invest_real_estate";
const DERIV_ASSETS_COLUMN: &str = "deriv_assets";
const INVEST_AS_RECEIV_COLUMN: &str = "invest_as_receiv";

const INCOME_COLUMNS: [&str; 8] = [
    NET_PROFIT_ATTR_P_COLUMN,
    INCOME_TAX_COLUMN,
    TOTAL_PROFIT_COLUMN,
    INT_EXP_COLUMN,
    OPERATE_PROFIT_COLUMN,
    FIN_EXP_COLUMN,
    INVEST_INCOME_COLUMN,
    FV_VALUE_CHG_GAIN_COLUMN,
];
const CASHFLOW_COLUMNS: [&str; 3] = [CFO_COLUMN, CAPEX_COLUMN, CASH_EQUIVALENTS_END_COLUMN];
const BALANCE_COLUMNS: [&str; 17] = [
    EQUITY_COLUMN,
    TOTAL_ASSETS_COLUMN,
    SHORT_BORROW_COLUMN,
    NON_CURRENT_LIAB_DUE_1Y_COLUMN,
    LONG_BORROW_COLUMN,
    BOND_PAYABLE_COLUMN,
    MONEY_CAP_COLUMN,
    TIME_DEPOSITS_COLUMN,
    TRAD_ASSET_COLUMN,
    DIV_RECEIV_COLUMN,
    INT_RECEIV_COLUMN,
    FA_AVAIL_FOR_SALE_COLUMN,
    HTM_INVEST_COLUMN,
    LT_EQT_INVEST_COLUMN,
    INVEST_REAL_ESTATE_COLUMN,
    DERIV_ASSETS_COLUMN,
    INVEST_AS_RECEIV_COLUMN,
];

pub struct StockDailyComprehensiveProfitability;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ComprehensiveProfitabilityOutput {
    ComprehensiveProfitability,
    StableRoe,
    StableRoic,
    StableRonoa,
    Fcffic,
}

#[derive(Default)]
pub struct ComprehensiveProfitabilityState {
    raw_cache: EventDrivenCrossSectionCache,
    snapshot_cache: InstrumentAlignedSnapshotCache<ProfitabilitySnapshot>,
}

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyComprehensiveProfitability)
}

impl Factor for StockDailyComprehensiveProfitability {
    fn spec(&self) -> FactorSpec {
        spec(ComprehensiveProfitabilityOutput::ComprehensiveProfitability)
    }

    fn compute_provider_key(&self) -> String {
        PROVIDER_KEY.to_string()
    }

    fn update_policy(&self) -> FactorUpdatePolicy {
        FactorUpdatePolicy::FinancialEventSnapshot
    }

    fn initial_compute_state(&self, _requested_ids: &[String]) -> Box<dyn Any + Send> {
        Box::new(ComprehensiveProfitabilityState::default())
    }

    fn compute(&self, context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let requested = [COMPREHENSIVE_PROFITABILITY_ID.to_string()];
        compute_requested(&requested, context, data)?
            .into_iter()
            .find(|series| series.spec.id == COMPREHENSIVE_PROFITABILITY_ID)
            .ok_or_else(|| err("comprehensive profitability provider did not return composite"))
    }

    fn compute_many(
        &self,
        requested_ids: &[String],
        context: &FactorContext,
        data: &DataPool,
    ) -> Result<Vec<FactorSeries>> {
        compute_requested(requested_ids, context, data)
    }

    fn compute_many_stateful(
        &self,
        requested_ids: &[String],
        context: &FactorContext,
        data: &DataPool,
        state: &mut (dyn Any + Send),
    ) -> Result<Vec<FactorSeries>> {
        let state = state
            .downcast_mut::<ComprehensiveProfitabilityState>()
            .ok_or_else(|| {
                err("comprehensive profitability provider received incompatible state")
            })?;
        compute_requested_stateful(requested_ids, context, data, state)
    }
}

pub fn spec(output: ComprehensiveProfitabilityOutput) -> FactorSpec {
    let (id, aliases, description) = match output {
        ComprehensiveProfitabilityOutput::ComprehensiveProfitability => (
            COMPREHENSIVE_PROFITABILITY_ID,
            vec![
                "Comprehensive Profitability".to_string(),
                "Deprecated Stable ROE ROIC RONOA FCFFIC Composite".to_string(),
            ],
            "Deprecated composite profitability factor retained for backward compatibility. Use stable_roe, stable_roic, stable_ronoa, and fcffic as separate factors.",
        ),
        ComprehensiveProfitabilityOutput::StableRoe => (
            STABLE_ROE_ID,
            vec!["Stable ROE".to_string(), "Leverage Residual Stable ROE".to_string()],
            "Stable ROE factor. It uses PIT single-quarter ROE, z-scores and clips both ROE and the equity multiplier to +/-3.5 before cross-sectional residualization, combines current ROE residual z-score with negative rolling residual volatility z-score over up to 12 quarters with min_periods=1, then neutralizes by Barra SIZE and SW sector and z-scores.",
        ),
        ComprehensiveProfitabilityOutput::StableRoic => (
            STABLE_ROIC_ID,
            vec!["Stable ROIC".to_string()],
            "Stable ROIC factor. It uses PIT single-quarter NOPLAT as total profit plus interest expense less income tax, computes non-annualized ROIC over invested capital, combines current ROIC z-score with negative rolling ROIC volatility z-score over up to 12 quarters with min_periods=1, then neutralizes by Barra SIZE and SW sector and z-scores.",
        ),
        ComprehensiveProfitabilityOutput::StableRonoa => (
            STABLE_RONOA_ID,
            vec!["Stable RONOA".to_string()],
            "Stable RONOA factor. It computes operating profit over positive net operating assets where NOA equals shareholder equity plus interest-bearing debt minus expanded financial assets, combines current RONOA z-score with negative rolling RONOA volatility z-score over up to 12 quarters with min_periods=1, then neutralizes by Barra SIZE and SW sector and z-scores.",
        ),
        ComprehensiveProfitabilityOutput::Fcffic => (
            FCFFIC_ID,
            vec!["FCFFIC".to_string(), "FCFF to Invested Capital".to_string()],
            "FCFFIC factor. It computes operating cash flow minus capex over invested capital, then neutralizes by Barra SIZE and SW sector and z-scores.",
        ),
    };

    FactorSpec {
        id: id.to_string(),
        aliases,
        name: id.to_string(),
        asset_class: AssetClass::Stock,
        frequency: Frequency::Daily,
        version: VERSION.to_string(),
        tags: tags_for_output(output),
        description: description.to_string(),
        dependencies: dependencies(),
        intraday_raw_dependencies: Vec::new(),
        lookback: Lookback { trading_days: 0 },
    }
}

pub fn compute_requested(
    requested_ids: &[String],
    context: &FactorContext,
    data: &DataPool,
) -> Result<Vec<FactorSeries>> {
    let mut state = ComprehensiveProfitabilityState::default();
    compute_requested_stateful(requested_ids, context, data, &mut state)
}

pub fn compute_requested_stateful(
    requested_ids: &[String],
    context: &FactorContext,
    data: &DataPool,
    state: &mut ComprehensiveProfitabilityState,
) -> Result<Vec<FactorSeries>> {
    let outputs = outputs_from_requested(requested_ids);
    if outputs.is_empty() {
        return Ok(Vec::new());
    }

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
    let raw_specs = raw_specs();
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
            compute_raw_series_with_prepared_inputs(
                data,
                &income,
                &cashflow,
                &balance,
                snapshot_cache,
            )
        },
    )?;
    finalize_raw_series(data, raw_series, &outputs)
}

fn compute_raw_series_with_prepared_inputs(
    data: &DataPool,
    income: &FinancialPitReader<'_>,
    cashflow: &FinancialPitReader<'_>,
    balance: &FinancialPitReader<'_>,
    snapshot_cache: &mut InstrumentAlignedSnapshotCache<ProfitabilitySnapshot>,
) -> Result<Vec<FactorSeries>> {
    let panel = data.stock_universe_panel()?;
    let columns = profitability_raw_columns(panel, income, cashflow, balance, snapshot_cache)?;
    Ok(columns.into_factor_series())
}

fn finalize_raw_series(
    data: &DataPool,
    raw_series: Vec<FactorSeries>,
    outputs: &[ComprehensiveProfitabilityOutput],
) -> Result<Vec<FactorSeries>> {
    let panel = data.stock_universe_panel()?;
    let raw = raw_columns_from_series(panel, raw_series)?;

    let stable_roe = sum_pair(
        &raw.roe_resid.cs(cs_zscore)?,
        &raw.roe_resid_stability.cs(cs_zscore)?,
    )?;
    let stable_roic = sum_pair(&raw.roic.cs(cs_zscore)?, &raw.roic_stability.cs(cs_zscore)?)?;
    let stable_ronoa = sum_pair(
        &raw.ronoa.cs(cs_zscore)?,
        &raw.ronoa_stability.cs(cs_zscore)?,
    )?;

    let processed_roe = postprocess_subfactor(&stable_roe, panel, data)?;
    let processed_roic = postprocess_subfactor(&stable_roic, panel, data)?;
    let processed_ronoa = postprocess_subfactor(&stable_ronoa, panel, data)?;
    let processed_fcffic = postprocess_subfactor(&raw.fcffic, panel, data)?;

    let mut series = Vec::new();
    for output in outputs {
        let column = match output {
            ComprehensiveProfitabilityOutput::ComprehensiveProfitability => {
                average_available_subfactors(
                    &processed_roe,
                    &processed_roic,
                    &processed_ronoa,
                    &processed_fcffic,
                )?
            }
            ComprehensiveProfitabilityOutput::StableRoe => processed_roe.clone(),
            ComprehensiveProfitabilityOutput::StableRoic => processed_roic.clone(),
            ComprehensiveProfitabilityOutput::StableRonoa => processed_ronoa.clone(),
            ComprehensiveProfitabilityOutput::Fcffic => processed_fcffic.clone(),
        };
        series.push(column.to_factor_series(spec(*output)));
    }
    Ok(series)
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct ProfitabilitySnapshot {
    quarters: [QuarterProfitability; HISTORY_WINDOW],
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct QuarterProfitability {
    roe: Option<f64>,
    equity_multiplier: Option<f64>,
    roic: Option<f64>,
    ronoa: Option<f64>,
    fcffic: Option<f64>,
}

struct RawColumns {
    roe_resid: PanelColumn,
    roe_resid_stability: PanelColumn,
    roic: PanelColumn,
    roic_stability: PanelColumn,
    ronoa: PanelColumn,
    ronoa_stability: PanelColumn,
    fcffic: PanelColumn,
}

impl RawColumns {
    fn into_factor_series(self) -> Vec<FactorSeries> {
        vec![
            self.roe_resid.to_factor_series(raw_spec(RAW_ROE_RESID_ID)),
            self.roe_resid_stability
                .to_factor_series(raw_spec(RAW_ROE_RESID_STABILITY_ID)),
            self.roic.to_factor_series(raw_spec(RAW_ROIC_ID)),
            self.roic_stability
                .to_factor_series(raw_spec(RAW_ROIC_STABILITY_ID)),
            self.ronoa.to_factor_series(raw_spec(RAW_RONOA_ID)),
            self.ronoa_stability
                .to_factor_series(raw_spec(RAW_RONOA_STABILITY_ID)),
            self.fcffic.to_factor_series(raw_spec(RAW_FCFFIC_ID)),
        ]
    }
}

fn profitability_raw_columns(
    panel: &DailyPanel,
    income: &FinancialPitReader<'_>,
    cashflow: &FinancialPitReader<'_>,
    balance: &FinancialPitReader<'_>,
    cache: &mut InstrumentAlignedSnapshotCache<ProfitabilitySnapshot>,
) -> Result<RawColumns> {
    let instrument_count = panel.instruments().len();
    let mut roe_resid = vec![None; panel.shape_len()];
    let mut roe_resid_stability = vec![None; panel.shape_len()];
    let mut roic = vec![None; panel.shape_len()];
    let mut roic_stability = vec![None; panel.shape_len()];
    let mut ronoa = vec![None; panel.shape_len()];
    let mut ronoa_stability = vec![None; panel.shape_len()];
    let mut fcffic = vec![None; panel.shape_len()];

    for (date_idx, trade_date) in panel.dates().iter().copied().enumerate() {
        if !panel.is_target_date(trade_date) {
            continue;
        }
        let snapshots = cached_financial_stock_snapshots_for_date(
            panel,
            trade_date,
            cache,
            |_, ts_code, offset| is_bj_stock(ts_code) || !panel.is_present_offset(offset),
            |trade_date, ts_code, _| {
                profitability_marker(ts_code, trade_date, income, cashflow, balance)
            },
            |trade_date, ts_code, _| {
                profitability_snapshot_for_stock(ts_code, trade_date, income, cashflow, balance)
            },
        );

        let mut roe_residuals_by_quarter = Vec::with_capacity(HISTORY_WINDOW);
        for quarter_idx in 0..HISTORY_WINDOW {
            let roe_values = snapshots
                .iter()
                .map(|snapshot| snapshot.and_then(|snapshot| snapshot.quarters[quarter_idx].roe))
                .collect::<Vec<_>>();
            let equity_multipliers = snapshots
                .iter()
                .map(|snapshot| {
                    snapshot.and_then(|snapshot| snapshot.quarters[quarter_idx].equity_multiplier)
                })
                .collect::<Vec<_>>();
            let roe_regression_values = zscore_clip_for_regression(&roe_values);
            let equity_multiplier_regression_values =
                zscore_clip_for_regression(&equity_multipliers);
            roe_residuals_by_quarter.push(cs_regression_residual(
                &roe_regression_values,
                &equity_multiplier_regression_values,
            ));
        }

        let date_offset = date_idx * instrument_count;
        for instrument_idx in 0..instrument_count {
            let Some(snapshot) = snapshots[instrument_idx] else {
                continue;
            };
            let offset = date_offset + instrument_idx;
            let current = snapshot.quarters[0];
            roe_resid[offset] = roe_residuals_by_quarter[0][instrument_idx];
            let resid_history = roe_residuals_by_quarter
                .iter()
                .map(|values| values[instrument_idx])
                .collect::<Vec<_>>();
            roe_resid_stability[offset] = stability_from_options(&resid_history);
            roic[offset] = current.roic;
            roic_stability[offset] =
                stability_from_quarters(&snapshot.quarters, |quarter| quarter.roic);
            ronoa[offset] = current.ronoa;
            ronoa_stability[offset] =
                stability_from_quarters(&snapshot.quarters, |quarter| quarter.ronoa);
            fcffic[offset] = current.fcffic;
        }
    }

    Ok(RawColumns {
        roe_resid: panel.column_from_values(roe_resid)?,
        roe_resid_stability: panel.column_from_values(roe_resid_stability)?,
        roic: panel.column_from_values(roic)?,
        roic_stability: panel.column_from_values(roic_stability)?,
        ronoa: panel.column_from_values(ronoa)?,
        ronoa_stability: panel.column_from_values(ronoa_stability)?,
        fcffic: panel.column_from_values(fcffic)?,
    })
}

fn profitability_marker(
    ts_code: &str,
    trade_date: i32,
    income: &FinancialPitReader<'_>,
    cashflow: &FinancialPitReader<'_>,
    balance: &FinancialPitReader<'_>,
) -> Option<FinancialEventMarker> {
    let mut current = income.latest_quarter_end_date(ts_code, trade_date)?;
    let mut builder = FinancialEventMarkerBuilder::new();
    for _ in 0..HISTORY_WINDOW {
        let prev = previous_quarter_end_date(current)?;
        builder.include_reader_record_for_end_date(
            FinancialStatementDataset::Income,
            income,
            ts_code,
            trade_date,
            current,
        );
        builder.include_reader_record_for_end_date(
            FinancialStatementDataset::CashFlow,
            cashflow,
            ts_code,
            trade_date,
            current,
        );
        builder.include_reader_record_for_end_date(
            FinancialStatementDataset::BalanceSheet,
            balance,
            ts_code,
            trade_date,
            current,
        );
        builder.include_reader_record_for_end_date(
            FinancialStatementDataset::BalanceSheet,
            balance,
            ts_code,
            trade_date,
            prev,
        );
        current = prev;
    }
    builder.build()
}

fn profitability_snapshot_for_stock(
    ts_code: &str,
    trade_date: i32,
    income: &FinancialPitReader<'_>,
    cashflow: &FinancialPitReader<'_>,
    balance: &FinancialPitReader<'_>,
) -> Option<ProfitabilitySnapshot> {
    let anchor = income.latest_quarter_end_date(ts_code, trade_date)?;
    let end_dates = quarter_chain(anchor, BALANCE_HISTORY_QUARTERS)?;
    let mut quarters = [empty_quarter_profitability(); HISTORY_WINDOW];
    for idx in 0..HISTORY_WINDOW {
        let income_t = income.record_for_end_date(ts_code, trade_date, end_dates[idx]);
        let cashflow_t = cashflow.record_for_end_date(ts_code, trade_date, end_dates[idx]);
        let balance_t = balance.record_for_end_date(ts_code, trade_date, end_dates[idx]);
        let balance_prev = balance.record_for_end_date(ts_code, trade_date, end_dates[idx + 1]);
        quarters[idx] = quarter_profitability(income_t, cashflow_t, balance_t, balance_prev);
    }
    Some(ProfitabilitySnapshot { quarters })
}

fn quarter_profitability(
    income: Option<PitFinancialRecordView<'_>>,
    cashflow: Option<PitFinancialRecordView<'_>>,
    balance: Option<PitFinancialRecordView<'_>>,
    balance_prev: Option<PitFinancialRecordView<'_>>,
) -> QuarterProfitability {
    if balance
        .and_then(|balance| positive_equity(balance.column(EQUITY_COLUMN)))
        .is_none()
    {
        return empty_quarter_profitability();
    }
    let roe =
        income
            .zip(balance)
            .zip(balance_prev)
            .and_then(|((income, balance), balance_prev)| {
                roe_for_records(income, balance, balance_prev)
            });
    let equity_multiplier = balance.and_then(equity_multiplier_for_record);
    let roic = income
        .zip(balance)
        .zip(cashflow)
        .and_then(|((income, balance), cashflow)| roic_for_records(income, balance, cashflow));
    let ronoa = income
        .zip(balance)
        .and_then(|(income, balance)| ronoa_for_records(income, balance));
    let fcffic = cashflow
        .zip(balance)
        .and_then(|(cashflow, balance)| fcffic_for_records(cashflow, balance));
    QuarterProfitability {
        roe,
        equity_multiplier,
        roic,
        ronoa,
        fcffic,
    }
}

fn empty_quarter_profitability() -> QuarterProfitability {
    QuarterProfitability {
        roe: None,
        equity_multiplier: None,
        roic: None,
        ronoa: None,
        fcffic: None,
    }
}

fn roe_for_records(
    income: PitFinancialRecordView<'_>,
    balance: PitFinancialRecordView<'_>,
    balance_prev: PitFinancialRecordView<'_>,
) -> Option<f64> {
    let net_profit = clean_or_zero(income.column(NET_PROFIT_ATTR_P_COLUMN));
    let equity = positive_equity(balance.column(EQUITY_COLUMN))?;
    let equity_prev = positive_equity(balance_prev.column(EQUITY_COLUMN))?;
    safe_div(net_profit, (equity + equity_prev) * 0.5)
}

fn equity_multiplier_for_record(balance: PitFinancialRecordView<'_>) -> Option<f64> {
    safe_div(
        clean(balance.column(TOTAL_ASSETS_COLUMN))?,
        positive_equity(balance.column(EQUITY_COLUMN))?,
    )
}

fn roic_for_records(
    income: PitFinancialRecordView<'_>,
    balance: PitFinancialRecordView<'_>,
    cashflow: PitFinancialRecordView<'_>,
) -> Option<f64> {
    let noplat = roic_noplat(income)?;
    let ic = invested_capital(balance, cashflow)?;
    safe_div(noplat, ic)
}

fn ronoa_for_records(
    income: PitFinancialRecordView<'_>,
    balance: PitFinancialRecordView<'_>,
) -> Option<f64> {
    let operating_profit = operating_profit(income)?;
    let noa = net_operating_assets(balance)?;
    safe_div(operating_profit, noa)
}

fn fcffic_for_records(
    cashflow: PitFinancialRecordView<'_>,
    balance: PitFinancialRecordView<'_>,
) -> Option<f64> {
    let cfo = clean_or_zero(cashflow.column(CFO_COLUMN));
    let capex = clean_or_zero(cashflow.column(CAPEX_COLUMN));
    let ic = invested_capital(balance, cashflow)?;
    safe_div(cfo - capex, ic)
}

fn operating_profit(income: PitFinancialRecordView<'_>) -> Option<f64> {
    let operate_profit = clean_or_zero(income.column(OPERATE_PROFIT_COLUMN));
    let fin_exp = clean_or_zero(income.column(FIN_EXP_COLUMN));
    let invest_income = clean_or_zero(income.column(INVEST_INCOME_COLUMN));
    let fv_value_chg_gain = clean_or_zero(income.column(FV_VALUE_CHG_GAIN_COLUMN));
    let value = operate_profit + fin_exp - invest_income - fv_value_chg_gain;
    value.is_finite().then_some(value)
}

fn roic_noplat(income: PitFinancialRecordView<'_>) -> Option<f64> {
    roic_noplat_from_values(
        income.column(TOTAL_PROFIT_COLUMN),
        income.column(INT_EXP_COLUMN),
        income.column(INCOME_TAX_COLUMN),
    )
}

fn roic_noplat_from_values(
    total_profit: Option<f64>,
    interest_expense: Option<f64>,
    income_tax: Option<f64>,
) -> Option<f64> {
    let value =
        clean_or_zero(total_profit) + clean_or_zero(interest_expense) - clean_or_zero(income_tax);
    value.is_finite().then_some(value)
}

fn invested_capital(
    balance: PitFinancialRecordView<'_>,
    cashflow: PitFinancialRecordView<'_>,
) -> Option<f64> {
    invested_capital_from_values(
        positive_equity(balance.column(EQUITY_COLUMN))?,
        interest_bearing_debt(balance),
        clean_or_zero(cashflow.column(CASH_EQUIVALENTS_END_COLUMN)),
    )
}

fn invested_capital_from_values(equity: f64, debt: f64, cash: f64) -> Option<f64> {
    let value = equity + debt - cash;
    value.is_finite().then_some(value)
}

fn net_operating_assets(balance: PitFinancialRecordView<'_>) -> Option<f64> {
    net_operating_assets_from_values(
        positive_equity(balance.column(EQUITY_COLUMN))?,
        interest_bearing_debt(balance),
        financial_assets(balance),
    )
}

fn net_operating_assets_from_values(
    equity: f64,
    financial_liabilities: f64,
    financial_assets: f64,
) -> Option<f64> {
    let value = equity + financial_liabilities - financial_assets;
    (value > EPS && value.is_finite()).then_some(value)
}

fn interest_bearing_debt(balance: PitFinancialRecordView<'_>) -> f64 {
    [
        SHORT_BORROW_COLUMN,
        NON_CURRENT_LIAB_DUE_1Y_COLUMN,
        LONG_BORROW_COLUMN,
        BOND_PAYABLE_COLUMN,
    ]
    .iter()
    .map(|column| clean_or_zero(balance.column(*column)))
    .sum()
}

fn financial_assets(balance: PitFinancialRecordView<'_>) -> f64 {
    [
        MONEY_CAP_COLUMN,
        TIME_DEPOSITS_COLUMN,
        TRAD_ASSET_COLUMN,
        DIV_RECEIV_COLUMN,
        INT_RECEIV_COLUMN,
        FA_AVAIL_FOR_SALE_COLUMN,
        HTM_INVEST_COLUMN,
        LT_EQT_INVEST_COLUMN,
        INVEST_REAL_ESTATE_COLUMN,
        DERIV_ASSETS_COLUMN,
        INVEST_AS_RECEIV_COLUMN,
    ]
    .iter()
    .map(|column| clean_or_zero(balance.column(*column)))
    .sum()
}

fn stability_from_quarters<F>(quarters: &[QuarterProfitability], mut f: F) -> Option<f64>
where
    F: FnMut(&QuarterProfitability) -> Option<f64>,
{
    let values = quarters
        .iter()
        .filter_map(|quarter| f(quarter))
        .collect::<Vec<_>>();
    negative_sample_std_min_periods(&values, STABILITY_MIN_PERIODS)
}

fn stability_from_options(values: &[Option<f64>]) -> Option<f64> {
    let values = values
        .iter()
        .filter_map(|value| clean(*value))
        .collect::<Vec<_>>();
    negative_sample_std_min_periods(&values, STABILITY_MIN_PERIODS)
}

fn negative_sample_std_min_periods(values: &[f64], min_periods: usize) -> Option<f64> {
    if values.len() < min_periods || values.iter().any(|value| !value.is_finite()) {
        return None;
    }
    sample_std(values).map(|std| -std)
}

fn sample_std(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    if values.len() == 1 {
        return Some(0.0);
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|value| {
            let diff = value - mean;
            diff * diff
        })
        .sum::<f64>()
        / (values.len() - 1) as f64;
    let std = variance.max(0.0).sqrt();
    std.is_finite().then_some(std)
}

fn postprocess_subfactor(
    values: &PanelColumn,
    panel: &DailyPanel,
    data: &DataPool,
) -> Result<PanelColumn> {
    neutralize_size_sector(values, panel, data)?.cs(cs_zscore)
}

fn zscore_clip_for_regression(values: &[Option<f64>]) -> Vec<Option<f64>> {
    cs_zscore(values)
        .into_iter()
        .map(|value| {
            clean(value).map(|value| value.clamp(-ROE_REGRESSION_CLIP, ROE_REGRESSION_CLIP))
        })
        .collect()
}

fn sum_pair(left: &PanelColumn, right: &PanelColumn) -> Result<PanelColumn> {
    left.zip_binary(right, |left, right| match (clean(left), clean(right)) {
        (Some(left), Some(right)) => Some(left + right),
        _ => None,
    })
}

fn average_available_subfactors(
    first: &PanelColumn,
    second: &PanelColumn,
    third: &PanelColumn,
    fourth: &PanelColumn,
) -> Result<PanelColumn> {
    first.zip_quaternary(second, third, fourth, |first, second, third, fourth| {
        average_clean_values(&[first, second, third, fourth])
    })
}

fn average_clean_values(values: &[Option<f64>]) -> Option<f64> {
    let mut sum = 0.0;
    let mut count = 0usize;
    for value in values.iter().filter_map(|value| clean(*value)) {
        sum += value;
        count += 1;
    }
    (count > 0).then_some(sum / count as f64)
}

fn raw_columns_from_series(
    panel: &DailyPanel,
    raw_series: Vec<FactorSeries>,
) -> Result<RawColumns> {
    Ok(RawColumns {
        roe_resid: raw_column(panel, &raw_series, RAW_ROE_RESID_ID)?,
        roe_resid_stability: raw_column(panel, &raw_series, RAW_ROE_RESID_STABILITY_ID)?,
        roic: raw_column(panel, &raw_series, RAW_ROIC_ID)?,
        roic_stability: raw_column(panel, &raw_series, RAW_ROIC_STABILITY_ID)?,
        ronoa: raw_column(panel, &raw_series, RAW_RONOA_ID)?,
        ronoa_stability: raw_column(panel, &raw_series, RAW_RONOA_STABILITY_ID)?,
        fcffic: raw_column(panel, &raw_series, RAW_FCFFIC_ID)?,
    })
}

fn raw_column(panel: &DailyPanel, raw_series: &[FactorSeries], id: &str) -> Result<PanelColumn> {
    let series = raw_series
        .iter()
        .find(|series| series.spec.id == id)
        .ok_or_else(|| {
            err(format!(
                "missing comprehensive profitability raw series {id}"
            ))
        })?;
    factor_series_to_panel_column(panel, series)
}

fn quarter_chain(anchor: i32, count: usize) -> Option<Vec<i32>> {
    let mut dates = Vec::with_capacity(count);
    let mut current = anchor;
    for _ in 0..count {
        dates.push(current);
        current = previous_quarter_end_date(current)?;
    }
    Some(dates)
}

fn clean(value: Option<f64>) -> Option<f64> {
    value.filter(|value| value.is_finite())
}

fn positive_equity(value: Option<f64>) -> Option<f64> {
    clean(value).filter(|value| *value > EPS)
}

fn clean_or_zero(value: Option<f64>) -> f64 {
    clean(value).unwrap_or(0.0)
}

fn safe_div(numerator: f64, denominator: f64) -> Option<f64> {
    if !numerator.is_finite() || !denominator.is_finite() || denominator.abs() <= EPS {
        return None;
    }
    let value = numerator / denominator;
    value.is_finite().then_some(value)
}

fn outputs_from_requested(requested_ids: &[String]) -> Vec<ComprehensiveProfitabilityOutput> {
    let mut outputs = Vec::new();
    for id in requested_ids {
        let Some(output) = ComprehensiveProfitabilityOutput::from_id(id) else {
            continue;
        };
        if !outputs.contains(&output) {
            outputs.push(output);
        }
    }
    outputs
}

impl ComprehensiveProfitabilityOutput {
    fn from_id(id: &str) -> Option<Self> {
        match id {
            COMPREHENSIVE_PROFITABILITY_ID => Some(Self::ComprehensiveProfitability),
            STABLE_ROE_ID => Some(Self::StableRoe),
            STABLE_ROIC_ID => Some(Self::StableRoic),
            STABLE_RONOA_ID => Some(Self::StableRonoa),
            FCFFIC_ID => Some(Self::Fcffic),
            _ => None,
        }
    }
}

fn raw_specs() -> Vec<FactorSpec> {
    [
        RAW_ROE_RESID_ID,
        RAW_ROE_RESID_STABILITY_ID,
        RAW_ROIC_ID,
        RAW_ROIC_STABILITY_ID,
        RAW_RONOA_ID,
        RAW_RONOA_STABILITY_ID,
        RAW_FCFFIC_ID,
    ]
    .iter()
    .map(|id| raw_spec(id))
    .collect()
}

fn raw_spec(id: &str) -> FactorSpec {
    FactorSpec {
        id: id.to_string(),
        aliases: Vec::new(),
        name: id.to_string(),
        asset_class: AssetClass::Stock,
        frequency: Frequency::Daily,
        version: VERSION.to_string(),
        tags: vec!["internal".to_string(), "financial_raw".to_string()],
        description: "Internal comprehensive profitability raw series.".to_string(),
        dependencies: Vec::new(),
        intraday_raw_dependencies: Vec::new(),
        lookback: Lookback { trading_days: 0 },
    }
}

fn dependencies() -> Vec<DataRequest> {
    vec![
        DataRequest::financial_quarters(DatasetId::StockIncome, &INCOME_COLUMNS, HISTORY_WINDOW),
        DataRequest::financial_quarters(
            DatasetId::StockCashFlow,
            &CASHFLOW_COLUMNS,
            HISTORY_WINDOW,
        ),
        DataRequest::financial_quarters(
            DatasetId::StockBalanceSheet,
            &BALANCE_COLUMNS,
            BALANCE_HISTORY_QUARTERS,
        ),
        DataRequest::new(DatasetId::StockBarraDaily, &["SIZE"]),
        DataRequest::new(DatasetId::StockSwClassification, &["l1_code"]),
    ]
}

fn tags_for_output(output: ComprehensiveProfitabilityOutput) -> Vec<String> {
    let mut tags = [
        "ZSZQ",
        "financial",
        "fundamental",
        "profitability",
        "quality",
        "stability",
        "pit",
        "neutralize",
        "barra",
        "size",
        "sector",
        "daily",
    ]
    .iter()
    .map(|value| value.to_string())
    .collect::<Vec<_>>();
    match output {
        ComprehensiveProfitabilityOutput::ComprehensiveProfitability => {
            tags.extend(
                ["composite", "deprecated"]
                    .iter()
                    .map(|value| value.to_string()),
            );
        }
        ComprehensiveProfitabilityOutput::StableRoe => {
            tags.push("roe".to_string());
            tags.push("deprecated".to_string());
        }
        ComprehensiveProfitabilityOutput::StableRoic => {
            tags.push("roic".to_string());
            tags.push("deprecated".to_string());
        }
        ComprehensiveProfitabilityOutput::StableRonoa => {
            tags.push("ronoa".to_string());
            tags.push("deprecated".to_string());
        }
        ComprehensiveProfitabilityOutput::Fcffic => {
            tags.push("fcffic".to_string());
            tags.push("deprecated".to_string());
        }
    }
    tags
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
    fn roic_noplat_uses_total_profit_interest_and_income_tax() {
        assert_close(
            roic_noplat_from_values(Some(100.0), Some(5.0), Some(20.0)),
            Some(85.0),
        );
        assert_close(
            roic_noplat_from_values(Some(100.0), Some(5.0), None),
            Some(105.0),
        );
    }

    #[test]
    fn invested_capital_uses_equity_debt_and_cash() {
        assert_close(
            invested_capital_from_values(1_000.0, 300.0, 120.0),
            Some(1_180.0),
        );
    }

    #[test]
    fn positive_equity_rejects_zero_and_negative_values() {
        assert_close(positive_equity(Some(10.0)), Some(10.0));
        assert_eq!(positive_equity(Some(0.0)), None);
        assert_eq!(positive_equity(Some(-1.0)), None);
    }

    #[test]
    fn net_operating_assets_uses_equity_debt_less_financial_assets() {
        assert_close(
            net_operating_assets_from_values(1_000.0, 250.0, 150.0),
            Some(1_100.0),
        );
        assert_eq!(net_operating_assets_from_values(100.0, 50.0, 150.0), None);
        assert_eq!(net_operating_assets_from_values(100.0, 50.0, 200.0), None);
    }

    #[test]
    fn stability_uses_min_periods_one_and_negative_sample_std() {
        assert_eq!(
            negative_sample_std_min_periods(&[], STABILITY_MIN_PERIODS),
            None
        );
        assert_close(
            negative_sample_std_min_periods(&[5.0], STABILITY_MIN_PERIODS),
            Some(0.0),
        );
        let values = [1.0, 2.0, 3.0];
        let mean = 2.0;
        let variance = values
            .iter()
            .map(|value| {
                let diff = value - mean;
                diff * diff
            })
            .sum::<f64>()
            / 2.0;
        assert_close(
            negative_sample_std_min_periods(&values, STABILITY_MIN_PERIODS),
            Some(-variance.sqrt()),
        );
    }

    #[test]
    fn residual_stability_uses_available_residuals() {
        let values = [
            Some(1.0),
            Some(2.0),
            Some(3.0),
            Some(4.0),
            Some(5.0),
            Some(6.0),
            Some(7.0),
            Some(8.0),
            Some(9.0),
            Some(10.0),
            Some(11.0),
            Some(12.0),
        ];
        assert!(stability_from_options(&values).is_some());
        let mut missing = values;
        missing[3] = None;
        assert!(stability_from_options(&missing).is_some());
        assert_eq!(stability_from_options(&[None, None]), None);
    }

    #[test]
    fn ronoa_stability_keeps_negative_values_when_denominator_was_valid() {
        let mut quarters = [empty_quarter_profitability(); HISTORY_WINDOW];
        quarters[0].ronoa = Some(-1.0);
        quarters[1].ronoa = Some(1.0);

        assert_close(
            stability_from_quarters(&quarters, |quarter| quarter.ronoa),
            Some(-(2.0_f64).sqrt()),
        );
    }

    #[test]
    fn roe_regression_inputs_are_zscored_and_clipped() {
        let mut values = vec![Some(0.0); 20];
        values[19] = Some(1000.0);

        let transformed = zscore_clip_for_regression(&values);

        assert_close(transformed[19], Some(ROE_REGRESSION_CLIP));
        assert!(transformed[..19]
            .iter()
            .all(|value| value.is_some_and(|value| value > -ROE_REGRESSION_CLIP && value < 0.0)));
    }

    #[test]
    fn average_available_subfactors_uses_non_null_values() {
        assert_close(
            average_clean_values(&[Some(1.0), Some(2.0), Some(3.0), Some(4.0)]),
            Some(2.5),
        );
        assert_close(
            average_clean_values(&[Some(1.0), None, Some(3.0), None]),
            Some(2.0),
        );
        assert_eq!(average_clean_values(&[None, None, None, None]), None);
    }

    #[test]
    fn outputs_are_parsed_from_requested_ids_in_order() {
        let requested = vec![
            STABLE_ROIC_ID.to_string(),
            "unknown".to_string(),
            STABLE_ROE_ID.to_string(),
            STABLE_ROIC_ID.to_string(),
        ];
        assert_eq!(
            outputs_from_requested(&requested),
            vec![
                ComprehensiveProfitabilityOutput::StableRoic,
                ComprehensiveProfitabilityOutput::StableRoe
            ]
        );
    }

    #[test]
    fn metadata_marks_composite_and_split_outputs_deprecated() {
        let old = spec(ComprehensiveProfitabilityOutput::ComprehensiveProfitability);
        assert_eq!(old.id, COMPREHENSIVE_PROFITABILITY_ID);
        assert!(old.tags.iter().any(|tag| tag == "deprecated"));

        for output in [
            ComprehensiveProfitabilityOutput::StableRoe,
            ComprehensiveProfitabilityOutput::StableRoic,
            ComprehensiveProfitabilityOutput::StableRonoa,
            ComprehensiveProfitabilityOutput::Fcffic,
        ] {
            let spec = spec(output);
            assert!(spec.tags.iter().any(|tag| tag == "ZSZQ"));
            assert!(spec.tags.iter().any(|tag| tag == "deprecated"));
        }
    }

    #[test]
    fn dependencies_include_expanded_ronoa_financial_assets() {
        let spec = spec(ComprehensiveProfitabilityOutput::StableRonoa);
        let balance = spec
            .dependencies
            .iter()
            .find(|request| request.dataset == DatasetId::StockBalanceSheet)
            .expect("balance request");
        for column in [
            MONEY_CAP_COLUMN,
            TIME_DEPOSITS_COLUMN,
            TRAD_ASSET_COLUMN,
            DIV_RECEIV_COLUMN,
            INT_RECEIV_COLUMN,
            FA_AVAIL_FOR_SALE_COLUMN,
            HTM_INVEST_COLUMN,
            LT_EQT_INVEST_COLUMN,
            INVEST_REAL_ESTATE_COLUMN,
            DERIV_ASSETS_COLUMN,
            INVEST_AS_RECEIV_COLUMN,
        ] {
            assert!(
                balance.columns.iter().any(|value| value == column),
                "missing balance column {column}"
            );
        }
    }
}

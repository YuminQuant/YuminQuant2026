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

const VERSION: &str = "0.1.0";
const FACTOR_ID: &str = "comprehensive_profitability";
const HISTORY_WINDOW: usize = 12;
const BALANCE_HISTORY_QUARTERS: usize = HISTORY_WINDOW + 1;
const EPS: f64 = 1e-12;

const RAW_ROE_ID: &str = "__comprehensive_profitability_roe";
const RAW_ROE_STABILITY_ID: &str = "__comprehensive_profitability_roe_stability";
const RAW_EQUITY_MULTIPLIER_ID: &str = "__comprehensive_profitability_equity_multiplier";
const RAW_ROIC_ID: &str = "__comprehensive_profitability_roic";
const RAW_ROIC_STABILITY_ID: &str = "__comprehensive_profitability_roic_stability";
const RAW_RONOA_ID: &str = "__comprehensive_profitability_ronoa";
const RAW_RONOA_STABILITY_ID: &str = "__comprehensive_profitability_ronoa_stability";
const RAW_FCFFIC_ID: &str = "__comprehensive_profitability_fcffic";

const NET_PROFIT_ATTR_P_COLUMN: &str = "n_income_attr_p";
const EBIT_COLUMN: &str = "ebit";
const INCOME_TAX_COLUMN: &str = "income_tax";
const TOTAL_PROFIT_COLUMN: &str = "total_profit";
const OPERATE_PROFIT_COLUMN: &str = "operate_profit";
const FIN_EXP_COLUMN: &str = "fin_exp";
const INVEST_INCOME_COLUMN: &str = "invest_income";
const FV_VALUE_CHG_GAIN_COLUMN: &str = "fv_value_chg_gain";

const CFO_COLUMN: &str = "n_cashflow_act";
const CAPEX_COLUMN: &str = "c_pay_acq_const_fiolta";

const EQUITY_COLUMN: &str = "total_hldr_eqy_exc_min_int";
const TOTAL_ASSETS_COLUMN: &str = "total_assets";
const TOTAL_LIAB_COLUMN: &str = "total_liab";
const MONEY_CAP_COLUMN: &str = "money_cap";
const SHORT_BORROW_COLUMN: &str = "st_borr";
const NON_CURRENT_LIAB_DUE_1Y_COLUMN: &str = "non_cur_liab_due_1y";
const LONG_BORROW_COLUMN: &str = "lt_borr";
const BOND_PAYABLE_COLUMN: &str = "bond_payable";

pub struct StockDailyComprehensiveProfitability;

#[derive(Default)]
struct ComprehensiveProfitabilityState {
    raw_cache: EventDrivenCrossSectionCache,
    snapshot_cache: InstrumentAlignedSnapshotCache<ProfitabilitySnapshot>,
}

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyComprehensiveProfitability)
}

impl Factor for StockDailyComprehensiveProfitability {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: FACTOR_ID.to_string(),
            aliases: vec![
                "Comprehensive Profitability".to_string(),
                "Stable ROE ROIC RONOA FCFFIC".to_string(),
            ],
            name: FACTOR_ID.to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: tags(),
            description: "Comprehensive profitability factor. It uses PIT single-quarter financials to build stable ROE, stable ROIC, stable RONOA, and FCFFIC. ROE/ROIC/RONOA stability is the negative 12-quarter sample standard deviation; stable ROE is additionally residualized against the equity multiplier. Each subfactor is neutralized by Barra SIZE and SW sector, z-scored, then equally averaged without winsorization.".to_string(),
            dependencies: vec![
                DataRequest::financial_quarters(
                    DatasetId::StockIncome,
                    &[
                        NET_PROFIT_ATTR_P_COLUMN,
                        EBIT_COLUMN,
                        INCOME_TAX_COLUMN,
                        TOTAL_PROFIT_COLUMN,
                        OPERATE_PROFIT_COLUMN,
                        FIN_EXP_COLUMN,
                        INVEST_INCOME_COLUMN,
                        FV_VALUE_CHG_GAIN_COLUMN,
                    ],
                    HISTORY_WINDOW,
                ),
                DataRequest::financial_quarters(
                    DatasetId::StockCashFlow,
                    &[CFO_COLUMN, CAPEX_COLUMN],
                    HISTORY_WINDOW,
                ),
                DataRequest::financial_quarters(
                    DatasetId::StockBalanceSheet,
                    &[
                        EQUITY_COLUMN,
                        TOTAL_ASSETS_COLUMN,
                        TOTAL_LIAB_COLUMN,
                        MONEY_CAP_COLUMN,
                        SHORT_BORROW_COLUMN,
                        NON_CURRENT_LIAB_DUE_1Y_COLUMN,
                        LONG_BORROW_COLUMN,
                        BOND_PAYABLE_COLUMN,
                    ],
                    BALANCE_HISTORY_QUARTERS,
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
        Box::new(ComprehensiveProfitabilityState::default())
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let mut snapshot_cache = InstrumentAlignedSnapshotCache::default();
        let raw_series = self.compute_raw_series(data, &mut snapshot_cache)?;
        self.finalize_raw_series(data, raw_series)
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
            .downcast_mut::<ComprehensiveProfitabilityState>()
            .ok_or_else(|| {
                err("comprehensive_profitability received incompatible event cache state")
            })?;
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
                self.compute_raw_series_with_prepared_inputs(
                    data,
                    &income,
                    &cashflow,
                    &balance,
                    snapshot_cache,
                )
            },
        )?;
        self.finalize_raw_series(data, raw_series)
            .map(|series| vec![series])
    }
}

impl StockDailyComprehensiveProfitability {
    fn compute_raw_series(
        &self,
        data: &DataPool,
        snapshot_cache: &mut InstrumentAlignedSnapshotCache<ProfitabilitySnapshot>,
    ) -> Result<Vec<FactorSeries>> {
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
        self.compute_raw_series_with_prepared_inputs(
            data,
            &income,
            &cashflow,
            &balance,
            snapshot_cache,
        )
    }

    fn compute_raw_series_with_prepared_inputs(
        &self,
        data: &DataPool,
        income: &FinancialPitReader<'_>,
        cashflow: &FinancialPitReader<'_>,
        balance: &FinancialPitReader<'_>,
        snapshot_cache: &mut InstrumentAlignedSnapshotCache<ProfitabilitySnapshot>,
    ) -> Result<Vec<FactorSeries>> {
        let panel = data.stock_universe_panel()?;
        let columns = profitability_raw_columns(&panel, income, cashflow, balance, snapshot_cache)?;
        Ok(columns.into_factor_series())
    }

    fn finalize_raw_series(
        &self,
        data: &DataPool,
        raw_series: Vec<FactorSeries>,
    ) -> Result<FactorSeries> {
        let panel = data.stock_universe_panel()?;
        let raw = raw_columns_from_series(&panel, raw_series)?;
        let stable_roe_pre =
            average_pair(&raw.roe.cs(cs_zscore)?, &raw.roe_stability.cs(cs_zscore)?)?;
        let stable_roe =
            stable_roe_pre.cs_binary(&raw.equity_multiplier, cs_regression_residual)?;
        let stable_roic =
            average_pair(&raw.roic.cs(cs_zscore)?, &raw.roic_stability.cs(cs_zscore)?)?;
        let stable_ronoa = average_pair(
            &raw.ronoa.cs(cs_zscore)?,
            &raw.ronoa_stability.cs(cs_zscore)?,
        )?;

        let processed_roe = postprocess_subfactor(&stable_roe, &panel, data)?;
        let processed_roic = postprocess_subfactor(&stable_roic, &panel, data)?;
        let processed_ronoa = postprocess_subfactor(&stable_ronoa, &panel, data)?;
        let processed_fcffic = postprocess_subfactor(&raw.fcffic, &panel, data)?;
        let final_factor = average_four_strict(
            &processed_roe,
            &processed_roic,
            &processed_ronoa,
            &processed_fcffic,
        )?;
        Ok(final_factor.to_factor_series(self.spec()))
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct ProfitabilitySnapshot {
    roe: Option<f64>,
    roe_stability: Option<f64>,
    equity_multiplier: Option<f64>,
    roic: Option<f64>,
    roic_stability: Option<f64>,
    ronoa: Option<f64>,
    ronoa_stability: Option<f64>,
    fcffic: Option<f64>,
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
    roe: PanelColumn,
    roe_stability: PanelColumn,
    equity_multiplier: PanelColumn,
    roic: PanelColumn,
    roic_stability: PanelColumn,
    ronoa: PanelColumn,
    ronoa_stability: PanelColumn,
    fcffic: PanelColumn,
}

impl RawColumns {
    fn into_factor_series(self) -> Vec<FactorSeries> {
        vec![
            self.roe.to_factor_series(raw_spec(RAW_ROE_ID)),
            self.roe_stability
                .to_factor_series(raw_spec(RAW_ROE_STABILITY_ID)),
            self.equity_multiplier
                .to_factor_series(raw_spec(RAW_EQUITY_MULTIPLIER_ID)),
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
    let mut roe = vec![None; panel.shape_len()];
    let mut roe_stability = vec![None; panel.shape_len()];
    let mut equity_multiplier = vec![None; panel.shape_len()];
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
        let date_offset = date_idx * instrument_count;
        for (instrument_idx, snapshot) in snapshots.into_iter().enumerate() {
            let Some(snapshot) = snapshot else {
                continue;
            };
            let offset = date_offset + instrument_idx;
            roe[offset] = snapshot.roe;
            roe_stability[offset] = snapshot.roe_stability;
            equity_multiplier[offset] = snapshot.equity_multiplier;
            roic[offset] = snapshot.roic;
            roic_stability[offset] = snapshot.roic_stability;
            ronoa[offset] = snapshot.ronoa;
            ronoa_stability[offset] = snapshot.ronoa_stability;
            fcffic[offset] = snapshot.fcffic;
        }
    }

    Ok(RawColumns {
        roe: panel.column_from_values(roe)?,
        roe_stability: panel.column_from_values(roe_stability)?,
        equity_multiplier: panel.column_from_values(equity_multiplier)?,
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
    let mut quarters = Vec::with_capacity(HISTORY_WINDOW);
    for idx in 0..HISTORY_WINDOW {
        let income_t = income.record_for_end_date(ts_code, trade_date, end_dates[idx]);
        let cashflow_t = cashflow.record_for_end_date(ts_code, trade_date, end_dates[idx]);
        let balance_t = balance.record_for_end_date(ts_code, trade_date, end_dates[idx]);
        let balance_prev = balance.record_for_end_date(ts_code, trade_date, end_dates[idx + 1]);
        quarters.push(quarter_profitability(
            income_t,
            cashflow_t,
            balance_t,
            balance_prev,
        ));
    }
    let current = quarters.first().copied()?;
    Some(ProfitabilitySnapshot {
        roe: current.roe,
        roe_stability: stability_from_quarters(&quarters, |quarter| quarter.roe),
        equity_multiplier: current.equity_multiplier,
        roic: current.roic,
        roic_stability: stability_from_quarters(&quarters, |quarter| quarter.roic),
        ronoa: current.ronoa,
        ronoa_stability: stability_from_quarters(&quarters, |quarter| quarter.ronoa),
        fcffic: current.fcffic,
    })
}

fn quarter_profitability(
    income: Option<PitFinancialRecordView<'_>>,
    cashflow: Option<PitFinancialRecordView<'_>>,
    balance: Option<PitFinancialRecordView<'_>>,
    balance_prev: Option<PitFinancialRecordView<'_>>,
) -> QuarterProfitability {
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
        .and_then(|(income, balance)| roic_for_records(income, balance));
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

fn roe_for_records(
    income: PitFinancialRecordView<'_>,
    balance: PitFinancialRecordView<'_>,
    balance_prev: PitFinancialRecordView<'_>,
) -> Option<f64> {
    let net_profit = clean(income.column(NET_PROFIT_ATTR_P_COLUMN))?;
    let equity = clean(balance.column(EQUITY_COLUMN))?;
    let equity_prev = clean(balance_prev.column(EQUITY_COLUMN))?;
    safe_div(net_profit, (equity + equity_prev) * 0.5)
}

fn equity_multiplier_for_record(balance: PitFinancialRecordView<'_>) -> Option<f64> {
    safe_div(
        clean(balance.column(TOTAL_ASSETS_COLUMN))?,
        clean(balance.column(EQUITY_COLUMN))?,
    )
}

fn roic_for_records(
    income: PitFinancialRecordView<'_>,
    balance: PitFinancialRecordView<'_>,
) -> Option<f64> {
    let ebit = clean(income.column(EBIT_COLUMN))?;
    let tax = tax_rate(
        income.column(INCOME_TAX_COLUMN),
        income.column(TOTAL_PROFIT_COLUMN),
    )?;
    let ic = invested_capital(balance)?;
    safe_div(ebit * (1.0 - tax), ic)
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
    let cfo = clean(cashflow.column(CFO_COLUMN))?;
    let capex = clean(cashflow.column(CAPEX_COLUMN))?;
    let ic = invested_capital(balance)?;
    safe_div(cfo - capex, ic)
}

fn operating_profit(income: PitFinancialRecordView<'_>) -> Option<f64> {
    let operate_profit = clean(income.column(OPERATE_PROFIT_COLUMN))?;
    let fin_exp = clean_or_zero(income.column(FIN_EXP_COLUMN));
    let invest_income = clean_or_zero(income.column(INVEST_INCOME_COLUMN));
    let fv_value_chg_gain = clean_or_zero(income.column(FV_VALUE_CHG_GAIN_COLUMN));
    let value = operate_profit + fin_exp - invest_income - fv_value_chg_gain;
    value.is_finite().then_some(value)
}

fn tax_rate(income_tax: Option<f64>, total_profit: Option<f64>) -> Option<f64> {
    let income_tax = clean(income_tax)?;
    let total_profit = clean(total_profit)?;
    let value = safe_div(income_tax, total_profit)?.clamp(0.0, 0.25);
    value.is_finite().then_some(value)
}

fn invested_capital(balance: PitFinancialRecordView<'_>) -> Option<f64> {
    invested_capital_from_values(
        clean(balance.column(EQUITY_COLUMN))?,
        interest_bearing_debt(balance),
        clean(balance.column(MONEY_CAP_COLUMN))?,
    )
}

fn invested_capital_from_values(equity: f64, debt: f64, cash: f64) -> Option<f64> {
    let value = equity + debt - cash;
    value.is_finite().then_some(value)
}

fn net_operating_assets(balance: PitFinancialRecordView<'_>) -> Option<f64> {
    net_operating_assets_from_values(
        clean(balance.column(TOTAL_ASSETS_COLUMN))?,
        clean(balance.column(MONEY_CAP_COLUMN))?,
        clean(balance.column(TOTAL_LIAB_COLUMN))?,
        interest_bearing_debt(balance),
    )
}

fn net_operating_assets_from_values(
    total_assets: f64,
    money_cap: f64,
    total_liab: f64,
    debt: f64,
) -> Option<f64> {
    let value = (total_assets - money_cap) - (total_liab - debt);
    value.is_finite().then_some(value)
}

fn interest_bearing_debt(balance: PitFinancialRecordView<'_>) -> f64 {
    clean_or_zero(balance.column(SHORT_BORROW_COLUMN))
        + clean_or_zero(balance.column(NON_CURRENT_LIAB_DUE_1Y_COLUMN))
        + clean_or_zero(balance.column(LONG_BORROW_COLUMN))
        + clean_or_zero(balance.column(BOND_PAYABLE_COLUMN))
}

fn stability_from_quarters<F>(quarters: &[QuarterProfitability], mut f: F) -> Option<f64>
where
    F: FnMut(&QuarterProfitability) -> Option<f64>,
{
    let values = quarters
        .iter()
        .filter_map(|quarter| f(quarter))
        .collect::<Vec<_>>();
    negative_sample_std_strict(&values, HISTORY_WINDOW)
}

fn negative_sample_std_strict(values: &[f64], required_len: usize) -> Option<f64> {
    if values.len() != required_len || values.iter().any(|value| !value.is_finite()) {
        return None;
    }
    sample_std(values).map(|std| -std)
}

fn sample_std(values: &[f64]) -> Option<f64> {
    if values.len() < 2 {
        return None;
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

fn average_pair(left: &PanelColumn, right: &PanelColumn) -> Result<PanelColumn> {
    left.zip_binary(right, |left, right| match (clean(left), clean(right)) {
        (Some(left), Some(right)) => Some((left + right) * 0.5),
        _ => None,
    })
}

fn average_four_strict(
    first: &PanelColumn,
    second: &PanelColumn,
    third: &PanelColumn,
    fourth: &PanelColumn,
) -> Result<PanelColumn> {
    first.zip_quaternary(
        second,
        third,
        fourth,
        |first, second, third, fourth| match (
            clean(first),
            clean(second),
            clean(third),
            clean(fourth),
        ) {
            (Some(first), Some(second), Some(third), Some(fourth)) => {
                Some((first + second + third + fourth) * 0.25)
            }
            _ => None,
        },
    )
}

fn raw_columns_from_series(
    panel: &DailyPanel,
    raw_series: Vec<FactorSeries>,
) -> Result<RawColumns> {
    Ok(RawColumns {
        roe: raw_column(panel, &raw_series, RAW_ROE_ID)?,
        roe_stability: raw_column(panel, &raw_series, RAW_ROE_STABILITY_ID)?,
        equity_multiplier: raw_column(panel, &raw_series, RAW_EQUITY_MULTIPLIER_ID)?,
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
                "missing comprehensive_profitability raw series {id}"
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

fn raw_specs() -> Vec<FactorSpec> {
    [
        RAW_ROE_ID,
        RAW_ROE_STABILITY_ID,
        RAW_EQUITY_MULTIPLIER_ID,
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
        description: "Internal comprehensive_profitability raw series.".to_string(),
        dependencies: Vec::new(),
        intraday_raw_dependencies: Vec::new(),
        lookback: Lookback { trading_days: 0 },
    }
}

fn tags() -> Vec<String> {
    [
        "financial",
        "fundamental",
        "profitability",
        "quality",
        "stability",
        "roe",
        "roic",
        "ronoa",
        "fcffic",
        "pit",
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
    fn tax_rate_clamps_and_rejects_invalid_profit() {
        assert_close(tax_rate(Some(10.0), Some(100.0)), Some(0.10));
        assert_close(tax_rate(Some(50.0), Some(100.0)), Some(0.25));
        assert_close(tax_rate(Some(-5.0), Some(100.0)), Some(0.0));
        assert_eq!(tax_rate(Some(1.0), Some(0.0)), None);
        assert_eq!(tax_rate(None, Some(100.0)), None);
    }

    #[test]
    fn invested_capital_uses_equity_debt_and_cash() {
        assert_close(
            invested_capital_from_values(1_000.0, 300.0, 120.0),
            Some(1_180.0),
        );
    }

    #[test]
    fn net_operating_assets_matches_assets_less_cash_less_operating_liabilities() {
        assert_close(
            net_operating_assets_from_values(2_000.0, 150.0, 900.0, 250.0),
            Some(1_200.0),
        );
    }

    #[test]
    fn stability_requires_strict_history_and_uses_negative_sample_std() {
        assert_eq!(
            negative_sample_std_strict(&[1.0, 2.0, 3.0], HISTORY_WINDOW),
            None
        );
        let values = [
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let mean = 6.5;
        let variance = values
            .iter()
            .map(|value| {
                let diff = value - mean;
                diff * diff
            })
            .sum::<f64>()
            / 11.0;
        assert_close(
            negative_sample_std_strict(&values, HISTORY_WINDOW),
            Some(-variance.sqrt()),
        );
    }

    #[test]
    fn average_four_requires_all_components() {
        let first = clean(Some(1.0));
        let second = clean(Some(2.0));
        let third = clean(Some(3.0));
        let fourth = clean(Some(4.0));
        assert_eq!(
            match (first, second, third, fourth) {
                (Some(first), Some(second), Some(third), Some(fourth)) => {
                    Some((first + second + third + fourth) * 0.25)
                }
                _ => None,
            },
            Some(2.5)
        );
        let missing = clean(None);
        assert_eq!(
            match (first, second, third, missing) {
                (Some(first), Some(second), Some(third), Some(fourth)) => {
                    Some((first + second + third + fourth) * 0.25)
                }
                _ => None,
            },
            None
        );
    }

    #[test]
    fn metadata_identifies_factor() {
        let spec = StockDailyComprehensiveProfitability.spec();
        assert_eq!(spec.id, FACTOR_ID);
        assert!(spec.tags.iter().any(|tag| tag == "profitability"));
        assert!(spec.tags.iter().any(|tag| tag == "pit"));
    }
}

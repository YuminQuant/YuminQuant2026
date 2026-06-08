use std::collections::BTreeMap;

use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::{err, Result};
use crate::factor::common::financial::previous_quarter_end_date;
use crate::factor::common::stock_daily_ops::{is_bj_stock, neutralize_size_only};
use crate::factor::common::{
    cached_financial_stock_snapshots_for_date, compute_financial_event_snapshot_streaming,
    factor_series_to_panel_column, ClassificationLevel, ClassificationMap, DailyPanel,
    EventDrivenCrossSectionCache, FinancialEventMarker, FinancialEventMarkerBuilder,
    FinancialEventSchedule, FinancialPitReader, FinancialStatementDataset,
    InstrumentAlignedSnapshotCache, PanelColumn, PitFinancialRecordView, ReportTypePreference,
};
use crate::operators::cs_zscore_by_group;

pub const PROVIDER_KEY: &str = "stock|daily|dbzq_financial_efficiency";
pub const ROE_EFFICIENCY_ID: &str = "roe_efficiency";
pub const CFO_EFFICIENCY_ID: &str = "cfo_efficiency";

const VERSION: &str = "0.1.0";
const FINANCIAL_QUARTERS: usize = 6;
const ROE_RAW_ID: &str = "__roe_efficiency_residual_raw";
const CFO_RAW_ID: &str = "__cfo_efficiency_residual_raw";
const EPS: f64 = 1e-12;

const REVENUE_COLUMN: &str = "revenue";
const PROFIT_COLUMN: &str = "n_income_attr_p";
const SELL_EXP_COLUMN: &str = "sell_exp";
const ADMIN_EXP_COLUMN: &str = "admin_exp";
const FIN_EXP_COLUMN: &str = "fin_exp";
const RD_EXP_COLUMN: &str = "rd_exp";

const CFO_COLUMN: &str = "n_cashflow_act";
const CAPEX_COLUMN: &str = "c_pay_acq_const_fiolta";

const EQUITY_COLUMN: &str = "total_hldr_eqy_exc_min_int";
const ACCOUNTS_RECEIV_COLUMN: &str = "accounts_receiv";
const PREPAYMENT_COLUMN: &str = "prepayment";
const INVENTORIES_COLUMN: &str = "inventories";
const FIX_ASSETS_COLUMN: &str = "fix_assets";
const ST_BORR_COLUMN: &str = "st_borr";
const LT_BORR_COLUMN: &str = "lt_borr";
const NON_CUR_LIAB_DUE_1Y_COLUMN: &str = "non_cur_liab_due_1y";
const BOND_PAYABLE_COLUMN: &str = "bond_payable";
const LT_PAYABLE_COLUMN: &str = "lt_payable";
const NOTES_PAYABLE_COLUMN: &str = "notes_payable";
const ACCT_PAYABLE_COLUMN: &str = "acct_payable";
const ADV_RECEIPTS_COLUMN: &str = "adv_receipts";

const INCOME_COLUMNS: [&str; 6] = [
    REVENUE_COLUMN,
    PROFIT_COLUMN,
    SELL_EXP_COLUMN,
    ADMIN_EXP_COLUMN,
    FIN_EXP_COLUMN,
    RD_EXP_COLUMN,
];
const CASHFLOW_COLUMNS: [&str; 2] = [CFO_COLUMN, CAPEX_COLUMN];
const BALANCE_COLUMNS: [&str; 13] = [
    EQUITY_COLUMN,
    ACCOUNTS_RECEIV_COLUMN,
    PREPAYMENT_COLUMN,
    INVENTORIES_COLUMN,
    FIX_ASSETS_COLUMN,
    ST_BORR_COLUMN,
    LT_BORR_COLUMN,
    NON_CUR_LIAB_DUE_1Y_COLUMN,
    BOND_PAYABLE_COLUMN,
    LT_PAYABLE_COLUMN,
    NOTES_PAYABLE_COLUMN,
    ACCT_PAYABLE_COLUMN,
    ADV_RECEIPTS_COLUMN,
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FinancialEfficiencyOutput {
    RoeEfficiency,
    CfoEfficiency,
}

#[derive(Default)]
pub struct FinancialEfficiencyComputeState {
    raw_cache: EventDrivenCrossSectionCache,
    snapshot_cache: InstrumentAlignedSnapshotCache<EfficiencySnapshot>,
}

#[derive(Clone, Copy, Debug, Default)]
struct EfficiencyNeeds {
    roe: bool,
    cfo: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct EfficiencySnapshot {
    roe: Option<RoeRow>,
    cfo: Option<CfoRow>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct RoeRow {
    y: f64,
    x: [f64; 4],
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct CfoRow {
    y: f64,
    x: [f64; 3],
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct RoeObservation {
    offset: usize,
    row: RoeRow,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct CfoObservation {
    offset: usize,
    row: CfoRow,
}

pub fn spec(output: FinancialEfficiencyOutput) -> FactorSpec {
    let (id, aliases, description) = match output {
        FinancialEfficiencyOutput::RoeEfficiency => (
            ROE_EFFICIENCY_ID,
            vec!["ROE Efficiency".to_string(), "ROE效率因子".to_string()],
            "DBZQ ROE efficiency factor. It uses PIT single-quarter ROE growth, capex-to-revenue, production asset turnover and interest-bearing liability growth, takes SW level-1 industry OLS residuals, industry-standardizes them and neutralizes SIZE.",
        ),
        FinancialEfficiencyOutput::CfoEfficiency => (
            CFO_EFFICIENCY_ID,
            vec!["CFO Efficiency".to_string(), "CFO效率因子".to_string()],
            "DBZQ CFO efficiency factor. It uses PIT single-quarter CFO scaled by equity, ROE, net accrual turnover change and opex change, takes SW level-1 industry OLS residuals, industry-standardizes them and neutralizes SIZE.",
        ),
    };
    FactorSpec {
        id: id.to_string(),
        aliases,
        name: id.to_string(),
        asset_class: AssetClass::Stock,
        frequency: Frequency::Daily,
        version: VERSION.to_string(),
        tags: tags(),
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
    let mut state = FinancialEfficiencyComputeState::default();
    compute_requested_stateful(requested_ids, context, data, &mut state)
}

pub fn compute_requested_stateful(
    requested_ids: &[String],
    context: &FactorContext,
    data: &DataPool,
    state: &mut FinancialEfficiencyComputeState,
) -> Result<Vec<FactorSeries>> {
    let needs = needs_from_requested(requested_ids);
    if !needs.roe && !needs.cfo {
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
    let sector_map = ClassificationMap::from_table(
        data.daily(DatasetId::StockSwClassification)?,
        ClassificationLevel::Sector,
    )?;
    let raw_specs = raw_specs(needs);
    let raw_series = compute_financial_event_snapshot_streaming(
        requested_ids,
        context,
        data,
        &mut state.raw_cache,
        &schedule,
        &raw_specs,
        |_, _, data| {
            compute_raw_with_prepared_financials(
                data,
                &income,
                &cashflow,
                &balance,
                &sector_map,
                &mut state.snapshot_cache,
                needs,
            )
        },
    )?;
    finalize_requested(data, raw_series, needs)
}

fn compute_raw_with_prepared_financials(
    data: &DataPool,
    income: &FinancialPitReader<'_>,
    cashflow: &FinancialPitReader<'_>,
    balance: &FinancialPitReader<'_>,
    sector_map: &ClassificationMap,
    snapshot_cache: &mut InstrumentAlignedSnapshotCache<EfficiencySnapshot>,
    needs: EfficiencyNeeds,
) -> Result<Vec<FactorSeries>> {
    let panel = data.daily_panel(DatasetId::StockDailyPv)?;
    let (roe_raw, cfo_raw) = efficiency_raw_columns(
        &panel,
        income,
        cashflow,
        balance,
        sector_map,
        snapshot_cache,
        needs,
    )?;
    let mut output = Vec::new();
    if needs.roe {
        output.push(
            roe_raw
                .ok_or_else(|| err("missing roe_efficiency raw column"))?
                .to_factor_series(raw_spec(ROE_RAW_ID)),
        );
    }
    if needs.cfo {
        output.push(
            cfo_raw
                .ok_or_else(|| err("missing cfo_efficiency raw column"))?
                .to_factor_series(raw_spec(CFO_RAW_ID)),
        );
    }
    Ok(output)
}

fn finalize_requested(
    data: &DataPool,
    raw_series: Vec<FactorSeries>,
    needs: EfficiencyNeeds,
) -> Result<Vec<FactorSeries>> {
    let panel = data.daily_panel(DatasetId::StockDailyPv)?;
    let raw_by_id = raw_series
        .into_iter()
        .map(|series| (series.spec.id.clone(), series))
        .collect::<BTreeMap<_, _>>();
    let mut output = Vec::new();
    if needs.roe {
        let raw = raw_by_id
            .get(ROE_RAW_ID)
            .ok_or_else(|| err("missing roe_efficiency raw series"))?;
        let raw = factor_series_to_panel_column(&panel, raw)?;
        let standardized = industry_zscore(&raw, data)?;
        let neutralized = neutralize_size_only(&standardized, &panel, data)?;
        output.push(neutralized.to_factor_series(spec(FinancialEfficiencyOutput::RoeEfficiency)));
    }
    if needs.cfo {
        let raw = raw_by_id
            .get(CFO_RAW_ID)
            .ok_or_else(|| err("missing cfo_efficiency raw series"))?;
        let raw = factor_series_to_panel_column(&panel, raw)?;
        let standardized = industry_zscore(&raw, data)?;
        let neutralized = neutralize_size_only(&standardized, &panel, data)?;
        output.push(neutralized.to_factor_series(spec(FinancialEfficiencyOutput::CfoEfficiency)));
    }
    Ok(output)
}

fn efficiency_raw_columns(
    panel: &DailyPanel,
    income: &FinancialPitReader<'_>,
    cashflow: &FinancialPitReader<'_>,
    balance: &FinancialPitReader<'_>,
    sector_map: &ClassificationMap,
    cache: &mut InstrumentAlignedSnapshotCache<EfficiencySnapshot>,
    needs: EfficiencyNeeds,
) -> Result<(Option<PanelColumn>, Option<PanelColumn>)> {
    let instrument_count = panel.instruments().len();
    let mut roe_values = needs.roe.then(|| vec![None; panel.shape_len()]);
    let mut cfo_values = needs.cfo.then(|| vec![None; panel.shape_len()]);

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
                efficiency_marker(ts_code, trade_date, income, cashflow, balance, needs)
            },
            |trade_date, ts_code, _| {
                efficiency_snapshot_for_stock(ts_code, trade_date, income, cashflow, balance, needs)
            },
        );
        let date_offset = date_idx * instrument_count;
        let mut roe_by_sector = BTreeMap::<String, Vec<RoeObservation>>::new();
        let mut cfo_by_sector = BTreeMap::<String, Vec<CfoObservation>>::new();
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
            if needs.roe {
                if let Some(row) = snapshot.roe.filter(row_is_finite_roe) {
                    roe_by_sector
                        .entry(group.to_string())
                        .or_default()
                        .push(RoeObservation { offset, row });
                }
            }
            if needs.cfo {
                if let Some(row) = snapshot.cfo.filter(row_is_finite_cfo) {
                    cfo_by_sector
                        .entry(group.to_string())
                        .or_default()
                        .push(CfoObservation { offset, row });
                }
            }
        }
        if let Some(values) = &mut roe_values {
            for (offset, residual) in grouped_roe_residuals(&roe_by_sector) {
                values[offset] = Some(residual);
            }
        }
        if let Some(values) = &mut cfo_values {
            for (offset, residual) in grouped_cfo_residuals(&cfo_by_sector) {
                values[offset] = Some(residual);
            }
        }
    }

    Ok((
        roe_values
            .map(|values| panel.column_from_values(values))
            .transpose()?,
        cfo_values
            .map(|values| panel.column_from_values(values))
            .transpose()?,
    ))
}

fn efficiency_marker(
    ts_code: &str,
    trade_date: i32,
    income: &FinancialPitReader<'_>,
    cashflow: &FinancialPitReader<'_>,
    balance: &FinancialPitReader<'_>,
    needs: EfficiencyNeeds,
) -> Option<FinancialEventMarker> {
    let ends = quarter_chain(
        income.latest_quarter_end_date(ts_code, trade_date)?,
        FINANCIAL_QUARTERS,
    )?;
    let mut builder = FinancialEventMarkerBuilder::new();
    for (idx, end_date) in ends.iter().copied().enumerate() {
        if idx <= 4 {
            builder.include_reader_record_for_end_date(
                FinancialStatementDataset::Income,
                income,
                ts_code,
                trade_date,
                end_date,
            );
        }
        if (needs.roe && idx <= 4) || (needs.cfo && idx == 0) {
            builder.include_reader_record_for_end_date(
                FinancialStatementDataset::CashFlow,
                cashflow,
                ts_code,
                trade_date,
                end_date,
            );
        }
        builder.include_reader_record_for_end_date(
            FinancialStatementDataset::BalanceSheet,
            balance,
            ts_code,
            trade_date,
            end_date,
        );
    }
    builder.build()
}

fn efficiency_snapshot_for_stock(
    ts_code: &str,
    trade_date: i32,
    income: &FinancialPitReader<'_>,
    cashflow: &FinancialPitReader<'_>,
    balance: &FinancialPitReader<'_>,
    needs: EfficiencyNeeds,
) -> Option<EfficiencySnapshot> {
    let ends = quarter_chain(
        income.latest_quarter_end_date(ts_code, trade_date)?,
        FINANCIAL_QUARTERS,
    )?;
    let income_records = records_for_ends(income, ts_code, trade_date, &ends);
    let cashflow_records = records_for_ends(cashflow, ts_code, trade_date, &ends);
    let balance_records = records_for_ends(balance, ts_code, trade_date, &ends);
    let roe = needs
        .roe
        .then(|| roe_row_from_records(&income_records, &cashflow_records, &balance_records))
        .flatten();
    let cfo = needs
        .cfo
        .then(|| cfo_row_from_records(&income_records, &cashflow_records, &balance_records))
        .flatten();
    Some(EfficiencySnapshot { roe, cfo })
}

fn roe_row_from_records(
    income_records: &[Option<PitFinancialRecordView<'_>>],
    cashflow_records: &[Option<PitFinancialRecordView<'_>>],
    balance_records: &[Option<PitFinancialRecordView<'_>>],
) -> Option<RoeRow> {
    let q_roe = q_roe_values(income_records, balance_records);
    let current_roe = q_roe[0]?;
    let mean_prev_roe = strict_mean(&q_roe[1..5])?;
    if mean_prev_roe.abs() <= EPS {
        return None;
    }
    let y = (current_roe - mean_prev_roe) / mean_prev_roe.abs();

    let mut capex_to_revenue = [None; 5];
    let mut operating_asset = [None; 6];
    let mut interest_liability = [None; 5];
    for idx in 0..5 {
        capex_to_revenue[idx] = capex_to_revenue_at(income_records[idx], cashflow_records[idx]);
        interest_liability[idx] = balance_records[idx].map(interest_liability_from_record);
    }
    for idx in 0..6 {
        operating_asset[idx] = balance_records[idx].map(operating_asset_from_record);
    }
    let current_capex = capex_to_revenue[0]?;
    let mean_capex = strict_mean(&capex_to_revenue[1..5])?;
    let turnover = turnover_from_values(
        clean(income_records[0]?.column(REVENUE_COLUMN))?,
        operating_asset[0]?,
        operating_asset[1]?,
    )?;
    let current_interest = interest_liability[0]?;
    let mean_interest = strict_mean(&interest_liability[1..5])?;
    if current_interest.abs() <= EPS && mean_interest.abs() <= EPS {
        return None;
    }
    if mean_interest.abs() <= EPS {
        return None;
    }
    let g_interest = current_interest / mean_interest;
    finite_row_roe(RoeRow {
        y,
        x: [current_capex, mean_capex, turnover, g_interest],
    })
}

fn cfo_row_from_records(
    income_records: &[Option<PitFinancialRecordView<'_>>],
    cashflow_records: &[Option<PitFinancialRecordView<'_>>],
    balance_records: &[Option<PitFinancialRecordView<'_>>],
) -> Option<CfoRow> {
    let q_roe = q_roe_values(income_records, balance_records);
    let mean_q_roe = strict_mean(&q_roe[0..4])?;
    let equity_t = equity_from_record(balance_records[0]?)?;
    let equity_t1 = equity_from_record(balance_records[1]?)?;
    let equity_sum = equity_t + equity_t1;
    if !equity_sum.is_finite() || equity_sum <= EPS {
        return None;
    }
    let y = clean(cashflow_records[0]?.column(CFO_COLUMN))? / equity_sum;

    let mut turnover = [None; 5];
    let mut opex = [None; 5];
    for idx in 0..5 {
        turnover[idx] = net_accrual_turnover_at(
            income_records[idx],
            balance_records[idx],
            balance_records[idx + 1],
        );
        opex[idx] = income_records[idx].map(opex_from_record);
    }
    let current_turnover = turnover[0]?;
    let mean_turnover = strict_mean(&turnover[1..5])?;
    let delta_turnover = current_turnover - mean_turnover;
    let current_opex = opex[0]?;
    let mean_opex = strict_mean(&opex[1..5])?;
    let delta_opex_scaled = (current_opex - mean_opex) / equity_sum;
    finite_row_cfo(CfoRow {
        y,
        x: [mean_q_roe, delta_turnover, delta_opex_scaled],
    })
}

fn grouped_roe_residuals(
    observations_by_sector: &BTreeMap<String, Vec<RoeObservation>>,
) -> Vec<(usize, f64)> {
    let mut output = Vec::new();
    for observations in observations_by_sector.values() {
        output.extend(roe_residuals(observations));
    }
    output
}

fn grouped_cfo_residuals(
    observations_by_sector: &BTreeMap<String, Vec<CfoObservation>>,
) -> Vec<(usize, f64)> {
    let mut output = Vec::new();
    for observations in observations_by_sector.values() {
        output.extend(cfo_residuals(observations));
    }
    output
}

fn roe_residuals(observations: &[RoeObservation]) -> Vec<(usize, f64)> {
    if observations.len() < 6 {
        return Vec::new();
    }
    let Some(beta) = roe_beta(observations) else {
        return Vec::new();
    };
    observations
        .iter()
        .filter_map(|observation| {
            let fitted = beta[0]
                + beta[1] * observation.row.x[0]
                + beta[2] * observation.row.x[1]
                + beta[3] * observation.row.x[2]
                + beta[4] * observation.row.x[3];
            let residual = observation.row.y - fitted;
            residual
                .is_finite()
                .then_some((observation.offset, residual))
        })
        .collect()
}

fn cfo_residuals(observations: &[CfoObservation]) -> Vec<(usize, f64)> {
    if observations.len() < 5 {
        return Vec::new();
    }
    let Some(beta) = cfo_beta(observations) else {
        return Vec::new();
    };
    observations
        .iter()
        .filter_map(|observation| {
            let fitted = beta[0]
                + beta[1] * observation.row.x[0]
                + beta[2] * observation.row.x[1]
                + beta[3] * observation.row.x[2];
            let residual = observation.row.y - fitted;
            residual
                .is_finite()
                .then_some((observation.offset, residual))
        })
        .collect()
}

fn roe_beta(observations: &[RoeObservation]) -> Option<[f64; 5]> {
    let mut xtx = [[0.0; 5]; 5];
    let mut xty = [0.0; 5];
    for observation in observations {
        let row = [
            1.0,
            observation.row.x[0],
            observation.row.x[1],
            observation.row.x[2],
            observation.row.x[3],
        ];
        for i in 0..5 {
            xty[i] += row[i] * observation.row.y;
            for j in 0..5 {
                xtx[i][j] += row[i] * row[j];
            }
        }
    }
    solve_linear_system(xtx, xty)
}

fn cfo_beta(observations: &[CfoObservation]) -> Option<[f64; 4]> {
    let mut xtx = [[0.0; 4]; 4];
    let mut xty = [0.0; 4];
    for observation in observations {
        let row = [
            1.0,
            observation.row.x[0],
            observation.row.x[1],
            observation.row.x[2],
        ];
        for i in 0..4 {
            xty[i] += row[i] * observation.row.y;
            for j in 0..4 {
                xtx[i][j] += row[i] * row[j];
            }
        }
    }
    solve_linear_system(xtx, xty)
}

fn solve_linear_system<const N: usize>(mut a: [[f64; N]; N], mut b: [f64; N]) -> Option<[f64; N]> {
    for pivot_idx in 0..N {
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
        for col_idx in pivot_idx..N {
            a[pivot_idx][col_idx] /= pivot;
        }
        b[pivot_idx] /= pivot;
        for row_idx in 0..N {
            if row_idx == pivot_idx {
                continue;
            }
            let factor = a[row_idx][pivot_idx];
            if factor.abs() <= f64::EPSILON {
                continue;
            }
            for col_idx in pivot_idx..N {
                a[row_idx][col_idx] -= factor * a[pivot_idx][col_idx];
            }
            b[row_idx] -= factor * b[pivot_idx];
        }
    }
    b.iter().all(|value| value.is_finite()).then_some(b)
}

fn q_roe_values(
    income_records: &[Option<PitFinancialRecordView<'_>>],
    balance_records: &[Option<PitFinancialRecordView<'_>>],
) -> [Option<f64>; 5] {
    let mut output = [None; 5];
    for idx in 0..5 {
        output[idx] = q_roe_at(income_records[idx], balance_records[idx]);
    }
    output
}

fn q_roe_at(
    income: Option<PitFinancialRecordView<'_>>,
    balance: Option<PitFinancialRecordView<'_>>,
) -> Option<f64> {
    let profit = clean(income?.column(PROFIT_COLUMN))?;
    let equity = equity_from_record(balance?)?;
    safe_div_positive_denominator(profit, equity)
}

fn capex_to_revenue_at(
    income: Option<PitFinancialRecordView<'_>>,
    cashflow: Option<PitFinancialRecordView<'_>>,
) -> Option<f64> {
    let revenue = clean(income?.column(REVENUE_COLUMN)).filter(|value| *value > 0.0)?;
    let capex = clean_or_zero(cashflow?.column(CAPEX_COLUMN));
    let value = capex / revenue;
    value.is_finite().then_some(value)
}

fn turnover_from_values(revenue: f64, asset_t: f64, asset_t1: f64) -> Option<f64> {
    let denominator = asset_t + asset_t1;
    if !denominator.is_finite() || denominator <= EPS {
        return None;
    }
    let value = 2.0 * revenue / denominator;
    value.is_finite().then_some(value)
}

fn net_accrual_turnover_at(
    income: Option<PitFinancialRecordView<'_>>,
    balance_t: Option<PitFinancialRecordView<'_>>,
    balance_t1: Option<PitFinancialRecordView<'_>>,
) -> Option<f64> {
    let revenue = clean(income?.column(REVENUE_COLUMN))?;
    let asset_t = net_accrual_asset_from_record(balance_t?);
    let asset_t1 = net_accrual_asset_from_record(balance_t1?);
    let denominator = 0.5 * (asset_t + asset_t1);
    if !denominator.is_finite() || denominator.abs() <= EPS {
        return None;
    }
    let value = revenue / denominator;
    value.is_finite().then_some(value)
}

fn operating_asset_from_record(record: PitFinancialRecordView<'_>) -> f64 {
    clean_or_zero(record.column(ACCOUNTS_RECEIV_COLUMN))
        + clean_or_zero(record.column(PREPAYMENT_COLUMN))
        + clean_or_zero(record.column(INVENTORIES_COLUMN))
        + clean_or_zero(record.column(FIX_ASSETS_COLUMN))
}

fn interest_liability_from_record(record: PitFinancialRecordView<'_>) -> f64 {
    clean_or_zero(record.column(ST_BORR_COLUMN))
        + clean_or_zero(record.column(LT_BORR_COLUMN))
        + clean_or_zero(record.column(NON_CUR_LIAB_DUE_1Y_COLUMN))
        + clean_or_zero(record.column(BOND_PAYABLE_COLUMN))
        + clean_or_zero(record.column(LT_PAYABLE_COLUMN))
}

fn net_accrual_asset_from_record(record: PitFinancialRecordView<'_>) -> f64 {
    clean_or_zero(record.column(ACCOUNTS_RECEIV_COLUMN))
        + clean_or_zero(record.column(PREPAYMENT_COLUMN))
        + clean_or_zero(record.column(INVENTORIES_COLUMN))
        - clean_or_zero(record.column(NOTES_PAYABLE_COLUMN))
        - clean_or_zero(record.column(ACCT_PAYABLE_COLUMN))
        - clean_or_zero(record.column(ADV_RECEIPTS_COLUMN))
}

fn opex_from_record(record: PitFinancialRecordView<'_>) -> f64 {
    clean_or_zero(record.column(SELL_EXP_COLUMN))
        + clean_or_zero(record.column(ADMIN_EXP_COLUMN))
        + clean_or_zero(record.column(FIN_EXP_COLUMN))
        + clean_or_zero(record.column(RD_EXP_COLUMN))
}

fn equity_from_record(record: PitFinancialRecordView<'_>) -> Option<f64> {
    clean(record.column(EQUITY_COLUMN)).filter(|value| *value > 0.0)
}

fn records_for_ends<'a>(
    reader: &'a FinancialPitReader<'a>,
    ts_code: &str,
    trade_date: i32,
    ends: &[i32],
) -> Vec<Option<PitFinancialRecordView<'a>>> {
    ends.iter()
        .map(|end_date| reader.record_for_end_date(ts_code, trade_date, *end_date))
        .collect()
}

fn quarter_chain(anchor: i32, len: usize) -> Option<Vec<i32>> {
    let mut output = Vec::with_capacity(len);
    let mut current = anchor;
    for _ in 0..len {
        output.push(current);
        current = previous_quarter_end_date(current)?;
    }
    Some(output)
}

fn strict_mean(values: &[Option<f64>]) -> Option<f64> {
    if values.is_empty() || values.iter().any(|value| clean(*value).is_none()) {
        return None;
    }
    let sum = values.iter().filter_map(|value| clean(*value)).sum::<f64>();
    let mean = sum / values.len() as f64;
    mean.is_finite().then_some(mean)
}

fn safe_div_positive_denominator(numerator: f64, denominator: f64) -> Option<f64> {
    if !numerator.is_finite() || !denominator.is_finite() || denominator <= EPS {
        return None;
    }
    let value = numerator / denominator;
    value.is_finite().then_some(value)
}

fn finite_row_roe(row: RoeRow) -> Option<RoeRow> {
    row_is_finite_roe(&row).then_some(row)
}

fn finite_row_cfo(row: CfoRow) -> Option<CfoRow> {
    row_is_finite_cfo(&row).then_some(row)
}

fn row_is_finite_roe(row: &RoeRow) -> bool {
    row.y.is_finite() && row.x.iter().all(|value| value.is_finite())
}

fn row_is_finite_cfo(row: &CfoRow) -> bool {
    row.y.is_finite() && row.x.iter().all(|value| value.is_finite())
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

fn needs_from_requested(requested_ids: &[String]) -> EfficiencyNeeds {
    EfficiencyNeeds {
        roe: requested_ids.iter().any(|id| id == ROE_EFFICIENCY_ID),
        cfo: requested_ids.iter().any(|id| id == CFO_EFFICIENCY_ID),
    }
}

fn raw_specs(needs: EfficiencyNeeds) -> Vec<FactorSpec> {
    let mut specs = Vec::new();
    if needs.roe {
        specs.push(raw_spec(ROE_RAW_ID));
    }
    if needs.cfo {
        specs.push(raw_spec(CFO_RAW_ID));
    }
    specs
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
        description: format!("Internal DBZQ financial efficiency raw series {id}."),
        dependencies: Vec::new(),
        intraday_raw_dependencies: Vec::new(),
        lookback: Lookback { trading_days: 0 },
    }
}

fn dependencies() -> Vec<DataRequest> {
    vec![
        DataRequest::new(DatasetId::StockDailyPv, &["close"]),
        DataRequest::financial_quarters(
            DatasetId::StockIncome,
            &INCOME_COLUMNS,
            FINANCIAL_QUARTERS,
        ),
        DataRequest::financial_quarters(
            DatasetId::StockCashFlow,
            &CASHFLOW_COLUMNS,
            FINANCIAL_QUARTERS,
        ),
        DataRequest::financial_quarters(
            DatasetId::StockBalanceSheet,
            &BALANCE_COLUMNS,
            FINANCIAL_QUARTERS,
        ),
        DataRequest::new(DatasetId::StockSwClassification, &["l1_code"]),
        DataRequest::new(DatasetId::StockBarraDaily, &["SIZE"]),
    ]
}

fn tags() -> Vec<String> {
    [
        "DBZQ",
        "financial",
        "fundamental",
        "pit",
        "efficiency",
        "residual",
        "industry_standardize",
        "size_neutralize",
        "daily",
    ]
    .iter()
    .map(|tag| (*tag).to_string())
    .collect()
}

fn clean(value: Option<f64>) -> Option<f64> {
    value.filter(|value| value.is_finite())
}

fn clean_or_zero(value: Option<f64>) -> f64 {
    clean(value).unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1e-10,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn roe_residuals_require_six_industry_observations() {
        let row = RoeRow {
            y: 1.0,
            x: [1.0, 2.0, 3.0, 4.0],
        };
        let observations = (0..5)
            .map(|offset| RoeObservation { offset, row })
            .collect::<Vec<_>>();
        assert!(roe_residuals(&observations).is_empty());
    }

    #[test]
    fn cfo_residuals_require_five_industry_observations() {
        let row = CfoRow {
            y: 1.0,
            x: [1.0, 2.0, 3.0],
        };
        let observations = (0..4)
            .map(|offset| CfoObservation { offset, row })
            .collect::<Vec<_>>();
        assert!(cfo_residuals(&observations).is_empty());
    }

    #[test]
    fn strict_mean_requires_all_values() {
        assert_eq!(strict_mean(&[Some(1.0), None]), None);
        assert_close(strict_mean(&[Some(1.0), Some(3.0)]).unwrap(), 2.0);
    }

    #[test]
    fn turnover_rejects_zero_asset_base() {
        assert_eq!(turnover_from_values(10.0, 0.0, 0.0), None);
        assert_close(turnover_from_values(10.0, 4.0, 6.0).unwrap(), 2.0);
    }

    #[test]
    fn specs_have_dbzq_tags() {
        for output in [
            FinancialEfficiencyOutput::RoeEfficiency,
            FinancialEfficiencyOutput::CfoEfficiency,
        ] {
            let spec = spec(output);
            assert!(spec.tags.contains(&"DBZQ".to_string()));
            assert!(spec.tags.contains(&"financial".to_string()));
            assert!(spec.tags.contains(&"efficiency".to_string()));
        }
    }

    #[test]
    fn requested_outputs_are_requested_aware() {
        let requested = vec![ROE_EFFICIENCY_ID.to_string()];
        let needs = needs_from_requested(&requested);
        assert!(needs.roe);
        assert!(!needs.cfo);
        let specs = raw_specs(needs);
        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].id, ROE_RAW_ID);
    }
}

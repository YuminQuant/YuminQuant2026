use std::any::Any;
use std::collections::{BTreeSet, HashMap};

use crate::barra::common::{
    add_months, align_table_column, arithmetic_return, average_columns, clean, expand_index_column,
    fy1_quarter, log_return, panel_from_target_stock_map, safe_div, sqrt_circ_mv_weights,
    standardize_panel_industry_filled_weighted, zscore_panel_weighted_filled_zero,
};
use crate::barra::BarraExposure;
use crate::core::{
    AssetClass, BarraSeries, BarraSpec, DataRequest, DatasetId, FactorContext, Frequency, Lookback,
};
use crate::data::{DataPool, Table};
use crate::error::Result;
use crate::factor::common::{
    cached_financial_stock_snapshots_for_date, DailyPanel, FinancialEventMarker,
    FinancialEventMarkerBuilder, FinancialStatementDataset, InstrumentAlignedSnapshotCache,
    PitFinancialData, ReportTypePreference,
};
use crate::operators::{ts_ew_regression_alpha_beta_residual_sigma, ts_ew_sum};

pub struct StockDailyBarraCne6Value;

const MODEL: &str = "CNE6";
const VERSION: &str = "0.3.0";
const MARKET_INDEX: &str = "000300.SH";
const LONG_WINDOW: usize = 1040;
const LONG_HALF_LIFE: f64 = 260.0;
const LONG_LAG: usize = 273;
const LONG_AVG_WINDOW: usize = 11;
const MAX_LOOKBACK: usize = LONG_WINDOW + LONG_LAG + LONG_AVG_WINDOW - 1;

pub fn create() -> Box<dyn BarraExposure> {
    Box::new(StockDailyBarraCne6Value)
}

#[derive(Default)]
struct ValueComputeState {
    cash_ep_cache: InstrumentAlignedSnapshotCache<CashEpSlowSnapshot>,
    ebit_ev_cache: InstrumentAlignedSnapshotCache<EbitEvSlowSnapshot>,
}

impl BarraExposure for StockDailyBarraCne6Value {
    fn family_id(&self) -> &'static str {
        "VALUE"
    }

    fn specs(&self) -> Vec<BarraSpec> {
        vec![
            value_spec("BTOP", "CNE6 book-to-price", "1 / daily pb.", MAX_LOOKBACK),
            value_spec(
                "Trailing_Earnings_To_Price",
                "CNE6 trailing earnings-to-price",
                "1 / daily pe_ttm.",
                MAX_LOOKBACK,
            ),
            value_spec(
                "Analyst_Predicted_Earnings_To_Price",
                "CNE6 analyst predicted earnings-to-price",
                "Mean FY1 analyst implied earnings yield from recent reports.",
                MAX_LOOKBACK,
            ),
            value_spec(
                "Cash_Earnings_To_Price",
                "CNE6 cash earnings-to-price",
                "TTM operating cash flow divided by current market value.",
                MAX_LOOKBACK,
            ),
            value_spec(
                "EBIT_To_EV",
                "CNE6 EBIT to enterprise value",
                "Latest annual EBIT divided by enterprise value.",
                MAX_LOOKBACK,
            ),
            value_spec(
                "Long_Term_Relative_Strength",
                "CNE6 long-term relative strength",
                "Negative lagged average of 1040-day EW log-return strength.",
                MAX_LOOKBACK,
            ),
            value_spec(
                "Long_Term_Historical_Alpha",
                "CNE6 long-term historical alpha",
                "Negative lagged average of 1040-day EW CAPM alpha.",
                MAX_LOOKBACK,
            ),
            value_spec(
                "Long_Term_Reversal",
                "CNE6 long-term reversal",
                "Composite of long-term relative strength and historical alpha.",
                MAX_LOOKBACK,
            ),
            value_spec(
                "Earnings_Yield",
                "CNE6 earnings yield",
                "Composite of trailing, analyst, cash and EBIT-to-EV yield signals.",
                MAX_LOOKBACK,
            ),
            value_spec(
                "VALUE",
                "CNE6 VALUE style exposure",
                "Composite of BTOP, Earnings_Yield, and Long_Term_Reversal.",
                MAX_LOOKBACK,
            ),
        ]
    }

    fn initial_compute_state(&self, _selected_ids: &BTreeSet<String>) -> Box<dyn Any + Send> {
        Box::new(ValueComputeState::default())
    }

    fn compute_stateful(
        &self,
        context: &FactorContext,
        data: &DataPool,
        state: &mut (dyn Any + Send),
    ) -> Result<Vec<BarraSeries>> {
        let state = state
            .downcast_mut::<ValueComputeState>()
            .expect("VALUE compute state type");
        self.compute_with_cache(
            context,
            data,
            &mut state.cash_ep_cache,
            &mut state.ebit_ev_cache,
        )
    }

    fn compute(&self, context: &FactorContext, data: &DataPool) -> Result<Vec<BarraSeries>> {
        let mut cash_ep_cache = InstrumentAlignedSnapshotCache::default();
        let mut ebit_ev_cache = InstrumentAlignedSnapshotCache::default();
        self.compute_with_cache(context, data, &mut cash_ep_cache, &mut ebit_ev_cache)
    }
}

impl StockDailyBarraCne6Value {
    fn compute_with_cache(
        &self,
        _context: &FactorContext,
        data: &DataPool,
        cash_ep_cache: &mut InstrumentAlignedSnapshotCache<CashEpSlowSnapshot>,
        ebit_ev_cache: &mut InstrumentAlignedSnapshotCache<EbitEvSlowSnapshot>,
    ) -> Result<Vec<BarraSeries>> {
        let panel = data.daily_panel(DatasetId::StockDailyPv)?;
        let index_panel = data.index_daily_panel(MARKET_INDEX)?;

        let basic_table = data.daily(DatasetId::StockDailyBasic)?;
        let pb = align_table_column(panel, basic_table, "pb")?;
        let pe_ttm = align_table_column(panel, basic_table, "pe_ttm")?;
        let total_mv = align_table_column(panel, basic_table, "total_mv")?;
        let weights = sqrt_circ_mv_weights(panel, data)?;
        let btop_raw = pb.map_values(|value| {
            clean(value).and_then(|value| (value.abs() > f64::EPSILON).then_some(1.0 / value))
        });
        let btop = standardize_panel_industry_filled_weighted(&btop_raw, &weights, data)?;
        let trailing_ep_raw = pe_ttm.map_values(|value| {
            clean(value).and_then(|value| (value.abs() > f64::EPSILON).then_some(1.0 / value))
        });
        let trailing_ep =
            standardize_panel_industry_filled_weighted(&trailing_ep_raw, &weights, data)?;

        let analyst_records = parse_analyst_records(data.daily(DatasetId::StockAnalystReport)?)?;
        let analyst_by_stock = index_analyst_records(&analyst_records);
        let analyst_ep_raw = analyst_ep_column(panel, &analyst_by_stock)?;
        let analyst_ep =
            standardize_panel_industry_filled_weighted(&analyst_ep_raw, &weights, data)?;

        let cashflow = PitFinancialData::from_table(
            data.daily(DatasetId::StockCashFlow)?,
            &["n_cashflow_act"],
            ReportTypePreference::income_single_quarter(),
        )?;
        let cash_ep_raw = cash_ep_raw_column(panel, &total_mv, &cashflow, cash_ep_cache)?;
        let cash_ep = standardize_panel_industry_filled_weighted(&cash_ep_raw, &weights, data)?;

        let income = PitFinancialData::from_table(
            data.daily(DatasetId::StockIncome)?,
            &["ebit"],
            ReportTypePreference::consolidated(),
        )?;
        let balance = PitFinancialData::from_table(
            data.daily(DatasetId::StockBalanceSheet)?,
            &["total_liab", "money_cap"],
            ReportTypePreference::balance_sheet_consolidated(),
        )?;
        let ebit_ev_raw = ebit_ev_raw_column(panel, &total_mv, &income, &balance, ebit_ev_cache)?;
        let ebit_ev = standardize_panel_industry_filled_weighted(&ebit_ev_raw, &weights, data)?;

        let stock_returns = panel
            .column("close")?
            .zip_binary(&panel.column("pre_close")?, arithmetic_return)?;
        let stock_log_returns = panel
            .column("close")?
            .zip_binary(&panel.column("pre_close")?, log_return)?;
        let index_returns = index_panel
            .column("close")?
            .zip_binary(&index_panel.column("pre_close")?, arithmetic_return)?;
        let market_returns = expand_index_column(panel, index_panel, &index_returns)?;

        let long_relative_strength_raw_col = stock_log_returns.ts(long_relative_strength_raw)?;
        let long_relative_strength = standardize_panel_industry_filled_weighted(
            &long_relative_strength_raw_col,
            &weights,
            data,
        )?;
        let long_historical_alpha_raw_col =
            stock_returns.ts_binary(&market_returns, long_historical_alpha_raw)?;
        let long_historical_alpha = standardize_panel_industry_filled_weighted(
            &long_historical_alpha_raw_col,
            &weights,
            data,
        )?;
        let long_term_reversal_raw =
            average_columns(panel, &[&long_relative_strength, &long_historical_alpha])?;
        let long_term_reversal =
            zscore_panel_weighted_filled_zero(&long_term_reversal_raw, &weights)?;
        let earnings_yield_raw =
            average_columns(panel, &[&trailing_ep, &analyst_ep, &cash_ep, &ebit_ev])?;
        let earnings_yield = zscore_panel_weighted_filled_zero(&earnings_yield_raw, &weights)?;
        let value_raw = average_columns(panel, &[&btop, &earnings_yield, &long_term_reversal])?;
        let value = zscore_panel_weighted_filled_zero(&value_raw, &weights)?;

        let specs = self.specs();
        Ok(vec![
            btop.to_barra_series(specs[0].clone()),
            trailing_ep.to_barra_series(specs[1].clone()),
            analyst_ep.to_barra_series(specs[2].clone()),
            cash_ep.to_barra_series(specs[3].clone()),
            ebit_ev.to_barra_series(specs[4].clone()),
            long_relative_strength.to_barra_series(specs[5].clone()),
            long_historical_alpha.to_barra_series(specs[6].clone()),
            long_term_reversal.to_barra_series(specs[7].clone()),
            earnings_yield.to_barra_series(specs[8].clone()),
            value.to_barra_series(specs[9].clone()),
        ])
    }
}

#[derive(Clone, Copy, Debug)]
struct CashEpSlowSnapshot {
    cash: f64,
    total_mv: f64,
}

#[derive(Clone, Copy, Debug)]
struct EbitEvSlowSnapshot {
    ebit: f64,
    total_liab: f64,
    money_cap: f64,
    total_mv: f64,
}

fn cash_ep_raw_column(
    panel: &DailyPanel,
    total_mv: &crate::factor::common::PanelColumn,
    cashflow: &PitFinancialData,
    cache: &mut InstrumentAlignedSnapshotCache<CashEpSlowSnapshot>,
) -> Result<crate::factor::common::PanelColumn> {
    let mut values = vec![None; panel.shape_len()];
    let instrument_count = panel.instruments().len();
    for (date_idx, trade_date) in panel.dates().iter().copied().enumerate() {
        if !panel.is_target_date(trade_date) {
            continue;
        }
        let snapshots = cached_financial_stock_snapshots_for_date(
            panel,
            trade_date,
            cache,
            |_, _, offset| !panel.is_present_offset(offset),
            |trade_date, ts_code, _| cash_ep_marker(ts_code, trade_date, cashflow),
            |trade_date, ts_code, offset| {
                let total_mv_snapshot =
                    clean(total_mv.values()[offset]).filter(|value| *value > 0.0);
                cash_ep_snapshot(ts_code, trade_date, cashflow, total_mv_snapshot)
            },
        );
        for (instrument_idx, snapshot) in snapshots.into_iter().enumerate() {
            let offset = date_idx * instrument_count + instrument_idx;
            values[offset] =
                snapshot.and_then(|snapshot| safe_div(snapshot.cash, snapshot.total_mv));
        }
    }
    panel.column_from_values(values)
}

fn ebit_ev_raw_column(
    panel: &DailyPanel,
    total_mv: &crate::factor::common::PanelColumn,
    income: &PitFinancialData,
    balance: &PitFinancialData,
    cache: &mut InstrumentAlignedSnapshotCache<EbitEvSlowSnapshot>,
) -> Result<crate::factor::common::PanelColumn> {
    let mut values = vec![None; panel.shape_len()];
    let instrument_count = panel.instruments().len();
    for (date_idx, trade_date) in panel.dates().iter().copied().enumerate() {
        if !panel.is_target_date(trade_date) {
            continue;
        }
        let snapshots = cached_financial_stock_snapshots_for_date(
            panel,
            trade_date,
            cache,
            |_, _, offset| !panel.is_present_offset(offset),
            |trade_date, ts_code, _| ebit_ev_marker(ts_code, trade_date, income, balance),
            |trade_date, ts_code, offset| {
                let total_mv_snapshot =
                    clean(total_mv.values()[offset]).filter(|value| *value > 0.0);
                ebit_ev_snapshot(ts_code, trade_date, income, balance, total_mv_snapshot)
            },
        );
        for (instrument_idx, snapshot) in snapshots.into_iter().enumerate() {
            let offset = date_idx * instrument_count + instrument_idx;
            values[offset] = snapshot.and_then(|snapshot| {
                let ev = snapshot.total_mv + snapshot.total_liab - snapshot.money_cap;
                safe_div(snapshot.ebit, ev)
            });
        }
    }
    panel.column_from_values(values)
}

fn cash_ep_marker(
    ts_code: &str,
    trade_date: i32,
    cashflow: &PitFinancialData,
) -> Option<FinancialEventMarker> {
    let mut builder = FinancialEventMarkerBuilder::new();
    builder.include_latest_ttm(
        FinancialStatementDataset::CashFlow,
        cashflow,
        ts_code,
        trade_date,
    );
    builder.build()
}

fn cash_ep_snapshot(
    ts_code: &str,
    trade_date: i32,
    cashflow: &PitFinancialData,
    total_mv_snapshot: Option<f64>,
) -> Option<CashEpSlowSnapshot> {
    Some(CashEpSlowSnapshot {
        cash: cashflow.ttm_sum(ts_code, trade_date, "n_cashflow_act")?,
        total_mv: total_mv_snapshot?,
    })
}

fn ebit_ev_marker(
    ts_code: &str,
    trade_date: i32,
    income: &PitFinancialData,
    balance: &PitFinancialData,
) -> Option<FinancialEventMarker> {
    let mut builder = FinancialEventMarkerBuilder::new();
    builder.include_latest_annual(
        FinancialStatementDataset::Income,
        income,
        ts_code,
        trade_date,
    );
    builder.include_latest_annual(
        FinancialStatementDataset::BalanceSheet,
        balance,
        ts_code,
        trade_date,
    );
    builder.build()
}

fn ebit_ev_snapshot(
    ts_code: &str,
    trade_date: i32,
    income: &PitFinancialData,
    balance: &PitFinancialData,
    total_mv_snapshot: Option<f64>,
) -> Option<EbitEvSlowSnapshot> {
    Some(EbitEvSlowSnapshot {
        ebit: income.latest_annual_value(ts_code, trade_date, "ebit")?,
        total_liab: balance
            .latest_annual_value(ts_code, trade_date, "total_liab")
            .unwrap_or(0.0),
        money_cap: balance
            .latest_annual_value(ts_code, trade_date, "money_cap")
            .unwrap_or(0.0),
        total_mv: total_mv_snapshot?,
    })
}

fn value_spec(id: &str, name: &str, description: &str, lookback: usize) -> BarraSpec {
    BarraSpec {
        id: id.to_string(),
        aliases: Vec::new(),
        name: name.to_string(),
        model: MODEL.to_string(),
        asset_class: AssetClass::Stock,
        frequency: Frequency::Daily,
        version: VERSION.to_string(),
        tags: ["barra", "cne6", "style", "value", "daily", "stock"]
            .iter()
            .map(|value| value.to_string())
            .collect(),
        description: description.to_string(),
        dependencies: vec![
            DataRequest::new(DatasetId::StockDailyPv, &["close", "pre_close"]),
            DataRequest::new(
                DatasetId::StockDailyBasic,
                &["pb", "pe_ttm", "total_mv", "circ_mv"],
            ),
            DataRequest::new(DatasetId::StockSwClassification, &["l1_code"]),
            DataRequest::financial_quarters(DatasetId::StockIncome, &["ebit"], 24),
            DataRequest::financial_quarters(DatasetId::StockCashFlow, &["n_cashflow_act"], 8),
            DataRequest::financial_quarters(
                DatasetId::StockBalanceSheet,
                &["total_liab", "money_cap"],
                24,
            ),
            DataRequest::new(
                DatasetId::StockAnalystReport,
                &["ts_code", "report_date", "quarter", "pe"],
            ),
            DataRequest::index_daily(MARKET_INDEX, &["close", "pre_close"]),
        ],
        lookback: Lookback {
            trading_days: lookback,
        },
    }
}

#[derive(Clone, Debug)]
struct AnalystRecord {
    ts_code: String,
    report_date: i32,
    quarter: String,
    pe: f64,
}

fn parse_analyst_records(table: &Table) -> Result<Vec<AnalystRecord>> {
    let ts_codes = table.required_utf8("ts_code")?;
    let report_dates = table.required_i32_date_cast("report_date")?;
    let quarters = table.required_utf8("quarter")?;
    let pe_values = table.required_f64_cast("pe")?;
    let mut records = Vec::new();
    for idx in 0..table.len {
        let (Some(ts_code), Some(report_date), Some(quarter), Some(pe)) = (
            ts_codes[idx].clone(),
            report_dates[idx],
            quarters[idx].clone(),
            clean(pe_values[idx]),
        ) else {
            continue;
        };
        if pe.abs() <= f64::EPSILON {
            continue;
        }
        records.push(AnalystRecord {
            ts_code,
            report_date,
            quarter,
            pe,
        });
    }
    Ok(records)
}

fn analyst_ep_column(
    panel: &DailyPanel,
    records_by_stock: &HashMap<&str, Vec<&AnalystRecord>>,
) -> Result<crate::factor::common::PanelColumn> {
    panel_from_target_stock_map(panel, |trade_date, ts_code| {
        let start_date = add_months(trade_date, -3);
        let fy1 = fy1_quarter(trade_date);
        let records = records_by_stock.get(ts_code)?;
        let mut sum = 0.0;
        let mut count = 0usize;
        for record in records {
            if record.quarter == fy1
                && record.report_date >= start_date
                && record.report_date <= trade_date
            {
                sum += 1.0 / record.pe;
                count += 1;
            }
        }
        (count > 0).then_some(sum / count as f64)
    })
}

fn index_analyst_records(records: &[AnalystRecord]) -> HashMap<&str, Vec<&AnalystRecord>> {
    let mut by_stock = HashMap::<&str, Vec<&AnalystRecord>>::new();
    for record in records {
        by_stock
            .entry(record.ts_code.as_str())
            .or_default()
            .push(record);
    }
    by_stock
}

fn long_relative_strength_raw(values: &[Option<f64>]) -> Vec<Option<f64>> {
    let raw = ts_ew_sum(values, LONG_WINDOW, LONG_WINDOW, LONG_HALF_LIFE);
    lagged_mean(&raw, LONG_LAG, LONG_AVG_WINDOW)
        .into_iter()
        .map(|value| clean(value).map(|value| -value))
        .collect()
}

fn long_historical_alpha_raw(
    stock_returns: &[Option<f64>],
    market_returns: &[Option<f64>],
) -> Vec<Option<f64>> {
    let (alpha, _beta, _sigma) = ts_ew_regression_alpha_beta_residual_sigma(
        stock_returns,
        market_returns,
        LONG_WINDOW,
        LONG_WINDOW,
        LONG_HALF_LIFE,
    );
    lagged_mean(&alpha, LONG_LAG, LONG_AVG_WINDOW)
        .into_iter()
        .map(|value| clean(value).map(|value| -value))
        .collect()
}

fn lagged_mean(values: &[Option<f64>], lag: usize, window: usize) -> Vec<Option<f64>> {
    let mut output = vec![None; values.len()];
    for idx in 0..values.len() {
        let Some(end) = idx.checked_sub(lag) else {
            continue;
        };
        let Some(start) = end.checked_sub(window - 1) else {
            continue;
        };
        let mut sum = 0.0;
        let mut count = 0usize;
        for value in &values[start..=end] {
            let Some(value) = clean(*value) else {
                count = 0;
                break;
            };
            sum += value;
            count += 1;
        }
        if count == window {
            output[idx] = Some(sum / window as f64);
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use crate::barra::BarraExposure;

    use super::StockDailyBarraCne6Value;

    #[test]
    fn cne6_value_family_registers_all_levels() {
        let specs = StockDailyBarraCne6Value.specs();
        let ids = specs
            .iter()
            .map(|spec| spec.id.as_str())
            .collect::<Vec<_>>();
        assert!(ids.contains(&"BTOP"));
        assert!(ids.contains(&"Trailing_Earnings_To_Price"));
        assert!(ids.contains(&"Long_Term_Reversal"));
        assert!(ids.contains(&"VALUE"));
    }
}

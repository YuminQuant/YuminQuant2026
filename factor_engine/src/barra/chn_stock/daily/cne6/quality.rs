use std::collections::HashMap;

use crate::barra::common::{
    align_table_column, average_columns, clean, panel_from_target_stock_map, safe_div, sample_std,
    slope_over_time, sqrt_circ_mv_weights, standardize_panel_industry_filled_weighted,
    zscore_panel_weighted_filled_zero, StatementData,
};
use crate::barra::BarraExposure;
use crate::core::{
    AssetClass, BarraSeries, BarraSpec, DataRequest, DatasetId, FactorContext, Frequency, Lookback,
};
use crate::data::{DataPool, Table};
use crate::error::Result;

pub struct StockDailyBarraCne6Quality;

const MODEL: &str = "CNE6";
const VERSION: &str = "0.3.0";
const LOOKBACK: usize = 1260;

pub fn create() -> Box<dyn BarraExposure> {
    Box::new(StockDailyBarraCne6Quality)
}

impl BarraExposure for StockDailyBarraCne6Quality {
    fn family_id(&self) -> &'static str {
        "QUALITY"
    }

    fn specs(&self) -> Vec<BarraSpec> {
        [
            "Market_Leverage",
            "Book_Leverage",
            "Debt_To_Asset",
            "Leverage",
            "Variation_In_Sales",
            "Variation_In_Earnings",
            "Variation_In_Cash_Flows",
            "Analyst_Forecast_EP_Std",
            "Earnings_Variability",
            "Accruals_Balance_Sheet",
            "Accruals_Cash_Flow",
            "Earnings_Quality",
            "Asset_Turnover",
            "Gross_Profitability",
            "Gross_Profit_Margin",
            "Return_On_Assets",
            "Profitability",
            "Total_Assets_Growth",
            "Issuance_Growth",
            "Capital_Expenditure_Growth",
            "Investment_Quality",
            "QUALITY",
        ]
        .iter()
        .map(|id| quality_spec(id))
        .collect()
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<Vec<BarraSeries>> {
        let panel = data.daily_panel(DatasetId::StockDailyPv)?;
        let basic_table = data.daily(DatasetId::StockDailyBasic)?;
        let total_mv = align_table_column(panel, basic_table, "total_mv")?;
        let close = panel.column("close")?;
        let weights = sqrt_circ_mv_weights(panel, data)?;

        let balance = StatementData::from_table(
            data.daily(DatasetId::StockBalanceSheet)?,
            &[
                "total_assets",
                "total_liab",
                "total_hldr_eqy_exc_min_int",
                "money_cap",
                "total_ncl",
                "st_borr",
                "lt_borr",
                "bond_payable",
                "non_cur_liab_due_1y",
                "total_share",
            ],
            &[1, 4],
        )?;
        let income = StatementData::from_table(
            data.daily(DatasetId::StockIncome)?,
            &["revenue", "oper_cost", "n_income_attr_p"],
            &[3, 2],
        )?;
        let income_annual = StatementData::from_table(
            data.daily(DatasetId::StockIncome)?,
            &["revenue", "oper_cost", "n_income_attr_p"],
            &[1, 4],
        )?;
        let cashflow_annual = StatementData::from_table(
            data.daily(DatasetId::StockCashFlow)?,
            &[
                "n_cashflow_act",
                "n_cashflow_inv_act",
                "n_incr_cash_cash_equ",
                "c_pay_acq_const_fiolta",
                "net_profit",
                "prov_depr_assets",
                "depr_fa_coga_dpba",
                "amort_intang_assets",
                "lt_amort_deferred_exp",
            ],
            &[1, 4],
        )?;
        let analyst_records = parse_analyst_records(data.daily(DatasetId::StockAnalystReport)?)?;
        let analyst_by_stock = index_analyst_records(&analyst_records);

        let market_leverage_raw = panel_from_target_stock_map(panel, |trade_date, ts_code| {
            let offset = panel_offset(panel, trade_date, ts_code)?;
            let me = clean(total_mv.values()[offset])?;
            let ld = balance
                .latest_annual_value(ts_code, trade_date, "total_ncl")
                .unwrap_or(0.0);
            safe_div(me + ld, me)
        })?;
        let market_leverage =
            standardize_panel_industry_filled_weighted(&market_leverage_raw, &weights, data)?;

        let book_leverage_raw = panel_from_target_stock_map(panel, |trade_date, ts_code| {
            let offset = panel_offset(panel, trade_date, ts_code)?;
            let me = clean(total_mv.values()[offset])?;
            let be =
                balance.latest_annual_value(ts_code, trade_date, "total_hldr_eqy_exc_min_int")?;
            let ld = balance
                .latest_annual_value(ts_code, trade_date, "total_ncl")
                .unwrap_or(0.0);
            safe_div(be + ld, me)
        })?;
        let book_leverage =
            standardize_panel_industry_filled_weighted(&book_leverage_raw, &weights, data)?;

        let debt_to_asset_raw = panel_from_target_stock_map(panel, |trade_date, ts_code| {
            let tl = balance.latest_annual_value(ts_code, trade_date, "total_liab")?;
            let ta = balance.latest_annual_value(ts_code, trade_date, "total_assets")?;
            safe_div(tl, ta)
        })?;
        let debt_to_asset =
            standardize_panel_industry_filled_weighted(&debt_to_asset_raw, &weights, data)?;
        let leverage_raw =
            average_columns(panel, &[&market_leverage, &book_leverage, &debt_to_asset])?;
        let leverage = zscore_panel_weighted_filled_zero(&leverage_raw, &weights)?;

        let variation_sales_raw = annual_cv(panel, &income_annual, "revenue")?;
        let variation_sales =
            standardize_panel_industry_filled_weighted(&variation_sales_raw, &weights, data)?;
        let variation_earnings_raw = annual_cv(panel, &income_annual, "n_income_attr_p")?;
        let variation_earnings =
            standardize_panel_industry_filled_weighted(&variation_earnings_raw, &weights, data)?;
        let variation_cash_flows_raw = annual_cv(panel, &cashflow_annual, "n_incr_cash_cash_equ")?;
        let variation_cash_flows =
            standardize_panel_industry_filled_weighted(&variation_cash_flows_raw, &weights, data)?;
        let analyst_forecast_std_raw = analyst_eps_std_column(panel, &close, &analyst_by_stock)?;
        let analyst_forecast_std =
            standardize_panel_industry_filled_weighted(&analyst_forecast_std_raw, &weights, data)?;
        let earnings_variability = average_columns(
            panel,
            &[
                &variation_sales,
                &variation_earnings,
                &variation_cash_flows,
                &analyst_forecast_std,
            ],
        )?;
        let earnings_variability =
            zscore_panel_weighted_filled_zero(&earnings_variability, &weights)?;

        let accruals_bs_raw = panel_from_target_stock_map(panel, |trade_date, ts_code| {
            let end_date = balance.latest_annual_end_date(ts_code, trade_date)?;
            let prev_end = (end_date / 10_000 - 1) * 10_000 + 12_31;
            let current_noa = noa(&balance, ts_code, trade_date, end_date)?;
            let prev_noa = noa(&balance, ts_code, trade_date, prev_end)?;
            let da = annual_da(&cashflow_annual, ts_code, trade_date, end_date).unwrap_or(0.0);
            let ta =
                balance.annual_value_for_end_date(ts_code, trade_date, end_date, "total_assets")?;
            safe_div(-(current_noa - prev_noa - da), ta)
        })?;
        let accruals_bs =
            standardize_panel_industry_filled_weighted(&accruals_bs_raw, &weights, data)?;
        let accruals_cf_raw = panel_from_target_stock_map(panel, |trade_date, ts_code| {
            let end_date = balance.latest_annual_end_date(ts_code, trade_date)?;
            let ni = cashflow_annual
                .annual_value_for_end_date(ts_code, trade_date, end_date, "net_profit")
                .or_else(|| {
                    income_annual.annual_value_for_end_date(
                        ts_code,
                        trade_date,
                        end_date,
                        "n_income_attr_p",
                    )
                })?;
            let cfo = cashflow_annual.annual_value_for_end_date(
                ts_code,
                trade_date,
                end_date,
                "n_cashflow_act",
            )?;
            let cfi = cashflow_annual.annual_value_for_end_date(
                ts_code,
                trade_date,
                end_date,
                "n_cashflow_inv_act",
            )?;
            let da = annual_da(&cashflow_annual, ts_code, trade_date, end_date).unwrap_or(0.0);
            let ta =
                balance.annual_value_for_end_date(ts_code, trade_date, end_date, "total_assets")?;
            safe_div(-(ni - (cfo + cfi) + da), ta)
        })?;
        let accruals_cf =
            standardize_panel_industry_filled_weighted(&accruals_cf_raw, &weights, data)?;
        let earnings_quality_raw = average_columns(panel, &[&accruals_bs, &accruals_cf])?;
        let earnings_quality = zscore_panel_weighted_filled_zero(&earnings_quality_raw, &weights)?;

        let asset_turnover_raw = panel_from_target_stock_map(panel, |trade_date, ts_code| {
            let sales = income.ttm_sum(ts_code, trade_date, "revenue")?;
            let ta = balance.latest_annual_value(ts_code, trade_date, "total_assets")?;
            safe_div(sales, ta)
        })?;
        let asset_turnover =
            standardize_panel_industry_filled_weighted(&asset_turnover_raw, &weights, data)?;
        let gross_profitability_raw = panel_from_target_stock_map(panel, |trade_date, ts_code| {
            let sales = income_annual.latest_annual_value(ts_code, trade_date, "revenue")?;
            let cogs = income_annual
                .latest_annual_value(ts_code, trade_date, "oper_cost")
                .unwrap_or(0.0);
            let ta = balance.latest_annual_value(ts_code, trade_date, "total_assets")?;
            safe_div(sales - cogs, ta)
        })?;
        let gross_profitability =
            standardize_panel_industry_filled_weighted(&gross_profitability_raw, &weights, data)?;
        let gross_profit_margin_raw = panel_from_target_stock_map(panel, |trade_date, ts_code| {
            let sales = income_annual.latest_annual_value(ts_code, trade_date, "revenue")?;
            let cogs = income_annual
                .latest_annual_value(ts_code, trade_date, "oper_cost")
                .unwrap_or(0.0);
            safe_div(sales - cogs, sales)
        })?;
        let gross_profit_margin =
            standardize_panel_industry_filled_weighted(&gross_profit_margin_raw, &weights, data)?;
        let return_on_assets_raw = panel_from_target_stock_map(panel, |trade_date, ts_code| {
            let earnings = income.ttm_sum(ts_code, trade_date, "n_income_attr_p")?;
            let ta = balance.latest_annual_value(ts_code, trade_date, "total_assets")?;
            safe_div(earnings, ta)
        })?;
        let return_on_assets =
            standardize_panel_industry_filled_weighted(&return_on_assets_raw, &weights, data)?;
        let profitability = average_columns(
            panel,
            &[
                &asset_turnover,
                &gross_profitability,
                &gross_profit_margin,
                &return_on_assets,
            ],
        )?;
        let profitability = zscore_panel_weighted_filled_zero(&profitability, &weights)?;

        let total_assets_growth_raw = annual_slope_ratio(panel, &balance, "total_assets", true)?;
        let total_assets_growth =
            standardize_panel_industry_filled_weighted(&total_assets_growth_raw, &weights, data)?;
        let issuance_growth_raw = annual_slope_ratio(panel, &balance, "total_share", true)?;
        let issuance_growth =
            standardize_panel_industry_filled_weighted(&issuance_growth_raw, &weights, data)?;
        let capex_growth_raw =
            annual_slope_ratio(panel, &cashflow_annual, "c_pay_acq_const_fiolta", true)?;
        let capex_growth =
            standardize_panel_industry_filled_weighted(&capex_growth_raw, &weights, data)?;
        let investment_quality = average_columns(
            panel,
            &[&total_assets_growth, &issuance_growth, &capex_growth],
        )?;
        let investment_quality = zscore_panel_weighted_filled_zero(&investment_quality, &weights)?;

        let quality = average_columns(
            panel,
            &[
                &leverage,
                &earnings_variability,
                &earnings_quality,
                &profitability,
                &investment_quality,
            ],
        )?;
        let quality = zscore_panel_weighted_filled_zero(&quality, &weights)?;

        let specs = self.specs();
        let columns = vec![
            market_leverage,
            book_leverage,
            debt_to_asset,
            leverage,
            variation_sales,
            variation_earnings,
            variation_cash_flows,
            analyst_forecast_std,
            earnings_variability,
            accruals_bs,
            accruals_cf,
            earnings_quality,
            asset_turnover,
            gross_profitability,
            gross_profit_margin,
            return_on_assets,
            profitability,
            total_assets_growth,
            issuance_growth,
            capex_growth,
            investment_quality,
            quality,
        ];
        Ok(columns
            .into_iter()
            .zip(specs)
            .map(|(column, spec)| column.to_barra_series(spec))
            .collect())
    }
}

fn quality_spec(id: &str) -> BarraSpec {
    BarraSpec {
        id: id.to_string(),
        aliases: Vec::new(),
        name: format!("CNE6 {id}"),
        model: MODEL.to_string(),
        asset_class: AssetClass::Stock,
        frequency: Frequency::Daily,
        version: VERSION.to_string(),
        tags: ["barra", "cne6", "style", "quality", "daily", "stock"]
            .iter()
            .map(|value| value.to_string())
            .collect(),
        description: format!("CNE6 QUALITY exposure component {id}."),
        dependencies: vec![
            DataRequest::new(DatasetId::StockDailyPv, &["close"]),
            DataRequest::new(DatasetId::StockDailyBasic, &["total_mv", "circ_mv"]),
            DataRequest::new(DatasetId::StockSwClassification, &["l1_code"]),
            DataRequest::financial_quarters(
                DatasetId::StockIncome,
                &["revenue", "oper_cost", "n_income_attr_p"],
                24,
            ),
            DataRequest::financial_quarters(
                DatasetId::StockBalanceSheet,
                &[
                    "total_assets",
                    "total_liab",
                    "total_hldr_eqy_exc_min_int",
                    "money_cap",
                    "total_ncl",
                    "st_borr",
                    "lt_borr",
                    "bond_payable",
                    "non_cur_liab_due_1y",
                    "total_share",
                ],
                24,
            ),
            DataRequest::financial_quarters(
                DatasetId::StockCashFlow,
                &[
                    "n_cashflow_act",
                    "n_cashflow_inv_act",
                    "n_incr_cash_cash_equ",
                    "c_pay_acq_const_fiolta",
                    "net_profit",
                    "prov_depr_assets",
                    "depr_fa_coga_dpba",
                    "amort_intang_assets",
                    "lt_amort_deferred_exp",
                ],
                24,
            ),
            DataRequest::new(
                DatasetId::StockAnalystReport,
                &["ts_code", "report_date", "quarter", "eps"],
            ),
        ],
        lookback: Lookback {
            trading_days: LOOKBACK,
        },
    }
}

fn annual_cv(
    panel: &crate::factor::common::DailyPanel,
    data: &StatementData,
    column: &str,
) -> Result<crate::factor::common::PanelColumn> {
    panel_from_target_stock_map(panel, |trade_date, ts_code| {
        let values = data.annual_values(ts_code, trade_date, column, 5)?;
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        sample_std(&values).and_then(|std| safe_div(std, mean.abs()))
    })
}

fn annual_slope_ratio(
    panel: &crate::factor::common::DailyPanel,
    data: &StatementData,
    column: &str,
    negate: bool,
) -> Result<crate::factor::common::PanelColumn> {
    panel_from_target_stock_map(panel, |trade_date, ts_code| {
        let values = data.annual_values(ts_code, trade_date, column, 5)?;
        let mean = values.iter().map(|value| value.abs()).sum::<f64>() / values.len() as f64;
        let slope = slope_over_time(&values)?;
        safe_div(slope, mean).map(|value| if negate { -value } else { value })
    })
}

fn total_debt(data: &StatementData, ts_code: &str, trade_date: i32, end_date: i32) -> f64 {
    ["st_borr", "lt_borr", "bond_payable", "non_cur_liab_due_1y"]
        .iter()
        .filter_map(|column| data.annual_value_for_end_date(ts_code, trade_date, end_date, column))
        .sum()
}

fn noa(data: &StatementData, ts_code: &str, trade_date: i32, end_date: i32) -> Option<f64> {
    let ta = data.annual_value_for_end_date(ts_code, trade_date, end_date, "total_assets")?;
    let cash = data
        .annual_value_for_end_date(ts_code, trade_date, end_date, "money_cap")
        .unwrap_or(0.0);
    let tl = data.annual_value_for_end_date(ts_code, trade_date, end_date, "total_liab")?;
    let td = total_debt(data, ts_code, trade_date, end_date);
    Some((ta - cash) - (tl - td))
}

fn annual_da(data: &StatementData, ts_code: &str, trade_date: i32, end_date: i32) -> Option<f64> {
    let mut sum = 0.0;
    let mut any = false;
    for column in [
        "prov_depr_assets",
        "depr_fa_coga_dpba",
        "amort_intang_assets",
        "lt_amort_deferred_exp",
    ] {
        if let Some(value) = data.annual_value_for_end_date(ts_code, trade_date, end_date, column) {
            sum += value;
            any = true;
        }
    }
    any.then_some(sum)
}

#[derive(Clone, Debug)]
struct AnalystRecord {
    ts_code: String,
    report_date: i32,
    quarter: String,
    eps: f64,
}

fn parse_analyst_records(table: &Table) -> Result<Vec<AnalystRecord>> {
    let ts_codes = table.required_utf8("ts_code")?;
    let report_dates = table.required_i32_date_cast("report_date")?;
    let quarters = table.required_utf8("quarter")?;
    let eps_values = table.required_f64_cast("eps")?;
    let mut records = Vec::new();
    for idx in 0..table.len {
        let (Some(ts_code), Some(report_date), Some(quarter), Some(eps)) = (
            ts_codes[idx].clone(),
            report_dates[idx],
            quarters[idx].clone(),
            clean(eps_values[idx]),
        ) else {
            continue;
        };
        records.push(AnalystRecord {
            ts_code,
            report_date,
            quarter,
            eps,
        });
    }
    Ok(records)
}

fn analyst_eps_std_column(
    panel: &crate::factor::common::DailyPanel,
    close: &crate::factor::common::PanelColumn,
    records_by_stock: &HashMap<&str, Vec<&AnalystRecord>>,
) -> Result<crate::factor::common::PanelColumn> {
    panel_from_target_stock_map(panel, |trade_date, ts_code| {
        let fy1 = crate::barra::common::fy1_quarter(trade_date);
        let start_date = crate::barra::common::add_months(trade_date, -3);
        let records = records_by_stock.get(ts_code)?;
        let mut values = Vec::new();
        for record in records {
            if record.quarter == fy1
                && record.report_date >= start_date
                && record.report_date <= trade_date
            {
                values.push(record.eps);
            }
        }
        let std = sample_std(&values)?;
        let close_value = close.values()[panel_offset(panel, trade_date, ts_code)?]?;
        clean(Some(close_value)).and_then(|value| safe_div(std, value))
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

fn panel_offset(
    panel: &crate::factor::common::DailyPanel,
    trade_date: i32,
    ts_code: &str,
) -> Option<usize> {
    let date_idx = panel.dates().iter().position(|date| *date == trade_date)?;
    let code_idx = panel
        .instruments()
        .iter()
        .position(|code| code == ts_code)?;
    Some(date_idx * panel.instruments().len() + code_idx)
}

#[cfg(test)]
mod tests {
    use crate::barra::BarraExposure;

    use super::StockDailyBarraCne6Quality;

    #[test]
    fn cne6_quality_family_registers_composite() {
        let specs = StockDailyBarraCne6Quality.specs();
        let ids = specs
            .iter()
            .map(|spec| spec.id.as_str())
            .collect::<Vec<_>>();
        assert!(ids.contains(&"Leverage"));
        assert!(ids.contains(&"Earnings_Quality"));
        assert!(ids.contains(&"QUALITY"));
    }
}

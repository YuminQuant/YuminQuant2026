use std::collections::HashMap;

use crate::barra::common::{
    add_months, average_columns, clean, fy_quarter, panel_from_target_stock_map, safe_div,
    slope_over_time, sqrt_circ_mv_weights, standardize_panel_industry_filled_weighted,
    zscore_panel_weighted_filled_zero,
};
use crate::barra::BarraExposure;
use crate::core::{
    AssetClass, BarraSeries, BarraSpec, DataRequest, DatasetId, FactorContext, Frequency, Lookback,
};
use crate::data::{DataPool, Table};
use crate::error::Result;
use crate::factor::common::{
    cached_financial_stock_snapshots, FinancialEventMarker, FinancialEventMarkerBuilder,
    FinancialStatementDataset, PanelColumn, PitFinancialData, ReportTypePreference,
};

pub struct StockDailyBarraCne6Growth;

const MODEL: &str = "CNE6";
const VERSION: &str = "0.3.0";
const LOOKBACK: usize = 1260;

pub fn create() -> Box<dyn BarraExposure> {
    Box::new(StockDailyBarraCne6Growth)
}

impl BarraExposure for StockDailyBarraCne6Growth {
    fn family_id(&self) -> &'static str {
        "GROWTH"
    }

    fn specs(&self) -> Vec<BarraSpec> {
        [
            "Predicted_Growth_3Y",
            "Historical_EPS_Growth",
            "Historical_Sales_Per_Share_Growth",
            "GROWTH",
        ]
        .iter()
        .map(|id| growth_spec(id))
        .collect()
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<Vec<BarraSeries>> {
        let panel = data.daily_panel(DatasetId::StockDailyPv)?;
        let income = PitFinancialData::from_table(
            data.daily(DatasetId::StockIncome)?,
            &["basic_eps", "revenue"],
            ReportTypePreference::consolidated(),
        )?;
        let balance = PitFinancialData::from_table(
            data.daily(DatasetId::StockBalanceSheet)?,
            &["total_share"],
            ReportTypePreference::balance_sheet_consolidated(),
        )?;
        let analyst_records = parse_analyst_records(data.daily(DatasetId::StockAnalystReport)?)?;
        let analyst_by_stock = index_analyst_records(&analyst_records);
        let weights = sqrt_circ_mv_weights(panel, data)?;

        let predicted_raw = predicted_growth_column(panel, &analyst_by_stock)?;
        let predicted = standardize_panel_industry_filled_weighted(&predicted_raw, &weights, data)?;
        let (eps_growth_raw, sales_growth_raw) =
            historical_growth_columns(panel, &income, &balance)?;
        let eps_growth =
            standardize_panel_industry_filled_weighted(&eps_growth_raw, &weights, data)?;
        let sales_growth =
            standardize_panel_industry_filled_weighted(&sales_growth_raw, &weights, data)?;
        let growth_raw = average_columns(panel, &[&predicted, &eps_growth, &sales_growth])?;
        let growth = zscore_panel_weighted_filled_zero(&growth_raw, &weights)?;

        let specs = self.specs();
        Ok(vec![
            predicted.to_barra_series(specs[0].clone()),
            eps_growth.to_barra_series(specs[1].clone()),
            sales_growth.to_barra_series(specs[2].clone()),
            growth.to_barra_series(specs[3].clone()),
        ])
    }
}

#[derive(Clone, Copy, Debug)]
struct GrowthSlowSnapshot {
    eps_growth: Option<f64>,
    sales_growth: Option<f64>,
}

fn historical_growth_columns(
    panel: &crate::factor::common::DailyPanel,
    income: &PitFinancialData,
    balance: &PitFinancialData,
) -> Result<(PanelColumn, PanelColumn)> {
    let mut eps_values = vec![None; panel.shape_len()];
    let mut sales_values = vec![None; panel.shape_len()];

    let snapshots = cached_financial_stock_snapshots(
        panel,
        |_, _, offset| !panel.is_present_offset(offset),
        |trade_date, ts_code, _| growth_marker(ts_code, trade_date, income, balance),
        |trade_date, ts_code, _| growth_snapshot(ts_code, trade_date, income, balance),
    );

    for (offset, snapshot) in snapshots.into_iter().enumerate() {
        let Some(snapshot) = snapshot else {
            continue;
        };
        eps_values[offset] = snapshot.eps_growth;
        sales_values[offset] = snapshot.sales_growth;
    }

    Ok((
        panel.column_from_values(eps_values)?,
        panel.column_from_values(sales_values)?,
    ))
}

fn growth_marker(
    ts_code: &str,
    trade_date: i32,
    income: &PitFinancialData,
    balance: &PitFinancialData,
) -> Option<FinancialEventMarker> {
    let mut builder = FinancialEventMarkerBuilder::new();
    builder.include_annual_chain(
        FinancialStatementDataset::Income,
        income,
        ts_code,
        trade_date,
        5,
    );
    builder.include_annual_chain(
        FinancialStatementDataset::BalanceSheet,
        balance,
        ts_code,
        trade_date,
        5,
    );
    builder.build()
}

fn growth_snapshot(
    ts_code: &str,
    trade_date: i32,
    income: &PitFinancialData,
    balance: &PitFinancialData,
) -> Option<GrowthSlowSnapshot> {
    let eps_growth = income
        .annual_values(ts_code, trade_date, "basic_eps", 5)
        .and_then(|values| {
            let mean = values.iter().map(|value| value.abs()).sum::<f64>() / values.len() as f64;
            slope_over_time(&values).and_then(|slope| safe_div(slope, mean))
        });
    let sales_growth = income
        .annual_values(ts_code, trade_date, "revenue", 5)
        .zip(balance.annual_values(ts_code, trade_date, "total_share", 5))
        .and_then(|(revenue, shares)| {
            let values = revenue
                .iter()
                .zip(shares.iter())
                .map(|(revenue, shares)| safe_div(*revenue, *shares))
                .collect::<Option<Vec<_>>>()?;
            let mean = values.iter().map(|value| value.abs()).sum::<f64>() / values.len() as f64;
            slope_over_time(&values).and_then(|slope| safe_div(slope, mean))
        });
    Some(GrowthSlowSnapshot {
        eps_growth,
        sales_growth,
    })
}

fn growth_spec(id: &str) -> BarraSpec {
    BarraSpec {
        id: id.to_string(),
        aliases: Vec::new(),
        name: format!("CNE6 {id}"),
        model: MODEL.to_string(),
        asset_class: AssetClass::Stock,
        frequency: Frequency::Daily,
        version: VERSION.to_string(),
        tags: ["barra", "cne6", "style", "growth", "daily", "stock"]
            .iter()
            .map(|value| value.to_string())
            .collect(),
        description: format!("CNE6 GROWTH exposure component {id}."),
        dependencies: vec![
            DataRequest::new(DatasetId::StockDailyPv, &["close"]),
            DataRequest::new(DatasetId::StockDailyBasic, &["circ_mv"]),
            DataRequest::new(DatasetId::StockSwClassification, &["l1_code"]),
            DataRequest::financial_quarters(DatasetId::StockIncome, &["basic_eps", "revenue"], 24),
            DataRequest::financial_quarters(DatasetId::StockBalanceSheet, &["total_share"], 24),
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

fn predicted_growth_column(
    panel: &crate::factor::common::DailyPanel,
    records_by_stock: &HashMap<&str, Vec<&AnalystRecord>>,
) -> Result<crate::factor::common::PanelColumn> {
    panel_from_target_stock_map(panel, |trade_date, ts_code| {
        let start_date = add_months(trade_date, -3);
        let fy1 = fy_quarter(trade_date, 0);
        let fy3 = fy_quarter(trade_date, 2);
        let eps1 = analyst_mean_eps(records_by_stock, ts_code, start_date, trade_date, &fy1)?;
        let eps3 = analyst_mean_eps(records_by_stock, ts_code, start_date, trade_date, &fy3)?;
        if eps1 <= 0.0 || eps3 <= 0.0 {
            return None;
        }
        Some((eps3 / eps1).powf(0.5) - 1.0)
    })
}

fn analyst_mean_eps(
    records_by_stock: &HashMap<&str, Vec<&AnalystRecord>>,
    ts_code: &str,
    start_date: i32,
    trade_date: i32,
    quarter: &str,
) -> Option<f64> {
    let records = records_by_stock.get(ts_code)?;
    let mut sum = 0.0;
    let mut count = 0usize;
    for record in records {
        if record.ts_code == ts_code
            && record.quarter == quarter
            && record.report_date >= start_date
            && record.report_date <= trade_date
        {
            sum += record.eps;
            count += 1;
        }
    }
    (count > 0).then_some(sum / count as f64)
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

#[cfg(test)]
mod tests {
    use crate::barra::BarraExposure;

    use super::StockDailyBarraCne6Growth;

    #[test]
    fn cne6_growth_family_registers_composite() {
        let specs = StockDailyBarraCne6Growth.specs();
        let ids = specs
            .iter()
            .map(|spec| spec.id.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            ids,
            vec![
                "Predicted_Growth_3Y",
                "Historical_EPS_Growth",
                "Historical_Sales_Per_Share_Growth",
                "GROWTH"
            ]
        );
    }
}

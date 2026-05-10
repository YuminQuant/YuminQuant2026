use std::collections::HashMap;

use crate::barra::common::{
    add_months, average_columns, clean, fy_quarter, panel_from_target_stock_map, safe_div,
    sqrt_circ_mv_weights, standardize_panel_industry_filled_weighted,
    zscore_panel_weighted_filled_zero,
};
use crate::barra::BarraExposure;
use crate::core::{
    AssetClass, BarraSeries, BarraSpec, DataRequest, DatasetId, FactorContext, Frequency, Lookback,
};
use crate::data::{DataPool, Table};
use crate::error::Result;

pub struct StockDailyBarraCne6Sentiment;

const MODEL: &str = "CNE6";
const VERSION: &str = "0.3.0";
const LOOKBACK: usize = 252;

pub fn create() -> Box<dyn BarraExposure> {
    Box::new(StockDailyBarraCne6Sentiment)
}

impl BarraExposure for StockDailyBarraCne6Sentiment {
    fn family_id(&self) -> &'static str {
        "SENTIMENT"
    }

    fn specs(&self) -> Vec<BarraSpec> {
        [
            "Revision_Ratio",
            "Change_In_Analyst_Predicted_EP",
            "Change_In_Analyst_Predicted_EPS",
            "SENTIMENT",
        ]
        .iter()
        .map(|id| sentiment_spec(id))
        .collect()
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<Vec<BarraSeries>> {
        let panel = data.daily_panel(DatasetId::StockDailyPv)?;
        let records = parse_analyst_records(data.daily(DatasetId::StockAnalystReport)?)?;
        let records_by_stock = index_analyst_records(&records);
        let weights = sqrt_circ_mv_weights(panel, data)?;

        let revision_raw = panel_from_target_stock_map(panel, |trade_date, ts_code| {
            revision_ratio(&records_by_stock, ts_code, trade_date)
        })?;
        let revision = standardize_panel_industry_filled_weighted(&revision_raw, &weights, data)?;
        let ep_change_raw = panel_from_target_stock_map(panel, |trade_date, ts_code| {
            weighted_forecast_change(&records_by_stock, ts_code, trade_date, ForecastField::Ep)
        })?;
        let ep_change = standardize_panel_industry_filled_weighted(&ep_change_raw, &weights, data)?;
        let eps_change_raw = panel_from_target_stock_map(panel, |trade_date, ts_code| {
            weighted_forecast_change(&records_by_stock, ts_code, trade_date, ForecastField::Eps)
        })?;
        let eps_change =
            standardize_panel_industry_filled_weighted(&eps_change_raw, &weights, data)?;
        let sentiment_raw = average_columns(panel, &[&revision, &ep_change, &eps_change])?;
        let sentiment = zscore_panel_weighted_filled_zero(&sentiment_raw, &weights)?;

        let specs = self.specs();
        Ok(vec![
            revision.to_barra_series(specs[0].clone()),
            ep_change.to_barra_series(specs[1].clone()),
            eps_change.to_barra_series(specs[2].clone()),
            sentiment.to_barra_series(specs[3].clone()),
        ])
    }
}

fn sentiment_spec(id: &str) -> BarraSpec {
    BarraSpec {
        id: id.to_string(),
        aliases: Vec::new(),
        name: format!("CNE6 {id}"),
        model: MODEL.to_string(),
        asset_class: AssetClass::Stock,
        frequency: Frequency::Daily,
        version: VERSION.to_string(),
        tags: ["barra", "cne6", "style", "sentiment", "daily", "stock"]
            .iter()
            .map(|value| value.to_string())
            .collect(),
        description: format!("CNE6 SENTIMENT exposure component {id}."),
        dependencies: vec![
            DataRequest::new(DatasetId::StockDailyPv, &["close"]),
            DataRequest::new(DatasetId::StockDailyBasic, &["circ_mv"]),
            DataRequest::new(DatasetId::StockSwClassification, &["l1_code"]),
            DataRequest::new(
                DatasetId::StockAnalystReport,
                &["ts_code", "report_date", "quarter", "eps", "pe"],
            ),
        ],
        lookback: Lookback {
            trading_days: LOOKBACK,
        },
    }
}

#[derive(Clone, Copy, Debug)]
enum ForecastField {
    Eps,
    Ep,
}

#[derive(Clone, Debug)]
struct AnalystRecord {
    ts_code: String,
    report_date: i32,
    quarter: String,
    eps: Option<f64>,
    pe: Option<f64>,
}

fn parse_analyst_records(table: &Table) -> Result<Vec<AnalystRecord>> {
    let ts_codes = table.required_utf8("ts_code")?;
    let report_dates = table.required_i32_date_cast("report_date")?;
    let quarters = table.required_utf8("quarter")?;
    let eps_values = table.required_f64_cast("eps")?;
    let pe_values = table.required_f64_cast("pe")?;
    let mut records = Vec::new();
    for idx in 0..table.len {
        let (Some(ts_code), Some(report_date), Some(quarter)) = (
            ts_codes[idx].clone(),
            report_dates[idx],
            quarters[idx].clone(),
        ) else {
            continue;
        };
        records.push(AnalystRecord {
            ts_code,
            report_date,
            quarter,
            eps: clean(eps_values[idx]),
            pe: clean(pe_values[idx]),
        });
    }
    Ok(records)
}

fn revision_ratio(
    records_by_stock: &HashMap<&str, Vec<&AnalystRecord>>,
    ts_code: &str,
    trade_date: i32,
) -> Option<f64> {
    let mut score = 0.0;
    let mut count = 0usize;
    for lag in 0..=2 {
        let current_date = add_months(trade_date, -lag);
        let previous_date = add_months(current_date, -1);
        let quarter = fy_quarter(current_date, 0);
        let current = mean_value(
            records_by_stock,
            ts_code,
            add_months(current_date, -1),
            current_date,
            &quarter,
            ForecastField::Eps,
        )?;
        let previous = mean_value(
            records_by_stock,
            ts_code,
            add_months(previous_date, -1),
            previous_date,
            &quarter,
            ForecastField::Eps,
        )?;
        let diff = current - previous;
        if diff.abs() <= f64::EPSILON {
            continue;
        }
        score += diff.signum();
        count += 1;
    }
    (count > 0).then_some(score / count as f64)
}

fn weighted_forecast_change(
    records_by_stock: &HashMap<&str, Vec<&AnalystRecord>>,
    ts_code: &str,
    trade_date: i32,
    field: ForecastField,
) -> Option<f64> {
    let mut sum = 0.0;
    let mut count = 0usize;
    for lag in 0..=3 {
        let current_date = add_months(trade_date, -3 * lag);
        let previous_date = add_months(trade_date, -3 * (lag + 1));
        let quarter = fy_quarter(current_date, 0);
        let current = mean_value(
            records_by_stock,
            ts_code,
            add_months(current_date, -3),
            current_date,
            &quarter,
            field,
        )?;
        let previous = mean_value(
            records_by_stock,
            ts_code,
            add_months(previous_date, -3),
            previous_date,
            &quarter,
            field,
        )?;
        if let Some(change) = safe_div(current - previous, previous) {
            sum += change;
            count += 1;
        }
    }
    (count > 0).then_some(sum / count as f64)
}

fn mean_value(
    records_by_stock: &HashMap<&str, Vec<&AnalystRecord>>,
    ts_code: &str,
    start_date: i32,
    end_date: i32,
    quarter: &str,
    field: ForecastField,
) -> Option<f64> {
    let records = records_by_stock.get(ts_code)?;
    let mut sum = 0.0;
    let mut count = 0usize;
    for record in records {
        if record.ts_code != ts_code
            || record.quarter != quarter
            || record.report_date < start_date
            || record.report_date > end_date
        {
            continue;
        }
        let value = match field {
            ForecastField::Eps => record.eps,
            ForecastField::Ep => record
                .pe
                .and_then(|pe| (pe.abs() > f64::EPSILON).then_some(1.0 / pe)),
        };
        if let Some(value) = value {
            sum += value;
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

    use super::StockDailyBarraCne6Sentiment;

    #[test]
    fn cne6_sentiment_family_registers_composite() {
        let specs = StockDailyBarraCne6Sentiment.specs();
        let ids = specs
            .iter()
            .map(|spec| spec.id.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            ids,
            vec![
                "Revision_Ratio",
                "Change_In_Analyst_Predicted_EP",
                "Change_In_Analyst_Predicted_EPS",
                "SENTIMENT"
            ]
        );
    }
}

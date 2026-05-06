use std::collections::HashMap;

use crate::core::{DataRequest, DatasetId};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::vector::clean;
use crate::factor::common::{ClassificationLevel, ClassificationMap, DailyPanel, PanelColumn};
use crate::operators::{ts_delay, ts_ew_sum, ts_mean};

pub const MARKET_INDEX: &str = "000985.CSI";
pub const RISK_WINDOW: usize = 10;
pub const UMR_WINDOW: usize = 60;
pub const UMR_MIN_PERIODS: usize = 1;
pub const UMR_HALF_LIFE: f64 = 30.0;
pub const UMR_LOOKBACK: usize = 69;

pub fn market_close_return_request() -> DataRequest {
    DataRequest::index_daily(MARKET_INDEX, &["close", "pre_close"])
}

pub fn market_intraday_return_request() -> DataRequest {
    DataRequest::index_daily(MARKET_INDEX, &["open", "close"])
}

pub fn finite(value: Option<f64>) -> Option<f64> {
    clean(value).filter(|value| value.is_finite())
}

pub fn ret(numerator: Option<f64>, denominator: Option<f64>) -> Option<f64> {
    match (finite(numerator), finite(denominator)) {
        (Some(numerator), Some(denominator)) if denominator.abs() > f64::EPSILON => {
            Some(numerator / denominator - 1.0)
        }
        _ => None,
    }
}

pub fn ratio(numerator: Option<f64>, denominator: Option<f64>) -> Option<f64> {
    match (finite(numerator), finite(denominator)) {
        (Some(numerator), Some(denominator)) if denominator.abs() > f64::EPSILON => {
            Some(numerator / denominator)
        }
        _ => None,
    }
}

pub fn multiply(left: Option<f64>, right: Option<f64>) -> Option<f64> {
    match (finite(left), finite(right)) {
        (Some(left), Some(right)) => Some(left * right),
        _ => None,
    }
}

pub fn subtract(left: Option<f64>, right: Option<f64>) -> Option<f64> {
    match (finite(left), finite(right)) {
        (Some(left), Some(right)) => Some(left - right),
        _ => None,
    }
}

pub fn abs_dev(left: Option<f64>, right: Option<f64>) -> Option<f64> {
    match (finite(left), finite(right)) {
        (Some(left), Some(right)) => Some((left - right).abs()),
        _ => None,
    }
}

pub fn percent_to_decimal(value: Option<f64>) -> Option<f64> {
    finite(value).map(|value| value / 100.0)
}

pub fn risk_coefficient(risk: &PanelColumn, higher_is_riskier: bool) -> Result<PanelColumn> {
    let prior_mean = risk.ts(|values| ts_delay(&ts_mean(values, RISK_WINDOW, 1), 1))?;
    if higher_is_riskier {
        prior_mean.zip_binary(risk, subtract)
    } else {
        risk.zip_binary(&prior_mean, subtract)
    }
}

pub fn umr_raw(
    risk: &PanelColumn,
    weighted_variable: &PanelColumn,
    higher_is_riskier: bool,
) -> Result<PanelColumn> {
    let coef = risk_coefficient(risk, higher_is_riskier)?;
    let weighted = coef.zip_binary(weighted_variable, multiply)?;
    weighted.ts(|values| ts_ew_sum(values, UMR_WINDOW, UMR_MIN_PERIODS, UMR_HALF_LIFE))
}

pub fn neutralize_size_sector(
    raw: &PanelColumn,
    panel: &DailyPanel,
    data: &DataPool,
) -> Result<PanelColumn> {
    let size = panel.column_from_table(data.daily(DatasetId::StockBarraDaily)?, "SIZE")?;
    let sector_map = ClassificationMap::from_table(
        data.daily(DatasetId::StockSwClassification)?,
        ClassificationLevel::Sector,
    )?;
    raw.cs_neutralize_regression_by_group(&[&size], None, |trade_date, ts_codes| {
        sector_map.groups_for(trade_date, ts_codes)
    })
}

pub fn stock_return(panel: &DailyPanel) -> Result<PanelColumn> {
    panel
        .column("close")?
        .zip_binary(&panel.column("pre_close")?, ret)
}

pub fn stock_intraday_return(panel: &DailyPanel) -> Result<PanelColumn> {
    panel
        .column("close")?
        .zip_binary(&panel.column("open")?, ret)
}

pub fn excess_return(panel: &DailyPanel, data: &DataPool) -> Result<PanelColumn> {
    let stock_ret = stock_return(panel)?;
    let market_ret = expanded_market_return(panel, data, false)?;
    stock_ret.zip_binary(&market_ret, subtract)
}

pub fn intraday_excess_return(panel: &DailyPanel, data: &DataPool) -> Result<PanelColumn> {
    let stock_ret = stock_intraday_return(panel)?;
    let market_ret = expanded_market_return(panel, data, true)?;
    stock_ret.zip_binary(&market_ret, subtract)
}

pub fn expanded_market_return(
    stock_panel: &DailyPanel,
    data: &DataPool,
    intraday: bool,
) -> Result<PanelColumn> {
    let index_panel = data.index_daily_panel(MARKET_INDEX)?;
    let index_return = if intraday {
        index_panel
            .column("close")?
            .zip_binary(&index_panel.column("open")?, ret)?
    } else {
        index_panel
            .column("close")?
            .zip_binary(&index_panel.column("pre_close")?, ret)?
    };
    expand_index_column(stock_panel, &index_panel, &index_return)
}

pub fn expand_index_column(
    stock_panel: &DailyPanel,
    index_panel: &DailyPanel,
    index_column: &PanelColumn,
) -> Result<PanelColumn> {
    let index_instrument_count = index_panel.instruments().len();
    if index_instrument_count == 0 {
        return stock_panel.column_from_values(vec![None; stock_panel.shape_len()]);
    }
    let mut by_date = HashMap::new();
    for (date_idx, trade_date) in index_panel.dates().iter().enumerate() {
        by_date.insert(
            *trade_date,
            index_column.values()[date_idx * index_instrument_count],
        );
    }

    let mut values = Vec::with_capacity(stock_panel.shape_len());
    for trade_date in stock_panel.dates() {
        let value = by_date.get(trade_date).copied().unwrap_or(None);
        for _ in stock_panel.instruments() {
            values.push(value);
        }
    }
    stock_panel.column_from_values(values)
}

#[cfg(test)]
mod tests {
    use crate::operators::{ts_delay, ts_mean};

    #[test]
    fn risk_mean_uses_prior_window_and_excludes_current_day() {
        let values = vec![Some(1.0), Some(2.0), Some(3.0)];
        let prior = ts_delay(&ts_mean(&values, 2, 1), 1);
        assert_eq!(prior, vec![None, Some(1.0), Some(1.5)]);
    }
}

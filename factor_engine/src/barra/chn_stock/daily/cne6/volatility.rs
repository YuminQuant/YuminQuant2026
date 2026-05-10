use std::collections::HashMap;

use crate::barra::common::{
    sqrt_circ_mv_weights, standardize_panel_industry_filled_weighted,
    zscore_panel_weighted_filled_zero,
};
use crate::barra::BarraExposure;
use crate::core::{
    AssetClass, BarraSeries, BarraSpec, DataRequest, DatasetId, FactorContext, Frequency, Lookback,
};
use crate::data::DataPool;
use crate::error::{err, Result};
use crate::factor::common::{DailyPanel, PanelColumn};
use crate::operators::{ts_ew_regression_beta_residual_sigma, ts_ew_std_dev};

pub struct StockDailyBarraCne6Volatility;

const MODEL: &str = "CNE6";
const VERSION: &str = "0.4.0";
const MARKET_INDEX: &str = "000300.SH";
const WINDOW: usize = 252;
const MIN_PERIODS: usize = 1;
const BETA_HALF_LIFE: f64 = 63.0;
const DASTD_HALF_LIFE: f64 = 42.0;
const TRADING_DAYS_PER_MONTH: usize = 21;
const MONTHS_PER_YEAR: usize = 12;

pub fn create() -> Box<dyn BarraExposure> {
    Box::new(StockDailyBarraCne6Volatility)
}

impl BarraExposure for StockDailyBarraCne6Volatility {
    fn family_id(&self) -> &'static str {
        "VOLATILITY"
    }

    fn specs(&self) -> Vec<BarraSpec> {
        vec![
            volatility_spec(
                "Historical_Sigma",
                &["HSIGMA"],
                "CNE6 historical sigma",
                "EW residual return volatility from the 252-day Beta regression against CSI 300 with min_periods=1.",
            ),
            volatility_spec(
                "Daily_Std",
                &["DASTD"],
                "CNE6 daily standard deviation",
                "EW standard deviation of daily stock returns over 252 trading days with half-life 42 and min_periods=1.",
            ),
            volatility_spec(
                "Cumulative_Range",
                &["CMRA"],
                "CNE6 cumulative range",
                "Max minus min of up to the past 12 monthly cumulative log returns with min_periods=1.",
            ),
            volatility_spec(
                "Beta",
                &[],
                "CNE6 Beta",
                "EW 252-day regression slope of stock returns on CSI 300 returns with half-life 63 and min_periods=1.",
            ),
            volatility_spec(
                "Residual_Volatility",
                &[],
                "CNE6 residual volatility",
                "Equal-weight composite of Historical_Sigma, Daily_Std, and Cumulative_Range.",
            ),
            volatility_spec(
                "VOLATILITY",
                &[],
                "CNE6 VOLATILITY style exposure",
                "Equal-weight composite of Beta and Residual_Volatility.",
            ),
        ]
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<Vec<BarraSeries>> {
        let stock_panel = data.daily_panel(DatasetId::StockDailyPv)?;
        let index_panel = data.index_daily_panel(MARKET_INDEX)?;
        let weights = sqrt_circ_mv_weights(stock_panel, data)?;

        let stock_returns = stock_panel.column("close")?.zip_binary(
            &stock_panel.column("pre_close")?,
            return_from_close_preclose,
        )?;
        let index_returns = index_panel.column("close")?.zip_binary(
            &index_panel.column("pre_close")?,
            return_from_close_preclose,
        )?;
        let market_returns = expand_index_column(stock_panel, index_panel, &index_returns)?;

        let (beta_raw, historical_sigma_raw) =
            beta_and_historical_sigma(stock_panel, &stock_returns, &market_returns)?;
        let beta = standardize_panel_industry_filled_weighted(&beta_raw, &weights, data)?;
        let historical_sigma =
            standardize_panel_industry_filled_weighted(&historical_sigma_raw, &weights, data)?;
        let daily_std_raw = stock_returns
            .ts(|values| ts_ew_std_dev(values, WINDOW, MIN_PERIODS, DASTD_HALF_LIFE))?;
        let daily_std = standardize_panel_industry_filled_weighted(&daily_std_raw, &weights, data)?;
        let cumulative_range_raw = stock_returns.ts(cumulative_range_12m)?;
        let cumulative_range =
            standardize_panel_industry_filled_weighted(&cumulative_range_raw, &weights, data)?;

        let residual_raw = historical_sigma.zip_ternary(
            &daily_std,
            &cumulative_range,
            |historical_sigma, daily_std, cumulative_range| match (
                clean(historical_sigma),
                clean(daily_std),
                clean(cumulative_range),
            ) {
                (Some(historical_sigma), Some(daily_std), Some(cumulative_range)) => {
                    Some((historical_sigma + daily_std + cumulative_range) / 3.0)
                }
                _ => None,
            },
        )?;
        let residual_volatility = zscore_panel_weighted_filled_zero(&residual_raw, &weights)?;

        let volatility_raw =
            beta.zip_binary(&residual_volatility, |beta, residual_volatility| {
                match (clean(beta), clean(residual_volatility)) {
                    (Some(beta), Some(residual_volatility)) => {
                        Some((beta + residual_volatility) / 2.0)
                    }
                    _ => None,
                }
            })?;
        let volatility = zscore_panel_weighted_filled_zero(&volatility_raw, &weights)?;

        let specs = self.specs();
        Ok(vec![
            historical_sigma.to_barra_series(specs[0].clone()),
            daily_std.to_barra_series(specs[1].clone()),
            cumulative_range.to_barra_series(specs[2].clone()),
            beta.to_barra_series(specs[3].clone()),
            residual_volatility.to_barra_series(specs[4].clone()),
            volatility.to_barra_series(specs[5].clone()),
        ])
    }
}

fn volatility_spec(id: &str, aliases: &[&str], name: &str, description: &str) -> BarraSpec {
    BarraSpec {
        id: id.to_string(),
        aliases: aliases.iter().map(|value| value.to_string()).collect(),
        name: name.to_string(),
        model: MODEL.to_string(),
        asset_class: AssetClass::Stock,
        frequency: Frequency::Daily,
        version: VERSION.to_string(),
        tags: ["barra", "cne6", "style", "volatility", "daily", "stock"]
            .iter()
            .map(|value| value.to_string())
            .collect(),
        description: description.to_string(),
        dependencies: vec![
            DataRequest::new(DatasetId::StockDailyPv, &["close", "pre_close"]),
            DataRequest::new(DatasetId::StockDailyBasic, &["circ_mv"]),
            DataRequest::new(DatasetId::StockSwClassification, &["l1_code"]),
            DataRequest::index_daily(MARKET_INDEX, &["close", "pre_close"]),
        ],
        lookback: Lookback {
            trading_days: WINDOW - 1,
        },
    }
}

fn return_from_close_preclose(close: Option<f64>, pre_close: Option<f64>) -> Option<f64> {
    match (clean(close), clean(pre_close)) {
        (Some(close), Some(pre_close)) if pre_close.abs() > f64::EPSILON => {
            Some(close / pre_close - 1.0)
        }
        _ => None,
    }
}

fn expand_index_column(
    stock_panel: &DailyPanel,
    index_panel: &DailyPanel,
    index_column: &PanelColumn,
) -> Result<PanelColumn> {
    let index_instrument_count = index_panel.instruments().len();
    if index_instrument_count == 0 {
        return Err(err("index daily panel has no instruments"));
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

fn beta_and_historical_sigma(
    panel: &DailyPanel,
    stock_returns: &PanelColumn,
    market_returns: &PanelColumn,
) -> Result<(PanelColumn, PanelColumn)> {
    let date_count = panel.dates().len();
    let instrument_count = panel.instruments().len();
    let mut beta_values = vec![None; panel.shape_len()];
    let mut sigma_values = vec![None; panel.shape_len()];

    for instrument_idx in 0..instrument_count {
        let mut stock_series = Vec::with_capacity(date_count);
        let mut market_series = Vec::with_capacity(date_count);
        for date_idx in 0..date_count {
            let offset = date_idx * instrument_count + instrument_idx;
            stock_series.push(stock_returns.values()[offset]);
            market_series.push(market_returns.values()[offset]);
        }
        let (beta, sigma) = ts_ew_regression_beta_residual_sigma(
            &stock_series,
            &market_series,
            WINDOW,
            MIN_PERIODS,
            BETA_HALF_LIFE,
        );
        for date_idx in 0..date_count {
            let offset = date_idx * instrument_count + instrument_idx;
            beta_values[offset] = beta[date_idx];
            sigma_values[offset] = sigma[date_idx];
        }
    }

    Ok((
        panel.column_from_values(beta_values)?,
        panel.column_from_values(sigma_values)?,
    ))
}

fn cumulative_range_12m(values: &[Option<f64>]) -> Vec<Option<f64>> {
    let mut output = vec![None; values.len()];
    let mut prefix_sum = vec![0.0; values.len() + 1];
    let mut prefix_count = vec![0usize; values.len() + 1];
    for (idx, value) in values.iter().enumerate() {
        prefix_sum[idx + 1] = prefix_sum[idx];
        prefix_count[idx + 1] = prefix_count[idx];
        if let Some(value) = clean(*value).and_then(|value| (value > -1.0).then_some(value)) {
            prefix_sum[idx + 1] += value.ln_1p();
            prefix_count[idx + 1] += 1;
        }
    }

    for idx in 0..values.len() {
        let mut month_values = Vec::with_capacity(MONTHS_PER_YEAR);
        let end = idx + 1;
        let available_months = end.div_ceil(TRADING_DAYS_PER_MONTH).min(MONTHS_PER_YEAR);
        for month in 1..=available_months {
            let window_len = (month * TRADING_DAYS_PER_MONTH).min(end);
            let start = end - window_len;
            if prefix_count[end] - prefix_count[start] >= MIN_PERIODS {
                month_values.push(prefix_sum[end] - prefix_sum[start]);
            }
        }
        if !month_values.is_empty() {
            let min = month_values.iter().copied().reduce(f64::min);
            let max = month_values.iter().copied().reduce(f64::max);
            output[idx] = match (min, max) {
                (Some(min), Some(max)) => Some(max - min),
                _ => None,
            };
        }
    }
    output
}

fn clean(value: Option<f64>) -> Option<f64> {
    value.filter(|value| !value.is_nan())
}

#[cfg(test)]
mod tests {
    use crate::barra::BarraExposure;

    use super::{cumulative_range_12m, StockDailyBarraCne6Volatility, WINDOW};

    #[test]
    fn cne6_volatility_family_registers_all_levels() {
        let exposure = StockDailyBarraCne6Volatility;
        let ids = exposure
            .specs()
            .iter()
            .map(|spec| spec.id.clone())
            .collect::<Vec<_>>();

        assert_eq!(
            ids,
            vec![
                "Historical_Sigma",
                "Daily_Std",
                "Cumulative_Range",
                "Beta",
                "Residual_Volatility",
                "VOLATILITY"
            ]
        );
    }

    #[test]
    fn cumulative_range_uses_available_history_with_min_periods_one() {
        let mut values = vec![Some(0.001); WINDOW];
        let output = cumulative_range_12m(&values);
        assert!(output[0].is_some());
        assert!(output[WINDOW - 2].is_some());
        assert!(output[WINDOW - 1].is_some());

        values[0] = None;
        let output = cumulative_range_12m(&values);
        assert!(output[WINDOW - 1].is_some());
    }
}

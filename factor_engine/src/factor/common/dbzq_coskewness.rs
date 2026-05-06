use std::collections::HashMap;

use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorSeries, FactorSpec, Frequency, Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::stock_daily_ops::neutralize_size_sector;
use crate::factor::common::vector::clean;
use crate::factor::common::{DailyPanel, PanelColumn};
use crate::operators::cs_zscore;

pub const VERSION: &str = "0.1.0";
pub const MARKET_INDEX: &str = "000985.CSI";
pub const WINDOW: usize = 120;

const EPS: f64 = f64::EPSILON;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CoskewnessMode {
    Full,
    Up,
    Down,
}

#[derive(Clone, Copy, Debug)]
pub struct CoskewnessFactorDef {
    pub id: &'static str,
    pub alias: &'static str,
    pub name: &'static str,
    pub mode: CoskewnessMode,
}

pub fn factor_spec(def: CoskewnessFactorDef) -> FactorSpec {
    FactorSpec {
        id: def.id.to_string(),
        aliases: vec![def.alias.to_string()],
        name: def.name.to_string(),
        asset_class: AssetClass::Stock,
        frequency: Frequency::Daily,
        version: VERSION.to_string(),
        tags: [
            "price",
            "return",
            "coskewness",
            "market",
            "neutralize",
            "barra",
            "size",
            "sector",
            "daily",
            "DBZQ",
        ]
        .iter()
        .map(|value| value.to_string())
        .collect(),
        description: format!(
            "{} from 120-day adjusted stock returns and 000985.CSI market returns, z-scored and neutralized by Barra SIZE and SW sector.",
            def.name
        ),
        dependencies: vec![
            DataRequest::new(DatasetId::StockDailyPv, &["close"]),
            DataRequest::new(DatasetId::StockAdjFactor, &["adj_factor"]),
            DataRequest::index_daily(MARKET_INDEX, &["close", "pre_close"]),
            DataRequest::new(DatasetId::StockBarraDaily, &["SIZE"]),
            DataRequest::new(DatasetId::StockSwClassification, &["l1_code"]),
        ],
        intraday_raw_dependencies: Vec::new(),
        lookback: Lookback {
            trading_days: WINDOW,
        },
    }
}

pub fn compute_factor(def: CoskewnessFactorDef, data: &DataPool) -> Result<FactorSeries> {
    let panel = data.daily_panel(DatasetId::StockDailyPv)?;
    let close = panel.column("close")?;
    let adj_factor =
        panel.column_from_table(data.daily(DatasetId::StockAdjFactor)?, "adj_factor")?;
    let adj_close = close.zip_binary(&adj_factor, multiply)?;
    let stock_return = adj_close.ts(daily_returns)?;

    let index_panel = data.index_daily_panel(MARKET_INDEX)?;
    let index_return = index_panel
        .column("close")?
        .zip_binary(&index_panel.column("pre_close")?, ret)?;
    let market_return = expand_index_column(&panel, &index_panel, &index_return)?;

    let raw = coskewness_column(&panel, &stock_return, &market_return, def.mode)?;
    let standardized = raw.cs(cs_zscore)?;
    let factor = neutralize_size_sector(&standardized, &panel, data)?;
    Ok(factor.to_factor_series(factor_spec(def)))
}

#[macro_export]
macro_rules! define_dbzq_coskewness_factor {
    ($struct_name:ident, $id:expr, $alias:expr, $name:expr, $mode:ident) => {
        const DEF: $crate::factor::common::dbzq_coskewness::CoskewnessFactorDef =
            $crate::factor::common::dbzq_coskewness::CoskewnessFactorDef {
                id: $id,
                alias: $alias,
                name: $name,
                mode: $crate::factor::common::dbzq_coskewness::CoskewnessMode::$mode,
            };

        pub struct $struct_name;

        pub fn create() -> Box<dyn $crate::factor::Factor> {
            Box::new($struct_name)
        }

        impl $crate::factor::Factor for $struct_name {
            fn spec(&self) -> $crate::core::FactorSpec {
                $crate::factor::common::dbzq_coskewness::factor_spec(DEF)
            }

            fn compute(
                &self,
                _context: &$crate::core::FactorContext,
                data: &$crate::data::DataPool,
            ) -> $crate::error::Result<$crate::core::FactorSeries> {
                $crate::factor::common::dbzq_coskewness::compute_factor(DEF, data)
            }
        }
    };
}

fn expand_index_column(
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

fn coskewness_column(
    panel: &DailyPanel,
    stock_return: &PanelColumn,
    market_return: &PanelColumn,
    mode: CoskewnessMode,
) -> Result<PanelColumn> {
    let date_count = panel.dates().len();
    let instrument_count = panel.instruments().len();
    let mut output = vec![None; panel.shape_len()];

    for instrument_idx in 0..instrument_count {
        let mut stock_series = Vec::with_capacity(date_count);
        let mut market_series = Vec::with_capacity(date_count);
        for date_idx in 0..date_count {
            let offset = date_idx * instrument_count + instrument_idx;
            stock_series.push(stock_return.values()[offset]);
            market_series.push(market_return.values()[offset]);
        }

        let computed = coskewness_series(&stock_series, &market_series, mode);
        for (date_idx, value) in computed.into_iter().enumerate() {
            output[date_idx * instrument_count + instrument_idx] = value;
        }
    }

    panel.column_from_values(output)
}

fn coskewness_series(
    stock_return: &[Option<f64>],
    market_return: &[Option<f64>],
    mode: CoskewnessMode,
) -> Vec<Option<f64>> {
    let mut output = vec![None; stock_return.len()];
    if stock_return.len() < WINDOW {
        return output;
    }

    for end in WINDOW - 1..stock_return.len() {
        let start = end + 1 - WINDOW;
        let mut x = Vec::with_capacity(WINDOW);
        let mut y = Vec::with_capacity(WINDOW);
        let mut valid = true;
        for idx in start..=end {
            let (Some(stock), Some(market)) =
                (finite(stock_return[idx]), finite(market_return[idx]))
            else {
                valid = false;
                break;
            };
            x.push(stock);
            y.push(market);
        }
        if !valid {
            continue;
        }
        output[end] = coskewness_value(&x, &y, mode);
    }

    output
}

fn coskewness_value(x: &[f64], y: &[f64], mode: CoskewnessMode) -> Option<f64> {
    if x.len() != y.len() || x.len() != WINDOW {
        return None;
    }

    let x_mean = mean(x)?;
    let y_mean = mean(y)?;
    match mode {
        CoskewnessMode::Full => full_coskewness(x, y, x_mean, y_mean).map(|value| -value),
        CoskewnessMode::Up => conditional_coskewness(x, y, x_mean, y_mean, |value| value > y_mean),
        CoskewnessMode::Down => {
            conditional_coskewness(x, y, x_mean, y_mean, |value| value < y_mean)
        }
    }
}

fn full_coskewness(x: &[f64], y: &[f64], x_mean: f64, y_mean: f64) -> Option<f64> {
    let x_std = population_std(x, x_mean)?;
    let y_std = population_std(y, y_mean)?;
    if x_std <= EPS || y_std <= EPS {
        return None;
    }

    let numerator = x
        .iter()
        .zip(y)
        .map(|(x_value, y_value)| (x_value - x_mean) * (y_value - y_mean).powi(2))
        .sum::<f64>()
        / x.len() as f64;
    let denominator = x_std * y_std.powi(2);
    if denominator.abs() <= EPS {
        return None;
    }
    finite(Some(numerator / denominator))
}

fn conditional_coskewness<F>(
    x: &[f64],
    y: &[f64],
    x_mean: f64,
    y_mean: f64,
    mut predicate: F,
) -> Option<f64>
where
    F: FnMut(f64) -> bool,
{
    let mut numerator = 0.0;
    let mut x_second = 0.0;
    let mut y_second = 0.0;
    let mut count = 0usize;

    for (x_value, y_value) in x.iter().zip(y) {
        if !predicate(*y_value) {
            continue;
        }
        let x_dev = x_value - x_mean;
        let y_dev = y_value - y_mean;
        numerator += x_dev * y_dev.powi(2);
        x_second += x_dev.powi(2);
        y_second += y_dev.powi(2);
        count += 1;
    }
    if count == 0 {
        return None;
    }

    let numerator = numerator / count as f64;
    let denominator = ((x_second / count as f64) * (y_second / count as f64)).sqrt();
    if denominator.abs() <= EPS {
        return None;
    }
    finite(Some(numerator / denominator))
}

fn daily_returns(values: &[Option<f64>]) -> Vec<Option<f64>> {
    let mut output = vec![None; values.len()];
    for idx in 1..values.len() {
        output[idx] = ret(values[idx], values[idx - 1]);
    }
    output
}

fn ret(numerator: Option<f64>, denominator: Option<f64>) -> Option<f64> {
    match (finite(numerator), finite(denominator)) {
        (Some(numerator), Some(denominator)) if denominator.abs() > EPS => {
            Some(numerator / denominator - 1.0)
        }
        _ => None,
    }
}

fn multiply(left: Option<f64>, right: Option<f64>) -> Option<f64> {
    match (finite(left), finite(right)) {
        (Some(left), Some(right)) => Some(left * right),
        _ => None,
    }
}

fn mean(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let sum = values.iter().sum::<f64>();
    finite(Some(sum / values.len() as f64))
}

fn population_std(values: &[f64], mean: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let variance = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / values.len() as f64;
    if variance <= EPS {
        return None;
    }
    finite(Some(variance.sqrt()))
}

fn finite(value: Option<f64>) -> Option<f64> {
    clean(value).filter(|value| value.is_finite())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: Option<f64>, expected: f64) {
        let actual = actual.expect("expected value");
        assert!(
            (actual - expected).abs() < 1e-10,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn dbzq_coskewness_daily_returns_use_adjusted_close_neighbors() {
        let returns = daily_returns(&[Some(100.0), Some(110.0), Some(99.0), None]);

        assert_eq!(returns[0], None);
        assert_close(returns[1], 0.1);
        assert_close(returns[2], -0.1);
        assert_eq!(returns[3], None);
    }

    #[test]
    fn dbzq_coskewness_full_mode_applies_negative_orientation() {
        let x = repeating_window(&[1.0, 2.0, 3.0]);
        let y = repeating_window(&[1.0, 2.0, 4.0]);

        let raw = full_coskewness(&x, &y, mean(&x).unwrap(), mean(&y).unwrap()).unwrap();
        let oriented = coskewness_value(&x, &y, CoskewnessMode::Full).unwrap();

        assert!((oriented + raw).abs() < 1e-10);
    }

    #[test]
    fn dbzq_coskewness_up_and_down_use_market_mean_filter() {
        let x = repeating_window(&[-2.0, -1.0, 1.0, 3.0]);
        let y = repeating_window(&[-3.0, -1.0, 1.0, 3.0]);
        let x_mean = mean(&x).unwrap();
        let y_mean = mean(&y).unwrap();

        let up = coskewness_value(&x, &y, CoskewnessMode::Up);
        let expected_up = conditional_coskewness(&x, &y, x_mean, y_mean, |value| value > y_mean);
        let down = coskewness_value(&x, &y, CoskewnessMode::Down);
        let expected_down = conditional_coskewness(&x, &y, x_mean, y_mean, |value| value < y_mean);

        assert_close(up, expected_up.unwrap());
        assert_close(down, expected_down.unwrap());
    }

    #[test]
    fn dbzq_coskewness_series_requires_full_window() {
        let stock = vec![Some(0.01); WINDOW];
        let market = vec![Some(0.02); WINDOW];

        let output = coskewness_series(&stock, &market, CoskewnessMode::Full);

        assert!(output[..WINDOW - 1].iter().all(Option::is_none));
        assert_eq!(output[WINDOW - 1], None);
    }

    fn repeating_window(pattern: &[f64]) -> Vec<f64> {
        (0..WINDOW)
            .map(|idx| pattern[idx % pattern.len()])
            .collect()
    }
}

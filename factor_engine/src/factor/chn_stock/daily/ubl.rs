use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::vector::clean;
use crate::factor::common::PanelColumn;
use crate::factor::Factor;
use crate::operators::{cs_zscore, ts_delay, ts_mean, ts_std_dev};

const VERSION: &str = "0.2.0";
const PREV_WINDOW: usize = 5;
const MONTH_WINDOW: usize = 20;
const MIN_PERIODS: usize = 1;

pub struct StockDailyUbl;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyUbl)
}

impl Factor for StockDailyUbl {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "ubl".to_string(),
            aliases: vec!["UBL".to_string()],
            name: "UBL".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: [
                "price_volume",
                "candlestick",
                "shadow",
                "neutralize",
                "barra",
                "size",
                "daily",
                "DWZQ",
            ]
            .iter()
            .map(|value| value.to_string())
            .collect(),
            description: "Up and Bottom Shadow Line factor from normalized upper candle shadow volatility and Williams lower shadow level after SIZE neutralization.".to_string(),
            dependencies: vec![
                DataRequest::new(DatasetId::StockDailyPv, &["open", "high", "low", "close"]),
                DataRequest::new(DatasetId::StockBarraDaily, &["SIZE"]),
            ],
            intraday_raw_dependencies: Vec::new(),
            lookback: Lookback {
                trading_days: PREV_WINDOW + MONTH_WINDOW - 1,
            },
        }
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let panel = data.daily_panel(DatasetId::StockDailyPv)?;
        let open = panel.column("open")?;
        let high = panel.column("high")?;
        let low = panel.column("low")?;
        let close = panel.column("close")?;
        let size = panel.column_from_table(data.daily(DatasetId::StockBarraDaily)?, "SIZE")?;

        let upper_shadow = high.zip_ternary(&open, &close, upper_shadow)?;
        let upper_mean5_prev =
            upper_shadow.ts(|values| previous_window_mean(values, PREV_WINDOW, MIN_PERIODS))?;
        let normalized_upper = upper_shadow.zip_binary(&upper_mean5_prev, safe_div)?;
        let upper_std20 =
            normalized_upper.ts(|values| ts_std_dev(values, MONTH_WINDOW, MIN_PERIODS))?;
        let upper_std_desize = upper_std20.cs_neutralize_regression(&[&size], None)?;

        let lower_shadow = close.zip_binary(&low, lower_shadow)?;
        let lower_mean5_prev =
            lower_shadow.ts(|values| previous_window_mean(values, PREV_WINDOW, MIN_PERIODS))?;
        let normalized_lower = lower_shadow.zip_binary(&lower_mean5_prev, safe_div)?;
        let lower_mean20 =
            normalized_lower.ts(|values| ts_mean(values, MONTH_WINDOW, MIN_PERIODS))?;
        let lower_mean_desize = lower_mean20.cs_neutralize_regression(&[&size], None)?;

        let factor = average_pair(
            &upper_std_desize.cs(cs_zscore)?,
            &lower_mean_desize.cs(cs_zscore)?,
        )?;
        Ok(factor.to_factor_series(self.spec()))
    }
}

fn upper_shadow(high: Option<f64>, open: Option<f64>, close: Option<f64>) -> Option<f64> {
    match (clean(high), clean(open), clean(close)) {
        (Some(high), Some(open), Some(close)) => Some(high - open.max(close)),
        _ => None,
    }
}

fn lower_shadow(close: Option<f64>, low: Option<f64>) -> Option<f64> {
    match (clean(close), clean(low)) {
        (Some(close), Some(low)) => Some(close - low),
        _ => None,
    }
}

fn previous_window_mean(
    values: &[Option<f64>],
    window: usize,
    min_periods: usize,
) -> Vec<Option<f64>> {
    let delayed = ts_delay(values, 1);
    ts_mean(&delayed, window, min_periods)
}

fn safe_div(numerator: Option<f64>, denominator: Option<f64>) -> Option<f64> {
    match (clean(numerator), clean(denominator)) {
        (Some(numerator), Some(denominator)) if denominator.abs() > f64::EPSILON => {
            Some(numerator / denominator)
        }
        _ => None,
    }
}

fn average_pair(left: &PanelColumn, right: &PanelColumn) -> Result<PanelColumn> {
    left.zip_binary(right, |left, right| match (clean(left), clean(right)) {
        (Some(left), Some(right)) => Some((left + right) / 2.0),
        _ => None,
    })
}

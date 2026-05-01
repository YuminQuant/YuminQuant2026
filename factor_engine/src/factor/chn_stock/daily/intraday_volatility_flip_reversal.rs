use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::vector::clean;
use crate::factor::Factor;
use crate::operators::{cs_mean, ts_std_dev};

pub struct StockDailyIntradayVolatilityFlipReversal;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyIntradayVolatilityFlipReversal)
}

impl Factor for StockDailyIntradayVolatilityFlipReversal {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "intraday_volatility_flip_reversal".to_string(),
            aliases: Vec::new(),
            name: "Intraday Volatility Flip Reversal".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: "0.1.0".to_string(),
            tags: [
                "price_volume",
                "intraday_return",
                "reversal",
                "volatility",
                "daily",
                "FZZQ",
            ]
            .iter()
            .map(|value| value.to_string())
            .collect(),
            description:
                "Close/open intraday return flipped when 20-day intraday volatility is below cross-sectional mean."
                    .to_string(),
            dependencies: vec![DataRequest::new(DatasetId::StockDailyPv, &["close", "open"])],
            intraday_raw_dependencies: Vec::new(),
            lookback: Lookback { trading_days: 19 },
        }
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let panel = data.daily_panel(DatasetId::StockDailyPv)?;
        let returns = panel
            .column("close")?
            .zip_binary(&panel.column("open")?, ret)?;
        let std20 = returns.ts(|values| ts_std_dev(values, 20, 20))?;
        let std20_mean = std20.cs(cs_mean)?;
        let factor = returns.zip_binary(
            &std20.zip_binary(&std20_mean, less_than)?,
            |ret, flip| match (clean(ret), clean(flip)) {
                (Some(ret), Some(flip)) => Some(if flip > 0.0 { -ret } else { ret }),
                _ => None,
            },
        )?;
        Ok(factor.to_factor_series(self.spec()))
    }
}

fn ret(close: Option<f64>, open: Option<f64>) -> Option<f64> {
    match (clean(close), clean(open)) {
        (Some(close), Some(open)) if open.abs() > f64::EPSILON => Some(close / open - 1.0),
        _ => None,
    }
}

fn less_than(left: Option<f64>, right: Option<f64>) -> Option<f64> {
    match (clean(left), clean(right)) {
        (Some(left), Some(right)) => Some((left < right) as i32 as f64),
        _ => None,
    }
}

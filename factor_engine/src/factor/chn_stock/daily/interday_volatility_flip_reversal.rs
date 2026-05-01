use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::vector::clean;
use crate::factor::Factor;
use crate::operators::{cs_mean, ts_pctchg, ts_std_dev};

pub struct StockDailyInterdayVolatilityFlipReversal;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyInterdayVolatilityFlipReversal)
}

impl Factor for StockDailyInterdayVolatilityFlipReversal {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "interday_volatility_flip_reversal".to_string(),
            aliases: Vec::new(),
            name: "Interday Volatility Flip Reversal".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: "0.1.0".to_string(),
            tags: ["price_volume", "return", "reversal", "volatility", "daily", "FZZQ"]
                .iter()
                .map(|value| value.to_string())
                .collect(),
            description:
                "Adjusted close return flipped when its 20-day volatility is below cross-sectional mean."
                    .to_string(),
            dependencies: vec![
                DataRequest::new(DatasetId::StockDailyPv, &["close"]),
                DataRequest::new(DatasetId::StockAdjFactor, &["adj_factor"]),
            ],
            intraday_raw_dependencies: Vec::new(),
            lookback: Lookback { trading_days: 20 },
        }
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let panel = data.daily_panel(DatasetId::StockDailyPv)?;
        let adj_factor =
            panel.column_from_table(data.daily(DatasetId::StockAdjFactor)?, "adj_factor")?;
        let adj_close = panel.column("close")?.zip_binary(&adj_factor, mul)?;
        let returns = adj_close.ts(|values| ts_pctchg(values, 1))?;
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

fn mul(left: Option<f64>, right: Option<f64>) -> Option<f64> {
    match (clean(left), clean(right)) {
        (Some(left), Some(right)) => Some(left * right),
        _ => None,
    }
}

fn less_than(left: Option<f64>, right: Option<f64>) -> Option<f64> {
    match (clean(left), clean(right)) {
        (Some(left), Some(right)) => Some((left < right) as i32 as f64),
        _ => None,
    }
}

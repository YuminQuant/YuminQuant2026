use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::vector::clean;
use crate::factor::Factor;
use crate::operators::{cs_mean, ts_diff, ts_mean, ts_pctchg};

pub struct StockDailyInterdayTurnoverFlipReversal20d;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyInterdayTurnoverFlipReversal20d)
}

impl Factor for StockDailyInterdayTurnoverFlipReversal20d {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "interday_turnover_flip_reversal_20d".to_string(),
            aliases: Vec::new(),
            name: "Interday Turnover Flip Reversal 20D".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: "0.1.0".to_string(),
            tags: [
                "price_volume",
                "return",
                "reversal",
                "turnover",
                "daily",
                "FZZQ",
            ]
            .iter()
            .map(|value| value.to_string())
            .collect(),
            description:
                "20-day mean of adjusted close returns flipped by cross-sectional turnover change."
                    .to_string(),
            dependencies: vec![
                DataRequest::new(DatasetId::StockDailyPv, &["close"]),
                DataRequest::new(DatasetId::StockAdjFactor, &["adj_factor"]),
                DataRequest::new(DatasetId::StockDailyBasic, &["turnover_rate_f"]),
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
        let turnover = panel
            .column_from_table(data.daily(DatasetId::StockDailyBasic)?, "turnover_rate_f")?
            .map_values(|value| clean(value).map(|value| value / 100.0));
        let turnover_delta = turnover.ts(|values| ts_diff(values, 1))?;
        let turnover_delta_mean = turnover_delta.cs(cs_mean)?;
        let flipped = returns.zip_binary(
            &turnover_delta.zip_binary(&turnover_delta_mean, less_than)?,
            |ret, flip| match (clean(ret), clean(flip)) {
                (Some(ret), Some(flip)) => Some(if flip > 0.0 { -ret } else { ret }),
                _ => None,
            },
        )?;
        let factor = flipped.ts(|values| ts_mean(values, 20, 20))?;
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

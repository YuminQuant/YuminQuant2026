use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    IntradayDailyRawRequest, Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::chn_stock::daily::treat_same::FAIR_RETURN_RAW_ID;
use crate::factor::Factor;
use crate::operators::ts_mean;

const VERSION: &str = "0.1.0";
const WINDOW: usize = 20;

pub struct StockDailyTreatSameReturnFair;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyTreatSameReturnFair)
}

impl Factor for StockDailyTreatSameReturnFair {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "treat_same_return_fair".to_string(),
            aliases: Vec::new(),
            name: "Treat Same Return Fair".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: [
                "price_volume",
                "return",
                "volume",
                "intraday",
                "minute_agg",
                "temporary",
                "daily",
                "FZZQ",
            ]
            .iter()
            .map(|value| value.to_string())
            .collect(),
            description:
                "Temporary output of the return fairness branch before Treat Same composite."
                    .to_string(),
            dependencies: vec![DataRequest::new(
                DatasetId::StockDailyPv,
                &["open", "close"],
            )],
            intraday_raw_dependencies: vec![IntradayDailyRawRequest::new(
                FAIR_RETURN_RAW_ID,
                WINDOW - 1,
            )],
            lookback: Lookback {
                trading_days: WINDOW - 1,
            },
        }
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let panel = data.intraday_daily_raw_panel(FAIR_RETURN_RAW_ID)?;
        let open = panel.column_from_table(data.daily(DatasetId::StockDailyPv)?, "open")?;
        let close = panel.column_from_table(data.daily(DatasetId::StockDailyPv)?, "close")?;
        let intraday_return = close.zip_binary(&open, ret)?;
        let fair_return = panel
            .column(FAIR_RETURN_RAW_ID)?
            .zip_binary(&intraday_return, multiply)?;
        let factor = fair_return.ts(|values| ts_mean(values, WINDOW, 1))?;

        Ok(factor.to_factor_series(self.spec()))
    }
}

fn ret(numerator: Option<f64>, denominator: Option<f64>) -> Option<f64> {
    match (clean(numerator), clean(denominator)) {
        (Some(numerator), Some(denominator)) if denominator.abs() > f64::EPSILON => {
            Some(numerator / denominator - 1.0)
        }
        _ => None,
    }
}

fn multiply(left: Option<f64>, right: Option<f64>) -> Option<f64> {
    Some(clean(left)? * clean(right)?)
}

fn clean(value: Option<f64>) -> Option<f64> {
    value.filter(|value| !value.is_nan())
}

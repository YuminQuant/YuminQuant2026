use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    IntradayDailyRawRequest, Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::chn_stock::daily::utd::INTRADAY_RETURN_VOLATILITY_RAW_ID;
use crate::factor::common::vector::clean;
use crate::factor::Factor;
use crate::operators::{ts_mean, ts_std_dev};

const VERSION: &str = "0.1.0";
const WINDOW: usize = 20;

pub struct StockDailyUid;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyUid)
}

impl Factor for StockDailyUid {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "uid".to_string(),
            aliases: vec!["UID".to_string()],
            name: "UID".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: [
                "price_volume",
                "return",
                "volatility",
                "distribution",
                "intraday",
                "minute_agg",
                "neutralize",
                "barra",
                "size",
                "daily",
                "DWZQ",
            ]
            .iter()
            .map(|value| value.to_string())
            .collect(),
            description: "Uniformity of Information Distribution factor, computed as the 20-day volatility-to-mean ratio of intraday return volatility after SIZE neutralization.".to_string(),
            dependencies: vec![DataRequest::new(DatasetId::StockBarraDaily, &["SIZE"])],
            intraday_raw_dependencies: vec![IntradayDailyRawRequest::new(
                INTRADAY_RETURN_VOLATILITY_RAW_ID,
                WINDOW - 1,
            )],
            lookback: Lookback {
                trading_days: WINDOW - 1,
            },
        }
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let panel = data.intraday_daily_raw_panel(INTRADAY_RETURN_VOLATILITY_RAW_ID)?;
        let raw = panel.column(INTRADAY_RETURN_VOLATILITY_RAW_ID)?;
        let size = panel.column_from_table(data.daily(DatasetId::StockBarraDaily)?, "SIZE")?;
        let mean20 = raw.ts(|values| ts_mean(values, WINDOW, WINDOW))?;
        let std20 = raw.ts(|values| ts_std_dev(values, WINDOW, WINDOW))?;
        let raw_factor = std20.zip_binary(&mean20, safe_div)?;
        let factor = raw_factor.cs_neutralize_regression(&[&size], None)?;
        Ok(factor.to_factor_series(self.spec()))
    }
}

fn safe_div(numerator: Option<f64>, denominator: Option<f64>) -> Option<f64> {
    match (clean(numerator), clean(denominator)) {
        (Some(numerator), Some(denominator)) if denominator.abs() > f64::EPSILON => {
            Some(numerator / denominator)
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uid_safe_div_rejects_zero_denominator() {
        assert_eq!(safe_div(Some(1.0), Some(0.0)), None);
        assert_eq!(safe_div(Some(3.0), Some(2.0)), Some(1.5));
    }
}

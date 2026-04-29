use crate::core::{
    AssetClass, FactorContext, FactorSeries, FactorSpec, Frequency, IntradayDailyRawRequest,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::chn_stock::daily::top20_centered_vol_ret_mean::RAW_ID as TOP20_RAW_ID;
use crate::factor::Factor;
use crate::operators::ts_mean;

pub struct StockDailyTop20CenteredVolRetMean20dMean;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyTop20CenteredVolRetMean20dMean)
}

impl Factor for StockDailyTop20CenteredVolRetMean20dMean {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "top20_centered_vol_ret_mean_20d_mean".to_string(),
            aliases: Vec::new(),
            name: "Stock 20-day mean of top centered-volume intraday return mean".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: "0.1.0".to_string(),
            tags: [
                "price_volume",
                "return",
                "volume",
                "intraday",
                "minute_agg",
                "mean",
                "daily",
            ]
            .iter()
            .map(|value| value.to_string())
            .collect(),
            description: "20-day mean of the mean return of the top 20 minutes ranked by centered 5-minute volume mean.".to_string(),
            dependencies: Vec::new(),
            intraday_raw_dependencies: vec![IntradayDailyRawRequest::new(TOP20_RAW_ID, 19)],
            lookback: Lookback { trading_days: 19 },
        }
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let panel = data.intraday_daily_raw_panel(TOP20_RAW_ID)?;
        let raw = panel.column(TOP20_RAW_ID)?;
        let factor = raw.ts(|values| ts_mean(values, 20, 20))?;
        Ok(factor.to_factor_series(self.spec()))
    }
}

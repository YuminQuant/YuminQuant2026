use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::Factor;
use crate::operators::{cs_pctrank, ts_covariance};

pub struct StockDailyWQAlpha016;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyWQAlpha016)
}

impl Factor for StockDailyWQAlpha016 {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "WQAlpha016".to_string(),
            aliases: Vec::new(),
            name: "WQAlpha016".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: "0.1.0".to_string(),
            tags: ["worldquant101alpha", "price_volume", "daily"]
                .iter()
                .map(|value| value.to_string())
                .collect(),
            description: "-rank(covariance(rank(high), rank(volume), 5))".to_string(),
            dependencies: vec![DataRequest::new(DatasetId::StockDailyPv, &["high", "vol"])],
            intraday_raw_dependencies: Vec::new(),
            lookback: Lookback { trading_days: 4 },
        }
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let panel = data.daily_panel(DatasetId::StockDailyPv)?;
        let ranked_high = panel
            .column("high")?
            .cs(|values| cs_pctrank(values, true))?;
        let ranked_volume = panel.column("vol")?.cs(|values| cs_pctrank(values, true))?;
        let factor = ranked_high
            .ts_binary(&ranked_volume, |high, volume| {
                ts_covariance(high, volume, 5, 5)
            })?
            .cs(|values| {
                cs_pctrank(values, true)
                    .into_iter()
                    .map(|value| value.map(|value| -value))
                    .collect()
            })?;
        Ok(factor.to_factor_series(self.spec()))
    }
}

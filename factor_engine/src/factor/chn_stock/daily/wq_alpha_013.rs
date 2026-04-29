use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::Factor;
use crate::operators::{cs_pctrank, ts_covariance};

pub struct StockDailyWQAlpha013;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyWQAlpha013)
}

impl Factor for StockDailyWQAlpha013 {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "WQAlpha013".to_string(),
            aliases: Vec::new(),
            name: "WQAlpha013".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: "0.1.0".to_string(),
            tags: ["worldquant101alpha", "price_volume", "daily"]
                .iter()
                .map(|value| value.to_string())
                .collect(),
            description: "-rank(covariance(rank(close), rank(volume), 5))".to_string(),
            dependencies: vec![DataRequest::new(DatasetId::StockDailyPv, &["close", "vol"])],
            intraday_raw_dependencies: Vec::new(),
            lookback: Lookback { trading_days: 4 },
        }
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let panel = data.daily_panel(DatasetId::StockDailyPv)?;
        let ranked_close = panel
            .column("close")?
            .cs(|values| cs_pctrank(values, true))?;
        let ranked_volume = panel.column("vol")?.cs(|values| cs_pctrank(values, true))?;
        let factor = ranked_close
            .ts_binary(&ranked_volume, |close, volume| {
                ts_covariance(close, volume, 5, 5)
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

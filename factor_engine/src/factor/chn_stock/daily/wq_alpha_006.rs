use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::Factor;
use crate::operators::ts_corr;

pub struct StockDailyWQAlpha006;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyWQAlpha006)
}

impl Factor for StockDailyWQAlpha006 {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "WQAlpha006".to_string(),
            aliases: Vec::new(),
            name: "WQAlpha006".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: "0.1.0".to_string(),
            tags: ["worldquant101alpha", "price_volume", "daily"]
                .iter()
                .map(|value| value.to_string())
                .collect(),
            description: "-correlation(open, volume, 10)".to_string(),
            dependencies: vec![DataRequest::new(DatasetId::StockDailyPv, &["open", "vol"])],
            intraday_raw_dependencies: Vec::new(),
            lookback: Lookback { trading_days: 9 },
        }
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let panel = data.daily_panel(DatasetId::StockDailyPv)?;
        let open = panel.column("open")?;
        let volume = panel.column("vol")?;
        let factor = open.ts_binary(&volume, |open, volume| {
            ts_corr(open, volume, 10, 10)
                .into_iter()
                .map(|value| value.map(|value| -value))
                .collect()
        })?;
        Ok(factor.to_factor_series(self.spec()))
    }
}

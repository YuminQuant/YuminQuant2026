use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::Factor;
use crate::operators::{cs_pctrank, ts_corr};

pub struct StockDailyWQAlpha003;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyWQAlpha003)
}

impl Factor for StockDailyWQAlpha003 {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "WQAlpha003".to_string(),
            aliases: Vec::new(),
            name: "WQAlpha003".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: "0.1.0".to_string(),
            tags: ["worldquant101alpha", "price_volume", "daily"]
                .iter()
                .map(|value| value.to_string())
                .collect(),
            description: "-correlation(rank(open), rank(volume), 10)".to_string(),
            dependencies: vec![DataRequest::new(DatasetId::StockDailyPv, &["open", "vol"])],
            intraday_raw_dependencies: Vec::new(),
            lookback: Lookback { trading_days: 9 },
        }
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let panel = data.daily_panel(DatasetId::StockDailyPv)?;
        let ranked_open = panel
            .column("open")?
            .cs(|values| cs_pctrank(values, true))?;
        let ranked_volume = panel.column("vol")?.cs(|values| cs_pctrank(values, true))?;
        let factor = ranked_open.ts_binary(&ranked_volume, |open, volume| {
            ts_corr(open, volume, 10, 10)
                .into_iter()
                .map(|value| value.map(|value| -value))
                .collect()
        })?;
        Ok(factor.to_factor_series(self.spec()))
    }
}

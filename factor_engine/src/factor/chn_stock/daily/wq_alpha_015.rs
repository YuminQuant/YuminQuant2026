use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::Factor;
use crate::operators::{cs_pctrank, ts_corr, ts_sum};

pub struct StockDailyWQAlpha015;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyWQAlpha015)
}

impl Factor for StockDailyWQAlpha015 {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "WQAlpha015".to_string(),
            aliases: Vec::new(),
            name: "WQAlpha015".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: "0.1.0".to_string(),
            tags: ["worldquant101alpha", "price_volume", "daily"]
                .iter()
                .map(|value| value.to_string())
                .collect(),
            description: "-sum(rank(correlation(rank(high), rank(volume), 3)), 3)".to_string(),
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
        let corr_rank = ranked_high
            .ts_binary(&ranked_volume, |high, volume| ts_corr(high, volume, 3, 3))?
            .cs(|values| cs_pctrank(values, true))?;
        let factor = corr_rank.ts(|values| {
            ts_sum(values, 3, 3)
                .into_iter()
                .map(|value| value.map(|value| -value))
                .collect()
        })?;
        Ok(factor.to_factor_series(self.spec()))
    }
}

use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::chip;
use crate::factor::Factor;

const VERSION: &str = "0.1.0";

pub struct StockDailyMktHoldingRet;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyMktHoldingRet)
}

impl Factor for StockDailyMktHoldingRet {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "mkt_holding_ret".to_string(),
            aliases: vec!["MktHoldingRet".to_string()],
            name: "Market Holding Return".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: ["price_volume", "chip", "market", "daily", "KYZQ"]
                .iter()
                .map(|value| value.to_string())
                .collect(),
            description:
                "Cross-sectional equal-weight mean of the 250-day retained chip holding return."
                    .to_string(),
            dependencies: vec![
                DataRequest::new(DatasetId::StockDailyPv, &["amount", "vol"]),
                DataRequest::new(DatasetId::StockDailyBasic, &["turnover_rate_f"]),
            ],
            intraday_raw_dependencies: Vec::new(),
            lookback: Lookback {
                trading_days: chip::CHIP_WINDOW,
            },
        }
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let (_panel, holding_ret) = chip::holding_ret_from_data(data)?;
        let factor = chip::cross_section_mean_constant(&holding_ret)?;
        Ok(factor.to_factor_series(self.spec()))
    }
}

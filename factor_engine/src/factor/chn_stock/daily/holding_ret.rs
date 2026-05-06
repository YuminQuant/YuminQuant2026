use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::chip;
use crate::factor::Factor;

const VERSION: &str = "0.1.0";

pub struct StockDailyHoldingRet;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyHoldingRet)
}

impl Factor for StockDailyHoldingRet {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "holding_ret".to_string(),
            aliases: vec!["HoldingRet".to_string()],
            name: "Holding Return".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: ["price_volume", "chip", "turnover", "vwap", "daily", "KYZQ"]
                .iter()
                .map(|value| value.to_string())
                .collect(),
            description: "Current daily VWAP relative to the 250-day retained chip weighted cost."
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
        let (_panel, factor) = chip::holding_ret_from_data(data)?;
        Ok(factor.to_factor_series(self.spec()))
    }
}

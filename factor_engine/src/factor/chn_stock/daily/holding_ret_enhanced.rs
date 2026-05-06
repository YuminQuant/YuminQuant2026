use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::chip;
use crate::factor::Factor;

const VERSION: &str = "0.1.0";

pub struct StockDailyHoldingRetEnhanced;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyHoldingRetEnhanced)
}

impl Factor for StockDailyHoldingRetEnhanced {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "holding_ret_enhanced".to_string(),
            aliases: vec!["HoldingRetEnhanced".to_string()],
            name: "Enhanced Holding Return".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: [
                "price_volume",
                "chip",
                "return",
                "composite",
                "daily",
                "KYZQ",
            ]
            .iter()
            .map(|value| value.to_string())
            .collect(),
            description:
                "Enhanced chip factor combining the -2% adjusted holding return with Ret20."
                    .to_string(),
            dependencies: vec![
                DataRequest::new(DatasetId::StockDailyPv, &["amount", "vol", "close"]),
                DataRequest::new(DatasetId::StockDailyBasic, &["turnover_rate_f"]),
                DataRequest::new(DatasetId::StockAdjFactor, &["adj_factor"]),
            ],
            intraday_raw_dependencies: Vec::new(),
            lookback: Lookback {
                trading_days: chip::CHIP_WINDOW,
            },
        }
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let (panel, holding_ret) = chip::holding_ret_from_data(data)?;
        let market = chip::cross_section_mean_constant(&holding_ret)?;
        let adjusted_holding = holding_ret.zip_binary(&market, chip::sign_adjust_minus_2pct)?;
        let ret20 = chip::ret20_from_data(&panel, data)?;
        let factor = ret20.zip_binary(&adjusted_holding, chip::enhanced)?;
        Ok(factor.to_factor_series(self.spec()))
    }
}

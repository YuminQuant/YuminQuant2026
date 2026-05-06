use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::chip;
use crate::factor::Factor;

const VERSION: &str = "0.1.0";

pub struct StockDailyHoldingRetAdjMinus2Pct;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyHoldingRetAdjMinus2Pct)
}

impl Factor for StockDailyHoldingRetAdjMinus2Pct {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "holding_ret_adj_minus_2pct".to_string(),
            aliases: vec![
                "HoldingRetAdjMinus2Pct".to_string(),
                "holding_ret_adj_-2%".to_string(),
            ],
            name: "Holding Return Adjusted -2pct".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: ["price_volume", "chip", "timing", "daily", "KYZQ"]
                .iter()
                .map(|value| value.to_string())
                .collect(),
            description: "Holding return multiplied by sign(market holding return + 2%)."
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
        let market = chip::cross_section_mean_constant(&holding_ret)?;
        let factor = holding_ret.zip_binary(&market, chip::sign_adjust_minus_2pct)?;
        Ok(factor.to_factor_series(self.spec()))
    }
}

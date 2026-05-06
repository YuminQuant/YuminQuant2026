use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::chip;
use crate::factor::Factor;

const VERSION: &str = "0.1.0";

pub struct StockDailyHoldingRetAdj;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyHoldingRetAdj)
}

impl Factor for StockDailyHoldingRetAdj {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "holding_ret_adj".to_string(),
            aliases: vec!["HoldingRetAdj".to_string()],
            name: "Holding Return Adjusted".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: ["price_volume", "chip", "timing", "daily", "KYZQ"]
                .iter()
                .map(|value| value.to_string())
                .collect(),
            description:
                "Holding return multiplied by the sign of market holding return at the zero threshold."
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
        let factor = holding_ret.zip_binary(&market, chip::sign_adjust)?;
        Ok(factor.to_factor_series(self.spec()))
    }
}

use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::chip;
use crate::factor::Factor;

const VERSION: &str = "0.1.0";

pub struct StockDailyRet20Adj;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyRet20Adj)
}

impl Factor for StockDailyRet20Adj {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "ret20_adj".to_string(),
            aliases: vec!["Ret20Adj".to_string()],
            name: "Ret20 Adjusted".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: ["price_volume", "chip", "return", "timing", "daily", "KYZQ"]
                .iter()
                .map(|value| value.to_string())
                .collect(),
            description: "Adjusted 20-day return multiplied by the sign of market holding return."
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
        let ret20 = chip::ret20_from_data(&panel, data)?;
        let factor = ret20.zip_binary(&market, chip::sign_adjust)?;
        Ok(factor.to_factor_series(self.spec()))
    }
}

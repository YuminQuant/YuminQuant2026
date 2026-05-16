use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::gfzq_behavioral;
use crate::factor::Factor;

const VERSION: &str = "0.1.0";

pub struct StockDailySalienceReturn;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailySalienceReturn)
}

impl Factor for StockDailySalienceReturn {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "salience_return".to_string(),
            aliases: vec!["STR_GFZQ".to_string()],
            name: "GFZQ Salience Return".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: ["GFZQ", "behavioral", "salience", "return", "neutralize", "daily"]
                .iter()
                .map(|value| value.to_string())
                .collect(),
            description: "GFZQ return-perspective salience factor from 20-day adjusted returns, followed by 3-sigma winsorization, z-score, and SIZE plus SW sector neutralization.".to_string(),
            dependencies: vec![
                DataRequest::new(DatasetId::StockDailyPv, &["close"]),
                DataRequest::new(DatasetId::StockAdjFactor, &["adj_factor"]),
                DataRequest::new(DatasetId::StockBarraDaily, &["SIZE"]),
                DataRequest::new(DatasetId::StockSwClassification, &["l1_code"]),
            ],
            intraday_raw_dependencies: Vec::new(),
            lookback: Lookback {
                trading_days: gfzq_behavioral::SALIENCE_WINDOW + 1,
            },
        }
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let (_panel, factor) = gfzq_behavioral::salience_return_from_data(data)?;
        Ok(factor.to_factor_series(self.spec()))
    }
}

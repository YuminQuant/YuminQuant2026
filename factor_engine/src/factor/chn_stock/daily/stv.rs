use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::gfzq_behavioral;
use crate::factor::Factor;

const VERSION: &str = "0.1.0";

pub struct StockDailyStv;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyStv)
}

impl Factor for StockDailyStv {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "stv".to_string(),
            aliases: vec!["STV".to_string()],
            name: "GFZQ Limit-Aware Turnover Salience".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: ["GFZQ", "behavioral", "salience", "turnover", "return", "neutralize", "daily"]
                .iter()
                .map(|value| value.to_string())
                .collect(),
            description: "GFZQ limit-aware turnover salience factor using a 7% return threshold, followed by 3-sigma winsorization, z-score, and SIZE plus SW sector neutralization.".to_string(),
            dependencies: vec![
                DataRequest::new(DatasetId::StockDailyPv, &["close"]),
                DataRequest::new(DatasetId::StockAdjFactor, &["adj_factor"]),
                DataRequest::new(DatasetId::StockDailyBasic, &["turnover_rate_f"]),
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
        let (_panel, factor) = gfzq_behavioral::stv_from_data(data)?;
        Ok(factor.to_factor_series(self.spec()))
    }
}

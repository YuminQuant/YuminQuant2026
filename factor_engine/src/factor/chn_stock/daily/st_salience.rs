use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::gfzq_behavioral;
use crate::factor::Factor;

const VERSION: &str = "0.1.0";

pub struct StockDailyStSalience;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyStSalience)
}

impl Factor for StockDailyStSalience {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "st_salience".to_string(),
            aliases: vec!["ST".to_string(), "SalienceTheory".to_string()],
            name: "Salience Theory".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: ["GFZQ", "behavioral", "salience", "return", "daily"]
                .iter()
                .map(|value| value.to_string())
                .collect(),
            description: "GFZQ salience theory factor from 20-day adjusted stock returns versus equal-weight market returns excluding BJ stocks.".to_string(),
            dependencies: vec![
                DataRequest::new(DatasetId::StockDailyPv, &["close"]),
                DataRequest::new(DatasetId::StockAdjFactor, &["adj_factor"]),
            ],
            intraday_raw_dependencies: Vec::new(),
            lookback: Lookback {
                trading_days: gfzq_behavioral::ST_WINDOW + 1,
            },
        }
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let (_panel, factor) = gfzq_behavioral::st_salience_from_data(data)?;
        Ok(factor.to_factor_series(self.spec()))
    }
}

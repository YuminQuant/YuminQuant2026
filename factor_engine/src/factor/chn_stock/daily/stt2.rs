use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::gfzq_behavioral;
use crate::factor::Factor;

const VERSION: &str = "0.1.0";

pub struct StockDailyStt2;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyStt2)
}

impl Factor for StockDailyStt2 {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "stt2".to_string(),
            aliases: vec!["STT2".to_string()],
            name: "GFZQ Full Turnover Salience".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: ["GFZQ", "behavioral", "salience", "turnover", "neutralize", "daily"]
                .iter()
                .map(|value| value.to_string())
                .collect(),
            description: "GFZQ full turnover salience factor using turnover salience weights and turnover payoff, followed by 3-sigma winsorization, z-score, and SIZE plus SW sector neutralization.".to_string(),
            dependencies: vec![
                DataRequest::new(DatasetId::StockDailyBasic, &["turnover_rate_f"]),
                DataRequest::new(DatasetId::StockBarraDaily, &["SIZE"]),
                DataRequest::new(DatasetId::StockSwClassification, &["l1_code"]),
            ],
            intraday_raw_dependencies: Vec::new(),
            lookback: Lookback {
                trading_days: gfzq_behavioral::SALIENCE_WINDOW,
            },
        }
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let (_panel, factor) = gfzq_behavioral::stt2_from_data(data)?;
        Ok(factor.to_factor_series(self.spec()))
    }
}

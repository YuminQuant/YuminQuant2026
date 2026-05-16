use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::gfzq_apl_beta;
use crate::factor::Factor;

const VERSION: &str = "0.1.0";

pub struct StockDailyAplBeta;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyAplBeta)
}

impl Factor for StockDailyAplBeta {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "apl_beta".to_string(),
            aliases: vec![
                "APL_beta".to_string(),
                "AbsolutePriceLimitBeta".to_string(),
            ],
            name: "Absolute Price Limit Beta".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: [
                "GFZQ",
                "behavioral",
                "price_limit",
                "regression",
                "daily",
                "neutralize",
            ]
            .iter()
            .map(|value| value.to_string())
            .collect(),
            description: "GFZQ APL beta factor: absolute rolling 20-day sensitivity of stock returns to non-BJ market price-limit hit ratio, controlling for id_vol_decorr-style MKT/SMB/HML, followed by z-score and SIZE plus SW sector neutralization.".to_string(),
            dependencies: vec![
                DataRequest::new(DatasetId::StockDailyPv, &["close", "pre_close"]),
                DataRequest::new(DatasetId::StockDailyBasic, &["circ_mv", "pb"]),
                DataRequest::new(DatasetId::StockDailyLimit, &["up_limit", "down_limit"]),
                DataRequest::new(DatasetId::StockBarraDaily, &["SIZE"]),
                DataRequest::new(DatasetId::StockSwClassification, &["l1_code"]),
                DataRequest::index_daily(gfzq_apl_beta::MARKET_INDEX, &["close", "pre_close"]),
            ],
            intraday_raw_dependencies: Vec::new(),
            lookback: Lookback {
                trading_days: gfzq_apl_beta::APL_REGRESSION_WINDOW - 1,
            },
        }
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let (_panel, factor) = gfzq_apl_beta::apl_beta_from_data(data)?;
        Ok(factor.to_factor_series(self.spec()))
    }
}

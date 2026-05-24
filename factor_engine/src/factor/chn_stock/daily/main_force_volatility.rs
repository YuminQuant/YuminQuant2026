use crate::core::{FactorContext, FactorSeries, FactorSpec, IntradayDailyRawSpec};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::mszq_main_force_volatility::{self, MszqMainForceVolatilityFactorDef};
use crate::factor::Factor;

const DEF: MszqMainForceVolatilityFactorDef = MszqMainForceVolatilityFactorDef {
    id: "main_force_volatility",
    alias: "main_force_volatility",
    name: "Main Force Volatility",
};

pub struct StockDailyMainForceVolatility;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyMainForceVolatility)
}

impl Factor for StockDailyMainForceVolatility {
    fn spec(&self) -> FactorSpec {
        mszq_main_force_volatility::factor_spec(DEF)
    }

    fn intraday_raw_specs(&self) -> Vec<IntradayDailyRawSpec> {
        mszq_main_force_volatility::raw_specs()
    }

    fn intraday_raw_provider_key(&self, _raw_id: &str) -> String {
        mszq_main_force_volatility::PROVIDER_KEY.to_string()
    }

    fn minute_compute_many(
        &self,
        raw_ids: &[String],
        context: &FactorContext,
        data: &DataPool,
    ) -> Result<Vec<crate::core::IntradayDailyRawSeries>> {
        mszq_main_force_volatility::minute_compute_many(raw_ids, context, data)
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        mszq_main_force_volatility::compute_factor(DEF, data)
    }
}

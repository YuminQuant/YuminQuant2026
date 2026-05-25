use crate::core::{FactorContext, FactorSeries, FactorSpec, IntradayDailyRawSpec};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::mszq_gravity_field::{self, MszqGravityFieldFactorDef};
use crate::factor::Factor;

const DEF: MszqGravityFieldFactorDef = MszqGravityFieldFactorDef {
    id: "gravity_field",
    alias: "gravity_field",
    name: "Gravity Field",
};

pub struct StockDailyGravityField;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyGravityField)
}

impl Factor for StockDailyGravityField {
    fn spec(&self) -> FactorSpec {
        mszq_gravity_field::factor_spec(DEF)
    }

    fn intraday_raw_specs(&self) -> Vec<IntradayDailyRawSpec> {
        mszq_gravity_field::raw_specs()
    }

    fn intraday_raw_provider_key(&self, _raw_id: &str) -> String {
        mszq_gravity_field::PROVIDER_KEY.to_string()
    }

    fn minute_compute_many(
        &self,
        raw_ids: &[String],
        context: &FactorContext,
        data: &DataPool,
    ) -> Result<Vec<crate::core::IntradayDailyRawSeries>> {
        mszq_gravity_field::minute_compute_many(raw_ids, context, data)
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        mszq_gravity_field::compute_factor(DEF, data)
    }
}

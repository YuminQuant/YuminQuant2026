use crate::core::{FactorContext, FactorSeries, FactorSpec, IntradayDailyRawSpec};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::mszq_momentum_pulse::{self, MszqMomentumPulseFactorDef};
use crate::factor::Factor;

const DEF: MszqMomentumPulseFactorDef = MszqMomentumPulseFactorDef {
    id: "momentum_pulse",
    alias: "momentum_pulse",
    name: "Momentum Pulse",
};

pub struct StockDailyMomentumPulse;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyMomentumPulse)
}

impl Factor for StockDailyMomentumPulse {
    fn spec(&self) -> FactorSpec {
        mszq_momentum_pulse::factor_spec(DEF)
    }

    fn intraday_raw_specs(&self) -> Vec<IntradayDailyRawSpec> {
        mszq_momentum_pulse::raw_specs()
    }

    fn intraday_raw_provider_key(&self, _raw_id: &str) -> String {
        mszq_momentum_pulse::PROVIDER_KEY.to_string()
    }

    fn minute_compute_many(
        &self,
        raw_ids: &[String],
        context: &FactorContext,
        data: &DataPool,
    ) -> Result<Vec<crate::core::IntradayDailyRawSeries>> {
        mszq_momentum_pulse::minute_compute_many(raw_ids, context, data)
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        mszq_momentum_pulse::compute_factor(DEF, data)
    }
}

use crate::core::{
    FactorContext, FactorSeries, FactorSpec, IntradayDailyRawAuxiliaryRequest,
    IntradayDailyRawSeries, IntradayDailyRawSpec,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::xyzq_crossday_spillover;
use crate::factor::{Factor, IntradayRawMaterializeMode};

pub struct StockDailyCrossdayIntradaySpillover;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyCrossdayIntradaySpillover)
}

impl Factor for StockDailyCrossdayIntradaySpillover {
    fn spec(&self) -> FactorSpec {
        xyzq_crossday_spillover::composite_factor_spec()
    }

    fn intraday_raw_specs(&self) -> Vec<IntradayDailyRawSpec> {
        xyzq_crossday_spillover::raw_specs()
    }

    fn intraday_raw_provider_key(&self, _raw_id: &str) -> String {
        xyzq_crossday_spillover::PROVIDER_KEY.to_string()
    }

    fn intraday_raw_materialize_mode(&self, _raw_ids: &[String]) -> IntradayRawMaterializeMode {
        xyzq_crossday_spillover::intraday_raw_materialize_mode()
    }

    fn initial_intraday_raw_state(&self, _raw_ids: &[String]) -> Box<dyn std::any::Any + Send> {
        xyzq_crossday_spillover::initial_intraday_raw_state()
    }

    fn intraday_raw_auxiliary_requirements(
        &self,
        raw_ids: &[String],
    ) -> Vec<IntradayDailyRawAuxiliaryRequest> {
        xyzq_crossday_spillover::intraday_raw_auxiliary_requirements(raw_ids)
    }

    fn minute_compute_stateful_many(
        &self,
        raw_ids: &[String],
        context: &FactorContext,
        data: &DataPool,
        state: &mut dyn std::any::Any,
    ) -> Result<Vec<IntradayDailyRawSeries>> {
        xyzq_crossday_spillover::minute_compute_stateful_many(raw_ids, context, data, state)
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        xyzq_crossday_spillover::compute_composite_factor(data)
    }
}

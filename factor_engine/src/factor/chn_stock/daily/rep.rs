use std::any::Any;

use crate::core::{FactorContext, FactorSeries, FactorSpec};
use crate::data::DataPool;
use crate::error::{err, Result};
use crate::factor::common::hazq_equity_composition::{
    compute_requested, compute_requested_stateful, spec, HazqEquityCompositionComputeState,
    HazqEquityCompositionOutput, PROVIDER_KEY, REP_ID,
};
use crate::factor::{Factor, FactorUpdatePolicy};

pub struct StockDailyRep;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyRep)
}

impl Factor for StockDailyRep {
    fn spec(&self) -> FactorSpec {
        spec(HazqEquityCompositionOutput::Rep)
    }

    fn compute_provider_key(&self) -> String {
        PROVIDER_KEY.to_string()
    }

    fn update_policy(&self) -> FactorUpdatePolicy {
        FactorUpdatePolicy::FinancialEventStateDailyFast
    }

    fn initial_compute_state(&self, _requested_ids: &[String]) -> Box<dyn Any + Send> {
        Box::new(HazqEquityCompositionComputeState::default())
    }

    fn compute(&self, context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let requested = [REP_ID.to_string()];
        compute_requested(&requested, context, data)?
            .into_iter()
            .find(|series| series.spec.id == REP_ID)
            .ok_or_else(|| err("HAZQ equity composition provider did not return rep"))
    }

    fn compute_many(
        &self,
        requested_ids: &[String],
        context: &FactorContext,
        data: &DataPool,
    ) -> Result<Vec<FactorSeries>> {
        compute_requested(requested_ids, context, data)
    }

    fn compute_many_stateful(
        &self,
        requested_ids: &[String],
        context: &FactorContext,
        data: &DataPool,
        state: &mut (dyn Any + Send),
    ) -> Result<Vec<FactorSeries>> {
        let state = state
            .downcast_mut::<HazqEquityCompositionComputeState>()
            .ok_or_else(|| err("HAZQ equity composition provider received incompatible state"))?;
        compute_requested_stateful(requested_ids, context, data, state)
    }
}

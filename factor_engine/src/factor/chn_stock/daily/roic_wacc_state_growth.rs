use std::any::Any;

use crate::core::{DataRequest, FactorContext, FactorSeries, FactorSpec};
use crate::data::DataPool;
use crate::error::{err, Result};
use crate::factor::common::dbzq_roic_wacc::{
    compute_requested, compute_requested_stateful, requirements_for_context, spec,
    RoicWaccComputeState, RoicWaccOutput, PROVIDER_KEY, ROIC_WACC_STATE_GROWTH_ID,
};
use crate::factor::{Factor, FactorUpdatePolicy};

pub struct StockDailyRoicWaccStateGrowth;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyRoicWaccStateGrowth)
}

impl Factor for StockDailyRoicWaccStateGrowth {
    fn spec(&self) -> FactorSpec {
        spec(RoicWaccOutput::StateGrowth)
    }

    fn compute_provider_key(&self) -> String {
        PROVIDER_KEY.to_string()
    }

    fn update_policy(&self) -> FactorUpdatePolicy {
        FactorUpdatePolicy::FinancialEventSnapshot
    }

    fn requirements_for_context(&self, context: &FactorContext) -> Vec<DataRequest> {
        requirements_for_context(context)
    }

    fn initial_compute_state(&self, _requested_ids: &[String]) -> Box<dyn Any + Send> {
        Box::new(RoicWaccComputeState::default())
    }

    fn compute(&self, context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let requested = [ROIC_WACC_STATE_GROWTH_ID.to_string()];
        compute_requested(&requested, context, data)?
            .into_iter()
            .find(|series| series.spec.id == ROIC_WACC_STATE_GROWTH_ID)
            .ok_or_else(|| err("ROIC-WACC provider did not return roic_wacc_state_growth"))
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
            .downcast_mut::<RoicWaccComputeState>()
            .ok_or_else(|| err("ROIC-WACC provider received incompatible state"))?;
        compute_requested_stateful(requested_ids, context, data, state)
    }
}

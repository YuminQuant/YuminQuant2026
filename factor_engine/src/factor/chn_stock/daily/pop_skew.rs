use std::any::Any;

use crate::core::{FactorContext, FactorSeries, FactorSpec};
use crate::data::DataPool;
use crate::error::{err, Result};
use crate::factor::common::dbzq_profit_skew::{
    compute_requested, compute_requested_stateful, spec, ProfitSkewComputeState, ProfitSkewOutput,
    POP_SKEW_ID, PROVIDER_KEY,
};
use crate::factor::{Factor, FactorUpdatePolicy};

pub struct StockDailyPopSkew;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyPopSkew)
}

impl Factor for StockDailyPopSkew {
    fn spec(&self) -> FactorSpec {
        spec(ProfitSkewOutput::PopSkew)
    }

    fn compute_provider_key(&self) -> String {
        PROVIDER_KEY.to_string()
    }

    fn update_policy(&self) -> FactorUpdatePolicy {
        FactorUpdatePolicy::FinancialEventStateDailyFast
    }

    fn initial_compute_state(&self, _requested_ids: &[String]) -> Box<dyn Any + Send> {
        Box::new(ProfitSkewComputeState::default())
    }

    fn compute(&self, context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let requested = [POP_SKEW_ID.to_string()];
        compute_requested(&requested, context, data)?
            .into_iter()
            .find(|series| series.spec.id == POP_SKEW_ID)
            .ok_or_else(|| err("profit skew provider did not return pop_skew"))
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
            .downcast_mut::<ProfitSkewComputeState>()
            .ok_or_else(|| err("profit skew provider received incompatible state"))?;
        compute_requested_stateful(requested_ids, context, data, state)
    }
}

use std::any::Any;

use crate::core::{FactorContext, FactorSeries, FactorSpec};
use crate::data::DataPool;
use crate::error::{err, Result};
use crate::factor::common::financial_similarity::{
    compute_requested, compute_requested_stateful, spec, FinancialSimilarityComputeState,
    FinancialSimilarityOutput, F_MOMENTUM_80PEC_ID, PROVIDER_KEY,
};
use crate::factor::{Factor, FactorUpdatePolicy};

pub struct StockDailyFMomentum80pec;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyFMomentum80pec)
}

impl Factor for StockDailyFMomentum80pec {
    fn spec(&self) -> FactorSpec {
        spec(FinancialSimilarityOutput::FMomentum80Pec)
    }

    fn compute_provider_key(&self) -> String {
        PROVIDER_KEY.to_string()
    }

    fn update_policy(&self) -> FactorUpdatePolicy {
        FactorUpdatePolicy::FinancialEventStateDailyFast
    }

    fn initial_compute_state(&self, _requested_ids: &[String]) -> Box<dyn Any + Send> {
        Box::new(FinancialSimilarityComputeState::default())
    }

    fn compute(&self, context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let requested = [F_MOMENTUM_80PEC_ID.to_string()];
        compute_requested(&requested, context, data)?
            .into_iter()
            .find(|series| series.spec.id == F_MOMENTUM_80PEC_ID)
            .ok_or_else(|| err("financial similarity provider did not return f_momentum_80pec"))
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
            .downcast_mut::<FinancialSimilarityComputeState>()
            .ok_or_else(|| err("financial similarity provider received incompatible state"))?;
        compute_requested_stateful(requested_ids, context, data, state)
    }
}

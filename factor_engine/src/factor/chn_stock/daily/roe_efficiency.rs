use std::any::Any;

use crate::core::{FactorContext, FactorSeries, FactorSpec};
use crate::data::DataPool;
use crate::error::{err, Result};
use crate::factor::common::dbzq_financial_efficiency::{
    compute_requested, compute_requested_stateful, spec, FinancialEfficiencyComputeState,
    FinancialEfficiencyOutput, PROVIDER_KEY, ROE_EFFICIENCY_ID,
};
use crate::factor::{Factor, FactorUpdatePolicy};

pub struct StockDailyRoeEfficiency;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyRoeEfficiency)
}

impl Factor for StockDailyRoeEfficiency {
    fn spec(&self) -> FactorSpec {
        spec(FinancialEfficiencyOutput::RoeEfficiency)
    }

    fn compute_provider_key(&self) -> String {
        PROVIDER_KEY.to_string()
    }

    fn update_policy(&self) -> FactorUpdatePolicy {
        FactorUpdatePolicy::FinancialEventSnapshot
    }

    fn initial_compute_state(&self, _requested_ids: &[String]) -> Box<dyn Any + Send> {
        Box::new(FinancialEfficiencyComputeState::default())
    }

    fn compute(&self, context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let requested = [ROE_EFFICIENCY_ID.to_string()];
        compute_requested(&requested, context, data)?
            .into_iter()
            .find(|series| series.spec.id == ROE_EFFICIENCY_ID)
            .ok_or_else(|| err("financial efficiency provider did not return roe_efficiency"))
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
            .downcast_mut::<FinancialEfficiencyComputeState>()
            .ok_or_else(|| err("financial efficiency provider received incompatible state"))?;
        compute_requested_stateful(requested_ids, context, data, state)
    }
}

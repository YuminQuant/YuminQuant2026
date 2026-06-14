use std::any::Any;

use crate::core::{FactorContext, FactorSeries, FactorSpec};
use crate::data::DataPool;
use crate::error::{err, Result};
use crate::factor::chn_stock::daily::comprehensive_profitability::{
    compute_requested, compute_requested_stateful, spec, ComprehensiveProfitabilityOutput,
    ComprehensiveProfitabilityState, PROVIDER_KEY, STABLE_ROE_ID,
};
use crate::factor::{Factor, FactorUpdatePolicy};

pub struct StockDailyStableRoe;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyStableRoe)
}

impl Factor for StockDailyStableRoe {
    fn spec(&self) -> FactorSpec {
        spec(ComprehensiveProfitabilityOutput::StableRoe)
    }

    fn compute_provider_key(&self) -> String {
        PROVIDER_KEY.to_string()
    }

    fn update_policy(&self) -> FactorUpdatePolicy {
        FactorUpdatePolicy::FinancialEventSnapshot
    }

    fn initial_compute_state(&self, _requested_ids: &[String]) -> Box<dyn Any + Send> {
        Box::new(ComprehensiveProfitabilityState::default())
    }

    fn compute(&self, context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let requested = [STABLE_ROE_ID.to_string()];
        compute_requested(&requested, context, data)?
            .into_iter()
            .find(|series| series.spec.id == STABLE_ROE_ID)
            .ok_or_else(|| err("comprehensive profitability provider did not return stable_roe"))
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
            .downcast_mut::<ComprehensiveProfitabilityState>()
            .ok_or_else(|| {
                err("comprehensive profitability provider received incompatible state")
            })?;
        compute_requested_stateful(requested_ids, context, data, state)
    }
}

use std::any::Any;

use crate::core::{FactorContext, FactorSeries, FactorSpec};
use crate::data::DataPool;
use crate::error::{err, Result};
use crate::factor::common::dbzq_profit_skew::{
    compute_requested, compute_requested_stateful, spec, ProfitSkewComputeState, ProfitSkewOutput,
    OP_MARGIN_TTM_SKEW_ID, PROVIDER_KEY,
};
use crate::factor::{Factor, FactorUpdatePolicy};

pub struct StockDailyOpMarginTtmSkew;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyOpMarginTtmSkew)
}

impl Factor for StockDailyOpMarginTtmSkew {
    fn spec(&self) -> FactorSpec {
        spec(ProfitSkewOutput::OpMarginTtmSkew)
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
        let requested = [OP_MARGIN_TTM_SKEW_ID.to_string()];
        compute_requested(&requested, context, data)?
            .into_iter()
            .find(|series| series.spec.id == OP_MARGIN_TTM_SKEW_ID)
            .ok_or_else(|| err("profit skew provider did not return op_margin_ttm_skew"))
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

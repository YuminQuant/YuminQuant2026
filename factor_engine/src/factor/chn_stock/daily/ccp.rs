use std::any::Any;

use crate::core::{FactorContext, FactorSeries, FactorSpec};
use crate::data::DataPool;
use crate::error::{err, Result};
use crate::factor::common::hazq_equity_composition::{
    compute_requested, compute_requested_stateful, spec, HazqEquityCompositionComputeState,
    HazqEquityCompositionOutput, CCP_ID, PROVIDER_KEY,
};
use crate::factor::{Factor, FactorUpdatePolicy};

pub struct StockDailyCcp;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyCcp)
}

impl Factor for StockDailyCcp {
    fn spec(&self) -> FactorSpec {
        spec(HazqEquityCompositionOutput::Ccp)
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
        let requested = [CCP_ID.to_string()];
        compute_requested(&requested, context, data)?
            .into_iter()
            .find(|series| series.spec.id == CCP_ID)
            .ok_or_else(|| err("HAZQ equity composition provider did not return ccp"))
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

#[cfg(test)]
mod tests {
    use crate::factor::chn_stock::daily::rep::StockDailyRep;

    use super::*;

    #[test]
    fn hazq_rep_and_ccp_wrappers_share_provider_key_state_and_daily_fast_policy() {
        let rep = StockDailyRep;
        let ccp = StockDailyCcp;

        assert_eq!(rep.compute_provider_key(), ccp.compute_provider_key());
        assert_eq!(rep.compute_provider_key(), PROVIDER_KEY);
        assert_eq!(
            rep.update_policy(),
            FactorUpdatePolicy::FinancialEventStateDailyFast
        );
        assert_eq!(
            ccp.update_policy(),
            FactorUpdatePolicy::FinancialEventStateDailyFast
        );
        assert!(rep
            .initial_compute_state(&[])
            .is::<HazqEquityCompositionComputeState>());
        assert!(ccp
            .initial_compute_state(&[])
            .is::<HazqEquityCompositionComputeState>());
    }
}

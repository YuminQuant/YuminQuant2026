use crate::core::{FactorContext, FactorSeries, FactorSpec};
use crate::data::DataPool;
use crate::error::{err, Result};
use crate::factor::common::financial_similarity::{
    compute_requested, spec, FinancialSimilarityOutput, F_MOMENTUM_80PEC_ID, PROVIDER_KEY,
};
use crate::factor::Factor;

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
}

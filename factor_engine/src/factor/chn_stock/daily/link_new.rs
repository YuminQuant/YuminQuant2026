use crate::core::{FactorContext, FactorSeries, FactorSpec};
use crate::data::DataPool;
use crate::error::{err, Result};
use crate::factor::common::financial_similarity::{
    compute_requested, spec, FinancialSimilarityOutput, LINK_NEW_ID, PROVIDER_KEY,
};
use crate::factor::Factor;

pub struct StockDailyLinkNew;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyLinkNew)
}

impl Factor for StockDailyLinkNew {
    fn spec(&self) -> FactorSpec {
        spec(FinancialSimilarityOutput::LinkNew)
    }

    fn compute_provider_key(&self) -> String {
        PROVIDER_KEY.to_string()
    }

    fn compute(&self, context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let requested = [LINK_NEW_ID.to_string()];
        compute_requested(&requested, context, data)?
            .into_iter()
            .find(|series| series.spec.id == LINK_NEW_ID)
            .ok_or_else(|| err("financial similarity provider did not return link_new"))
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

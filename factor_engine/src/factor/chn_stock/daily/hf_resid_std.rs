use crate::core::{
    FactorContext, FactorSeries, FactorSpec, IntradayDailyRawSeries, IntradayDailyRawSpec,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::gfzq_hf_resid_std;
use crate::factor::Factor;

pub struct StockDailyHfResidStd;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyHfResidStd)
}

impl Factor for StockDailyHfResidStd {
    fn spec(&self) -> FactorSpec {
        gfzq_hf_resid_std::factor_spec()
    }

    fn intraday_raw_specs(&self) -> Vec<IntradayDailyRawSpec> {
        gfzq_hf_resid_std::raw_specs()
    }

    fn intraday_raw_provider_key(&self, _raw_id: &str) -> String {
        gfzq_hf_resid_std::PROVIDER_KEY.to_string()
    }

    fn minute_compute_many(
        &self,
        raw_ids: &[String],
        context: &FactorContext,
        data: &DataPool,
    ) -> Result<Vec<IntradayDailyRawSeries>> {
        gfzq_hf_resid_std::minute_compute_many(raw_ids, context, data)
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        gfzq_hf_resid_std::compute_factor(data)
    }
}

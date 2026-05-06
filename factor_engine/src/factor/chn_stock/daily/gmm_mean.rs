use crate::core::{
    FactorContext, FactorSeries, FactorSpec, IntradayDailyRawSeries, IntradayDailyRawSpec,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::stock_daily_raw_ids::GMM_MEAN_RAW_ID;
use crate::factor::common::xyzq_extreme_gmm::{
    self, XyzqExtremeGmmFactorDef, XyzqExtremeGmmRawFamily,
};
use crate::factor::Factor;

const DEF: XyzqExtremeGmmFactorDef = XyzqExtremeGmmFactorDef {
    id: "gmm_mean",
    alias: "gmm_mean",
    name: "gmm_mean",
    raw_id: GMM_MEAN_RAW_ID,
    smooth_window: xyzq_extreme_gmm::default_smooth_window(),
};

pub struct StockDailyGmmMean;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyGmmMean)
}

impl Factor for StockDailyGmmMean {
    fn spec(&self) -> FactorSpec {
        xyzq_extreme_gmm::factor_spec(DEF)
    }

    fn intraday_raw_specs(&self) -> Vec<IntradayDailyRawSpec> {
        xyzq_extreme_gmm::gmm_raw_specs()
    }

    fn intraday_raw_provider_key(&self, _raw_id: &str) -> String {
        "xyzq_gmm_return_provider".to_string()
    }

    fn minute_compute(
        &self,
        raw_id: &str,
        context: &FactorContext,
        data: &DataPool,
    ) -> Result<Option<IntradayDailyRawSeries>> {
        let raw_ids = vec![raw_id.to_string()];
        Ok(xyzq_extreme_gmm::minute_compute_many_for(
            &raw_ids,
            context,
            data,
            XyzqExtremeGmmRawFamily::GmmReturn,
        )?
        .into_iter()
        .next())
    }

    fn minute_compute_many(
        &self,
        raw_ids: &[String],
        context: &FactorContext,
        data: &DataPool,
    ) -> Result<Vec<IntradayDailyRawSeries>> {
        xyzq_extreme_gmm::minute_compute_many_for(
            raw_ids,
            context,
            data,
            XyzqExtremeGmmRawFamily::GmmReturn,
        )
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        xyzq_extreme_gmm::compute_factor(DEF, data)
    }
}

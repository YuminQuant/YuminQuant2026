use crate::core::{
    FactorContext, FactorSeries, FactorSpec, IntradayDailyRawSeries, IntradayDailyRawSpec,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::stock_daily_raw_ids::LH_RTN_DIFF_RAW_ID;
use crate::factor::common::xyzq_intraday_contrast::{
    self, XyzqIntradayContrastFactorDef, XyzqIntradayContrastRawFamily,
};
use crate::factor::Factor;

const DEF: XyzqIntradayContrastFactorDef = XyzqIntradayContrastFactorDef {
    id: "lh_rtn_diff",
    alias: "lh_rtnDiff",
    name: "lh_rtnDiff",
    raw_id: LH_RTN_DIFF_RAW_ID,
};

pub struct StockDailyLhRtnDiff;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyLhRtnDiff)
}

impl Factor for StockDailyLhRtnDiff {
    fn spec(&self) -> FactorSpec {
        xyzq_intraday_contrast::factor_spec(DEF)
    }

    fn intraday_raw_specs(&self) -> Vec<IntradayDailyRawSpec> {
        xyzq_intraday_contrast::lh_raw_specs()
    }

    fn intraday_raw_provider_key(&self, _raw_id: &str) -> String {
        "xyzq_lh_intraday_diff_provider".to_string()
    }

    fn minute_compute(
        &self,
        raw_id: &str,
        context: &FactorContext,
        data: &DataPool,
    ) -> Result<Option<IntradayDailyRawSeries>> {
        let raw_ids = vec![raw_id.to_string()];
        Ok(xyzq_intraday_contrast::minute_compute_many_for(
            &raw_ids,
            context,
            data,
            XyzqIntradayContrastRawFamily::LhIntradayDiff,
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
        xyzq_intraday_contrast::minute_compute_many_for(
            raw_ids,
            context,
            data,
            XyzqIntradayContrastRawFamily::LhIntradayDiff,
        )
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        xyzq_intraday_contrast::compute_factor(DEF, data)
    }
}

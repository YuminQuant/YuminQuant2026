use crate::core::{
    FactorContext, FactorSeries, FactorSpec, IntradayDailyRawSeries, IntradayDailyRawSpec,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::stock_daily_raw_ids::DIFF_IDX_RAW_ID;
use crate::factor::common::xyzq_intraday_contrast::{
    self, XyzqIntradayContrastFactorDef, XyzqIntradayContrastRawFamily,
};
use crate::factor::Factor;

const DEF: XyzqIntradayContrastFactorDef = XyzqIntradayContrastFactorDef {
    id: "diff_idx",
    alias: "diff_idx",
    name: "diff_idx",
    raw_id: DIFF_IDX_RAW_ID,
};

pub struct StockDailyDiffIdx;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyDiffIdx)
}

impl Factor for StockDailyDiffIdx {
    fn spec(&self) -> FactorSpec {
        xyzq_intraday_contrast::factor_spec(DEF)
    }

    fn intraday_raw_specs(&self) -> Vec<IntradayDailyRawSpec> {
        xyzq_intraday_contrast::high_low_raw_specs()
    }

    fn intraday_raw_provider_key(&self, _raw_id: &str) -> String {
        "xyzq_high_low_timing_provider".to_string()
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
            XyzqIntradayContrastRawFamily::HighLowTiming,
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
            XyzqIntradayContrastRawFamily::HighLowTiming,
        )
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        xyzq_intraday_contrast::compute_factor(DEF, data)
    }
}

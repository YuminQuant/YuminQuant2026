use crate::core::{
    FactorContext, FactorSeries, FactorSpec, IntradayDailyRawSeries, IntradayDailyRawSpec,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::stock_daily_raw_ids::EX_RTN_MAX_VAL_RAW_ID;
use crate::factor::common::xyzq_extreme_gmm::{self, XyzqExtremeGmmFactorDef};
use crate::factor::Factor;

const DEF: XyzqExtremeGmmFactorDef = XyzqExtremeGmmFactorDef {
    id: "ex_rtn_max_val",
    alias: "exRtn_maxVal",
    name: "exRtn_maxVal",
    raw_id: EX_RTN_MAX_VAL_RAW_ID,
    smooth_window: xyzq_extreme_gmm::default_smooth_window(),
};

pub struct StockDailyExRtnMaxVal;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyExRtnMaxVal)
}

impl Factor for StockDailyExRtnMaxVal {
    fn spec(&self) -> FactorSpec {
        xyzq_extreme_gmm::factor_spec(DEF)
    }

    fn intraday_raw_specs(&self) -> Vec<IntradayDailyRawSpec> {
        xyzq_extreme_gmm::raw_specs()
    }

    fn minute_compute(
        &self,
        raw_id: &str,
        context: &FactorContext,
        data: &DataPool,
    ) -> Result<Option<IntradayDailyRawSeries>> {
        let raw_ids = vec![raw_id.to_string()];
        Ok(
            xyzq_extreme_gmm::minute_compute_many(&raw_ids, context, data)?
                .into_iter()
                .next(),
        )
    }

    fn minute_compute_many(
        &self,
        raw_ids: &[String],
        context: &FactorContext,
        data: &DataPool,
    ) -> Result<Vec<IntradayDailyRawSeries>> {
        xyzq_extreme_gmm::minute_compute_many(raw_ids, context, data)
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        xyzq_extreme_gmm::compute_factor(DEF, data)
    }
}

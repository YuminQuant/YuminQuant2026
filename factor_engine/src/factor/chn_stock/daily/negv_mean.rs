use crate::core::{
    FactorContext, FactorSeries, FactorSpec, IntradayDailyRawSeries, IntradayDailyRawSpec,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::stock_daily_raw_ids::NEGV_MEAN_RAW_ID;
use crate::factor::common::xyzq_vshape_structure::{
    self, XyzqVshapeFactorDef, XyzqVshapeFactorKind,
};
use crate::factor::Factor;

const DEF: XyzqVshapeFactorDef = XyzqVshapeFactorDef {
    id: "negv_mean",
    alias: "negV_mean",
    name: "negV_mean",
    kind: XyzqVshapeFactorKind::RollingMean {
        raw_id: NEGV_MEAN_RAW_ID,
        window: xyzq_vshape_structure::default_window(),
    },
};

pub struct StockDailyNegvMean;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyNegvMean)
}

impl Factor for StockDailyNegvMean {
    fn spec(&self) -> FactorSpec {
        xyzq_vshape_structure::factor_spec(DEF)
    }

    fn intraday_raw_specs(&self) -> Vec<IntradayDailyRawSpec> {
        xyzq_vshape_structure::raw_specs()
    }

    fn intraday_raw_provider_key(&self, _raw_id: &str) -> String {
        "xyzq_vshape_structure_provider".to_string()
    }

    fn minute_compute(
        &self,
        raw_id: &str,
        context: &FactorContext,
        data: &DataPool,
    ) -> Result<Option<IntradayDailyRawSeries>> {
        let raw_ids = vec![raw_id.to_string()];
        Ok(
            xyzq_vshape_structure::minute_compute_many(&raw_ids, context, data)?
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
        xyzq_vshape_structure::minute_compute_many(raw_ids, context, data)
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        xyzq_vshape_structure::compute_factor(DEF, data)
    }
}

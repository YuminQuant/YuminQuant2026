use crate::core::{
    FactorContext, FactorSeries, FactorSpec, IntradayDailyRawSeries, IntradayDailyRawSpec,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::stock_daily_raw_ids::VOLROC_SKEW_RAW_ID;
use crate::factor::common::xyzq_volume_shape::{
    self, default_window, XyzqVolumeAggregation, XyzqVolumeFactorDef, XyzqVolumeRawFamily,
};
use crate::factor::Factor;

const DEF: XyzqVolumeFactorDef = XyzqVolumeFactorDef {
    id: "volroc_skew",
    alias: "volroc_skew",
    name: "volroc_skew",
    raw_id: VOLROC_SKEW_RAW_ID,
    window: default_window(),
    aggregation: XyzqVolumeAggregation::Mean,
};

pub struct StockDailyVolrocSkew;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyVolrocSkew)
}

impl Factor for StockDailyVolrocSkew {
    fn spec(&self) -> FactorSpec {
        xyzq_volume_shape::factor_spec(DEF)
    }

    fn intraday_raw_specs(&self) -> Vec<IntradayDailyRawSpec> {
        xyzq_volume_shape::volroc_raw_specs()
    }

    fn intraday_raw_provider_key(&self, _raw_id: &str) -> String {
        "xyzq_volroc_shape_provider".to_string()
    }

    fn minute_compute(
        &self,
        raw_id: &str,
        context: &FactorContext,
        data: &DataPool,
    ) -> Result<Option<IntradayDailyRawSeries>> {
        let raw_ids = vec![raw_id.to_string()];
        Ok(xyzq_volume_shape::minute_compute_many_for(
            &raw_ids,
            context,
            data,
            XyzqVolumeRawFamily::VolrocShape,
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
        xyzq_volume_shape::minute_compute_many_for(
            raw_ids,
            context,
            data,
            XyzqVolumeRawFamily::VolrocShape,
        )
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        xyzq_volume_shape::compute_factor(DEF, data)
    }
}

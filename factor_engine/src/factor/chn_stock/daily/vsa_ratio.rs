use crate::core::{
    FactorContext, FactorSeries, FactorSpec, IntradayDailyRawSeries, IntradayDailyRawSpec,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::stock_daily_raw_ids::VSA_RATIO_RAW_ID;
use crate::factor::common::xyzq_volume_shape::{
    self, default_window, XyzqVolumeAggregation, XyzqVolumeFactorDef, XyzqVolumeRawFamily,
};
use crate::factor::Factor;

const DEF: XyzqVolumeFactorDef = XyzqVolumeFactorDef {
    id: "vsa_ratio",
    alias: "vsa_ratio",
    name: "vsa_ratio",
    raw_id: VSA_RATIO_RAW_ID,
    window: default_window(),
    aggregation: XyzqVolumeAggregation::Mean,
};

pub struct StockDailyVsaRatio;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyVsaRatio)
}

impl Factor for StockDailyVsaRatio {
    fn spec(&self) -> FactorSpec {
        xyzq_volume_shape::factor_spec(DEF)
    }

    fn intraday_raw_specs(&self) -> Vec<IntradayDailyRawSpec> {
        xyzq_volume_shape::vsa_raw_specs()
    }

    fn intraday_raw_provider_key(&self, _raw_id: &str) -> String {
        "xyzq_vsa_provider".to_string()
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
            XyzqVolumeRawFamily::Vsa,
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
        xyzq_volume_shape::minute_compute_many_for(raw_ids, context, data, XyzqVolumeRawFamily::Vsa)
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        xyzq_volume_shape::compute_factor(DEF, data)
    }
}

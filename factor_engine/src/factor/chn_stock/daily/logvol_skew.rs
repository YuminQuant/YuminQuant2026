use crate::core::{
    FactorContext, FactorSeries, FactorSpec, IntradayDailyRawSeries, IntradayDailyRawSpec,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::stock_daily_raw_ids::LOGVOL_SKEW_RAW_ID;
use crate::factor::common::xyzq_volume_shape::{
    self, default_window, XyzqVolumeAggregation, XyzqVolumeFactorDef, XyzqVolumeRawFamily,
};
use crate::factor::Factor;

const DEF: XyzqVolumeFactorDef = XyzqVolumeFactorDef {
    id: "logvol_skew",
    alias: "logvol_skew",
    name: "logvol_skew",
    raw_id: LOGVOL_SKEW_RAW_ID,
    window: default_window(),
    aggregation: XyzqVolumeAggregation::Mean,
};

pub struct StockDailyLogvolSkew;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyLogvolSkew)
}

impl Factor for StockDailyLogvolSkew {
    fn spec(&self) -> FactorSpec {
        xyzq_volume_shape::factor_spec(DEF)
    }

    fn intraday_raw_specs(&self) -> Vec<IntradayDailyRawSpec> {
        xyzq_volume_shape::logvol_raw_specs()
    }

    fn intraday_raw_provider_key(&self, _raw_id: &str) -> String {
        "xyzq_logvol_shape_provider".to_string()
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
            XyzqVolumeRawFamily::LogvolShape,
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
            XyzqVolumeRawFamily::LogvolShape,
        )
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        xyzq_volume_shape::compute_factor(DEF, data)
    }
}

use crate::core::{
    FactorContext, FactorSeries, FactorSpec, IntradayDailyRawSeries, IntradayDailyRawSpec,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::stock_daily_raw_ids::HIGH_STD_RTN_MEAN_RAW_ID;
use crate::factor::common::xyzq_serial_structure::{
    self, XyzqSerialAggregation, XyzqSerialFactorDef, XyzqSerialRawFamily,
};
use crate::factor::Factor;

const DEF: XyzqSerialFactorDef = XyzqSerialFactorDef {
    id: "high_std_rtn_mean",
    alias: "highStdRtn_mean",
    name: "highStdRtn_mean",
    raw_id: HIGH_STD_RTN_MEAN_RAW_ID,
    aggregation: XyzqSerialAggregation::Mean,
};

pub struct StockDailyHighStdRtnMean;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyHighStdRtnMean)
}

impl Factor for StockDailyHighStdRtnMean {
    fn spec(&self) -> FactorSpec {
        xyzq_serial_structure::factor_spec(DEF)
    }

    fn intraday_raw_specs(&self) -> Vec<IntradayDailyRawSpec> {
        xyzq_serial_structure::high_std_rtn_raw_specs()
    }

    fn intraday_raw_provider_key(&self, _raw_id: &str) -> String {
        "xyzq_high_std_rtn_provider".to_string()
    }

    fn minute_compute(
        &self,
        raw_id: &str,
        context: &FactorContext,
        data: &DataPool,
    ) -> Result<Option<IntradayDailyRawSeries>> {
        let raw_ids = vec![raw_id.to_string()];
        Ok(xyzq_serial_structure::minute_compute_many_for(
            &raw_ids,
            context,
            data,
            XyzqSerialRawFamily::HighStdRtn,
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
        xyzq_serial_structure::minute_compute_many_for(
            raw_ids,
            context,
            data,
            XyzqSerialRawFamily::HighStdRtn,
        )
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        xyzq_serial_structure::compute_factor(DEF, data)
    }
}

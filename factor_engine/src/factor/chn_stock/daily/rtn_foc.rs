use crate::core::{
    FactorContext, FactorSeries, FactorSpec, IntradayDailyRawSeries, IntradayDailyRawSpec,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::stock_daily_raw_ids::RTN_FOC_RAW_ID;
use crate::factor::common::xyzq_serial_structure::{
    self, XyzqSerialAggregation, XyzqSerialFactorDef,
};
use crate::factor::Factor;

const DEF: XyzqSerialFactorDef = XyzqSerialFactorDef {
    id: "rtn_foc",
    alias: "rtn_foc",
    name: "rtn_foc",
    raw_id: RTN_FOC_RAW_ID,
    aggregation: XyzqSerialAggregation::Mean,
};

pub struct StockDailyRtnFoc;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyRtnFoc)
}

impl Factor for StockDailyRtnFoc {
    fn spec(&self) -> FactorSpec {
        xyzq_serial_structure::factor_spec(DEF)
    }

    fn intraday_raw_specs(&self) -> Vec<IntradayDailyRawSpec> {
        xyzq_serial_structure::raw_specs()
    }

    fn minute_compute(
        &self,
        raw_id: &str,
        context: &FactorContext,
        data: &DataPool,
    ) -> Result<Option<IntradayDailyRawSeries>> {
        let raw_ids = vec![raw_id.to_string()];
        Ok(
            xyzq_serial_structure::minute_compute_many(&raw_ids, context, data)?
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
        xyzq_serial_structure::minute_compute_many(raw_ids, context, data)
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        xyzq_serial_structure::compute_factor(DEF, data)
    }
}

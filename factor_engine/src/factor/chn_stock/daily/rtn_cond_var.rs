use crate::core::{
    FactorContext, FactorSeries, FactorSpec, IntradayDailyRawSeries, IntradayDailyRawSpec,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::stock_daily_raw_ids::RTN_COND_VAR_RAW_ID;
use crate::factor::common::xyzq_serial_structure::{
    self, XyzqSerialAggregation, XyzqSerialFactorDef, XyzqSerialRawFamily,
};
use crate::factor::Factor;

const DEF: XyzqSerialFactorDef = XyzqSerialFactorDef {
    id: "rtn_cond_var",
    alias: "rtn_condVaR",
    name: "rtn_condVaR",
    raw_id: RTN_COND_VAR_RAW_ID,
    aggregation: XyzqSerialAggregation::Std,
};

pub struct StockDailyRtnCondVar;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyRtnCondVar)
}

impl Factor for StockDailyRtnCondVar {
    fn spec(&self) -> FactorSpec {
        xyzq_serial_structure::factor_spec(DEF)
    }

    fn intraday_raw_specs(&self) -> Vec<IntradayDailyRawSpec> {
        xyzq_serial_structure::cond_var_raw_specs()
    }

    fn intraday_raw_provider_key(&self, _raw_id: &str) -> String {
        "xyzq_cond_var_provider".to_string()
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
            XyzqSerialRawFamily::CondVar,
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
            XyzqSerialRawFamily::CondVar,
        )
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        xyzq_serial_structure::compute_factor(DEF, data)
    }
}

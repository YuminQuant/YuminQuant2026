use crate::core::{
    FactorContext, FactorSeries, FactorSpec, IntradayDailyRawSeries, IntradayDailyRawSpec,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::stock_daily_raw_ids::RVC_COR_RAW_ID;
use crate::factor::common::xyzq_flow_structure::{self, XyzqFlowFactorDef, XyzqFlowRawFamily};
use crate::factor::Factor;

const DEF: XyzqFlowFactorDef = XyzqFlowFactorDef {
    id: "rvc_cor",
    alias: "rvc_cor",
    name: "rvc_cor",
    raw_id: RVC_COR_RAW_ID,
    window: xyzq_flow_structure::default_window(),
};

pub struct StockDailyRvcCor;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyRvcCor)
}

impl Factor for StockDailyRvcCor {
    fn spec(&self) -> FactorSpec {
        xyzq_flow_structure::factor_spec(DEF)
    }

    fn intraday_raw_specs(&self) -> Vec<IntradayDailyRawSpec> {
        xyzq_flow_structure::correlation_raw_specs()
    }

    fn intraday_raw_provider_key(&self, _raw_id: &str) -> String {
        "xyzq_flow_correlation_provider".to_string()
    }

    fn minute_compute(
        &self,
        raw_id: &str,
        context: &FactorContext,
        data: &DataPool,
    ) -> Result<Option<IntradayDailyRawSeries>> {
        let raw_ids = vec![raw_id.to_string()];
        Ok(xyzq_flow_structure::minute_compute_many_for(
            &raw_ids,
            context,
            data,
            XyzqFlowRawFamily::Correlation,
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
        xyzq_flow_structure::minute_compute_many_for(
            raw_ids,
            context,
            data,
            XyzqFlowRawFamily::Correlation,
        )
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        xyzq_flow_structure::compute_factor(DEF, data)
    }
}

use crate::core::{
    FactorContext, FactorSeries, FactorSpec, IntradayDailyRawSeries, IntradayDailyRawSpec,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::stock_daily_raw_ids::TE_V2R_RAW_ID;
use crate::factor::common::xyzq_flow_structure::{self, XyzqFlowFactorDef, XyzqFlowRawFamily};
use crate::factor::Factor;

const DEF: XyzqFlowFactorDef = XyzqFlowFactorDef {
    id: "te_v2r",
    alias: "te_v2r",
    name: "te_v2r",
    raw_id: TE_V2R_RAW_ID,
    window: xyzq_flow_structure::te_window(),
};

pub struct StockDailyTeV2r;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyTeV2r)
}

impl Factor for StockDailyTeV2r {
    fn spec(&self) -> FactorSpec {
        xyzq_flow_structure::factor_spec(DEF)
    }

    fn intraday_raw_specs(&self) -> Vec<IntradayDailyRawSpec> {
        xyzq_flow_structure::transfer_entropy_raw_specs()
    }

    fn intraday_raw_provider_key(&self, _raw_id: &str) -> String {
        "xyzq_flow_transfer_entropy_provider".to_string()
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
            XyzqFlowRawFamily::TransferEntropy,
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
            XyzqFlowRawFamily::TransferEntropy,
        )
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        xyzq_flow_structure::compute_factor(DEF, data)
    }
}

use crate::core::{
    FactorContext, FactorSeries, FactorSpec, IntradayDailyRawSeries, IntradayDailyRawSpec,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::xyzq_domain_structure::{
    self, XyzqDomainFactorDef, XyzqDomainFactorKind, XyzqDomainFeature, XyzqDomainRawFamily,
};
use crate::factor::Factor;

const DEF: XyzqDomainFactorDef = XyzqDomainFactorDef {
    id: "intra_ds_rtn_by_t",
    alias: "intraDSRtn_byT",
    name: "intraDSRtn_byT",
    kind: XyzqDomainFactorKind::IntraDs {
        feature: XyzqDomainFeature::TimeRtn,
    },
};

pub struct StockDailyIntraDsRtnByT;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyIntraDsRtnByT)
}

impl Factor for StockDailyIntraDsRtnByT {
    fn spec(&self) -> FactorSpec {
        xyzq_domain_structure::factor_spec(DEF)
    }

    fn intraday_raw_specs(&self) -> Vec<IntradayDailyRawSpec> {
        xyzq_domain_structure::raw_specs_for_family(XyzqDomainRawFamily::Time)
    }

    fn intraday_raw_provider_key(&self, _raw_id: &str) -> String {
        "xyzq_domain_time_provider".to_string()
    }

    fn minute_compute(
        &self,
        raw_id: &str,
        context: &FactorContext,
        data: &DataPool,
    ) -> Result<Option<IntradayDailyRawSeries>> {
        let raw_ids = vec![raw_id.to_string()];
        Ok(xyzq_domain_structure::minute_compute_many_for(
            &raw_ids,
            context,
            data,
            XyzqDomainRawFamily::Time,
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
        xyzq_domain_structure::minute_compute_many_for(
            raw_ids,
            context,
            data,
            XyzqDomainRawFamily::Time,
        )
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        xyzq_domain_structure::compute_factor(DEF, data)
    }
}

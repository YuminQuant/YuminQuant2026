use crate::core::{FactorContext, FactorSeries, FactorSpec, IntradayDailyRawSpec};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::kyzq_apm::{self, KyzqApmFactorDef, KyzqApmKind};
use crate::factor::Factor;

const DEF: KyzqApmFactorDef = KyzqApmFactorDef {
    id: "apm",
    alias: "APM",
    name: "APM",
    kind: KyzqApmKind::Apm,
};

pub struct StockDailyApm;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyApm)
}

impl Factor for StockDailyApm {
    fn spec(&self) -> FactorSpec {
        kyzq_apm::factor_spec(DEF)
    }

    fn intraday_raw_specs(&self) -> Vec<IntradayDailyRawSpec> {
        kyzq_apm::raw_specs()
    }

    fn intraday_raw_provider_key(&self, _raw_id: &str) -> String {
        kyzq_apm::PROVIDER_KEY.to_string()
    }

    fn minute_compute_many(
        &self,
        raw_ids: &[String],
        context: &FactorContext,
        data: &DataPool,
    ) -> Result<Vec<crate::core::IntradayDailyRawSeries>> {
        kyzq_apm::minute_compute_many(raw_ids, context, data)
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        kyzq_apm::compute_factor(DEF, data)
    }
}

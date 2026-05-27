use crate::core::{FactorContext, FactorSeries, FactorSpec};
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

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        kyzq_apm::compute_factor(DEF, data)
    }
}

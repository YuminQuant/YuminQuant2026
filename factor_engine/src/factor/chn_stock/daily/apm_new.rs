use crate::core::{FactorContext, FactorSeries, FactorSpec};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::kyzq_apm::{self, KyzqApmFactorDef, KyzqApmKind};
use crate::factor::Factor;

const DEF: KyzqApmFactorDef = KyzqApmFactorDef {
    id: "apm_new",
    alias: "APMnew",
    name: "APM New",
    kind: KyzqApmKind::ApmNew,
};

pub struct StockDailyApmNew;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyApmNew)
}

impl Factor for StockDailyApmNew {
    fn spec(&self) -> FactorSpec {
        kyzq_apm::factor_spec(DEF)
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        kyzq_apm::compute_factor(DEF, data)
    }
}

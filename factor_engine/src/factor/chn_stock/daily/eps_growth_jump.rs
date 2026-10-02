use crate::core::{FactorContext, FactorSeries, FactorSpec};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::dbzq_financial_surprise::{self, SurpriseKind};
use crate::factor::{Factor, FactorUpdatePolicy};

pub struct StockDailyEpsGrowthJump;
pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyEpsGrowthJump)
}
impl Factor for StockDailyEpsGrowthJump {
    fn spec(&self) -> FactorSpec {
        dbzq_financial_surprise::spec(SurpriseKind::EpsJump)
    }
    fn update_policy(&self) -> FactorUpdatePolicy {
        FactorUpdatePolicy::FinancialEventStateDailyFast
    }
    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        dbzq_financial_surprise::compute(SurpriseKind::EpsJump, data)
    }
}

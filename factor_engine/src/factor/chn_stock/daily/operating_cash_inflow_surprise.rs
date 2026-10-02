use crate::core::{FactorContext, FactorSeries, FactorSpec};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::dbzq_financial_surprise::{self, SurpriseKind};
use crate::factor::{Factor, FactorUpdatePolicy};

pub struct StockDailyOperatingCashInflowSurprise;
pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyOperatingCashInflowSurprise)
}
impl Factor for StockDailyOperatingCashInflowSurprise {
    fn spec(&self) -> FactorSpec {
        dbzq_financial_surprise::spec(SurpriseKind::CashInflow)
    }
    fn update_policy(&self) -> FactorUpdatePolicy {
        FactorUpdatePolicy::FinancialEventStateDailyFast
    }
    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        dbzq_financial_surprise::compute(SurpriseKind::CashInflow, data)
    }
}

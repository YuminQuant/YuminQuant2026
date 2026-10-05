use crate::factor::common::gdzq_financial_ts_residual::{GdzqFinancialTsResidual, Output};
use crate::factor::Factor;

pub type StockDailyRoeSurplusReserveTsResid = GdzqFinancialTsResidual;

pub fn create() -> Box<dyn Factor> {
    Box::new(GdzqFinancialTsResidual::new(Output::RoeSurplusReserve))
}

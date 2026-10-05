use crate::factor::common::financial_profit_trend::{FinancialProfitTrend, Output};
use crate::factor::Factor;

pub type StockDailyOcfa = FinancialProfitTrend;

pub fn create() -> Box<dyn Factor> {
    Box::new(FinancialProfitTrend::new(Output::Ocfa))
}

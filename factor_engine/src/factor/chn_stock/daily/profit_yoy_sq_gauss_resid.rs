use crate::factor::common::gaussian_financial::{GaussianFinancialFactor, GaussianFinancialOutput};
use crate::factor::Factor;

pub type StockDailyProfitYoySqGaussResid = GaussianFinancialFactor;

pub fn create() -> Box<dyn Factor> {
    Box::new(GaussianFinancialFactor::new(
        GaussianFinancialOutput::ProfitYoySq,
    ))
}

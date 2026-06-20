use crate::factor::common::gaussian_financial_ext::{
    GaussianFinancialExtFactor, GaussianFinancialExtOutput,
};
use crate::factor::Factor;

pub type StockDailyNetprofitComprParentGaussResid = GaussianFinancialExtFactor;

pub fn create() -> Box<dyn Factor> {
    Box::new(GaussianFinancialExtFactor::new(
        GaussianFinancialExtOutput::NetprofitComprParent,
    ))
}

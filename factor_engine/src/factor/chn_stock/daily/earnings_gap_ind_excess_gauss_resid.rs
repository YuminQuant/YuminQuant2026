use crate::factor::common::earnings_reaction_gaussian::{
    EarningsReactionGaussianFactor, EarningsReactionOutput,
};
use crate::factor::Factor;

pub type StockDailyEarningsGapIndExcessGaussResid = EarningsReactionGaussianFactor;

pub fn create() -> Box<dyn Factor> {
    Box::new(EarningsReactionGaussianFactor::new(
        EarningsReactionOutput::GapIndustryExcess,
    ))
}

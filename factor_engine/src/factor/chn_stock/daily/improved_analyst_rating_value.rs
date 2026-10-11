use crate::factor::common::htzq_analyst::{HtzqAnalyst, Output};
use crate::factor::Factor;

pub fn create() -> Box<dyn Factor> {
    Box::new(HtzqAnalyst(Output::RatingValue))
}

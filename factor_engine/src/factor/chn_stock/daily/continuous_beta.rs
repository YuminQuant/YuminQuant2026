use crate::factor::common::dfzq_high_frequency_beta::{HighFrequencyBeta, Output};
use crate::factor::Factor;

pub fn create() -> Box<dyn Factor> {
    Box::new(HighFrequencyBeta(Output::Continuous))
}

use crate::factor::common::dfzq_analyst_coverage::{AnalystCoverage, Output};
use crate::factor::Factor;

pub fn create() -> Box<dyn Factor> {
    Box::new(AnalystCoverage(Output::Anncov))
}

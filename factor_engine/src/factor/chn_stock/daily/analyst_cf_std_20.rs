use crate::factor::common::htzq_coverage_network::{CoverageNetwork, Kind, Output};
use crate::factor::Factor;

pub fn create() -> Box<dyn Factor> {
    Box::new(CoverageNetwork(Output(Kind::Volatility, 20)))
}

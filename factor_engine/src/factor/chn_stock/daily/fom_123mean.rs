use crate::factor::common::dfzq_analyst_monthly::{AnalystMonthly, Output};
use crate::factor::Factor;

pub fn create() -> Box<dyn Factor> {
    Box::new(AnalystMonthly(Output::Fom))
}

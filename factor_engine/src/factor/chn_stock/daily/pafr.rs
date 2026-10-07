use crate::factor::common::fzzq_pafr::{Output, Pafr};
use crate::factor::Factor;

pub fn create() -> Box<dyn Factor> {
    Box::new(Pafr(Output::Raw))
}

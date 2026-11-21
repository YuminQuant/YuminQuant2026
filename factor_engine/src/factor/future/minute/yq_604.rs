use crate::factor::{future::reversal, Factor};

pub fn create() -> Box<dyn Factor> {
    reversal::create(604)
}

use crate::factor::{future::flow, Factor};

pub fn create() -> Box<dyn Factor> {
    flow::create(12)
}

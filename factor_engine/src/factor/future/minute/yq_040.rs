use crate::factor::{future::trend, Factor};

pub fn create() -> Box<dyn Factor> {
    trend::create(14)
}

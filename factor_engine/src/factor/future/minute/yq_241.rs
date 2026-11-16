use crate::factor::{future::skewness, Factor};

pub fn create() -> Box<dyn Factor> {
    skewness::create(241)
}

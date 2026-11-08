use crate::factor::{future::aroon, Factor};

pub fn create() -> Box<dyn Factor> {
    aroon::create(1)
}

use crate::factor::{future::rsi, Factor};

pub fn create() -> Box<dyn Factor> {
    rsi::create(2)
}

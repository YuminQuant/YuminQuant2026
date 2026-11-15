use crate::factor::{future::variance_amplitude, Factor};

pub fn create() -> Box<dyn Factor> {
    variance_amplitude::create(3, 0)
}

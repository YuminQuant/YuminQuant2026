use crate::factor::{future::variance_amplitude, Factor};

pub fn create() -> Box<dyn Factor> {
    variance_amplitude::create(8, 0)
}

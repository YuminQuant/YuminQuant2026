use crate::factor::{future::moving_average, Factor};

pub fn create() -> Box<dyn Factor> {
    moving_average::create(24)
}

use crate::factor::{future::variance_timing, Factor};

pub fn create() -> Box<dyn Factor> {
    variance_timing::create(308)
}

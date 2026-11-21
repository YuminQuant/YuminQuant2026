use crate::factor::{future::snr, Factor};

pub fn create() -> Box<dyn Factor> {
    snr::create(266)
}

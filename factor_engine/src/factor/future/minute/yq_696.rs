use crate::factor::{future::cci, Factor};

pub fn create() -> Box<dyn Factor> {
    cci::create(696)
}

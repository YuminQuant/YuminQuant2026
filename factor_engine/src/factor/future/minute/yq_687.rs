use crate::factor::{future::wait_and_see, Factor};

pub fn create() -> Box<dyn Factor> {
    wait_and_see::create(687)
}

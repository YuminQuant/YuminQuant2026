use crate::factor::{future::runs, Factor};

pub fn create() -> Box<dyn Factor> {
    runs::create(540)
}

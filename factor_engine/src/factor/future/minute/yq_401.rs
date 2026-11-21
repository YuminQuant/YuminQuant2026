use crate::factor::{future::long_memory, Factor};

pub fn create() -> Box<dyn Factor> {
    long_memory::create(401)
}

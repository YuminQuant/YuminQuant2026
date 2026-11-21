use crate::factor::{future::ljung_box, Factor};

pub fn create() -> Box<dyn Factor> {
    ljung_box::create(318)
}

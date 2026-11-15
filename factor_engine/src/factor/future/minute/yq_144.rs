use crate::factor::{future::multiscale_volatility, Factor};

pub fn create() -> Box<dyn Factor> {
    multiscale_volatility::create_variant(2, 0)
}

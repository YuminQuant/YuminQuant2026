use crate::factor::common::price_state_signals::PriceState;
use crate::factor::Factor;

pub fn create() -> Box<dyn Factor> {
    Box::new(PriceState(0))
}

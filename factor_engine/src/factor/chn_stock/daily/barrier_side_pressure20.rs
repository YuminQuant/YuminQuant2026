use crate::factor::common::ghzq_price_digits::PriceDigits;
use crate::factor::Factor;

pub fn create() -> Box<dyn Factor> {
    Box::new(PriceDigits(9))
}

use crate::factor::common::dfzq_seasonal_surprise::{Output, SeasonalSurprise};
use crate::factor::Factor;

pub fn create() -> Box<dyn Factor> {
    Box::new(SeasonalSurprise(Output::Sur))
}

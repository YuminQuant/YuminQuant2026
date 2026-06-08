pub use crate::factor::chn_stock::daily::special_roa2::StockDailySpecialRoa1;
use crate::factor::Factor;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailySpecialRoa1)
}

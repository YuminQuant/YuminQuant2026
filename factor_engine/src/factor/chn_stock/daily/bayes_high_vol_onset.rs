use crate::factor::common::xyzq_trade_disagreement::TradeDisagreement;
use crate::factor::Factor;

pub fn create() -> Box<dyn Factor> {
    Box::new(TradeDisagreement(8))
}

use crate::factor::common::dfzq_daily_leaderboard::{DailyLeaderboard, Side};
use crate::factor::Factor;
pub fn create() -> Box<dyn Factor> {
    Box::new(DailyLeaderboard(Side::Loser))
}

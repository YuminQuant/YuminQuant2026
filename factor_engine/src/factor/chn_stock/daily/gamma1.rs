use crate::factor::common::stock_daily_raw_ids::GAMMA1_RAW_ID;
use crate::factor::common::xyzq_liquidity::LiquidityFamily;

crate::define_xyzq_liquidity_factor!(
    StockDailyGamma1,
    "gamma1",
    "Gamma1",
    "Gamma1",
    GAMMA1_RAW_ID,
    LiquidityFamily::OneMinute,
    40,
    40
);

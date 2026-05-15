use crate::factor::common::stock_daily_raw_ids::GAMMA3_RAW_ID;
use crate::factor::common::xyzq_liquidity::LiquidityFamily;

crate::define_xyzq_liquidity_factor!(
    StockDailyGamma3,
    "gamma3",
    "Gamma3",
    "Gamma3",
    GAMMA3_RAW_ID,
    LiquidityFamily::OneMinute,
    3,
    3
);

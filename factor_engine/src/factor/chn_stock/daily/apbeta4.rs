use crate::factor::common::stock_daily_raw_ids::APBETA4_RAW_ID;
use crate::factor::common::xyzq_liquidity::LiquidityFamily;

crate::define_xyzq_liquidity_factor!(
    StockDailyApbeta4,
    "apbeta4",
    "APBeta4",
    "APBeta4",
    APBETA4_RAW_ID,
    LiquidityFamily::OneMinute,
    5,
    5
);

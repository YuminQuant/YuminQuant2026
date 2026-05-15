use crate::factor::common::stock_daily_raw_ids::APBETA1_RAW_ID;
use crate::factor::common::xyzq_liquidity::LiquidityFamily;

crate::define_xyzq_liquidity_factor!(
    StockDailyApbeta1,
    "apbeta1",
    "APBeta1",
    "APBeta1",
    APBETA1_RAW_ID,
    LiquidityFamily::OneMinute,
    40,
    40
);

use crate::factor::common::stock_daily_raw_ids::AMIHUD_1MIN_RAW_ID;
use crate::factor::common::xyzq_liquidity::LiquidityFamily;

crate::define_xyzq_liquidity_factor!(
    StockDailyAmihud1min5d,
    "amihud_1min_5d",
    "Amihud_1min_5d",
    "Amihud 1min 5d",
    AMIHUD_1MIN_RAW_ID,
    LiquidityFamily::OneMinute,
    5,
    5
);

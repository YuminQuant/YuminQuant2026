use crate::factor::common::stock_daily_raw_ids::GAMMA4_RAW_ID;
use crate::factor::common::xyzq_liquidity::LiquidityFamily;

crate::define_xyzq_liquidity_factor!(
    StockDailyGamma4,
    "gamma4",
    "Gamma4",
    "Gamma4",
    GAMMA4_RAW_ID,
    LiquidityFamily::OneMinute,
    5,
    5
);

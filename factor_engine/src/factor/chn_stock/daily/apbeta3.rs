use crate::factor::common::stock_daily_raw_ids::APBETA3_RAW_ID;
use crate::factor::common::xyzq_liquidity::LiquidityFamily;

crate::define_xyzq_liquidity_factor!(
    StockDailyApbeta3,
    "apbeta3",
    "APBeta3",
    "APBeta3",
    APBETA3_RAW_ID,
    LiquidityFamily::Crossday5m,
    1,
    20
);

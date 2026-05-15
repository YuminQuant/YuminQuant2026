use crate::factor::common::stock_daily_raw_ids::APBETA2_RAW_ID;
use crate::factor::common::xyzq_liquidity::LiquidityFamily;

crate::define_xyzq_liquidity_factor!(
    StockDailyApbeta2,
    "apbeta2",
    "APBeta2",
    "APBeta2",
    APBETA2_RAW_ID,
    LiquidityFamily::Crossday5m,
    1,
    20
);

use crate::factor::common::stock_daily_raw_ids::CLOSE_APBETA4_RAW_ID;
use crate::factor::common::xyzq_liquidity::LiquidityFamily;

crate::define_xyzq_liquidity_factor!(
    StockDailyCloseApbeta4,
    "close_apbeta4",
    "close_APBeta4",
    "close APBeta4",
    CLOSE_APBETA4_RAW_ID,
    LiquidityFamily::Operator,
    1,
    20
);

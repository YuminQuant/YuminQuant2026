use crate::factor::common::stock_daily_raw_ids::RV_APBETA3_RAW_ID;
use crate::factor::common::xyzq_liquidity::LiquidityFamily;

crate::define_xyzq_liquidity_factor!(
    StockDailyRvApbeta3,
    "rv_apbeta3",
    "rv_APBeta3",
    "rv APBeta3",
    RV_APBETA3_RAW_ID,
    LiquidityFamily::Operator,
    5,
    5
);

use crate::factor::common::stock_daily_raw_ids::LIQRESID_RAW_ID;
use crate::factor::common::xyzq_liquidity::LiquidityFamily;

crate::define_xyzq_liquidity_factor!(
    StockDailyLiqresid,
    "liqresid",
    "Liqresid",
    "Liqresid",
    LIQRESID_RAW_ID,
    LiquidityFamily::OneMinute,
    1,
    20
);

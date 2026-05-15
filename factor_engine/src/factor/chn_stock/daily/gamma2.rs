use crate::factor::common::stock_daily_raw_ids::GAMMA2_RAW_ID;
use crate::factor::common::xyzq_liquidity::LiquidityFamily;

crate::define_xyzq_liquidity_factor!(
    StockDailyGamma2,
    "gamma2",
    "Gamma2",
    "Gamma2",
    GAMMA2_RAW_ID,
    LiquidityFamily::Crossday5m,
    1,
    20
);

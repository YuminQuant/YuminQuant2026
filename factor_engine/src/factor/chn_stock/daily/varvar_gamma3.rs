use crate::factor::common::stock_daily_raw_ids::VARVAR_GAMMA3_RAW_ID;
use crate::factor::common::xyzq_liquidity::LiquidityFamily;

crate::define_xyzq_liquidity_factor!(
    StockDailyVarvarGamma3,
    "varvar_gamma3",
    "varvar_Gamma3",
    "varvar Gamma3",
    VARVAR_GAMMA3_RAW_ID,
    LiquidityFamily::Operator,
    50,
    50
);

use crate::factor::common::stock_daily_raw_ids::CLOSEVOLCORR_GAMMA3_RAW_ID;
use crate::factor::common::xyzq_liquidity::LiquidityFamily;

crate::define_xyzq_liquidity_factor!(
    StockDailyClosevolcorrGamma3,
    "closevolcorr_gamma3",
    "closevolcorr_Gamma3",
    "closevolcorr Gamma3",
    CLOSEVOLCORR_GAMMA3_RAW_ID,
    LiquidityFamily::Operator,
    20,
    20
);

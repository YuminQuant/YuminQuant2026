use crate::factor::common::stock_daily_raw_ids::RSI_GAMMA3_RAW_ID;
use crate::factor::common::xyzq_liquidity::LiquidityFamily;

crate::define_xyzq_liquidity_factor!(
    StockDailyRsiGamma3,
    "rsi_gamma3",
    "rsi_Gamma3",
    "rsi Gamma3",
    RSI_GAMMA3_RAW_ID,
    LiquidityFamily::Operator,
    3,
    3
);

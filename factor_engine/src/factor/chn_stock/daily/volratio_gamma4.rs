use crate::factor::common::stock_daily_raw_ids::VOLRATIO_GAMMA4_RAW_ID;
use crate::factor::common::xyzq_liquidity::LiquidityFamily;

crate::define_xyzq_liquidity_factor!(
    StockDailyVolratioGamma4,
    "volratio_gamma4",
    "volratio_Gamma4",
    "volratio Gamma4",
    VOLRATIO_GAMMA4_RAW_ID,
    LiquidityFamily::Operator,
    2,
    2
);

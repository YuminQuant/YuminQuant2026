use crate::factor::common::stock_daily_raw_ids::CVAR95_5MIN_RAW_ID;

crate::define_dbzq_5min_factor!(
    StockDailyVocvar95,
    "vocvar95",
    "VOcVaR95",
    "VOcVaR95",
    CVAR95_5MIN_RAW_ID,
    Uncertainty
);

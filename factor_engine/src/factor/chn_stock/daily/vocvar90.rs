use crate::factor::common::stock_daily_raw_ids::CVAR90_5MIN_RAW_ID;

crate::define_dbzq_5min_factor!(
    StockDailyVocvar90,
    "vocvar90",
    "VOcVaR90",
    "VOcVaR90",
    CVAR90_5MIN_RAW_ID,
    Uncertainty
);

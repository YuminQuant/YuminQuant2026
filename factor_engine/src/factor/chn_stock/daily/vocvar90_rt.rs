use crate::factor::common::stock_daily_raw_ids::CVAR90_RT_5MIN_RAW_ID;

crate::define_dbzq_5min_factor!(
    StockDailyVocvar90Rt,
    "vocvar90_rt",
    "VOcVaR90_RT",
    "VOcVaR90 RT",
    CVAR90_RT_5MIN_RAW_ID,
    Uncertainty
);

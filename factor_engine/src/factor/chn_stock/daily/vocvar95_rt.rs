use crate::factor::common::stock_daily_raw_ids::CVAR95_RT_5MIN_RAW_ID;

crate::define_dbzq_5min_factor!(
    StockDailyVocvar95Rt,
    "vocvar95_rt",
    "VOcVaR95_RT",
    "VOcVaR95 RT",
    CVAR95_RT_5MIN_RAW_ID,
    Uncertainty
);

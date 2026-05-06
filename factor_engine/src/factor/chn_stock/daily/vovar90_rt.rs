use crate::factor::common::stock_daily_raw_ids::VAR90_RT_5MIN_RAW_ID;

crate::define_dbzq_5min_factor!(
    StockDailyVovar90Rt,
    "vovar90_rt",
    "VOVaR90_RT",
    "VOVaR90 RT",
    VAR90_RT_5MIN_RAW_ID,
    Uncertainty
);

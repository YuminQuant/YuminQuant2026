use crate::factor::common::stock_daily_raw_ids::CVAR90_RT_5MIN_RAW_ID;

crate::define_dbzq_5min_factor!(
    StockDailyCvar90RtWeek,
    "cvar90_rt_week",
    "cVaR90_RT_week",
    "cVaR90 RT Week",
    CVAR90_RT_5MIN_RAW_ID,
    WeekMean
);

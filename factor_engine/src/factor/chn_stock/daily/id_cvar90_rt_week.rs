use crate::factor::common::stock_daily_raw_ids::ID_CVAR90_RT_5MIN_RAW_ID;

crate::define_dbzq_5min_factor!(
    StockDailyIdCvar90RtWeek,
    "id_cvar90_rt_week",
    "ID_cVaR90_RT_week",
    "ID cVaR90 RT Week",
    ID_CVAR90_RT_5MIN_RAW_ID,
    WeekMean
);

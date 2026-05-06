use crate::factor::common::stock_daily_raw_ids::ID_CVAR95_RT_5MIN_RAW_ID;

crate::define_dbzq_5min_factor!(
    StockDailyIdCvar95RtWeek,
    "id_cvar95_rt_week",
    "ID_cVaR95_RT_week",
    "ID cVaR95 RT Week",
    ID_CVAR95_RT_5MIN_RAW_ID,
    WeekMean
);

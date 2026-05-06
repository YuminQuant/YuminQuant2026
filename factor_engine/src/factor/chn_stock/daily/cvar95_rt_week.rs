use crate::factor::common::stock_daily_raw_ids::CVAR95_RT_5MIN_RAW_ID;

crate::define_dbzq_5min_factor!(
    StockDailyCvar95RtWeek,
    "cvar95_rt_week",
    "cVaR95_RT_week",
    "cVaR95 RT Week",
    CVAR95_RT_5MIN_RAW_ID,
    WeekMean
);

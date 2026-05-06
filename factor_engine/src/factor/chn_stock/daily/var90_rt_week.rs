use crate::factor::common::stock_daily_raw_ids::VAR90_RT_5MIN_RAW_ID;

crate::define_dbzq_5min_factor!(
    StockDailyVar90RtWeek,
    "var90_rt_week",
    "VaR90_RT_week",
    "VaR90 RT Week",
    VAR90_RT_5MIN_RAW_ID,
    WeekMean
);

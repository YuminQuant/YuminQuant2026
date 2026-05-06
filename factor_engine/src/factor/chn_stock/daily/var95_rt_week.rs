use crate::factor::common::stock_daily_raw_ids::VAR95_RT_5MIN_RAW_ID;

crate::define_dbzq_5min_factor!(
    StockDailyVar95RtWeek,
    "var95_rt_week",
    "VaR95_RT_week",
    "VaR95 RT Week",
    VAR95_RT_5MIN_RAW_ID,
    WeekMean
);

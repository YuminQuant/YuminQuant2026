use crate::factor::common::stock_daily_raw_ids::ID_VAR95_RT_5MIN_RAW_ID;

crate::define_dbzq_5min_factor!(
    StockDailyIdVar95RtWeek,
    "id_var95_rt_week",
    "ID_VaR95_RT_week",
    "ID VaR95 RT Week",
    ID_VAR95_RT_5MIN_RAW_ID,
    WeekMean
);

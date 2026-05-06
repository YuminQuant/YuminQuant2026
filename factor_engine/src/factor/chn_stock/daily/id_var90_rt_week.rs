use crate::factor::common::stock_daily_raw_ids::ID_VAR90_RT_5MIN_RAW_ID;

crate::define_dbzq_5min_factor!(
    StockDailyIdVar90RtWeek,
    "id_var90_rt_week",
    "ID_VaR90_RT_week",
    "ID VaR90 RT Week",
    ID_VAR90_RT_5MIN_RAW_ID,
    WeekMean
);

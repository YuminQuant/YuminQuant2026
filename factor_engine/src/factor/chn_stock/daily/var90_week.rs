use crate::factor::common::stock_daily_raw_ids::VAR90_5MIN_RAW_ID;

crate::define_dbzq_5min_factor!(
    StockDailyVar90Week,
    "var90_week",
    "VaR90_week",
    "VaR90 Week",
    VAR90_5MIN_RAW_ID,
    WeekMean
);

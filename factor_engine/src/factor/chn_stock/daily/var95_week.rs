use crate::factor::common::stock_daily_raw_ids::VAR95_5MIN_RAW_ID;

crate::define_dbzq_5min_factor!(
    StockDailyVar95Week,
    "var95_week",
    "VaR95_week",
    "VaR95 Week",
    VAR95_5MIN_RAW_ID,
    WeekMean
);

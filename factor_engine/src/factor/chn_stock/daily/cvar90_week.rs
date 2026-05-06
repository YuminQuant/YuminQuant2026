use crate::factor::common::stock_daily_raw_ids::CVAR90_5MIN_RAW_ID;

crate::define_dbzq_5min_factor!(
    StockDailyCvar90Week,
    "cvar90_week",
    "cVaR90_week",
    "cVaR90 Week",
    CVAR90_5MIN_RAW_ID,
    WeekMean
);

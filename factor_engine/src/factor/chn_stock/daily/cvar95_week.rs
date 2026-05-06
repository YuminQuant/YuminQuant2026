use crate::factor::common::stock_daily_raw_ids::CVAR95_5MIN_RAW_ID;

crate::define_dbzq_5min_factor!(
    StockDailyCvar95Week,
    "cvar95_week",
    "cVaR95_week",
    "cVaR95 Week",
    CVAR95_5MIN_RAW_ID,
    WeekMean
);

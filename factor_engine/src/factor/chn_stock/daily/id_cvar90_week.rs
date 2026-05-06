use crate::factor::common::stock_daily_raw_ids::ID_CVAR90_5MIN_RAW_ID;

crate::define_dbzq_5min_factor!(
    StockDailyIdCvar90Week,
    "id_cvar90_week",
    "ID_cVaR90_week",
    "ID cVaR90 Week",
    ID_CVAR90_5MIN_RAW_ID,
    WeekMean
);

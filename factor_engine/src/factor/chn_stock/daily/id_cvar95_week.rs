use crate::factor::common::stock_daily_raw_ids::ID_CVAR95_5MIN_RAW_ID;

crate::define_dbzq_5min_factor!(
    StockDailyIdCvar95Week,
    "id_cvar95_week",
    "ID_cVaR95_week",
    "ID cVaR95 Week",
    ID_CVAR95_5MIN_RAW_ID,
    WeekMean
);

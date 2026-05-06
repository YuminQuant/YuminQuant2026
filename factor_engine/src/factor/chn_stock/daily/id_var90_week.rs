use crate::factor::common::stock_daily_raw_ids::ID_VAR90_5MIN_RAW_ID;

crate::define_dbzq_5min_factor!(
    StockDailyIdVar90Week,
    "id_var90_week",
    "ID_VaR90_week",
    "ID VaR90 Week",
    ID_VAR90_5MIN_RAW_ID,
    WeekMean
);

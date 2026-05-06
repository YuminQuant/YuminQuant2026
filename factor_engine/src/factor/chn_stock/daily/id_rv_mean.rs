use crate::factor::common::stock_daily_raw_ids::ID_RV_5MIN_RAW_ID;

crate::define_dbzq_5min_factor!(
    StockDailyIdRvMean,
    "id_rv_mean",
    "ID_RV_mean",
    "ID RV Mean",
    ID_RV_5MIN_RAW_ID,
    WeekMean
);

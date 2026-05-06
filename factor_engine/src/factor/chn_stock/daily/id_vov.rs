use crate::factor::common::stock_daily_raw_ids::ID_RV_5MIN_RAW_ID;

crate::define_dbzq_5min_factor!(
    StockDailyIdVov,
    "id_vov",
    "ID_VOV",
    "ID VOV",
    ID_RV_5MIN_RAW_ID,
    Uncertainty
);

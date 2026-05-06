use crate::factor::common::stock_daily_raw_ids::ID_CVAR90_5MIN_RAW_ID;

crate::define_dbzq_5min_factor!(
    StockDailyIdVocvar90,
    "id_vocvar90",
    "ID_VOcVaR90",
    "ID VOcVaR90",
    ID_CVAR90_5MIN_RAW_ID,
    Uncertainty
);

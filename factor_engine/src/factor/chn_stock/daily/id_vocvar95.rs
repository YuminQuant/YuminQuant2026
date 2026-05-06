use crate::factor::common::stock_daily_raw_ids::ID_CVAR95_5MIN_RAW_ID;

crate::define_dbzq_5min_factor!(
    StockDailyIdVocvar95,
    "id_vocvar95",
    "ID_VOcVaR95",
    "ID VOcVaR95",
    ID_CVAR95_5MIN_RAW_ID,
    Uncertainty
);

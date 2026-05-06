use crate::factor::common::stock_daily_raw_ids::ID_CVAR90_RT_5MIN_RAW_ID;

crate::define_dbzq_5min_factor!(
    StockDailyIdVocvar90Rt,
    "id_vocvar90_rt",
    "ID_VOcVaR90_RT",
    "ID VOcVaR90 RT",
    ID_CVAR90_RT_5MIN_RAW_ID,
    Uncertainty
);

use crate::factor::common::stock_daily_raw_ids::ID_CVAR95_RT_5MIN_RAW_ID;

crate::define_dbzq_5min_factor!(
    StockDailyIdVocvar95Rt,
    "id_vocvar95_rt",
    "ID_VOcVaR95_RT",
    "ID VOcVaR95 RT",
    ID_CVAR95_RT_5MIN_RAW_ID,
    Uncertainty
);

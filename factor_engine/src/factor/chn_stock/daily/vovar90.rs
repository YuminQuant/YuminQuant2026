use crate::factor::common::stock_daily_raw_ids::VAR90_5MIN_RAW_ID;

crate::define_dbzq_5min_factor!(
    StockDailyVovar90,
    "vovar90",
    "VOVaR90",
    "VOVaR90",
    VAR90_5MIN_RAW_ID,
    Uncertainty
);

use crate::factor::common::stock_daily_raw_ids::VAR95_5MIN_RAW_ID;

crate::define_dbzq_5min_factor!(
    StockDailyVovar95,
    "vovar95",
    "VOVaR95",
    "VOVaR95",
    VAR95_5MIN_RAW_ID,
    Uncertainty
);

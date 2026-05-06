use crate::factor::common::stock_daily_raw_ids::VAR95_RT_5MIN_RAW_ID;

crate::define_dbzq_5min_factor!(
    StockDailyVovar95Rt,
    "vovar95_rt",
    "VOVaR95_RT",
    "VOVaR95 RT",
    VAR95_RT_5MIN_RAW_ID,
    Uncertainty
);

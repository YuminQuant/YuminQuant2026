use crate::factor::common::stock_daily_raw_ids::VSA_HIGH2MIN_RAW_ID;

crate::define_xyzq_volume_shape_factor!(
    StockDailyVsaHigh2min,
    "vsa_high2min",
    "vsa_high2min",
    "vsa_high2min",
    VSA_HIGH2MIN_RAW_ID,
    crate::factor::common::xyzq_volume_shape::default_window(),
    Mean
);

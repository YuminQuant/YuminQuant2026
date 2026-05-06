use crate::factor::common::stock_daily_raw_ids::VSA_LOW2MAX_RAW_ID;

crate::define_xyzq_volume_shape_factor!(
    StockDailyVsaLow2max,
    "vsa_low2max",
    "vsa_low2max",
    "vsa_low2max",
    VSA_LOW2MAX_RAW_ID,
    crate::factor::common::xyzq_volume_shape::default_window(),
    Mean
);

use crate::factor::common::stock_daily_raw_ids::VSA_RATIO_RAW_ID;

crate::define_xyzq_volume_shape_factor!(
    StockDailyVsaRatio,
    "vsa_ratio",
    "vsa_ratio",
    "vsa_ratio",
    VSA_RATIO_RAW_ID,
    crate::factor::common::xyzq_volume_shape::default_window(),
    Mean
);

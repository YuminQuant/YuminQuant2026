use crate::factor::common::stock_daily_raw_ids::VOLROC_SKEW_RAW_ID;

crate::define_xyzq_volume_shape_factor!(
    StockDailyVolrocSkew,
    "volroc_skew",
    "volroc_skew",
    "volroc_skew",
    VOLROC_SKEW_RAW_ID,
    crate::factor::common::xyzq_volume_shape::default_window(),
    Mean
);

use crate::factor::common::stock_daily_raw_ids::VOLROC_KURT_RAW_ID;

crate::define_xyzq_volume_shape_factor!(
    StockDailyVolrocKurt,
    "volroc_kurt",
    "volroc_kurt",
    "volroc_kurt",
    VOLROC_KURT_RAW_ID,
    crate::factor::common::xyzq_volume_shape::default_window(),
    Mean
);

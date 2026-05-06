use crate::factor::common::stock_daily_raw_ids::VOL_MAXSTD_RAW_ID;

crate::define_xyzq_volume_shape_factor!(
    StockDailyVolMaxstd,
    "vol_maxstd",
    "vol_maxstd",
    "vol_maxstd",
    VOL_MAXSTD_RAW_ID,
    crate::factor::common::xyzq_volume_shape::default_window(),
    Std
);

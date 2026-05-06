use crate::factor::common::stock_daily_raw_ids::VOL_MAXMEAN_RAW_ID;

crate::define_xyzq_volume_shape_factor!(
    StockDailyVolMaxmean,
    "vol_maxmean",
    "vol_maxmean",
    "vol_maxmean",
    VOL_MAXMEAN_RAW_ID,
    crate::factor::common::xyzq_volume_shape::default_window(),
    Std
);

use crate::factor::common::stock_daily_raw_ids::VOL_ENTROPY_SHAPE_RAW_ID;

crate::define_xyzq_volume_shape_factor!(
    StockDailyVolEntropy,
    "vol_entropy",
    "vol_entropy",
    "vol_entropy",
    VOL_ENTROPY_SHAPE_RAW_ID,
    crate::factor::common::xyzq_volume_shape::entropy_window(),
    Std
);

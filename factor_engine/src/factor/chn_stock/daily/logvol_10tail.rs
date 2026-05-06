use crate::factor::common::stock_daily_raw_ids::LOGVOL_10TAIL_RAW_ID;

crate::define_xyzq_volume_shape_factor!(
    StockDailyLogvol10tail,
    "logvol_10tail",
    "logvol_10tail",
    "logvol_10tail",
    LOGVOL_10TAIL_RAW_ID,
    crate::factor::common::xyzq_volume_shape::default_window(),
    Mean
);

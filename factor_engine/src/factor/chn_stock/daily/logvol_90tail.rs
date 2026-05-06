use crate::factor::common::stock_daily_raw_ids::LOGVOL_90TAIL_RAW_ID;

crate::define_xyzq_volume_shape_factor!(
    StockDailyLogvol90tail,
    "logvol_90tail",
    "logvol_90tail",
    "logvol_90tail",
    LOGVOL_90TAIL_RAW_ID,
    crate::factor::common::xyzq_volume_shape::default_window(),
    Mean
);

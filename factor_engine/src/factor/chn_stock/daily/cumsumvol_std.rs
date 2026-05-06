use crate::factor::common::stock_daily_raw_ids::CUMSUMVOL_STD_RAW_ID;

crate::define_xyzq_volume_shape_factor!(
    StockDailyCumsumvolStd,
    "cumsumvol_std",
    "cumsumvol_std",
    "cumsumvol_std",
    CUMSUMVOL_STD_RAW_ID,
    crate::factor::common::xyzq_volume_shape::default_window(),
    Mean
);

use crate::factor::common::stock_daily_raw_ids::CUMSUMVOL_MEAN_RAW_ID;

crate::define_xyzq_volume_shape_factor!(
    StockDailyCumsumvolMean,
    "cumsumvol_mean",
    "cumsumvol_mean",
    "cumsumvol_mean",
    CUMSUMVOL_MEAN_RAW_ID,
    crate::factor::common::xyzq_volume_shape::default_window(),
    Mean
);

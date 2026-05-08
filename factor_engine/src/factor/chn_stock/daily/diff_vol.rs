use crate::factor::common::stock_daily_raw_ids::DIFF_VOL_RAW_ID;

crate::define_xyzq_intraday_contrast_factor!(
    StockDailyDiffVol,
    "diff_vol",
    "diff_vol",
    "diff_vol",
    DIFF_VOL_RAW_ID
);

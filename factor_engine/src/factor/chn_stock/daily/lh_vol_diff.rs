use crate::factor::common::stock_daily_raw_ids::LH_VOL_DIFF_RAW_ID;

crate::define_xyzq_intraday_contrast_factor!(
    StockDailyLhVolDiff,
    "lh_vol_diff",
    "lh_volDiff",
    "lh_volDiff",
    LH_VOL_DIFF_RAW_ID
);

use crate::factor::common::stock_daily_raw_ids::DIFF_STD_RAW_ID;

crate::define_xyzq_intraday_contrast_factor!(
    StockDailyDiffStd,
    "diff_std",
    "diff_std",
    "diff_std",
    DIFF_STD_RAW_ID
);

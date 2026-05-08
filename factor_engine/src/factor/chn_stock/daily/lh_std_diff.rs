use crate::factor::common::stock_daily_raw_ids::LH_STD_DIFF_RAW_ID;

crate::define_xyzq_intraday_contrast_factor!(
    StockDailyLhStdDiff,
    "lh_std_diff",
    "lh_stdDiff",
    "lh_stdDiff",
    LH_STD_DIFF_RAW_ID
);

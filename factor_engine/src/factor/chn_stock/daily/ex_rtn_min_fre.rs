use crate::factor::common::stock_daily_raw_ids::EX_RTN_MIN_FRE_RAW_ID;

crate::define_xyzq_extreme_gmm_factor!(
    StockDailyExRtnMinFre,
    "ex_rtn_min_fre",
    "exRtn_minFre",
    "exRtn_minFre",
    EX_RTN_MIN_FRE_RAW_ID,
    crate::factor::common::xyzq_extreme_gmm::default_smooth_window()
);

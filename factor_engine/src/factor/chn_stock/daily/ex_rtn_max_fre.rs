use crate::factor::common::stock_daily_raw_ids::EX_RTN_MAX_FRE_RAW_ID;

crate::define_xyzq_extreme_gmm_factor!(
    StockDailyExRtnMaxFre,
    "ex_rtn_max_fre",
    "exRtn_maxFre",
    "exRtn_maxFre",
    EX_RTN_MAX_FRE_RAW_ID,
    crate::factor::common::xyzq_extreme_gmm::default_smooth_window()
);

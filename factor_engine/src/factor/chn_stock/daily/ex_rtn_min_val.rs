use crate::factor::common::stock_daily_raw_ids::EX_RTN_MIN_VAL_RAW_ID;

crate::define_xyzq_extreme_gmm_factor!(
    StockDailyExRtnMinVal,
    "ex_rtn_min_val",
    "exRtn_minVal",
    "exRtn_minVal",
    EX_RTN_MIN_VAL_RAW_ID,
    crate::factor::common::xyzq_extreme_gmm::default_smooth_window()
);

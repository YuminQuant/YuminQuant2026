use crate::factor::common::stock_daily_raw_ids::RTN_COND_VAR_RAW_ID;

crate::define_xyzq_serial_structure_factor!(
    StockDailyRtnCondVar,
    "rtn_cond_var",
    "rtn_condVaR",
    "rtn_condVaR",
    RTN_COND_VAR_RAW_ID,
    Std
);

use crate::factor::common::stock_daily_raw_ids::CUTVOL_RTN_VAR_RAW_ID;

crate::define_xyzq_flow_structure_factor!(
    StockDailyCutvolRtnVar,
    "cutvol_rtn_var",
    "cutVol_rtnVar",
    "cutVol_rtnVar",
    CUTVOL_RTN_VAR_RAW_ID,
    crate::factor::common::xyzq_flow_structure::default_window()
);

use crate::factor::common::stock_daily_raw_ids::CUTVOL_TIME_VAR_RAW_ID;

crate::define_xyzq_flow_structure_factor!(
    StockDailyCutvolTimeVar,
    "cutvol_time_var",
    "cutVol_timeVar",
    "cutVol_timeVar",
    CUTVOL_TIME_VAR_RAW_ID,
    crate::factor::common::xyzq_flow_structure::default_window()
);

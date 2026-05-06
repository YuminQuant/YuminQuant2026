use crate::factor::common::stock_daily_raw_ids::CUTVOL_TIME_COR_RAW_ID;

crate::define_xyzq_flow_structure_factor!(
    StockDailyCutvolTimeCor,
    "cutvol_time_cor",
    "cutVol_timeCor",
    "cutVol_timeCor",
    CUTVOL_TIME_COR_RAW_ID,
    crate::factor::common::xyzq_flow_structure::te_window()
);

use crate::factor::common::stock_daily_raw_ids::CUTVOL_TIME_MEAN_RAW_ID;

crate::define_xyzq_flow_structure_factor!(
    StockDailyCutvolTimeMean,
    "cutvol_time_mean",
    "cutVol_timeMean",
    "cutVol_timeMean",
    CUTVOL_TIME_MEAN_RAW_ID,
    crate::factor::common::xyzq_flow_structure::default_window()
);

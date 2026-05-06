use crate::factor::common::stock_daily_raw_ids::CUTVOL_ENTROPY_RAW_ID;

crate::define_xyzq_flow_structure_factor!(
    StockDailyCutvolEntropy,
    "cutvol_entropy",
    "cutVol_entropy",
    "cutVol_entropy",
    CUTVOL_ENTROPY_RAW_ID,
    crate::factor::common::xyzq_flow_structure::te_window()
);

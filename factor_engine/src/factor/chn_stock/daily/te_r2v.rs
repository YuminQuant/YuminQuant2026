use crate::factor::common::stock_daily_raw_ids::TE_R2V_RAW_ID;

crate::define_xyzq_flow_structure_factor!(
    StockDailyTeR2v,
    "te_r2v",
    "te_r2v",
    "te_r2v",
    TE_R2V_RAW_ID,
    crate::factor::common::xyzq_flow_structure::te_window()
);

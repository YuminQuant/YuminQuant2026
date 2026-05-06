use crate::factor::common::stock_daily_raw_ids::VOH_COR_RAW_ID;

crate::define_xyzq_flow_structure_factor!(
    StockDailyVohCor,
    "voh_cor",
    "voh_cor",
    "voh_cor",
    VOH_COR_RAW_ID,
    crate::factor::common::xyzq_flow_structure::default_window()
);

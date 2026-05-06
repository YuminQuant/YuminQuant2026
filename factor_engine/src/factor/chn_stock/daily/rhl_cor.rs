use crate::factor::common::stock_daily_raw_ids::RHL_COR_RAW_ID;

crate::define_xyzq_flow_structure_factor!(
    StockDailyRhlCor,
    "rhl_cor",
    "rhl_cor",
    "rhl_cor",
    RHL_COR_RAW_ID,
    crate::factor::common::xyzq_flow_structure::default_window()
);

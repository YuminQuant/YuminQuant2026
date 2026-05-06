use crate::factor::common::stock_daily_raw_ids::VOL_COR_RAW_ID;

crate::define_xyzq_flow_structure_factor!(
    StockDailyVolCor,
    "vol_cor",
    "vol_cor",
    "vol_cor",
    VOL_COR_RAW_ID,
    crate::factor::common::xyzq_flow_structure::default_window()
);

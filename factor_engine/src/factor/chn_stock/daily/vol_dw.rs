use crate::factor::common::stock_daily_raw_ids::VOL_DW_RAW_ID;

crate::define_xyzq_serial_structure_factor!(
    StockDailyVolDw,
    "vol_dw",
    "vol_DW",
    "vol_DW",
    VOL_DW_RAW_ID,
    Mean
);

use crate::factor::common::stock_daily_raw_ids::VOL_FOC_RAW_ID;

crate::define_xyzq_serial_structure_factor!(
    StockDailyVolFoc,
    "vol_foc",
    "vol_foc",
    "vol_foc",
    VOL_FOC_RAW_ID,
    Mean
);

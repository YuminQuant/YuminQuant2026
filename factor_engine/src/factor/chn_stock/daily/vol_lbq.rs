use crate::factor::common::stock_daily_raw_ids::VOL_LBQ_RAW_ID;

crate::define_xyzq_serial_structure_factor!(
    StockDailyVolLbq,
    "vol_lbq",
    "vol_LBQ",
    "vol_LBQ",
    VOL_LBQ_RAW_ID,
    Mean
);

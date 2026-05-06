use crate::factor::common::stock_daily_raw_ids::FLASH_CRASH_PROB_RAW_ID;

crate::define_xyzq_serial_structure_factor!(
    StockDailyFlashCrashProb,
    "flash_crash_prob",
    "flashCrashProb",
    "flashCrashProb",
    FLASH_CRASH_PROB_RAW_ID,
    Mean
);

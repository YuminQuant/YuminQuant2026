use std::path::Path;

use crate::data::parquet_io::read_parquet;
use crate::error::{err, Result};

#[derive(Clone, Debug)]
pub struct TradingCalendar {
    open_dates: Vec<i32>,
}

impl TradingCalendar {
    pub fn load(data_root: &Path, exchange: &str) -> Result<Self> {
        let path = data_root
            .join("calendar")
            .join(format!("trade_cal_{}.parquet", exchange));
        if !path.exists() {
            return Err(err(format!("calendar file not found: {}", path.display())));
        }
        let columns = vec!["cal_date".to_string(), "is_open".to_string()];
        let table = read_parquet(&path, Some(&columns))?;
        let cal_dates = table.required_i32("cal_date")?;
        let is_open = table.required_i64_cast("is_open")?;

        let mut dates = Vec::new();
        for (date, open) in cal_dates.iter().zip(is_open.iter()) {
            if let (Some(date), Some(open)) = (date, open) {
                if *open == 1 {
                    dates.push(*date);
                }
            }
        }
        dates.sort_unstable();
        dates.dedup();
        Ok(Self { open_dates: dates })
    }

    pub fn open_dates_between(&self, start_date: i32, end_date: i32) -> Vec<i32> {
        self.open_dates
            .iter()
            .copied()
            .filter(|date| *date >= start_date && *date <= end_date)
            .collect()
    }

    pub fn warmup_start(&self, start_date: i32, trading_days: usize) -> i32 {
        if self.open_dates.is_empty() || trading_days == 0 {
            return start_date;
        }
        let first_target_idx = match self.open_dates.binary_search(&start_date) {
            Ok(idx) => idx,
            Err(idx) => idx.min(self.open_dates.len().saturating_sub(1)),
        };
        let warmup_idx = first_target_idx.saturating_sub(trading_days);
        self.open_dates[warmup_idx]
    }
}

use std::path::Path;

use crate::data::parquet_io::read_parquet;
use crate::error::{err, Result};

#[derive(Clone, Debug)]
pub struct TradingCalendar {
    open_dates: Vec<i32>,
}

impl TradingCalendar {
    #[cfg(test)]
    pub(crate) fn from_open_dates(mut open_dates: Vec<i32>) -> Self {
        open_dates.sort_unstable();
        open_dates.dedup();
        Self { open_dates }
    }

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

    pub fn first_open_on_or_after(&self, date: i32) -> Option<i32> {
        let idx = match self.open_dates.binary_search(&date) {
            Ok(idx) | Err(idx) => idx,
        };
        self.open_dates.get(idx).copied()
    }

    pub fn last_open_on_or_before(&self, date: i32) -> Option<i32> {
        match self.open_dates.binary_search(&date) {
            Ok(idx) => self.open_dates.get(idx).copied(),
            Err(0) => None,
            Err(idx) => self.open_dates.get(idx - 1).copied(),
        }
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

    pub fn open_date_after(&self, date: i32, trading_days: usize) -> Option<i32> {
        let idx = self.open_dates.binary_search(&date).ok()?;
        self.open_dates.get(idx + trading_days).copied()
    }

    pub fn has_open_date_after(&self, date: i32, trading_days: usize) -> bool {
        self.open_date_after(date, trading_days).is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::TradingCalendar;

    fn calendar() -> TradingCalendar {
        TradingCalendar::from_open_dates(vec![20100104, 20100105, 20100106, 20110104])
    }

    #[test]
    fn aligns_to_nearest_open_dates_inside_requested_range() {
        let calendar = calendar();

        assert_eq!(calendar.first_open_on_or_after(20100101), Some(20100104));
        assert_eq!(calendar.last_open_on_or_before(20100110), Some(20100106));
    }

    #[test]
    fn returns_none_when_no_open_date_can_satisfy_boundary() {
        let calendar = calendar();

        assert_eq!(calendar.first_open_on_or_after(20120101), None);
        assert_eq!(calendar.last_open_on_or_before(20091231), None);
    }

    #[test]
    fn warmup_can_cross_year_boundary() {
        let calendar = calendar();

        assert_eq!(calendar.warmup_start(20110104, 2), 20100105);
    }

    #[test]
    fn returns_future_open_date_by_offset() {
        let calendar = calendar();

        assert_eq!(calendar.open_date_after(20100104, 0), Some(20100104));
        assert_eq!(calendar.open_date_after(20100104, 2), Some(20100106));
        assert_eq!(calendar.open_date_after(20100104, 3), Some(20110104));
        assert_eq!(calendar.open_date_after(20100105, 3), None);
        assert_eq!(calendar.open_date_after(20100107, 1), None);
    }
}

use std::collections::{BTreeMap, HashMap};

use crate::data::Table;
use crate::error::{err, Result};
use crate::factor::common::{DailyPanel, PanelColumn};
use crate::operators::{cs_winsorize_quantile, cs_zscore};

#[derive(Clone, Debug)]
pub struct StatementRecord {
    pub end_date: i32,
    pub disclosure_date: i32,
    pub report_type: i64,
    pub update_flag: i64,
    values: BTreeMap<String, Option<f64>>,
}

impl StatementRecord {
    pub fn column(&self, name: &str) -> Option<f64> {
        self.values
            .get(name)
            .copied()
            .flatten()
            .and_then(clean_value)
    }
}

#[derive(Clone, Debug)]
pub struct StatementData {
    preference: Vec<i64>,
    by_ts_code: BTreeMap<String, BTreeMap<i32, BTreeMap<i64, Vec<StatementRecord>>>>,
}

impl StatementData {
    pub fn from_table(table: &Table, value_columns: &[&str], preference: &[i64]) -> Result<Self> {
        let ts_codes = table.required_utf8("ts_code")?;
        let ann_dates = table.required_i32_date_cast("ann_date")?;
        let f_ann_dates = table.required_i32_date_cast("f_ann_date")?;
        let end_dates = table.required_i32_date_cast("end_date")?;
        let report_types = table.required_i64_cast("report_type")?;
        let update_flags = table.required_i64_cast("update_flag")?;
        let mut value_data = BTreeMap::new();
        for column in value_columns {
            value_data.insert((*column).to_string(), table.required_f64_cast(column)?);
        }

        let mut by_ts_code = BTreeMap::new();
        for idx in 0..table.len {
            let (Some(ts_code), Some(end_date), Some(disclosure_date), Some(report_type)) = (
                ts_codes[idx].clone(),
                end_dates[idx],
                f_ann_dates[idx].or(ann_dates[idx]),
                report_types[idx],
            ) else {
                continue;
            };
            if !preference.contains(&report_type) {
                continue;
            }
            let values = value_data
                .iter()
                .map(|(name, values)| (name.clone(), values[idx]))
                .collect::<BTreeMap<_, _>>();
            by_ts_code
                .entry(ts_code)
                .or_insert_with(BTreeMap::new)
                .entry(end_date)
                .or_insert_with(BTreeMap::new)
                .entry(report_type)
                .or_insert_with(Vec::new)
                .push(StatementRecord {
                    end_date,
                    disclosure_date,
                    report_type,
                    update_flag: update_flags[idx].unwrap_or(0),
                    values,
                });
        }

        for by_end_date in by_ts_code.values_mut() {
            for by_report_type in by_end_date.values_mut() {
                for versions in by_report_type.values_mut() {
                    versions.sort_by(|left, right| {
                        right
                            .disclosure_date
                            .cmp(&left.disclosure_date)
                            .then_with(|| right.update_flag.cmp(&left.update_flag))
                    });
                }
            }
        }

        Ok(Self {
            preference: preference.to_vec(),
            by_ts_code,
        })
    }

    pub fn record_for_end_date(
        &self,
        ts_code: &str,
        trade_date: i32,
        end_date: i32,
    ) -> Option<&StatementRecord> {
        let by_report_type = self.by_ts_code.get(ts_code)?.get(&end_date)?;
        for report_type in &self.preference {
            let Some(records) = by_report_type.get(report_type) else {
                continue;
            };
            if let Some(record) = records
                .iter()
                .find(|record| record.disclosure_date <= trade_date)
            {
                return Some(record);
            }
        }
        None
    }

    pub fn ttm_sum(&self, ts_code: &str, trade_date: i32, column: &str) -> Option<f64> {
        let by_end_date = self.by_ts_code.get(ts_code)?;
        for (&anchor, _) in by_end_date.iter().rev() {
            let mut current = anchor;
            let mut sum = 0.0;
            let mut valid = true;
            for _ in 0..4 {
                let Some(record) = self.record_for_end_date(ts_code, trade_date, current) else {
                    valid = false;
                    break;
                };
                let Some(value) = record.column(column) else {
                    valid = false;
                    break;
                };
                sum += value;
                current = previous_quarter_end_date(current)?;
            }
            if valid {
                return Some(sum);
            }
        }
        None
    }

    pub fn latest_annual_value(&self, ts_code: &str, trade_date: i32, column: &str) -> Option<f64> {
        let end_date = self.latest_annual_end_date(ts_code, trade_date)?;
        self.record_for_end_date(ts_code, trade_date, end_date)?
            .column(column)
    }

    pub fn latest_annual_end_date(&self, ts_code: &str, trade_date: i32) -> Option<i32> {
        let by_end_date = self.by_ts_code.get(ts_code)?;
        for (&end_date, _) in by_end_date.iter().rev() {
            if end_date % 10_000 == 12_31
                && self
                    .record_for_end_date(ts_code, trade_date, end_date)
                    .is_some()
            {
                return Some(end_date);
            }
        }
        None
    }

    pub fn annual_value_for_end_date(
        &self,
        ts_code: &str,
        trade_date: i32,
        end_date: i32,
        column: &str,
    ) -> Option<f64> {
        self.record_for_end_date(ts_code, trade_date, end_date)?
            .column(column)
    }

    pub fn annual_values(
        &self,
        ts_code: &str,
        trade_date: i32,
        column: &str,
        count: usize,
    ) -> Option<Vec<f64>> {
        let by_end_date = self.by_ts_code.get(ts_code)?;
        for (&anchor, _) in by_end_date.iter().rev() {
            if anchor % 10_000 != 12_31 {
                continue;
            }
            let mut year = anchor / 10_000;
            let mut values = Vec::with_capacity(count);
            let mut valid = true;
            for _ in 0..count {
                let end_date = year * 10_000 + 12_31;
                let Some(record) = self.record_for_end_date(ts_code, trade_date, end_date) else {
                    valid = false;
                    break;
                };
                let Some(value) = record.column(column) else {
                    valid = false;
                    break;
                };
                values.push(value);
                year -= 1;
            }
            if valid {
                values.reverse();
                return Some(values);
            }
        }
        None
    }
}

pub fn clean_value(value: f64) -> Option<f64> {
    (!value.is_nan()).then_some(value)
}

pub fn clean(value: Option<f64>) -> Option<f64> {
    value.and_then(clean_value)
}

pub fn safe_div(numerator: f64, denominator: f64) -> Option<f64> {
    (denominator.abs() > f64::EPSILON).then_some(numerator / denominator)
}

pub fn standardize_cross_section(values: &[Option<f64>]) -> Vec<Option<f64>> {
    cs_zscore(&cs_winsorize_quantile(values, 0.01, 0.99))
}

pub fn average_available(values: &[Option<f64>]) -> Option<f64> {
    let mut sum = 0.0;
    let mut count = 0usize;
    for value in values {
        if let Some(value) = clean(*value) {
            sum += value;
            count += 1;
        }
    }
    (count > 0).then_some(sum / count as f64)
}

pub fn average_columns(panel: &DailyPanel, columns: &[&PanelColumn]) -> Result<PanelColumn> {
    if columns.is_empty() {
        return panel.column_from_values(vec![None; panel.shape_len()]);
    }
    let mut values = Vec::with_capacity(panel.shape_len());
    for idx in 0..panel.shape_len() {
        let row = columns
            .iter()
            .map(|column| column.values()[idx])
            .collect::<Vec<_>>();
        values.push(average_available(&row));
    }
    panel.column_from_values(values)
}

pub fn align_table_column(panel: &DailyPanel, table: &Table, column: &str) -> Result<PanelColumn> {
    panel.column_from_table(table, column)
}

pub fn arithmetic_return(close: Option<f64>, pre_close: Option<f64>) -> Option<f64> {
    match (clean(close), clean(pre_close)) {
        (Some(close), Some(pre_close)) if pre_close.abs() > f64::EPSILON => {
            Some(close / pre_close - 1.0)
        }
        _ => None,
    }
}

pub fn log_return(close: Option<f64>, pre_close: Option<f64>) -> Option<f64> {
    match (clean(close), clean(pre_close)) {
        (Some(close), Some(pre_close)) if close > 0.0 && pre_close > 0.0 => {
            Some((close / pre_close).ln())
        }
        _ => None,
    }
}

pub fn expand_index_column(
    stock_panel: &DailyPanel,
    index_panel: &DailyPanel,
    index_column: &PanelColumn,
) -> Result<PanelColumn> {
    let index_instrument_count = index_panel.instruments().len();
    if index_instrument_count == 0 {
        return Err(err("index daily panel has no instruments"));
    }
    let mut by_date = HashMap::new();
    for (date_idx, trade_date) in index_panel.dates().iter().enumerate() {
        by_date.insert(
            *trade_date,
            index_column.values()[date_idx * index_instrument_count],
        );
    }
    let mut values = Vec::with_capacity(stock_panel.shape_len());
    for trade_date in stock_panel.dates() {
        let value = by_date.get(trade_date).copied().unwrap_or(None);
        for _ in stock_panel.instruments() {
            values.push(value);
        }
    }
    stock_panel.column_from_values(values)
}

pub fn panel_from_stock_map<F>(panel: &DailyPanel, mut f: F) -> Result<PanelColumn>
where
    F: FnMut(i32, &str) -> Option<f64>,
{
    let mut values = Vec::with_capacity(panel.shape_len());
    for trade_date in panel.dates() {
        for ts_code in panel.instruments() {
            values.push(f(*trade_date, ts_code));
        }
    }
    panel.column_from_values(values)
}

pub fn panel_from_target_stock_map<F>(panel: &DailyPanel, mut f: F) -> Result<PanelColumn>
where
    F: FnMut(i32, &str) -> Option<f64>,
{
    let mut values = Vec::with_capacity(panel.shape_len());
    for trade_date in panel.dates() {
        if !panel.is_target_date(*trade_date) {
            values.extend(std::iter::repeat_n(None, panel.instruments().len()));
            continue;
        }
        for ts_code in panel.instruments() {
            values.push(f(*trade_date, ts_code));
        }
    }
    panel.column_from_values(values)
}

pub fn sample_std(values: &[f64]) -> Option<f64> {
    if values.len() < 2 {
        return None;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let var = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / (values.len() as f64 - 1.0);
    Some(var.max(0.0).sqrt())
}

pub fn slope_over_time(values: &[f64]) -> Option<f64> {
    if values.len() < 2 {
        return None;
    }
    let n = values.len() as f64;
    let mean_x = (n - 1.0) / 2.0;
    let mean_y = values.iter().sum::<f64>() / n;
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    for (idx, value) in values.iter().enumerate() {
        let x = idx as f64;
        numerator += (x - mean_x) * (value - mean_y);
        denominator += (x - mean_x).powi(2);
    }
    safe_div(numerator, denominator)
}

pub fn previous_quarter_end_date(end_date: i32) -> Option<i32> {
    let year = end_date / 10_000;
    match end_date % 10_000 {
        331 => Some((year - 1) * 10_000 + 12_31),
        630 => Some(year * 10_000 + 3_31),
        930 => Some(year * 10_000 + 6_30),
        1231 => Some(year * 10_000 + 9_30),
        _ => None,
    }
}

pub fn fy1_quarter(trade_date: i32) -> String {
    let year = trade_date / 10_000;
    if trade_date <= year * 10_000 + 4_30 {
        format!("{year}Q4")
    } else {
        format!("{}Q4", year + 1)
    }
}

pub fn fy_quarter(trade_date: i32, offset: i32) -> String {
    let base_year = if trade_date <= (trade_date / 10_000) * 10_000 + 4_30 {
        trade_date / 10_000
    } else {
        trade_date / 10_000 + 1
    };
    format!("{}Q4", base_year + offset)
}

pub fn add_months(date: i32, months: i32) -> i32 {
    let year = date / 10_000;
    let month = (date / 100) % 100;
    let day = date % 100;
    let total_months = year * 12 + month - 1 + months;
    let new_year = total_months.div_euclid(12);
    let new_month = total_months.rem_euclid(12) + 1;
    let new_day = day.min(days_in_month(new_year, new_month));
    new_year * 10_000 + new_month * 100 + new_day
}

fn days_in_month(year: i32, month: i32) -> i32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if is_leap_year(year) => 29,
        2 => 28,
        _ => 30,
    }
}

fn is_leap_year(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}

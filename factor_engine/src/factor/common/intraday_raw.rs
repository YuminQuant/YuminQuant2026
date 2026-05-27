use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

use crate::core::{
    AssetClass, DatasetId, FactorRowKey, IntradayDailyRawSeries, IntradayDailyRawSpec,
};
use crate::data::{ColumnData, Table};
use crate::error::{err, Result};

#[derive(Clone, Debug)]
pub struct RequestedRawIds<'a> {
    raw_ids: BTreeSet<&'a str>,
}

impl<'a> RequestedRawIds<'a> {
    pub fn new(raw_ids: &'a [String], known_raw_ids: &[&str]) -> Self {
        let known = known_raw_ids.iter().copied().collect::<BTreeSet<_>>();
        let raw_ids = raw_ids
            .iter()
            .map(String::as_str)
            .filter(|raw_id| known.contains(raw_id))
            .collect::<BTreeSet<_>>();
        Self { raw_ids }
    }

    pub fn contains(&self, raw_id: &str) -> bool {
        self.raw_ids.contains(raw_id)
    }

    pub fn contains_any(&self, raw_ids: &[&str]) -> bool {
        raw_ids.iter().any(|raw_id| self.contains(raw_id))
    }

    pub fn is_empty(&self) -> bool {
        self.raw_ids.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &'a str> + '_ {
        self.raw_ids.iter().copied()
    }
}

pub fn stock_minute_raw_spec(
    raw_id: &str,
    version: &str,
    columns: &[&str],
    window_days: usize,
) -> IntradayDailyRawSpec {
    IntradayDailyRawSpec {
        raw_id: raw_id.to_string(),
        version: version.to_string(),
        asset_class: AssetClass::Stock,
        source_dataset: DatasetId::StockMinute1m,
        columns: columns.iter().map(|value| (*value).to_string()).collect(),
        window_days,
    }
}

pub fn intraday_daily_raw_series_to_table(series_list: &[IntradayDailyRawSeries]) -> Result<Table> {
    let mut raw_columns = BTreeSet::new();
    let mut rows: BTreeMap<(i32, String), BTreeMap<String, Option<f64>>> = BTreeMap::new();

    for series in series_list {
        raw_columns.insert(series.spec.raw_id.clone());
        for item in &series.values {
            let FactorRowKey::Daily {
                trade_date,
                ts_code,
            } = &item.key
            else {
                return Err(err(
                    "intraday daily raw series only supports daily row keys",
                ));
            };
            rows.entry((*trade_date, ts_code.clone()))
                .or_default()
                .insert(series.spec.raw_id.clone(), item.value);
        }
    }

    let raw_columns = raw_columns.into_iter().collect::<Vec<_>>();
    let len = rows.len();
    let mut trade_dates = Vec::with_capacity(len);
    let mut ts_codes = Vec::with_capacity(len);
    let mut values = raw_columns
        .iter()
        .map(|column| (column.clone(), Vec::with_capacity(len)))
        .collect::<BTreeMap<_, _>>();

    for ((trade_date, ts_code), row) in rows {
        trade_dates.push(Some(trade_date));
        ts_codes.push(Some(ts_code));
        for column in &raw_columns {
            values
                .get_mut(column)
                .expect("raw column initialized")
                .push(row.get(column).copied().unwrap_or(None));
        }
    }

    let mut table = Table::empty();
    table.insert("trade_date", ColumnData::I32(trade_dates))?;
    table.insert("ts_code", ColumnData::Utf8(ts_codes))?;
    for (column, values) in values {
        table.insert(column, ColumnData::F64(values))?;
    }
    Ok(table)
}

pub fn clean(value: Option<f64>) -> Option<f64> {
    value.filter(|value| !value.is_nan())
}

pub fn pct_change_at(values: &[Option<f64>], idx: usize) -> Option<f64> {
    if idx == 0 {
        return None;
    }
    let current = clean(values[idx])?;
    let previous = clean(values[idx - 1])?;
    (previous.abs() > f64::EPSILON).then_some(current / previous - 1.0)
}

pub fn quantile_linear(values: &mut [f64], quantile: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let q = quantile.clamp(0.0, 1.0);
    let position = q * (values.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    let cmp = |left: &f64, right: &f64| left.partial_cmp(right).unwrap_or(Ordering::Equal);
    let lower_value = *values.select_nth_unstable_by(lower, cmp).1;
    if lower == upper {
        return Some(lower_value);
    }
    let upper_value = *values.select_nth_unstable_by(upper, cmp).1;
    let weight = position - lower as f64;
    Some(lower_value * (1.0 - weight) + upper_value * weight)
}

pub fn mean(values: impl IntoIterator<Item = f64>) -> Option<f64> {
    let mut sum = 0.0;
    let mut count = 0usize;
    for value in values {
        if value.is_nan() {
            continue;
        }
        sum += value;
        count += 1;
    }
    (count > 0).then_some(sum / count as f64)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use crate::core::{FactorRowKey, FactorValue, IntradayDailyRawSeries};

    use super::{intraday_daily_raw_series_to_table, quantile_linear, stock_minute_raw_spec};

    #[test]
    fn quantile_linear_uses_interpolation_without_full_sort_requirement() {
        let mut values = vec![10.0, 1.0, 5.0, 9.0, 7.0];
        assert_eq!(quantile_linear(&mut values, 0.5), Some(7.0));

        let mut values = (1..=10).map(|value| value as f64).collect::<Vec<_>>();
        assert_eq!(quantile_linear(&mut values, 0.9), Some(9.1));
    }

    #[test]
    fn raw_series_to_table_merges_raw_columns_by_date_and_code() {
        let left = IntradayDailyRawSeries {
            spec: stock_minute_raw_spec("left", "0.1.0", &["close"], 1),
            values: vec![FactorValue {
                key: FactorRowKey::Daily {
                    trade_date: 20260424,
                    ts_code: "000001.SZ".to_string(),
                },
                value: Some(1.0),
            }],
        };
        let right = IntradayDailyRawSeries {
            spec: stock_minute_raw_spec("right", "0.1.0", &["close"], 1),
            values: vec![FactorValue {
                key: FactorRowKey::Daily {
                    trade_date: 20260424,
                    ts_code: "000001.SZ".to_string(),
                },
                value: Some(2.0),
            }],
        };

        let table = intraday_daily_raw_series_to_table(&[left, right]).unwrap();
        assert_eq!(table.len, 1);
        assert!(matches!(
            table.columns.get("left"),
            Some(crate::data::ColumnData::F64(values)) if values == &vec![Some(1.0)]
        ));
        assert!(matches!(
            table.columns.get("right"),
            Some(crate::data::ColumnData::F64(values)) if values == &vec![Some(2.0)]
        ));
        assert_eq!(
            table.columns.keys().cloned().collect::<Vec<_>>(),
            BTreeMap::from([
                ("left".to_string(), ()),
                ("right".to_string(), ()),
                ("trade_date".to_string(), ()),
                ("ts_code".to_string(), ())
            ])
            .keys()
            .cloned()
            .collect::<Vec<_>>()
        );
    }
}

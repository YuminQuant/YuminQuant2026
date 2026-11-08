use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow_array::{ArrayRef, Float64Array, Int32Array, RecordBatch, StringArray};
use arrow_schema::DataType;
use parquet::basic::Compression;
use parquet::file::{metadata::KeyValue, properties::WriterProperties};

use crate::calendar::TradingCalendar;
use crate::config::EngineConfig;
use crate::data::parquet_io::{read_parquet_with_null_types, write_parquet_batch};
use crate::data::Table;
use crate::error::{err, Result};
use crate::factor::common::intraday_raw::quantile_linear;
use crate::factor::future::minute_input::product;
use crate::operators::time_series::pearson_corr_finite;

#[derive(Clone, Debug)]
pub struct FutureMinuteBacktestRequest {
    pub start_date: i32,
    pub end_date: i32,
    pub factor_columns: Vec<String>,
    pub label_column: String,
    pub products: Vec<String>,
    pub factor_root: Option<PathBuf>,
    pub label_root: Option<PathBuf>,
    pub output_dir: Option<PathBuf>,
}

#[derive(Default)]
struct Mean {
    sum: f64,
    count: usize,
}

impl Mean {
    fn push(&mut self, value: Option<f64>) {
        if let Some(value) = value {
            self.sum += value;
            self.count += 1;
        }
    }

    fn value(&self) -> Option<f64> {
        (self.count > 0).then(|| self.sum / self.count as f64)
    }
}

#[derive(Default)]
struct Metrics {
    ic: Mean,
    cosine_similarity: Mean,
    upper: Mean,
    lower: Mean,
}

impl Metrics {
    fn add_pair_metrics(&mut self, rows: &[usize], factor: &[Option<f64>], label: &[Option<f64>]) {
        let (x, y): (Vec<_>, Vec<_>) = rows
            .iter()
            .filter_map(|&row| factor[row].zip(label[row]))
            .unzip();
        if x.len() >= 2 {
            self.ic.push(pearson_corr_finite(&x, &y));
            self.cosine_similarity.push(cosine_similarity(&x, &y));
        }
    }

    fn add_tails(&mut self, rows: &[usize], factor: &[Option<f64>], label: &[Option<f64>]) {
        let mut values: Vec<_> = rows.iter().filter_map(|&row| factor[row]).collect();
        let Some(lower) = quantile_linear(&mut values, 0.1) else {
            return;
        };
        let upper = quantile_linear(&mut values, 0.9).expect("nonempty factors");
        if lower >= upper {
            if rows
                .iter()
                .any(|&row| factor[row].is_some() && label[row].is_some())
            {
                self.upper.push(Some(0.0));
                self.lower.push(Some(0.0));
            }
            return;
        }
        for &row in rows {
            if let Some(value) = factor[row] {
                if value >= upper {
                    self.upper.push(label[row]);
                } else if value <= lower {
                    self.lower.push(label[row]);
                }
            }
        }
    }

    fn average_tails(&mut self, other: &Self) {
        self.upper.push(other.upper.value());
        self.lower.push(other.lower.value());
    }
}

fn cosine_similarity(x: &[f64], y: &[f64]) -> Option<f64> {
    // Scale each vector before squaring to keep the result stable for large or tiny values.
    let max_abs = |values: &[f64]| values.iter().map(|v| v.abs()).fold(0.0, f64::max);
    let x_scale = max_abs(x);
    let y_scale = max_abs(y);
    if x_scale == 0.0 || y_scale == 0.0 {
        return None;
    }
    let (mut dot, mut x_squared, mut y_squared) = (0.0, 0.0, 0.0);
    for (&x, &y) in x.iter().zip(y) {
        let x = x / x_scale;
        let y = y / y_scale;
        dot += x * y;
        x_squared += x * x;
        y_squared += y * y;
    }
    Some((dot / (x_squared.sqrt() * y_squared.sqrt())).clamp(-1.0, 1.0))
}

struct Observation {
    date: i32,
    code: String,
    metrics: Metrics,
}

pub fn run(config: &EngineConfig, request: &FutureMinuteBacktestRequest) -> Result<Vec<PathBuf>> {
    if request.start_date > request.end_date {
        return Err(err("start date must be on or before end date"));
    }
    let factors = selected_columns(&request.factor_columns, "factor")?;
    let mut directory_names = BTreeSet::new();
    for &factor in &factors {
        if !factor
            .bytes()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, b'_' | b'-'))
            || !directory_names.insert(factor.to_ascii_lowercase())
        {
            return Err(err(format!(
                "invalid or ambiguous factor directory name: {factor}"
            )));
        }
    }
    let label = selected_columns(std::slice::from_ref(&request.label_column), "label")?[0];
    let products: BTreeSet<_> = request
        .products
        .iter()
        .map(|code| code.to_ascii_uppercase())
        .collect();
    let calendar = TradingCalendar::load(&config.data_root, &config.future_calendar_exchange)?;
    let dates = calendar.open_dates_between(request.start_date, request.end_date);
    if dates.is_empty() {
        return Err(err("no futures trading dates in the requested range"));
    }
    let factor_root = request
        .factor_root
        .as_deref()
        .unwrap_or(&config.factor_root);
    let label_root = request.label_root.as_deref().unwrap_or(&config.label_root);
    let mut summaries: Vec<Vec<Observation>> = (0..factors.len()).map(|_| Vec::new()).collect();
    let mut seen_products = BTreeSet::new();
    for date in dates {
        let factor_table = read_day(factor_root, date, &factors)?;
        let label_table = read_day(label_root, date, &[label])?;
        let factor_keys = minute_keys(&factor_table, date)?;
        let label_keys = minute_keys(&label_table, date)?;
        let label_index: HashMap<_, _> = label_keys
            .into_iter()
            .enumerate()
            .map(|(i, key)| (key, i))
            .collect();
        let aligned: Vec<_> = factor_keys
            .iter()
            .map(|key| label_index.get(key).copied())
            .collect();
        if !aligned.iter().any(Option::is_some) {
            return Err(err(format!(
                "no matching futures factor and label minute keys on {date}"
            )));
        }

        let mut by_minute = BTreeMap::<&str, Vec<usize>>::new();
        let mut product_rows = BTreeMap::<&str, Vec<usize>>::new();
        let mut product_codes = HashMap::new();
        for (row, &(time, contract)) in factor_keys.iter().enumerate() {
            let code = product_codes
                .entry(contract)
                .or_insert_with(|| product(contract));
            let Some(code) = code.as_deref() else {
                return Err(err(format!(
                    "invalid futures contract code {contract} on {date}"
                )));
            };
            if products.is_empty()
                || products.contains(code)
                || products.contains(code.split_once('.').unwrap().0)
            {
                by_minute.entry(time).or_default().push(row);
            }
        }
        for rows in by_minute.values() {
            for &row in rows {
                let code = product_codes[factor_keys[row].1]
                    .as_deref()
                    .expect("validated product");
                product_rows.entry(code).or_default().push(row);
            }
        }
        seen_products.extend(product_rows.keys().map(|code| code.to_string()));
        let label_values = finite_column(&label_table, label)?;
        let returns: Vec<_> = aligned
            .iter()
            .map(|index| index.and_then(|i| label_values[i]))
            .collect();

        for (&factor, summary) in factors.iter().zip(&mut summaries) {
            let values = finite_column(&factor_table, factor)?;
            let mut all = Metrics::default();
            for rows in by_minute.values() {
                all.add_pair_metrics(rows, &values, &returns);
            }
            for (&code, rows) in &product_rows {
                let mut local = Metrics::default();
                local.add_pair_metrics(rows, &values, &returns);
                local.add_tails(rows, &values, &returns);
                all.average_tails(&local);
                summary.push(Observation {
                    date,
                    code: code.to_string(),
                    metrics: local,
                });
            }
            summary.push(Observation {
                date,
                code: "all".to_string(),
                metrics: all,
            });
        }
    }
    for code in &products {
        if !seen_products
            .iter()
            .any(|seen| seen == code || seen.split_once('.').unwrap().0 == code)
        {
            return Err(err(format!(
                "no futures factor rows for product {code} in the requested range"
            )));
        }
    }
    let output_dir = request
        .output_dir
        .clone()
        .unwrap_or_else(|| config.data_root.join("backtest").join("future"));
    factors
        .into_iter()
        .zip(summaries)
        .map(|(factor, rows)| {
            let path = output_dir.join(factor).join("summary.parquet");
            let props = WriterProperties::builder()
                .set_compression(Compression::SNAPPY)
                .set_key_value_metadata(Some(vec![KeyValue::new(
                    "label_id".to_string(),
                    label.to_string(),
                )]))
                .build();
            write_parquet_batch(&path, &observations_batch(&rows)?, props)?;
            Ok(path)
        })
        .collect()
}

fn selected_columns<'a>(columns: &'a [String], kind: &str) -> Result<Vec<&'a str>> {
    let selected: BTreeSet<_> = columns.iter().map(String::as_str).collect();
    if selected.is_empty()
        || selected
            .iter()
            .any(|name| name.is_empty() || matches!(*name, "trade_date" | "trade_time" | "ts_code"))
    {
        return Err(err(format!("provide one or more {kind} value columns")));
    }
    Ok(selected.into_iter().collect())
}

fn read_day(root: &Path, date: i32, values: &[&str]) -> Result<Table> {
    let path = root
        .join("future")
        .join("minute_1m")
        .join((date / 10000).to_string())
        .join(format!("{date}.parquet"));
    let projection: Vec<String> = ["trade_date", "trade_time", "ts_code"]
        .into_iter()
        .chain(values.iter().copied())
        .map(str::to_string)
        .collect();
    let null_types = values
        .iter()
        .map(|&name| (name.to_string(), DataType::Float64))
        .collect();
    let table = read_parquet_with_null_types(&path, &projection, &null_types)?;
    for column in &projection {
        if !table.columns.contains_key(column) {
            return Err(err(format!(
                "missing column {column} in {}",
                path.display()
            )));
        }
    }
    Ok(table)
}

fn minute_keys(table: &Table, date: i32) -> Result<Vec<(&str, &str)>> {
    let dates = table.required_i32_date_cast("trade_date")?;
    let times = table.required_utf8("trade_time")?;
    let codes = table.required_utf8("ts_code")?;
    let mut seen = BTreeSet::new();
    dates
        .iter()
        .zip(times)
        .zip(codes)
        .map(
            |((day, time), code)| match (day, time.as_deref(), code.as_deref()) {
                (Some(day), Some(time), Some(code))
                    if *day == date && !time.is_empty() && !code.is_empty() =>
                {
                    if !seen.insert((time, code)) {
                        return Err(err(format!(
                            "duplicate futures minute key on {date}: {time} {code}"
                        )));
                    }
                    Ok((time, code))
                }
                _ => Err(err(format!(
                    "invalid futures minute key in {date} partition"
                ))),
            },
        )
        .collect()
}

fn finite_column(table: &Table, name: &str) -> Result<Vec<Option<f64>>> {
    Ok(table
        .required_f64_cast(name)?
        .into_iter()
        .map(|value| value.filter(|v| v.is_finite()))
        .collect())
}

fn observations_batch(rows: &[Observation]) -> Result<RecordBatch> {
    let columns: [(&str, ArrayRef); 6] = [
        (
            "trade_date",
            Arc::new(Int32Array::from_iter_values(
                rows.iter().map(|row| row.date),
            )),
        ),
        (
            "ts_code",
            Arc::new(StringArray::from_iter_values(
                rows.iter().map(|row| row.code.as_str()),
            )),
        ),
        (
            "ic",
            Arc::new(Float64Array::from_iter(
                rows.iter().map(|row| row.metrics.ic.value()),
            )),
        ),
        (
            "cosic",
            Arc::new(Float64Array::from_iter(
                rows.iter().map(|row| row.metrics.cosine_similarity.value()),
            )),
        ),
        (
            "long_rtn",
            Arc::new(Float64Array::from_iter(
                rows.iter().map(|row| row.metrics.upper.value()),
            )),
        ),
        (
            "short_rtn",
            Arc::new(Float64Array::from_iter(
                rows.iter().map(|row| row.metrics.lower.value()),
            )),
        ),
    ];
    Ok(RecordBatch::try_from_iter(columns)?)
}

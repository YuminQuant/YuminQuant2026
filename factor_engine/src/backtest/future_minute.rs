use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
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
    pub quantile_mode: QuantileMode,
}

#[derive(Clone, Copy, Debug, Default)]
pub enum QuantileMode {
    Daily,
    #[default]
    Trailing5,
}

impl QuantileMode {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "daily" => Ok(Self::Daily),
            "trailing-5" => Ok(Self::Trailing5),
            _ => Err(err("--quantile-mode must be daily or trailing-5")),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Daily => "daily",
            Self::Trailing5 => "trailing-5",
        }
    }
}

const TRAILING_DAYS: usize = 5;

#[derive(Clone, Copy)]
struct TailThresholds {
    lower: f64,
    upper: f64,
}

impl TailThresholds {
    fn for_day(rows: &[usize], factor: &[Option<f64>]) -> Option<Self> {
        let mut values: Vec<_> = rows.iter().filter_map(|&row| factor[row]).collect();
        Some(Self {
            lower: quantile_linear(&mut values, 0.1)?,
            upper: quantile_linear(&mut values, 0.9)?,
        })
    }
}

type DailyThresholds = BTreeMap<String, TailThresholds>;

#[derive(Default)]
struct ThresholdHistory {
    days: VecDeque<DailyThresholds>,
}

impl ThresholdHistory {
    fn mean(&self, code: &str) -> Option<TailThresholds> {
        if self.days.len() != TRAILING_DAYS {
            return None;
        }
        let (mut lower, mut upper) = (0.0, 0.0);
        for day in &self.days {
            let thresholds = day.get(code)?;
            lower += thresholds.lower;
            upper += thresholds.upper;
        }
        Some(TailThresholds {
            lower: lower / TRAILING_DAYS as f64,
            upper: upper / TRAILING_DAYS as f64,
        })
    }

    fn advance(&mut self, today: DailyThresholds) {
        if self.days.len() == TRAILING_DAYS {
            self.days.pop_front();
        }
        self.days.push_back(today);
    }
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

    fn add_tails(
        &mut self,
        rows: &[usize],
        factor: &[Option<f64>],
        label: &[Option<f64>],
        TailThresholds { lower, upper }: TailThresholds,
        zero_empty_tails: bool,
    ) {
        if lower < upper {
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
        if (lower >= upper || zero_empty_tails)
            && (self.upper.count == 0 || self.lower.count == 0)
            && rows
                .iter()
                .any(|&row| factor[row].is_some() && label[row].is_some())
        {
            if self.upper.count == 0 {
                self.upper.push(Some(0.0));
            }
            if self.lower.count == 0 {
                self.lower.push(Some(0.0));
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

#[derive(Default)]
struct DayGroups<'a> {
    by_minute: BTreeMap<&'a str, Vec<usize>>,
    by_product: BTreeMap<String, Vec<usize>>,
}

struct FactorSummary {
    rows: Vec<Observation>,
    history: Option<ThresholdHistory>,
}

impl FactorSummary {
    fn new(mode: QuantileMode) -> Self {
        Self {
            rows: Vec::new(),
            history: matches!(mode, QuantileMode::Trailing5).then(ThresholdHistory::default),
        }
    }

    fn add_day(
        &mut self,
        date: i32,
        groups: &DayGroups<'_>,
        factor: &[Option<f64>],
        returns: Option<&[Option<f64>]>,
    ) {
        let mut today = DailyThresholds::new();
        let mut all = Metrics::default();
        for (code, rows) in &groups.by_product {
            let daily = TailThresholds::for_day(rows, factor);
            if self.history.is_some() {
                if let Some(daily) = daily {
                    today.insert(code.clone(), daily);
                }
            }
            let Some(returns) = returns else { continue };
            let thresholds = match &self.history {
                Some(history) => history.mean(code),
                None => daily,
            };
            let mut local = Metrics::default();
            local.add_pair_metrics(rows, factor, returns);
            if let Some(thresholds) = thresholds {
                local.add_tails(rows, factor, returns, thresholds, self.history.is_some());
            }
            all.average_tails(&local);
            self.rows.push(Observation {
                date,
                code: code.clone(),
                metrics: local,
            });
        }
        if let Some(returns) = returns {
            for rows in groups.by_minute.values() {
                all.add_pair_metrics(rows, factor, returns);
            }
            self.rows.push(Observation {
                date,
                code: "all".to_string(),
                metrics: all,
            });
        }
        // Publish today's thresholds only after using the previous sessions.
        if let Some(history) = &mut self.history {
            history.advance(today);
        }
    }
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
    let dates = evaluation_dates(&calendar, request);
    if dates.last().is_none_or(|&date| date < request.start_date) {
        return Err(err("no futures trading dates in the requested range"));
    }
    let factor_root = request
        .factor_root
        .as_deref()
        .unwrap_or(&config.factor_root);
    let label_root = request.label_root.as_deref().unwrap_or(&config.label_root);
    let mut summaries: Vec<_> = factors
        .iter()
        .map(|_| FactorSummary::new(request.quantile_mode))
        .collect();
    let mut seen_products = BTreeSet::new();
    for date in dates {
        let evaluate = date >= request.start_date;
        let factor_path = minute_path(factor_root, date);
        if !evaluate && !crate::artifact_io::exists(&factor_path)? {
            for summary in &mut summaries {
                if let Some(history) = &mut summary.history {
                    history.advance(DailyThresholds::new());
                }
            }
            continue;
        }

        let factor_table = read_day(&factor_path, &factors)?;
        let factor_keys = minute_keys(&factor_table, date)?;
        let groups = group_rows(&factor_keys, &products, date)?;
        let returns = if evaluate {
            seen_products.extend(groups.by_product.keys().cloned());
            Some(aligned_returns(label_root, label, date, &factor_keys)?)
        } else {
            None
        };
        for (&factor, summary) in factors.iter().zip(&mut summaries) {
            let values = finite_column(&factor_table, factor)?;
            summary.add_day(date, &groups, &values, returns.as_deref());
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
    let output_dir = request.output_dir.clone().unwrap_or_else(|| {
        let root = config.data_root.join("backtest").join("future");
        match request.quantile_mode {
            QuantileMode::Daily => root,
            QuantileMode::Trailing5 => root.join("trailing-5"),
        }
    });
    factors
        .into_iter()
        .zip(summaries)
        .map(|(factor, summary)| {
            let path = output_dir.join(factor).join("summary.parquet");
            let props = WriterProperties::builder()
                .set_compression(Compression::SNAPPY)
                .set_key_value_metadata(Some(vec![
                    KeyValue::new("label_id".to_string(), label.to_string()),
                    KeyValue::new(
                        "quantile_mode".to_string(),
                        request.quantile_mode.as_str().to_string(),
                    ),
                ]))
                .build();
            write_parquet_batch(&path, &observations_batch(&summary.rows)?, props)?;
            Ok(path)
        })
        .collect()
}

fn evaluation_dates(calendar: &TradingCalendar, request: &FutureMinuteBacktestRequest) -> Vec<i32> {
    let start = match request.quantile_mode {
        QuantileMode::Daily => request.start_date,
        QuantileMode::Trailing5 => calendar.warmup_start(request.start_date, TRAILING_DAYS),
    };
    calendar.open_dates_between(start, request.end_date)
}

fn group_rows<'a>(
    keys: &[(&'a str, &str)],
    products: &BTreeSet<String>,
    date: i32,
) -> Result<DayGroups<'a>> {
    let mut groups = DayGroups::default();
    let mut product_codes = HashMap::new();
    for (row, &(time, contract)) in keys.iter().enumerate() {
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
            groups.by_minute.entry(time).or_default().push(row);
        }
    }
    let mut product_rows = BTreeMap::<&str, Vec<usize>>::new();
    for rows in groups.by_minute.values() {
        for &row in rows {
            let code = product_codes[keys[row].1]
                .as_deref()
                .expect("validated product");
            product_rows.entry(code).or_default().push(row);
        }
    }
    groups.by_product = product_rows
        .into_iter()
        .map(|(code, rows)| (code.to_string(), rows))
        .collect();
    Ok(groups)
}

fn aligned_returns(
    root: &Path,
    label: &str,
    date: i32,
    factor_keys: &[(&str, &str)],
) -> Result<Vec<Option<f64>>> {
    let label_table = read_day(&minute_path(root, date), &[label])?;
    let label_index: HashMap<_, _> = minute_keys(&label_table, date)?
        .into_iter()
        .enumerate()
        .map(|(i, key)| (key, i))
        .collect();
    let values = finite_column(&label_table, label)?;
    let mut matched = false;
    let returns = factor_keys
        .iter()
        .map(|key| {
            let index = label_index.get(key)?;
            matched = true;
            values[*index]
        })
        .collect();
    if !matched {
        return Err(err(format!(
            "no matching futures factor and label minute keys on {date}"
        )));
    }
    Ok(returns)
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

fn minute_path(root: &Path, date: i32) -> PathBuf {
    root.join("future")
        .join("minute_1m")
        .join((date / 10000).to_string())
        .join(format!("{date}.parquet"))
}

fn read_day(path: &Path, values: &[&str]) -> Result<Table> {
    let projection: Vec<String> = ["trade_date", "trade_time", "ts_code"]
        .into_iter()
        .chain(values.iter().copied())
        .map(str::to_string)
        .collect();
    let null_types = values
        .iter()
        .map(|&name| (name.to_string(), DataType::Float64))
        .collect();
    let table = read_parquet_with_null_types(path, &projection, &null_types)?;
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

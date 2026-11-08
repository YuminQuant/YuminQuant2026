use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow_array::{ArrayRef, Float64Array, Int32Array, RecordBatch, StringArray, UInt64Array};
use arrow_schema::DataType;
use parquet::basic::Compression;
use parquet::file::{metadata::KeyValue, properties::WriterProperties};
use rayon::prelude::*;

use crate::backtest::data::{
    active_factor_matches_tags, external_factor_ids_from_root, factor_metadata_matches,
};
use crate::backtest::request::DEFAULT_BACKTEST_THREADS;
use crate::calendar::TradingCalendar;
use crate::config::EngineConfig;
use crate::core::{AssetClass, Frequency};
use crate::data::parquet_io::{read_parquet_with_null_types, write_parquet_batch};
use crate::data::Table;
use crate::error::{err, Result};
use crate::factor::common::intraday_raw::quantile_linear;
use crate::factor::future::minute_input::product;
use crate::operators::time_series::pearson_corr_finite;
use crate::storage::FactorMetadata;

#[derive(Clone, Debug)]
pub struct FutureMinuteBacktestRequest {
    pub start_date: i32,
    pub end_date: i32,
    pub factor_selection: FactorSelection,
    pub label_column: String,
    pub products: Vec<String>,
    pub factor_root: Option<PathBuf>,
    pub label_root: Option<PathBuf>,
    pub output_dir: Option<PathBuf>,
    pub quantile_mode: QuantileMode,
    pub quantile: u8,
    pub factor_batch_size: usize,
    pub date_batch_size: usize,
    pub threads: Option<usize>,
    pub profile: bool,
}

#[derive(Clone, Debug)]
pub enum FactorSelection {
    Columns(Vec<String>),
    Tags(Vec<String>),
    All,
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
pub const DEFAULT_TAIL_QUANTILE: u8 = 95;

#[derive(Clone, Copy)]
struct TailThresholds {
    lower: f64,
    upper: f64,
}

impl TailThresholds {
    fn for_day(rows: &[usize], factor: &[Option<f64>], quantile: u8) -> Option<Self> {
        let mut values: Vec<_> = rows.iter().filter_map(|&row| factor[row]).collect();
        Some(Self {
            lower: quantile_linear(&mut values, f64::from(100 - quantile) / 100.0)?,
            upper: quantile_linear(&mut values, f64::from(quantile) / 100.0)?,
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
    trigger_count: Option<u64>,
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
        split_at_median: bool,
        zero_empty_tails: bool,
    ) {
        let mut has_pair = false;
        if lower < upper || split_at_median {
            for &row in rows {
                if let Some((value, returns)) = factor[row].zip(label[row]) {
                    has_pair = true;
                    if value > upper || (!split_at_median && value == upper) {
                        self.upper.push(Some(returns));
                    } else if value < lower || (!split_at_median && value == lower) {
                        self.lower.push(Some(returns));
                    }
                }
            }
        } else {
            has_pair = rows
                .iter()
                .any(|&row| factor[row].is_some() && label[row].is_some());
        }
        if !has_pair {
            return;
        }
        self.trigger_count = Some((self.upper.count + self.lower.count) as u64);
        if lower >= upper || zero_empty_tails {
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
        if let Some(count) = other.trigger_count {
            *self.trigger_count.get_or_insert(0) += count;
        }
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
struct DayGroups {
    all: Vec<usize>,
    by_product: BTreeMap<String, Vec<usize>>,
    by_sector: BTreeMap<&'static str, Vec<usize>>,
}

struct FactorSummary {
    rows: Vec<Observation>,
    history: Option<ThresholdHistory>,
    quantile: u8,
}

impl FactorSummary {
    fn new(mode: QuantileMode, quantile: u8) -> Self {
        Self {
            rows: Vec::new(),
            history: matches!(mode, QuantileMode::Trailing5).then(ThresholdHistory::default),
            quantile,
        }
    }

    fn add_day(
        &mut self,
        date: i32,
        groups: &DayGroups,
        factor: &[Option<f64>],
        returns: Option<&[Option<f64>]>,
    ) {
        let mut today = DailyThresholds::new();
        let mut all = Metrics::default();
        for (code, rows) in &groups.by_product {
            let daily = TailThresholds::for_day(rows, factor, self.quantile);
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
                local.add_tails(
                    rows,
                    factor,
                    returns,
                    thresholds,
                    self.quantile == 50,
                    self.history.is_some(),
                );
            }
            all.average_tails(&local);
            self.rows.push(Observation {
                date,
                code: code.clone(),
                metrics: local,
            });
        }
        if let Some(returns) = returns {
            all.add_pair_metrics(&groups.all, factor, returns);
            self.rows.push(Observation {
                date,
                code: "all".to_string(),
                metrics: all,
            });
            for (sector, rows) in &groups.by_sector {
                let mut metrics = Metrics::default();
                metrics.add_pair_metrics(rows, factor, returns);
                self.rows.push(Observation {
                    date,
                    code: format!("sector:{sector}"),
                    metrics,
                });
            }
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
    if !(50..=99).contains(&request.quantile) {
        return Err(err("--quantile must be an integer from 50 to 99"));
    }
    if request.factor_batch_size == 0 || request.date_batch_size == 0 {
        return Err(err("backtest batch sizes must be greater than 0"));
    }
    let started = std::time::Instant::now();
    let resolved =
        resolve_factor_columns(&request.factor_selection, request.factor_root.as_deref())?;
    let factors = selected_columns(&resolved, "factor")?;
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
    let sectors = sector_by_product()?;
    let inputs = EvaluationInputs {
        start_date: request.start_date,
        factor_root,
        label_root,
        label,
        products: &products,
        sectors: &sectors,
    };
    let pool =
        crate::resources::thread_pool(Some(request.threads.unwrap_or(DEFAULT_BACKTEST_THREADS)))?;
    let output_dir = request.output_dir.clone().unwrap_or_else(|| {
        let root = config
            .data_root
            .join("backtest")
            .join("future")
            .join("minute_1m");
        match request.quantile_mode {
            QuantileMode::Daily => root.join("daily"),
            QuantileMode::Trailing5 => root,
        }
    });
    let mut paths = Vec::new();
    for batch in factors.chunks(request.factor_batch_size) {
        let summaries = pool.install(|| evaluate_batch(&inputs, batch, &dates, request))?;
        for (&factor, summary) in batch.iter().zip(summaries) {
            let path = output_dir.join(factor).join("summary.parquet");
            write_summary(&path, &summary.rows, label, request)?;
            paths.push(path);
        }
    }
    if request.profile {
        eprintln!(
            "future_backtest factors={} factor_batches={} date_batches_per_factor_batch={} threads={} elapsed_ms={}",
            factors.len(),
            factors.len().div_ceil(request.factor_batch_size),
            dates.len().div_ceil(request.date_batch_size),
            pool.current_num_threads(),
            started.elapsed().as_millis(),
        );
    }
    Ok(paths)
}

fn resolve_factor_columns(
    selection: &FactorSelection,
    external: Option<&Path>,
) -> Result<Vec<String>> {
    match (selection, external) {
        (FactorSelection::Columns(columns), Some(_)) => return Ok(columns.clone()),
        (FactorSelection::Tags(_), Some(_)) => {
            return Err(err(
                "--tags cannot be used with --factor-root; use --factors or --all-factors",
            ));
        }
        (FactorSelection::All, Some(root)) => {
            return external_factor_ids_from_root(root, AssetClass::Future, Frequency::Minute1);
        }
        _ => {}
    }

    let metadata = crate::engine::available_specs()
        .iter()
        .filter(|spec| {
            spec.asset_class == AssetClass::Future && spec.frequency == Frequency::Minute1
        })
        .map(FactorMetadata::from_spec)
        .collect::<Vec<_>>();
    if let FactorSelection::Columns(columns) = selection {
        // Explicit columns also support locally supplied factors outside the registry.
        return columns
            .iter()
            .map(|column| {
                let Some(row) = metadata
                    .iter()
                    .find(|row| factor_metadata_matches(row, column))
                else {
                    return Ok(column.clone());
                };
                if !active_factor_matches_tags(row, &[]) {
                    return Err(err(format!(
                        "deprecated factor cannot be backtested explicitly: {column}"
                    )));
                }
                Ok(row.output_column.clone())
            })
            .collect();
    }
    let tags = match selection {
        FactorSelection::Tags(tags) => tags.as_slice(),
        _ => &[],
    };
    let columns = metadata
        .into_iter()
        .filter(|row| active_factor_matches_tags(row, tags))
        .map(|row| row.output_column)
        .collect::<Vec<_>>();
    if columns.is_empty() {
        return Err(err("no active futures minute factors match the selection"));
    }
    Ok(columns)
}

struct EvaluationInputs<'a> {
    start_date: i32,
    factor_root: &'a Path,
    label_root: &'a Path,
    label: &'a str,
    products: &'a BTreeSet<String>,
    sectors: &'a HashMap<String, &'static str>,
}

struct EvaluationDay {
    date: i32,
    table: Table,
    groups: DayGroups,
    returns: Option<Vec<Option<f64>>>,
}

impl EvaluationInputs<'_> {
    fn read_day(&self, date: i32, factors: &[&str]) -> Result<Option<EvaluationDay>> {
        let evaluate = date >= self.start_date;
        let path = minute_path(self.factor_root, date);
        if !evaluate && !crate::artifact_io::exists(&path)? {
            return Ok(None);
        }
        let table = read_day(&path, factors)?;
        let keys = minute_keys(&table, date)?;
        let groups = group_rows(&keys, self.products, evaluate.then_some(self.sectors), date)?;
        let returns = evaluate
            .then(|| aligned_returns(self.label_root, self.label, date, &keys))
            .transpose()?;
        Ok(Some(EvaluationDay {
            date,
            table,
            groups,
            returns,
        }))
    }
}

fn evaluate_batch(
    inputs: &EvaluationInputs<'_>,
    factors: &[&str],
    dates: &[i32],
    request: &FutureMinuteBacktestRequest,
) -> Result<Vec<FactorSummary>> {
    let mut summaries = factors
        .iter()
        .map(|_| FactorSummary::new(request.quantile_mode, request.quantile))
        .collect::<Vec<_>>();
    let mut seen_products = BTreeSet::new();
    for date_batch in dates.chunks(request.date_batch_size) {
        // Limit resident minute tables to the thread budget, even for long date batches.
        for wave in date_batch.chunks(rayon::current_num_threads()) {
            let days = wave
                .par_iter()
                .map(|&date| inputs.read_day(date, factors))
                .collect::<Result<Vec<_>>>()?;
            for day in days {
                let Some(day) = day else {
                    for summary in &mut summaries {
                        if let Some(history) = &mut summary.history {
                            history.advance(DailyThresholds::new());
                        }
                    }
                    continue;
                };
                if day.returns.is_some() {
                    seen_products.extend(day.groups.by_product.keys().cloned());
                }
                // Consume sessions in date order so the rolling window spans batches.
                factors
                    .par_iter()
                    .zip(summaries.par_iter_mut())
                    .try_for_each(|(&factor, summary)| -> Result<()> {
                        let values = finite_column(&day.table, factor)?;
                        summary.add_day(day.date, &day.groups, &values, day.returns.as_deref());
                        Ok(())
                    })?;
            }
        }
    }
    for code in inputs.products {
        if !seen_products
            .iter()
            .any(|seen| seen == code || seen.split_once('.').unwrap().0 == code)
        {
            return Err(err(format!(
                "no futures factor rows for product {code} in the requested range"
            )));
        }
    }
    Ok(summaries)
}

fn write_summary(
    path: &Path,
    rows: &[Observation],
    label: &str,
    request: &FutureMinuteBacktestRequest,
) -> Result<()> {
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .set_key_value_metadata(Some(vec![
            KeyValue::new("label_id".to_string(), label.to_string()),
            KeyValue::new(
                "quantile_mode".to_string(),
                request.quantile_mode.as_str().to_string(),
            ),
            KeyValue::new("tail_quantile".to_string(), request.quantile.to_string()),
        ]))
        .build();
    write_parquet_batch(path, &observations_batch(rows)?, props)
}

fn evaluation_dates(calendar: &TradingCalendar, request: &FutureMinuteBacktestRequest) -> Vec<i32> {
    let start = match request.quantile_mode {
        QuantileMode::Daily => request.start_date,
        QuantileMode::Trailing5 => calendar.warmup_start(request.start_date, TRAILING_DAYS),
    };
    calendar.open_dates_between(start, request.end_date)
}

fn group_rows(
    keys: &[(&str, &str)],
    products: &BTreeSet<String>,
    sectors: Option<&HashMap<String, &'static str>>,
    date: i32,
) -> Result<DayGroups> {
    let mut groups = DayGroups::default();
    let contracts: BTreeSet<_> = keys.iter().map(|&(_, contract)| contract).collect();
    let mut codes = HashMap::new();
    for contract in contracts {
        let code = product(contract).ok_or_else(|| {
            err(format!(
                "invalid futures contract code {contract} on {date}"
            ))
        })?;
        let selected = products.is_empty()
            || products.contains(&code)
            || products.contains(code.split_once('.').unwrap().0);
        codes.insert(contract, selected.then_some(code));
    }
    let mut product_rows = BTreeMap::<&str, Vec<usize>>::new();
    for (row, &(_, contract)) in keys.iter().enumerate() {
        let Some(code) = codes[contract].as_deref() else {
            continue;
        };
        groups.all.push(row);
        product_rows.entry(code).or_default().push(row);
        if let Some(sector) = sectors.and_then(|mapping| mapping.get(code)) {
            groups.by_sector.entry(*sector).or_default().push(row);
        }
    }
    groups.by_product = product_rows
        .into_iter()
        .map(|(code, rows)| (code.to_string(), rows))
        .collect();
    Ok(groups)
}

fn sector_by_product() -> Result<HashMap<String, &'static str>> {
    // The analysis package reads the same sector and exchange tables.
    let doc = include_str!("../../docs/FUTURE_SECTORS.md");
    let mut sectors = HashMap::new();
    let mut exchanges = HashMap::new();
    for line in doc.lines().filter(|line| line.starts_with("| ")) {
        let cells: Vec<_> = line.trim_matches('|').split('|').map(str::trim).collect();
        match cells.as_slice() {
            [abbr, _, _, kind, codes] if matches!(*kind, "商品" | "金融" | "运价") => {
                for code in codes.split(',').map(str::trim) {
                    if sectors.insert(code, *abbr).is_some() {
                        return Err(err(format!("duplicate futures sector for {code}")));
                    }
                }
            }
            [code, _, _, exchange]
                if code.bytes().all(|b| b.is_ascii_uppercase()) && !code.is_empty() =>
            {
                if exchanges.insert(*code, *exchange).is_some() {
                    return Err(err(format!("duplicate futures exchange for {code}")));
                }
            }
            _ => {}
        }
    }
    if sectors.is_empty() || sectors.len() != exchanges.len() {
        return Err(err("futures sector and exchange tables do not match"));
    }
    sectors
        .into_iter()
        .map(|(code, sector)| {
            let exchange = exchanges
                .get(code)
                .ok_or_else(|| err(format!("missing futures exchange for {code}")))?;
            Ok((format!("{code}.{exchange}"), sector))
        })
        .collect()
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
    let columns: [(&str, ArrayRef); 7] = [
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
        (
            "trigger_count",
            Arc::new(UInt64Array::from_iter(
                rows.iter().map(|row| row.metrics.trigger_count),
            )),
        ),
    ];
    Ok(RecordBatch::try_from_iter(columns)?)
}

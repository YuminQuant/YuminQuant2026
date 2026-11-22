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
use crate::backtest::metrics::{daily_factor_stats, FactorStatsDaily};
use crate::backtest::preprocess::coverage_stats;
use crate::backtest::request::DEFAULT_BACKTEST_THREADS;
use crate::calendar::TradingCalendar;
use crate::config::EngineConfig;
use crate::core::{AssetClass, Frequency};
use crate::data::parquet_io::{read_parquet_with_null_types, write_parquet, write_parquet_batch};
use crate::data::{ColumnData, Table};
use crate::error::{err, Result};
use crate::factor::common::intraday_raw::quantile_linear;
use crate::factor::future::minute_input::{
    adjacent_minute, product, FutureMinuteInput, FutureMinuteOptions,
};
use crate::operators::time_series::pearson_corr_finite;
use crate::storage::FactorMetadata;

#[derive(Clone, Debug)]
pub struct FutureMinuteBacktestRequest {
    pub start_date: i32,
    pub end_date: i32,
    pub factor_selection: FactorSelection,
    pub label_column: String,
    pub products: Vec<String>,
    pub pool_id: Option<String>,
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
const HOLDING_LABEL: &str = "future_open_return_1m";
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
    eligible_contract_count: Option<u64>,
    holding_rtn: Mean,
    holding_duration: HoldingDuration,
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
        if let Some(count) = other.eligible_contract_count {
            *self.eligible_contract_count.get_or_insert(0) += count;
        }
    }

    fn add_holding(&mut self, return_bp: Option<f64>, duration: HoldingDuration) {
        self.holding_rtn.push(return_bp);
        self.holding_duration.minutes += duration.minutes;
        self.holding_duration.positions += duration.positions;
    }
}

#[derive(Clone, Copy, Default)]
struct HoldingDuration {
    minutes: u64,
    positions: u64,
}

impl HoldingDuration {
    fn average(self) -> Option<f64> {
        (self.positions > 0).then(|| self.minutes as f64 / self.positions as f64)
    }
}

struct HoldingMinute {
    signal_row: Option<usize>,
    return_bp: Option<f64>,
    continuous: bool,
}

fn evaluate_holding(
    timeline: &[HoldingMinute],
    factor: &[Option<f64>],
    thresholds: TailThresholds,
) -> (Option<f64>, HoldingDuration) {
    let mut direction = 0.0;
    let mut return_bp = 0.0;
    let mut duration = HoldingDuration::default();
    let mut has_signal = false;
    let mut has_return = false;
    let mut missing_held_return = false;
    // A signal on bar i controls the interval from open i+1 to open i+2.
    // Close at the final bar's open and discard signals with no remaining interval.
    for minute in timeline.iter().take(timeline.len().saturating_sub(2)) {
        if let Some(value) = minute.signal_row.and_then(|row| factor[row]) {
            has_signal = true;
            let next_direction = if value > thresholds.upper {
                1.0
            } else if value < thresholds.lower {
                -1.0
            } else {
                direction
            };
            if next_direction != direction {
                duration.positions += 1;
                direction = next_direction;
            }
        }
        has_return |= minute.return_bp.is_some();
        if direction != 0.0 {
            duration.minutes += u64::from(minute.continuous);
            if let Some(value) = minute.return_bp {
                return_bp += direction * value;
            } else {
                missing_held_return = true;
            }
        }
    }
    (
        (has_signal && has_return && !missing_held_return).then_some(return_bp),
        duration,
    )
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
    by_product: BTreeMap<String, ProductRows>,
    by_sector: BTreeMap<&'static str, Vec<usize>>,
}

#[derive(Default)]
struct ProductRows {
    pooled: Vec<usize>,
    contracts: BTreeMap<String, Vec<usize>>,
    sector: Option<&'static str>,
}

impl ProductRows {
    fn eligible_contracts(&self, factor: &[Option<f64>], returns: &[Option<f64>]) -> u64 {
        self.contracts
            .values()
            .filter(|rows| {
                rows.iter()
                    .any(|&row| factor[row].is_some() && returns[row].is_some())
            })
            .count() as u64
    }
}

struct FactorDistribution {
    code: String,
    stats: FactorStatsDaily,
}

struct FactorSummary {
    rows: Vec<Observation>,
    distributions: Vec<FactorDistribution>,
    history: Option<ThresholdHistory>,
    quantile: u8,
}

impl FactorSummary {
    fn new(mode: QuantileMode, quantile: u8) -> Self {
        Self {
            rows: Vec::new(),
            distributions: Vec::new(),
            history: matches!(mode, QuantileMode::Trailing5).then(ThresholdHistory::default),
            quantile,
        }
    }

    fn add_distributions(&mut self, date: i32, factor: &str, groups: &DayGroups, raw: &[Option<f64>]) {
        let pooled = std::iter::once(("all".to_owned(), groups.all.as_slice()))
            .chain(groups.by_product.iter().map(|(code, rows)| (code.clone(), rows.pooled.as_slice())))
            .chain(groups.by_sector.iter().map(|(sector, rows)| (format!("sector:{sector}"), rows.as_slice())));
        for (code, rows) in pooled {
            let values = rows.iter().map(|&row| raw[row]).collect::<Vec<_>>();
            let coverage = coverage_stats(&values);
            self.distributions.push(FactorDistribution {
                code,
                stats: daily_factor_stats(factor.to_owned(), date, &values, coverage.coverage, coverage.inf_rate),
            });
        }
    }

    fn orient_holding_returns(&mut self, sectors: &HashMap<String, &'static str>) {
        let is_product = |code: &str| code != "all" && !code.starts_with("sector:");
        let mut product_ic = BTreeMap::<String, Mean>::new();
        for row in self.rows.iter().filter(|row| is_product(&row.code)) {
            product_ic.entry(row.code.clone()).or_default().push(row.metrics.ic.value());
        }
        let mut aggregate_returns = BTreeMap::<(i32, String), Mean>::new();
        for row in self.rows.iter_mut().filter(|row| is_product(&row.code)) {
            // Every minute's position is reversed for a negative mean product IC.
            // The linear return changes sign; entries, reversals and durations are identical.
            if product_ic[&row.code].value().is_some_and(|ic| ic < 0.0) {
                row.metrics.holding_rtn.sum = -row.metrics.holding_rtn.sum;
            }
            let value = row.metrics.holding_rtn.value();
            aggregate_returns.entry((row.date, "all".to_owned())).or_default().push(value);
            if let Some(sector) = sectors.get(&row.code) {
                aggregate_returns.entry((row.date, format!("sector:{sector}"))).or_default().push(value);
            }
        }
        for row in self.rows.iter_mut().filter(|row| !is_product(&row.code)) {
            row.metrics.holding_rtn = aggregate_returns.remove(&(row.date, row.code.clone())).unwrap_or_default();
        }
    }

    fn add_day(
        &mut self,
        date: i32,
        groups: &DayGroups,
        history_groups: Option<&DayGroups>,
        factor: &[Option<f64>],
        returns: Option<&DayReturns>,
    ) {
        let mut all = Metrics::default();
        let mut sector_holdings = BTreeMap::<&str, Metrics>::new();
        for (code, group) in &groups.by_product {
            let rows = &group.pooled;
            let Some(returns) = returns else { continue };
            let thresholds = match &self.history {
                Some(history) => history.mean(code),
                None => TailThresholds::for_day(rows, factor, self.quantile),
            };
            let mut local = Metrics::default();
            local.add_pair_metrics(rows, factor, &returns.tails);
            if let Some(thresholds) = thresholds {
                local.add_tails(
                    rows,
                    factor,
                    &returns.tails,
                    thresholds,
                    self.quantile == 50,
                    self.history.is_some(),
                );
                if local.trigger_count.is_some() {
                    local.eligible_contract_count =
                        Some(group.eligible_contracts(factor, &returns.tails));
                }
                for contract in group.contracts.keys() {
                    if let Some(timeline) = returns.holding.get(contract) {
                        let (return_bp, duration) = evaluate_holding(timeline, factor, thresholds);
                        local.add_holding(return_bp, duration);
                    }
                }
            }
            all.average_tails(&local);
            let holding_rtn = local.holding_rtn.value();
            all.add_holding(holding_rtn, local.holding_duration);
            if let Some(sector) = group.sector {
                sector_holdings
                    .entry(sector)
                    .or_default()
                    .add_holding(holding_rtn, local.holding_duration);
            }
            self.rows.push(Observation {
                date,
                code: code.clone(),
                metrics: local,
            });
        }
        if let Some(returns) = returns {
            all.add_pair_metrics(&groups.all, factor, &returns.tails);
            self.rows.push(Observation {
                date,
                code: "all".to_string(),
                metrics: all,
            });
            for (sector, rows) in &groups.by_sector {
                let mut metrics = sector_holdings.remove(sector).unwrap_or_default();
                metrics.add_pair_metrics(rows, factor, &returns.tails);
                self.rows.push(Observation {
                    date,
                    code: format!("sector:{sector}"),
                    metrics,
                });
            }
        }
        // Publish today's thresholds only after using the previous sessions.
        if let Some(history) = &mut self.history {
            let today = history_groups
                .unwrap_or(groups)
                .by_product
                .iter()
                .filter_map(|(code, group)| {
                    let threshold = TailThresholds::for_day(&group.pooled, factor, self.quantile)?;
                    Some((code.clone(), threshold))
                })
                .collect();
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
    let target_dates = &dates[dates.partition_point(|&date| date < request.start_date)..];
    let liquidity_pool = if let Some(id) = &request.pool_id {
        if id.is_empty()
            || !id
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
        {
            return Err(err("invalid futures pool ID"));
        }
        let pool = FutureMinuteInput::load_pool(&config.data_root, id, target_dates)?;
        let missing: Vec<_> = target_dates
            .iter()
            .filter(|&&date| pool.selection(date).is_none())
            .collect();
        if !missing.is_empty() {
            return Err(err(format!(
                "futures pool {id} has no mapping on {missing:?}"
            )));
        }
        Some(pool)
    } else {
        None
    };
    let historical_main = (liquidity_pool.is_some()
        && matches!(request.quantile_mode, QuantileMode::Trailing5))
    .then(|| FutureMinuteInput::load(FutureMinuteOptions::default(), &config.data_root, &dates))
    .transpose()?;
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
        liquidity_pool: liquidity_pool.as_ref(),
        historical_main: historical_main.as_ref(),
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
        let mut jobs: Vec<Box<dyn FnOnce() -> Result<PathBuf> + Send + '_>> = Vec::new();
        for (&factor, summary) in batch.iter().zip(summaries) {
            let summary_path = output_dir.join(factor).join("summary.parquet");
            jobs.push(Box::new(move || {
                write_summary(&summary_path, &summary.rows, label, request)?;
                Ok(summary_path)
            }));
            let stats_path = output_dir.join(factor).join("factor_stats.parquet");
            jobs.push(Box::new(move || {
                write_factor_stats(&stats_path, &summary.distributions)?;
                Ok(stats_path)
            }));
        }
        paths.extend(crate::data::read_ahead::write_files(&output_dir, jobs, |job| job())?);
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
        eprintln!("{}", crate::artifact_io::profile());
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
    liquidity_pool: Option<&'a FutureMinuteInput>,
    historical_main: Option<&'a FutureMinuteInput>,
}

struct EvaluationDay {
    date: i32,
    table: Table,
    groups: DayGroups,
    history_groups: Option<DayGroups>,
    returns: Option<DayReturns>,
}

struct DayReturns {
    tails: Vec<Option<f64>>,
    holding: HashMap<String, Vec<HoldingMinute>>,
}

impl EvaluationInputs<'_> {
    fn read_day(&self, date: i32, factors: &[&str]) -> Result<Option<EvaluationDay>> {
        let evaluate = date >= self.start_date;
        let path = minute_path(self.factor_root, date);
        if !evaluate && !crate::artifact_io::exists(&path)? {
            return Ok(None);
        }
        let table = read_day(&path, factors, None)?;
        let keys = minute_keys(&table, date)?;
        let pool_mapping = self.liquidity_pool.filter(|_| evaluate).map(|pool| {
            pool.selection(date)
                .expect("pool coverage checked before evaluation")
        });
        let groups = if evaluate || self.historical_main.is_none() {
            group_rows(
                &keys,
                self.products,
                evaluate.then_some(self.sectors),
                pool_mapping,
                date,
            )?
        } else {
            DayGroups::default()
        };
        let history_groups = self
            .historical_main
            .map(|ordinary| match ordinary.selection(date) {
                Some(mapping) => group_rows(&keys, self.products, None, Some(mapping), date),
                None => Ok(DayGroups::default()),
            })
            .transpose()?;
        let returns = evaluate
            .then(|| {
                aligned_returns(
                    self.label_root,
                    self.label,
                    date,
                    &keys,
                    pool_mapping.map(|_| groups.all.as_slice()),
                    &groups.all,
                )
            })
            .transpose()?;
        Ok(Some(EvaluationDay {
            date,
            table,
            groups,
            history_groups,
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
        crate::data::read_ahead::visit_filtered_inputs(
            inputs.factor_root,
            date_batch,
            |&date| inputs.read_day(date, factors),
            |day| {
                let Some(day) = day else {
                    for summary in &mut summaries {
                        if let Some(history) = &mut summary.history {
                            history.advance(DailyThresholds::new());
                        }
                    }
                    return Ok(());
                };
                if day.returns.is_some() {
                    seen_products.extend(day.groups.by_product.keys().cloned());
                }
                // Consume sessions in date order so the rolling window spans batches.
                factors
                    .par_iter()
                    .zip(summaries.par_iter_mut())
                    .try_for_each(|(&factor, summary)| -> Result<()> {
                        let raw = day.table.required_f64_cast(factor)?;
                        if day.date >= inputs.start_date {
                            summary.add_distributions(day.date, factor, &day.groups, &raw);
                        }
                        let values = raw.into_iter().map(|value| value.filter(|v| v.is_finite())).collect::<Vec<_>>();
                        summary.add_day(
                            day.date,
                            &day.groups,
                            day.history_groups.as_ref(),
                            &values,
                            day.returns.as_ref(),
                        );
                        Ok(())
                    })
            },
        )?;
    }
    for summary in &mut summaries {
        summary.orient_holding_returns(inputs.sectors);
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

fn write_factor_stats(path: &Path, rows: &[FactorDistribution]) -> Result<()> {
    let stats = rows.iter().map(|row| &row.stats).collect::<Vec<_>>();
    let mut table = crate::backtest::storage::factor_stats_table(&stats)?;
    table.columns.insert("ts_code".to_owned(), ColumnData::Utf8(
        rows.iter().map(|row| Some(row.code.clone())).collect(),
    ));
    write_parquet(path, &table)
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
            KeyValue::new("holding_label_id".to_string(), HOLDING_LABEL.to_string()),
            KeyValue::new("holding_direction".to_string(), "product_mean_daily_ic".to_string()),
            KeyValue::new(
                "quantile_mode".to_string(),
                request.quantile_mode.as_str().to_string(),
            ),
            KeyValue::new("tail_quantile".to_string(), request.quantile.to_string()),
            KeyValue::new(
                "pool_id".to_string(),
                request.pool_id.as_deref().unwrap_or("none").to_string(),
            ),
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
    mapping: Option<&BTreeMap<String, String>>,
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
        let selected_product = products.is_empty()
            || products.contains(&code)
            || products.contains(code.split_once('.').unwrap().0);
        let selected = selected_product
            && mapping
                .is_none_or(|mapping| mapping.get(&code).is_some_and(|main| main == contract));
        codes.insert(contract, selected.then_some(code));
    }
    let mut product_rows = BTreeMap::<&str, ProductRows>::new();
    let mut contract_rows = HashMap::<&str, Vec<usize>>::new();
    for (row, &(_, contract)) in keys.iter().enumerate() {
        let Some(code) = codes[contract].as_deref() else {
            continue;
        };
        groups.all.push(row);
        let product_rows = product_rows.entry(code).or_default();
        product_rows.pooled.push(row);
        contract_rows.entry(contract).or_default().push(row);
        if let Some(sector) = sectors.and_then(|mapping| mapping.get(code)) {
            product_rows.sector = Some(*sector);
            groups.by_sector.entry(*sector).or_default().push(row);
        }
    }
    for (contract, rows) in contract_rows {
        let code = codes[contract].as_deref().expect("selected contract");
        product_rows
            .get_mut(code)
            .expect("selected product")
            .contracts
            .insert(contract.to_string(), rows);
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
    selected_rows: Option<&[usize]>,
    holding_rows: &[usize],
) -> Result<DayReturns> {
    let label_table = read_day(&minute_path(root, date), &[label], Some(HOLDING_LABEL))?;
    let label_keys = minute_keys(&label_table, date)?;
    let label_index: HashMap<_, _> = label_keys
        .iter()
        .copied()
        .enumerate()
        .map(|(i, key)| (key, i))
        .collect();
    let values = finite_column(&label_table, label)?;
    let mut matched = false;
    let mut returns = vec![None; factor_keys.len()];
    let mut align = |row: usize| {
        if let Some(&index) = label_index.get(&factor_keys[row]) {
            matched = true;
            returns[row] = values[index];
        }
    };
    if let Some(rows) = selected_rows {
        rows.iter().copied().for_each(&mut align);
    } else {
        (0..factor_keys.len()).for_each(&mut align);
    }
    if !matched && !selected_rows.is_some_and(|rows| rows.is_empty()) {
        return Err(err(format!(
            "no matching futures factor and label minute keys on {date}"
        )));
    }
    let holding = if label_table.columns.contains_key(HOLDING_LABEL) {
        let minute_returns = finite_column(&label_table, HOLDING_LABEL)?;
        holding_timelines(factor_keys, holding_rows, &label_keys, minute_returns)
    } else {
        HashMap::new()
    };
    Ok(DayReturns {
        tails: returns,
        holding,
    })
}

fn holding_timelines(
    factor_keys: &[(&str, &str)],
    holding_rows: &[usize],
    label_keys: &[(&str, &str)],
    returns: Vec<Option<f64>>,
) -> HashMap<String, Vec<HoldingMinute>> {
    let signals: HashMap<_, _> = holding_rows
        .iter()
        .map(|&row| (factor_keys[row], row))
        .collect();
    let contracts: BTreeSet<_> = holding_rows.iter().map(|&row| factor_keys[row].1).collect();
    let mut timelines = HashMap::<&str, Vec<(&str, HoldingMinute)>>::new();
    for (&(time, code), return_bp) in label_keys.iter().zip(returns) {
        if contracts.contains(code) {
            timelines.entry(code).or_default().push((
                time,
                HoldingMinute {
                    signal_row: signals.get(&(time, code)).copied(),
                    return_bp,
                    continuous: false,
                },
            ));
        }
    }
    timelines
        .into_iter()
        .map(|(code, mut rows)| {
            rows.sort_unstable_by_key(|(time, _)| *time);
            for index in 0..rows.len().saturating_sub(2) {
                // The one-minute label at index starts at the next bar's open.
                rows[index].1.continuous =
                    continuous_holding_interval(rows[index + 1].0, rows[index + 2].0);
            }
            (
                code.to_string(),
                rows.into_iter().map(|(_, minute)| minute).collect(),
            )
        })
        .collect()
}

fn continuous_holding_interval(start: &str, end: &str) -> bool {
    if start.len() == 8 && end.len() == 8 {
        use crate::strategy::market::clock_minute;
        return clock_minute(start)
            .zip(clock_minute(end))
            .is_some_and(|(from, to)| {
                from + 1 == to && start.as_bytes()[6..] == end.as_bytes()[6..]
            });
    }
    adjacent_minute(start, end)
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

fn read_day(path: &Path, values: &[&str], optional: Option<&str>) -> Result<Table> {
    let required: Vec<String> = ["trade_date", "trade_time", "ts_code"]
        .into_iter()
        .chain(values.iter().copied())
        .map(str::to_string)
        .collect();
    let optional = optional.filter(|name| !values.contains(name));
    let mut projection = required.clone();
    projection.extend(optional.map(str::to_string));
    let null_types = values
        .iter()
        .copied()
        .chain(optional)
        .map(|name| (name.to_string(), DataType::Float64))
        .collect();
    let table = read_parquet_with_null_types(path, &projection, &null_types)?;
    for column in &required {
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
    let columns: [(&str, ArrayRef); 10] = [
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
        (
            "eligible_contract_count",
            Arc::new(UInt64Array::from_iter(
                rows.iter().map(|row| row.metrics.eligible_contract_count),
            )),
        ),
        (
            "holding_rtn",
            Arc::new(Float64Array::from_iter(
                rows.iter().map(|row| row.metrics.holding_rtn.value()),
            )),
        ),
        (
            "holding_minutes",
            Arc::new(Float64Array::from_iter(
                rows.iter()
                    .map(|row| row.metrics.holding_duration.average()),
            )),
        ),
    ];
    Ok(RecordBatch::try_from_iter(columns)?)
}

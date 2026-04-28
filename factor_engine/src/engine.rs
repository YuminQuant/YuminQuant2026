use std::collections::BTreeSet;
use std::ops::Range;
use std::path::PathBuf;
use std::time::Instant;

use crate::calendar::TradingCalendar;
use crate::config::EngineConfig;
use crate::core::{
    factor_registry_key, AssetClass, DataRequest, FactorContext, FactorSeries, FactorSpec,
    Frequency,
};
use crate::data::{DataCatalog, DataPool, MarketDataLoader};
use crate::error::{err, Result};
use crate::factor::registry::{all_factors, factor_map};
use crate::factor::Factor;
use crate::storage::{FactorMetadata, FactorStorage};
use rayon::prelude::*;

pub const DEFAULT_FACTOR_BATCH_SIZE: usize = 64;

#[derive(Clone, Debug)]
pub struct RunRequest {
    pub asset_class: AssetClass,
    pub frequency: Frequency,
    pub start_date: i32,
    pub end_date: i32,
    pub factor_ids: Option<Vec<String>>,
    pub tags: Option<Vec<String>>,
    pub config_path: Option<PathBuf>,
    pub dry_run: bool,
    pub factor_batch_size: usize,
    pub threads: Option<usize>,
    pub profile: bool,
}

#[derive(Clone, Debug)]
pub struct RunReport {
    pub factor_count: usize,
    pub output_file_count: usize,
    pub load_start_date: i32,
    pub target_dates: Vec<i32>,
    pub effective_start_date: Option<i32>,
    pub effective_end_date: Option<i32>,
    pub date_batch_count: usize,
    pub factor_batch_count: usize,
    pub execution_batch_count: usize,
    pub selected_factor_ids: Vec<String>,
    pub loaded_requests: Vec<DataRequest>,
    pub profiles: Vec<BatchProfile>,
    pub status_message: Option<String>,
}

#[derive(Clone, Debug)]
pub struct BatchProfile {
    pub date_batch_index: usize,
    pub factor_batch_index: usize,
    pub start_date: i32,
    pub end_date: i32,
    pub factor_count: usize,
    pub load_ms: u128,
    pub compute_ms: u128,
    pub write_ms: u128,
    pub factors: Vec<FactorProfile>,
}

#[derive(Clone, Debug)]
pub struct FactorProfile {
    pub factor_id: String,
    pub row_count: usize,
    pub non_null_count: usize,
}

pub struct Engine {
    config: EngineConfig,
}

impl Engine {
    pub fn new(config: EngineConfig) -> Self {
        Self { config }
    }

    pub fn from_request(request: &RunRequest) -> Result<Self> {
        Ok(Self::new(EngineConfig::discover(
            request.config_path.clone(),
        )?))
    }

    pub fn plan(&self, request: &RunRequest) -> Result<RunReport> {
        self.execute(request, true)
    }

    pub fn run(&self, request: &RunRequest) -> Result<RunReport> {
        self.execute(request, request.dry_run)
    }

    pub fn write_metadata(&self) -> Result<usize> {
        let specs = available_specs();
        let storage = FactorStorage::new(self.config.factor_root.clone());
        storage.write_metadata(&specs)?;
        Ok(specs.len())
    }

    pub fn read_metadata(&self) -> Result<Vec<FactorMetadata>> {
        FactorStorage::new(self.config.factor_root.clone()).read_metadata()
    }

    fn execute(&self, request: &RunRequest, dry_run: bool) -> Result<RunReport> {
        validate_date_value(request.start_date, "start-date")?;
        validate_date_value(request.end_date, "end-date")?;

        let metadata = self.read_metadata()?;
        let selected_metadata = select_metadata(request, &metadata);
        if let SelectionResult::Empty(message) = selected_metadata {
            return Ok(empty_report(request, message));
        }
        let SelectionResult::Selected(selected_metadata) = selected_metadata else {
            unreachable!("selection result handled");
        };

        let mut registry = factor_map();
        let mut factors = Vec::new();
        let mut stale_ids = Vec::new();
        for metadata in &selected_metadata {
            let key = factor_registry_key(
                &metadata.asset_class,
                &metadata.frequency,
                &metadata.factor_id,
            );
            if let Some(factor) = registry.remove(&key) {
                factors.push(factor);
            } else {
                stale_ids.push(metadata.factor_id.clone());
            }
        }
        if !stale_ids.is_empty() {
            return Err(err(format!(
                "factor_metadata.parquet is stale; missing registered implementation(s): {}. Run `metadata` again.",
                stale_ids.join(",")
            )));
        }
        if factors.is_empty() {
            return Ok(empty_report(
                request,
                format!(
                    "No factors selected for asset={} frequency={}.",
                    request.asset_class, request.frequency
                ),
            ));
        }

        let specs = factors
            .iter()
            .map(|factor| factor.spec())
            .collect::<Vec<_>>();
        let max_lookback = specs
            .iter()
            .map(|spec| spec.lookback.trading_days)
            .max()
            .unwrap_or(0);
        let calendar_exchange = match request.asset_class {
            AssetClass::Stock => &self.config.stock_calendar_exchange,
            AssetClass::Future => &self.config.future_calendar_exchange,
        };
        let calendar = TradingCalendar::load(&self.config.data_root, calendar_exchange)?;
        let Some(effective_start_date) = calendar.first_open_on_or_after(request.start_date) else {
            return Ok(empty_report(
                request,
                format!(
                    "No open trading dates found on or after start_date {}.",
                    request.start_date
                ),
            ));
        };
        let Some(effective_end_date) = calendar.last_open_on_or_before(request.end_date) else {
            return Ok(empty_report(
                request,
                format!(
                    "No open trading dates found on or before end_date {}.",
                    request.end_date
                ),
            ));
        };
        if effective_start_date > effective_end_date {
            return Ok(empty_report(
                request,
                format!(
                    "No open trading dates found between {} and {} after calendar alignment.",
                    request.start_date, request.end_date
                ),
            ));
        }

        let target_dates = calendar.open_dates_between(effective_start_date, effective_end_date);
        if target_dates.is_empty() {
            return Ok(empty_report(
                request,
                format!(
                    "No open trading dates found between {} and {}.",
                    request.start_date, request.end_date
                ),
            ));
        }

        let date_batches = split_dates_by_year(&target_dates);
        let factor_batch_size = request.factor_batch_size.max(1);
        let factor_ranges = factor_batch_ranges(factors.len(), factor_batch_size);
        let factor_batch_count = factor_ranges.len();
        let execution_batch_count = date_batches.len() * factor_batch_count;
        let load_start_date = calendar.warmup_start(effective_start_date, max_lookback);

        let loaded_requests =
            merge_requests(specs.iter().flat_map(|spec| spec.dependencies.clone()));
        if dry_run {
            return Ok(RunReport {
                factor_count: specs.len(),
                output_file_count: 0,
                load_start_date,
                target_dates,
                effective_start_date: Some(effective_start_date),
                effective_end_date: Some(effective_end_date),
                date_batch_count: date_batches.len(),
                factor_batch_count,
                execution_batch_count,
                selected_factor_ids: specs.iter().map(|spec| spec.id.clone()).collect(),
                loaded_requests,
                profiles: Vec::new(),
                status_message: None,
            });
        }

        let catalog = DataCatalog::new(self.config.data_root.clone())
            .with_stock_sw_classification_path(self.config.stock_sw_classification_path.clone());
        let loader = MarketDataLoader::new(catalog);
        let storage = FactorStorage::new(self.config.factor_root.clone());
        let thread_pool = build_thread_pool(request.threads)?;
        let mut output_paths = BTreeSet::new();
        let mut profiles = Vec::new();

        for (date_batch_index, date_batch) in date_batches.iter().enumerate() {
            let batch_start_date = *date_batch
                .first()
                .expect("date batches are never empty after split");
            let batch_end_date = *date_batch
                .last()
                .expect("date batches are never empty after split");
            let batch_load_start_date = calendar.warmup_start(batch_start_date, max_lookback);
            let context = FactorContext {
                asset_class: request.asset_class,
                frequency: request.frequency,
                start_date: batch_start_date,
                end_date: batch_end_date,
                load_start_date: batch_load_start_date,
                target_dates: date_batch.clone(),
            };

            for (factor_batch_index, range) in factor_ranges.iter().enumerate() {
                let batch_specs = specs[range.clone()].to_vec();
                let batch_requests = merge_requests(
                    batch_specs
                        .iter()
                        .flat_map(|spec| spec.dependencies.clone()),
                );
                let load_started = Instant::now();
                let pool = DataPool::load(&loader, &batch_requests, &context)?;
                let load_ms = load_started.elapsed().as_millis();
                let compute_started = Instant::now();
                let results = compute_factor_batch(
                    &factors[range.clone()],
                    &context,
                    &pool,
                    thread_pool.as_ref(),
                )?;
                let compute_ms = compute_started.elapsed().as_millis();
                let factor_profiles = results
                    .iter()
                    .map(|series| FactorProfile {
                        factor_id: series.spec.id.clone(),
                        row_count: series.values.len(),
                        non_null_count: series
                            .values
                            .iter()
                            .filter(|item| item.value.is_some())
                            .count(),
                    })
                    .collect::<Vec<_>>();
                let write_started = Instant::now();
                let written_paths = storage.write_results(&results)?;
                let write_ms = write_started.elapsed().as_millis();
                output_paths.extend(written_paths);
                if request.profile {
                    profiles.push(BatchProfile {
                        date_batch_index: date_batch_index + 1,
                        factor_batch_index: factor_batch_index + 1,
                        start_date: batch_start_date,
                        end_date: batch_end_date,
                        factor_count: batch_specs.len(),
                        load_ms,
                        compute_ms,
                        write_ms,
                        factors: factor_profiles,
                    });
                }
            }
        }

        Ok(RunReport {
            factor_count: specs.len(),
            output_file_count: output_paths.len(),
            load_start_date,
            target_dates,
            effective_start_date: Some(effective_start_date),
            effective_end_date: Some(effective_end_date),
            date_batch_count: date_batches.len(),
            factor_batch_count,
            execution_batch_count,
            selected_factor_ids: specs.iter().map(|spec| spec.id.clone()).collect(),
            loaded_requests,
            profiles,
            status_message: None,
        })
    }
}

enum SelectionResult {
    Selected(Vec<FactorMetadata>),
    Empty(String),
}

fn select_metadata(request: &RunRequest, metadata: &[FactorMetadata]) -> SelectionResult {
    let base = metadata
        .iter()
        .filter(|row| {
            row.asset_class == request.asset_class.as_str()
                && row.frequency == request.frequency.as_str()
        })
        .cloned()
        .collect::<Vec<_>>();

    if let Some(tags) = &request.tags {
        let selected = base
            .into_iter()
            .filter(|row| {
                tags.iter()
                    .all(|tag| row.tags.iter().any(|item| item == tag))
            })
            .collect::<Vec<_>>();
        if selected.is_empty() {
            return SelectionResult::Empty(format!(
                "No factors found for tag(s): {}",
                tags.join(",")
            ));
        }
        return SelectionResult::Selected(selected);
    }

    if let Some(factor_ids) = &request.factor_ids {
        let mut selected = Vec::new();
        let mut missing = Vec::new();
        for factor_id_or_name in factor_ids {
            if let Some(row) = base.iter().find(|row| {
                row.factor_id == *factor_id_or_name
                    || row.name == *factor_id_or_name
                    || row.aliases.iter().any(|alias| alias == factor_id_or_name)
            }) {
                selected.push(row.clone());
            } else {
                missing.push(factor_id_or_name.clone());
            }
        }
        if !missing.is_empty() {
            return SelectionResult::Empty(format!(
                "No factors found in metadata for: {}",
                missing.join(",")
            ));
        }
        return SelectionResult::Selected(selected);
    }

    if base.is_empty() {
        return SelectionResult::Empty(format!(
            "No factors found in metadata for asset={} frequency={}.",
            request.asset_class, request.frequency
        ));
    }
    SelectionResult::Selected(base)
}

fn empty_report(request: &RunRequest, message: String) -> RunReport {
    RunReport {
        factor_count: 0,
        output_file_count: 0,
        load_start_date: request.start_date,
        target_dates: Vec::new(),
        effective_start_date: None,
        effective_end_date: None,
        date_batch_count: 0,
        factor_batch_count: 0,
        execution_batch_count: 0,
        selected_factor_ids: Vec::new(),
        loaded_requests: Vec::new(),
        profiles: Vec::new(),
        status_message: Some(message),
    }
}

fn merge_requests<I>(requests: I) -> Vec<DataRequest>
where
    I: IntoIterator<Item = DataRequest>,
{
    use std::collections::{BTreeSet, HashMap};

    let mut grouped: HashMap<_, (BTreeSet<String>, Option<usize>)> = HashMap::new();
    for request in requests {
        let entry = grouped.entry(request.dataset).or_default();
        entry.0.extend(request.columns.into_iter());
        entry.1 = match (entry.1, request.financial_quarters) {
            (Some(left), Some(right)) => Some(left.max(right)),
            (None, Some(right)) => Some(right),
            (left, None) => left,
        };
    }
    let mut merged = grouped
        .into_iter()
        .map(|(dataset, (columns, financial_quarters))| DataRequest {
            dataset,
            columns: columns.into_iter().collect(),
            financial_quarters,
        })
        .collect::<Vec<_>>();
    merged.sort_by_key(|request| request.dataset);
    merged
}

fn validate_date_value(date: i32, name: &str) -> Result<()> {
    let value = date.to_string();
    if value.len() != 8 {
        return Err(err(format!(
            "--{name} must be an 8-digit YYYYMMDD date, got {date}"
        )));
    }
    Ok(())
}

fn split_dates_by_year(dates: &[i32]) -> Vec<Vec<i32>> {
    let mut batches = Vec::new();
    let mut current = Vec::new();
    let mut current_year = None;

    for date in dates {
        let year = date / 10_000;
        if current_year.is_some_and(|value| value != year) {
            batches.push(current);
            current = Vec::new();
        }
        current_year = Some(year);
        current.push(*date);
    }

    if !current.is_empty() {
        batches.push(current);
    }
    batches
}

fn factor_batch_ranges(factor_count: usize, factor_batch_size: usize) -> Vec<Range<usize>> {
    if factor_count == 0 {
        return Vec::new();
    }

    let batch_size = factor_batch_size.max(1);
    (0..factor_count)
        .step_by(batch_size)
        .map(|start| start..(start + batch_size).min(factor_count))
        .collect()
}

fn build_thread_pool(threads: Option<usize>) -> Result<Option<rayon::ThreadPool>> {
    match threads {
        Some(0) => Err(err("threads must be greater than 0")),
        Some(threads) => Ok(Some(
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()?,
        )),
        None => Ok(None),
    }
}

fn compute_factor_batch(
    factors: &[Box<dyn Factor>],
    context: &FactorContext,
    pool: &DataPool,
    thread_pool: Option<&rayon::ThreadPool>,
) -> Result<Vec<FactorSeries>> {
    let compute = || {
        factors
            .par_iter()
            .map(|factor| factor.compute(context, pool))
            .collect::<Vec<_>>()
    };
    let results = match thread_pool {
        Some(thread_pool) => thread_pool.install(compute),
        None => compute(),
    };
    results.into_iter().collect()
}

pub fn available_specs() -> Vec<FactorSpec> {
    all_factors()
        .into_iter()
        .map(|factor| factor.spec())
        .collect()
}

#[cfg(test)]
mod tests {
    use crate::core::{AssetClass, Frequency};
    use crate::storage::FactorMetadata;

    use super::{
        factor_batch_ranges, select_metadata, split_dates_by_year, validate_date_value, RunRequest,
        SelectionResult,
    };

    #[test]
    fn validates_eight_digit_dates() {
        assert!(validate_date_value(20260424, "end-date").is_ok());
        assert!(validate_date_value(2026424, "end-date").is_err());
    }

    #[test]
    fn splits_dates_by_natural_year() {
        let batches = split_dates_by_year(&[20100104, 20100105, 20110104, 20110105, 20120104]);

        assert_eq!(
            batches,
            vec![
                vec![20100104, 20100105],
                vec![20110104, 20110105],
                vec![20120104]
            ]
        );
    }

    #[test]
    fn chunks_factors_by_configured_batch_size() {
        assert_eq!(
            factor_batch_ranges(0, 64),
            Vec::<std::ops::Range<usize>>::new()
        );
        assert_eq!(factor_batch_ranges(63, 64), vec![0..63]);
        assert_eq!(factor_batch_ranges(65, 64), vec![0..64, 64..65]);
        assert_eq!(factor_batch_ranges(3, 0), vec![0..1, 1..2, 2..3]);
    }

    #[test]
    fn selection_matches_short_id_and_legacy_alias_inside_asset_frequency() {
        let request = RunRequest {
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            start_date: 20260105,
            end_date: 20260105,
            factor_ids: Some(vec!["stock.daily.pv.return_1d".to_string()]),
            tags: None,
            config_path: None,
            dry_run: false,
            factor_batch_size: 64,
            threads: None,
            profile: false,
        };
        let metadata = vec![
            metadata_row("return_1d", "stock", "daily", &["stock.daily.pv.return_1d"]),
            metadata_row(
                "return_1d",
                "future",
                "daily",
                &["future.daily.pv.return_1d"],
            ),
        ];

        let SelectionResult::Selected(selected) = select_metadata(&request, &metadata) else {
            panic!("expected selected");
        };
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].asset_class, "stock");
        assert_eq!(selected[0].factor_id, "return_1d");
    }

    fn metadata_row(
        factor_id: &str,
        asset_class: &str,
        frequency: &str,
        aliases: &[&str],
    ) -> FactorMetadata {
        FactorMetadata {
            factor_id: factor_id.to_string(),
            aliases: aliases.iter().map(|alias| (*alias).to_string()).collect(),
            aliases_json: String::new(),
            version: "0.1.0".to_string(),
            output_column: factor_id.to_string(),
            name: factor_id.to_string(),
            asset_class: asset_class.to_string(),
            frequency: frequency.to_string(),
            tags: Vec::new(),
            tags_json: String::new(),
            dependencies_json: String::new(),
            description: String::new(),
            updated_at: String::new(),
        }
    }
}

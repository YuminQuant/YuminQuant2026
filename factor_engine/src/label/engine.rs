use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::ops::Range;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use rayon::prelude::*;

use crate::calendar::TradingCalendar;
use crate::config::EngineConfig;
use crate::core::{
    label_registry_key, AssetClass, DataRequest, DatasetId, FactorContext, Frequency,
    IntradayDailyRawRequest, IntradayDailyRawSeries, IntradayDailyRawSpec, LabelSeries, LabelSpec,
};
use crate::data::{DataCatalog, DataPool, MarketDataLoader, Table};
use crate::engine::{BatchProfile, FactorProfile};
use crate::error::{err, Result};
use crate::label::registry::{all_labels, label_map};
use crate::label::Label;
use crate::progress::ProgressBar;
use crate::storage::{IntradayDailyRawStorage, LabelMetadata, LabelStorage};

#[derive(Clone, Debug)]
pub struct LabelRunRequest {
    pub asset_class: AssetClass,
    pub frequency: Frequency,
    pub start_date: i32,
    pub end_date: i32,
    pub label_ids: Option<Vec<String>>,
    pub tags: Option<Vec<String>>,
    pub config_path: Option<PathBuf>,
    pub dry_run: bool,
    pub label_batch_size: usize,
    pub date_batch_size: usize,
    pub threads: Option<usize>,
    pub profile: bool,
    pub refresh_label_cache: bool,
}

#[derive(Clone, Debug)]
pub struct LabelRunReport {
    pub label_count: usize,
    pub output_file_count: usize,
    pub target_dates: Vec<i32>,
    pub skipped_dates: Vec<i32>,
    pub effective_start_date: Option<i32>,
    pub effective_end_date: Option<i32>,
    pub max_lookahead: usize,
    pub date_batch_count: usize,
    pub label_batch_count: usize,
    pub execution_batch_count: usize,
    pub selected_label_ids: Vec<String>,
    pub loaded_requests: Vec<DataRequest>,
    pub loaded_intraday_raw_requests: Vec<IntradayDailyRawRequest>,
    pub profiles: Vec<BatchProfile>,
    pub status_message: Option<String>,
}

pub struct LabelEngine {
    config: EngineConfig,
}

impl LabelEngine {
    pub fn new(config: EngineConfig) -> Self {
        Self { config }
    }

    pub fn from_request(request: &LabelRunRequest) -> Result<Self> {
        Ok(Self::new(EngineConfig::discover(
            request.config_path.clone(),
        )?))
    }

    pub fn write_metadata(&self) -> Result<usize> {
        let specs = available_label_specs();
        let storage = LabelStorage::new(self.config.label_root.clone());
        storage.write_metadata(&specs)?;
        Ok(specs.len())
    }

    pub fn read_metadata(&self) -> Result<Vec<LabelMetadata>> {
        LabelStorage::new(self.config.label_root.clone()).read_metadata()
    }

    pub fn plan(&self, request: &LabelRunRequest) -> Result<LabelRunReport> {
        self.execute(request, true)
    }

    pub fn run(&self, request: &LabelRunRequest) -> Result<LabelRunReport> {
        self.execute(request, request.dry_run)
    }

    fn execute(&self, request: &LabelRunRequest, dry_run: bool) -> Result<LabelRunReport> {
        validate_date_value(request.start_date, "start-date")?;
        validate_date_value(request.end_date, "end-date")?;
        if request.frequency != Frequency::Daily {
            return Ok(empty_label_report(
                request,
                "Label engine currently supports daily labels only.".to_string(),
            ));
        }

        let metadata = self.read_metadata()?;
        let selected_metadata = select_label_metadata(request, &metadata);
        if let SelectionResult::Empty(message) = selected_metadata {
            return Ok(empty_label_report(request, message));
        }
        let SelectionResult::Selected(selected_metadata) = selected_metadata else {
            unreachable!("selection result handled");
        };

        let mut registry = label_map();
        let mut labels = Vec::new();
        let mut stale_ids = Vec::new();
        for metadata in &selected_metadata {
            let key = label_registry_key(
                &metadata.asset_class,
                &metadata.frequency,
                &metadata.label_id,
            );
            if let Some(label) = registry.remove(&key) {
                labels.push(label);
            } else {
                stale_ids.push(metadata.label_id.clone());
            }
        }
        if !stale_ids.is_empty() {
            return Err(err(format!(
                "label_metadata.parquet is stale; missing registered implementation(s): {}. Run `label-metadata` again.",
                stale_ids.join(",")
            )));
        }
        if labels.is_empty() {
            return Ok(empty_label_report(
                request,
                format!(
                    "No labels selected for asset={} frequency={}.",
                    request.asset_class, request.frequency
                ),
            ));
        }

        let specs = labels.iter().map(|label| label.spec()).collect::<Vec<_>>();
        let max_lookahead = specs
            .iter()
            .map(|spec| spec.lookahead.trading_days)
            .max()
            .unwrap_or(0);
        let calendar_exchange = match request.asset_class {
            AssetClass::Stock => &self.config.stock_calendar_exchange,
            AssetClass::Future => &self.config.future_calendar_exchange,
        };
        let calendar = TradingCalendar::load(&self.config.data_root, calendar_exchange)?;
        let Some(effective_start_date) = calendar.first_open_on_or_after(request.start_date) else {
            return Ok(empty_label_report(
                request,
                format!(
                    "No open trading dates found on or after start_date {}.",
                    request.start_date
                ),
            ));
        };
        let Some(effective_end_date) = calendar.last_open_on_or_before(request.end_date) else {
            return Ok(empty_label_report(
                request,
                format!(
                    "No open trading dates found on or before end_date {}.",
                    request.end_date
                ),
            ));
        };
        if effective_start_date > effective_end_date {
            return Ok(empty_label_report(
                request,
                format!(
                    "No open trading dates found between {} and {} after calendar alignment.",
                    request.start_date, request.end_date
                ),
            ));
        }

        let requested_target_dates =
            calendar.open_dates_between(effective_start_date, effective_end_date);
        let target_dates =
            eligible_label_target_dates(&requested_target_dates, &calendar, max_lookahead);
        if target_dates.is_empty() {
            return Ok(empty_label_report(
                request,
                format!(
                    "No label target dates have enough future trading days for lookahead {}.",
                    max_lookahead
                ),
            ));
        }

        let label_batch_size = request.label_batch_size.max(1);
        let date_batches = split_dates_by_chunk(&target_dates, request.date_batch_size.max(1));
        let execution_stages = label_execution_stages(&labels, &specs, label_batch_size);
        let label_batch_count = execution_stages
            .iter()
            .map(|stage| stage.batch_ranges.len())
            .sum::<usize>();
        let loaded_requests =
            merge_requests(specs.iter().flat_map(|spec| spec.dependencies.clone()));
        let loaded_intraday_raw_requests = label_raw_requests(&labels);
        let execution_batch_count = date_batches.len() * label_batch_count;
        let date_batch_count = date_batches.len();
        if dry_run {
            return Ok(LabelRunReport {
                label_count: specs.len(),
                output_file_count: 0,
                target_dates,
                skipped_dates: Vec::new(),
                effective_start_date: Some(effective_start_date),
                effective_end_date: Some(effective_end_date),
                max_lookahead,
                date_batch_count,
                label_batch_count,
                execution_batch_count,
                selected_label_ids: specs.iter().map(|spec| spec.id.clone()).collect(),
                loaded_requests,
                loaded_intraday_raw_requests,
                profiles: Vec::new(),
                status_message: None,
            });
        }

        let catalog = DataCatalog::new(self.config.data_root.clone())
            .with_stock_sw_classification_path(self.config.stock_sw_classification_path.clone())
            .with_stock_ci_classification_path(self.config.stock_ci_classification_path.clone());
        let loader = MarketDataLoader::new(catalog);
        let storage = LabelStorage::new(self.config.label_root.clone());
        let raw_storage = IntradayDailyRawStorage::new(self.config.label_root.clone());
        let raw_providers = label_raw_providers();
        let thread_pool = build_thread_pool(request.threads)?;
        let progress = ProgressBar::new("label-run", execution_batch_count, true);
        let mut output_paths = BTreeSet::new();
        let mut profiles = Vec::new();
        let mut materialized_intraday_raw_dates = BTreeSet::new();

        for (date_batch_index, date_batch) in date_batches.iter().enumerate() {
            for stage in &execution_stages {
                for (label_batch_index, range) in stage.batch_ranges.iter().enumerate() {
                    let batch_indices = &stage.label_indices[range.clone()];
                    let batch_specs = batch_indices
                        .iter()
                        .map(|idx| specs[*idx].clone())
                        .collect::<Vec<_>>();
                    let batch_labels = batch_indices
                        .iter()
                        .map(|idx| labels[*idx].as_ref())
                        .collect::<Vec<_>>();
                    let batch_max_lookahead = batch_specs
                        .iter()
                        .map(|spec| spec.lookahead.trading_days)
                        .max()
                        .unwrap_or(stage.max_lookahead);
                    if date_batch.is_empty() {
                        progress.tick(format!(
                            "stage={} dates={}..{} skipped=empty_date_batch labels={}",
                            stage.name,
                            date_batch.first().copied().unwrap_or_default(),
                            date_batch.last().copied().unwrap_or_default(),
                            batch_specs.len()
                        ));
                        continue;
                    }

                    let batch_start_date = *date_batch
                        .first()
                        .expect("date batch is not empty after check");
                    let batch_last_target_date = *date_batch
                        .last()
                        .expect("date batch is not empty after check");
                    let batch_load_end_date = calendar
                        .open_date_after(batch_last_target_date, batch_max_lookahead)
                        .expect("target dates are filtered by global lookahead");
                    let batch_load_dates =
                        calendar.open_dates_between(batch_start_date, batch_load_end_date);
                    let batch_context = FactorContext {
                        asset_class: request.asset_class,
                        frequency: request.frequency,
                        start_date: batch_start_date,
                        end_date: batch_load_end_date,
                        load_start_date: batch_start_date,
                        load_dates: batch_load_dates,
                        target_dates: date_batch.clone(),
                    };

                    let raw_ids = raw_ids_for_label_indices(&labels, batch_indices);
                    let (raw_table, raw_profiles) = if raw_ids.is_empty() {
                        (None, Vec::new())
                    } else {
                        let (table, raw_profiles) = materialize_label_intraday_raw_table(
                            &raw_ids,
                            &raw_providers,
                            &loader,
                            &raw_storage,
                            &calendar,
                            request,
                            &batch_context,
                            date_batch_index + 1,
                            thread_pool.as_ref(),
                            &mut materialized_intraday_raw_dates,
                        )?;
                        (Some(table), raw_profiles)
                    };
                    profiles.extend(raw_profiles);

                    let batch_requests = merge_requests(
                        batch_specs
                            .iter()
                            .flat_map(|spec| spec.dependencies.clone()),
                    );
                    let load_started = Instant::now();
                    let mut pool = DataPool::load(&loader, &batch_requests, &batch_context)?;
                    if let Some(raw_table) = raw_table.clone() {
                        pool.set_intraday_daily_raw(raw_table, &batch_context)?;
                    }
                    let load_ms = load_started.elapsed().as_millis();
                    let compute_started = Instant::now();
                    let results = compute_label_batch(
                        &batch_labels,
                        &batch_context,
                        &pool,
                        thread_pool.as_ref(),
                    )?;
                    let compute_ms = compute_started.elapsed().as_millis();
                    let label_profiles = results
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
                            stage: stage.name.to_string(),
                            date_batch_index: date_batch_index + 1,
                            factor_batch_index: label_batch_index + 1,
                            start_date: batch_start_date,
                            end_date: batch_last_target_date,
                            factor_count: batch_specs.len(),
                            load_ms,
                            compute_ms,
                            write_ms,
                            factors: label_profiles,
                        });
                    }
                    progress.tick(format!(
                        "stage={} dates={}..{} labels={}",
                        stage.name,
                        batch_start_date,
                        batch_last_target_date,
                        batch_specs.len()
                    ));
                }
            }
        }
        progress.finish();

        Ok(LabelRunReport {
            label_count: specs.len(),
            output_file_count: output_paths.len(),
            target_dates,
            skipped_dates: Vec::new(),
            effective_start_date: Some(effective_start_date),
            effective_end_date: Some(effective_end_date),
            max_lookahead,
            date_batch_count,
            label_batch_count,
            execution_batch_count,
            selected_label_ids: specs.iter().map(|spec| spec.id.clone()).collect(),
            loaded_requests,
            loaded_intraday_raw_requests,
            profiles,
            status_message: None,
        })
    }
}

enum SelectionResult {
    Selected(Vec<LabelMetadata>),
    Empty(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct LabelExecutionStage {
    name: &'static str,
    label_indices: Vec<usize>,
    batch_ranges: Vec<Range<usize>>,
    max_lookahead: usize,
}

fn label_execution_stages(
    labels: &[Box<dyn Label>],
    specs: &[LabelSpec],
    label_batch_size: usize,
) -> Vec<LabelExecutionStage> {
    let raw_dependency_flags = labels
        .iter()
        .map(|label| !label.intraday_raw_dependencies().is_empty())
        .collect::<Vec<_>>();
    label_execution_stages_with_raw_flags(specs, &raw_dependency_flags, label_batch_size)
}

fn label_execution_stages_with_raw_flags(
    specs: &[LabelSpec],
    raw_dependency_flags: &[bool],
    label_batch_size: usize,
) -> Vec<LabelExecutionStage> {
    let mut daily_indices = Vec::new();
    let mut minute_indices = Vec::new();
    for (idx, spec) in specs.iter().enumerate() {
        if label_requires_minute(spec) || raw_dependency_flags.get(idx).copied().unwrap_or(false) {
            minute_indices.push(idx);
        } else {
            daily_indices.push(idx);
        }
    }

    let mut stages = Vec::new();
    if !daily_indices.is_empty() {
        stages.push(label_execution_stage(
            "daily_label_no_minute",
            daily_indices,
            specs,
            label_batch_size,
        ));
    }
    if !minute_indices.is_empty() {
        stages.push(label_execution_stage(
            "minute_label_postprocess",
            minute_indices,
            specs,
            label_batch_size,
        ));
    }
    stages
}

fn label_execution_stage(
    name: &'static str,
    label_indices: Vec<usize>,
    specs: &[LabelSpec],
    label_batch_size: usize,
) -> LabelExecutionStage {
    let max_lookahead = label_indices
        .iter()
        .map(|idx| specs[*idx].lookahead.trading_days)
        .max()
        .unwrap_or(0);
    let batch_ranges = label_batch_ranges(label_indices.len(), label_batch_size);
    LabelExecutionStage {
        name,
        label_indices,
        batch_ranges,
        max_lookahead,
    }
}

#[derive(Clone)]
struct LabelRawProvider {
    label: Arc<dyn Label>,
    spec: IntradayDailyRawSpec,
}

fn label_raw_providers() -> BTreeMap<String, LabelRawProvider> {
    let mut providers = BTreeMap::new();
    for label in all_labels() {
        let label: Arc<dyn Label> = Arc::from(label);
        for spec in label.intraday_raw_specs() {
            providers.insert(
                spec.raw_id.clone(),
                LabelRawProvider {
                    label: Arc::clone(&label),
                    spec,
                },
            );
        }
    }
    providers
}

fn label_raw_requests(labels: &[Box<dyn Label>]) -> Vec<IntradayDailyRawRequest> {
    let mut requests = BTreeMap::<String, usize>::new();
    for label in labels {
        for request in label.intraday_raw_dependencies() {
            requests
                .entry(request.raw_id)
                .and_modify(|lookback| *lookback = (*lookback).max(request.daily_lookback))
                .or_insert(request.daily_lookback);
        }
    }
    requests
        .into_iter()
        .map(|(raw_id, daily_lookback)| IntradayDailyRawRequest {
            raw_id,
            daily_lookback,
        })
        .collect()
}

fn raw_ids_for_label_indices(labels: &[Box<dyn Label>], label_indices: &[usize]) -> Vec<String> {
    let mut raw_ids = BTreeSet::new();
    for idx in label_indices {
        for request in labels[*idx].intraday_raw_dependencies() {
            raw_ids.insert(request.raw_id);
        }
    }
    raw_ids.into_iter().collect()
}

fn materialize_label_intraday_raw_table(
    raw_ids: &[String],
    providers: &BTreeMap<String, LabelRawProvider>,
    loader: &MarketDataLoader,
    storage: &IntradayDailyRawStorage,
    calendar: &TradingCalendar,
    request: &LabelRunRequest,
    context: &FactorContext,
    date_batch_index: usize,
    thread_pool: Option<&rayon::ThreadPool>,
    materialized_intraday_raw_dates: &mut BTreeSet<(String, i32)>,
) -> Result<(Table, Vec<BatchProfile>)> {
    let raw_id_set = raw_ids.iter().cloned().collect::<BTreeSet<_>>();
    let mut missing_by_date = BTreeMap::<i32, Vec<IntradayDailyRawSpec>>::new();
    for raw_id in raw_ids {
        let provider = providers
            .get(raw_id)
            .ok_or_else(|| err(format!("label intraday raw provider not found: {raw_id}")))?;
        let mut missing_dates = storage
            .missing_dates(
                &provider.spec,
                &context.load_dates,
                request.refresh_label_cache,
            )?
            .into_iter()
            .collect::<BTreeSet<_>>();
        missing_dates.retain(|date| {
            !materialized_intraday_raw_dates.contains(&(provider.spec.raw_id.clone(), *date))
        });
        for date in missing_dates {
            missing_by_date
                .entry(date)
                .or_default()
                .push(provider.spec.clone());
        }
    }

    let mut profiles = Vec::new();
    let mut materialized_specs = Vec::new();
    for (date, specs) in missing_by_date {
        let source_dataset = specs
            .first()
            .map(|spec| spec.source_dataset)
            .ok_or_else(|| err("empty label raw materialization plan"))?;
        let columns = specs
            .iter()
            .flat_map(|spec| spec.columns.iter().cloned())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        let raw_context = FactorContext {
            asset_class: request.asset_class,
            frequency: Frequency::Daily,
            start_date: date,
            end_date: date,
            load_start_date: date,
            load_dates: calendar.open_dates_between(date, date),
            target_dates: vec![date],
        };
        let batch_requests = vec![DataRequest {
            dataset: source_dataset,
            entity_id: None,
            bar_size: None,
            columns,
            financial_quarters: None,
        }];
        let load_started = Instant::now();
        let raw_pool = DataPool::load(loader, &batch_requests, &raw_context)?;
        let load_ms = load_started.elapsed().as_millis();

        let jobs = specs
            .iter()
            .map(|spec| {
                let provider = providers
                    .get(&spec.raw_id)
                    .expect("provider already validated");
                (Arc::clone(&provider.label), spec.raw_id.clone())
            })
            .collect::<Vec<_>>();

        let compute_started = Instant::now();
        let compute_raw = || -> Result<Vec<Vec<IntradayDailyRawSeries>>> {
            jobs.par_iter()
                .map(|(label, raw_id)| {
                    let raw_series = label.minute_compute_many(
                        std::slice::from_ref(raw_id),
                        &raw_context,
                        &raw_pool,
                    )?;
                    Ok(raw_series)
                })
                .collect()
        };
        let computed = match thread_pool {
            Some(thread_pool) => thread_pool.install(compute_raw),
            None => compute_raw(),
        }?;
        let mut chunk_series = Vec::new();
        for item in computed {
            let mut raw_series_list: Vec<_> = item
                .into_iter()
                .filter(|series| raw_id_set.contains(&series.spec.raw_id))
                .collect();
            chunk_series.append(&mut raw_series_list);
        }
        let returned = chunk_series
            .iter()
            .map(|series| series.spec.raw_id.clone())
            .collect::<BTreeSet<_>>();
        let expected = specs
            .iter()
            .map(|spec| spec.raw_id.clone())
            .collect::<BTreeSet<_>>();
        let missing = expected.difference(&returned).cloned().collect::<Vec<_>>();
        if !missing.is_empty() {
            return Err(err(format!(
                "label intraday raw provider did not return requested raw(s): {}",
                missing.join(",")
            )));
        }
        let compute_ms = compute_started.elapsed().as_millis();

        let raw_profiles = chunk_series
            .iter()
            .map(|raw_series| FactorProfile {
                factor_id: raw_series.spec.raw_id.clone(),
                row_count: raw_series.values.len(),
                non_null_count: raw_series
                    .values
                    .iter()
                    .filter(|item| item.value.is_some())
                    .count(),
            })
            .collect::<Vec<_>>();

        let write_started = Instant::now();
        if !chunk_series.is_empty() {
            storage.write_results(&chunk_series)?;
        }
        for raw_series in &chunk_series {
            materialized_intraday_raw_dates.insert((raw_series.spec.raw_id.clone(), date));
        }
        materialized_specs.extend(chunk_series.iter().map(|series| series.spec.clone()));
        let write_ms = write_started.elapsed().as_millis();
        if request.profile && !raw_profiles.is_empty() {
            profiles.push(BatchProfile {
                stage: "label_intraday_raw_materialize_window_1".to_string(),
                date_batch_index,
                factor_batch_index: 1,
                start_date: date,
                end_date: date,
                factor_count: raw_profiles.len(),
                load_ms,
                compute_ms,
                write_ms,
                factors: raw_profiles,
            });
        }
    }
    if !materialized_specs.is_empty() {
        storage.write_metadata(&materialized_specs)?;
    }

    Ok((
        storage.load_raw_by_dates(context.asset_class, raw_ids, &context.load_dates)?,
        profiles,
    ))
}

fn select_label_metadata(request: &LabelRunRequest, metadata: &[LabelMetadata]) -> SelectionResult {
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
                "No labels found for tag(s): {}",
                tags.join(",")
            ));
        }
        return SelectionResult::Selected(selected);
    }

    if let Some(label_ids) = &request.label_ids {
        let mut selected = Vec::new();
        let mut missing = Vec::new();
        for label_id_or_name in label_ids {
            if let Some(row) = base.iter().find(|row| {
                row.label_id == *label_id_or_name
                    || row.name == *label_id_or_name
                    || row.aliases.iter().any(|alias| alias == label_id_or_name)
            }) {
                selected.push(row.clone());
            } else {
                missing.push(label_id_or_name.clone());
            }
        }
        if !missing.is_empty() {
            return SelectionResult::Empty(format!(
                "No labels found in metadata for: {}",
                missing.join(",")
            ));
        }
        return SelectionResult::Selected(selected);
    }

    if base.is_empty() {
        return SelectionResult::Empty(format!(
            "No labels found in metadata for asset={} frequency={}.",
            request.asset_class, request.frequency
        ));
    }
    SelectionResult::Selected(base)
}

fn empty_label_report(_request: &LabelRunRequest, message: String) -> LabelRunReport {
    LabelRunReport {
        label_count: 0,
        output_file_count: 0,
        target_dates: Vec::new(),
        skipped_dates: Vec::new(),
        effective_start_date: None,
        effective_end_date: None,
        max_lookahead: 0,
        date_batch_count: 0,
        label_batch_count: 0,
        execution_batch_count: 0,
        selected_label_ids: Vec::new(),
        loaded_requests: Vec::new(),
        loaded_intraday_raw_requests: Vec::new(),
        profiles: Vec::new(),
        status_message: Some(message),
    }
}

fn merge_requests<I>(requests: I) -> Vec<DataRequest>
where
    I: IntoIterator<Item = DataRequest>,
{
    let mut grouped: HashMap<_, BTreeSet<String>> = HashMap::new();
    for request in requests {
        let key = (request.dataset, request.entity_id.clone(), request.bar_size);
        grouped
            .entry(key)
            .or_default()
            .extend(request.columns.into_iter());
    }
    let mut merged = grouped
        .into_iter()
        .map(|((dataset, entity_id, bar_size), columns)| DataRequest {
            dataset,
            entity_id,
            bar_size,
            columns: columns.into_iter().collect(),
            financial_quarters: None,
        })
        .collect::<Vec<_>>();
    merged.sort_by(|left, right| {
        left.dataset
            .cmp(&right.dataset)
            .then_with(|| left.entity_id.cmp(&right.entity_id))
            .then_with(|| left.bar_size.cmp(&right.bar_size))
    });
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

fn label_batch_ranges(label_count: usize, label_batch_size: usize) -> Vec<Range<usize>> {
    if label_count == 0 {
        return Vec::new();
    }
    let batch_size = label_batch_size.max(1);
    (0..label_count)
        .step_by(batch_size)
        .map(|start| start..(start + batch_size).min(label_count))
        .collect()
}

fn split_dates_by_chunk(dates: &[i32], chunk_size: usize) -> Vec<Vec<i32>> {
    let chunk_size = chunk_size.max(1);
    dates
        .chunks(chunk_size)
        .map(|chunk| chunk.to_vec())
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

fn compute_label_batch(
    labels: &[&dyn Label],
    context: &FactorContext,
    pool: &DataPool,
    thread_pool: Option<&rayon::ThreadPool>,
) -> Result<Vec<LabelSeries>> {
    let compute = || {
        labels
            .par_iter()
            .map(|label| label.compute(context, pool))
            .collect::<Vec<_>>()
    };
    let results = match thread_pool {
        Some(thread_pool) => thread_pool.install(compute),
        None => compute(),
    };
    results.into_iter().collect()
}

fn label_requires_minute(spec: &LabelSpec) -> bool {
    spec.dependencies
        .iter()
        .any(|request| request.dataset == DatasetId::StockMinute1m)
}

fn eligible_label_target_dates(
    requested_target_dates: &[i32],
    calendar: &TradingCalendar,
    max_lookahead: usize,
) -> Vec<i32> {
    requested_target_dates
        .iter()
        .copied()
        .filter(|date| calendar.has_open_date_after(*date, max_lookahead))
        .collect()
}

pub fn available_label_specs() -> Vec<LabelSpec> {
    all_labels().into_iter().map(|label| label.spec()).collect()
}

#[cfg(test)]
mod tests {
    use crate::core::{AssetClass, DataRequest, DatasetId, Frequency, LabelSpec, Lookahead};
    use crate::storage::LabelMetadata;

    use super::{
        eligible_label_target_dates, label_batch_ranges, label_execution_stages_with_raw_flags,
        merge_requests, select_label_metadata, LabelRunRequest, SelectionResult,
    };
    use crate::calendar::TradingCalendar;

    #[test]
    fn chunks_labels_by_configured_batch_size() {
        assert_eq!(
            label_batch_ranges(0, 10),
            Vec::<std::ops::Range<usize>>::new()
        );
        assert_eq!(label_batch_ranges(3, 10), vec![0..3]);
        assert_eq!(label_batch_ranges(11, 10), vec![0..10, 10..11]);
    }

    #[test]
    fn execution_stages_run_daily_labels_before_minute_labels() {
        let specs = vec![
            label_spec(
                "future_open_return_1d",
                vec![
                    DataRequest::new(DatasetId::StockDailyPv, &["open"]),
                    DataRequest::new(DatasetId::StockAdjFactor, &["adj_factor"]),
                ],
                2,
            ),
            label_spec(
                "future_open_5m_vwap_return_20d",
                vec![
                    DataRequest::new(DatasetId::StockDailyPv, &["close"]),
                    DataRequest::new(DatasetId::StockAdjFactor, &["adj_factor"]),
                    DataRequest::new(DatasetId::StockMinute1m, &["amount", "vol"]),
                ],
                21,
            ),
            label_spec(
                "future_close_return_5d",
                vec![
                    DataRequest::new(DatasetId::StockDailyPv, &["close"]),
                    DataRequest::new(DatasetId::StockAdjFactor, &["adj_factor"]),
                ],
                6,
            ),
        ];

        let raw_dependency_flags = vec![false, true, false];
        let stages = label_execution_stages_with_raw_flags(&specs, &raw_dependency_flags, 1);

        assert_eq!(stages.len(), 2);
        assert_eq!(stages[0].name, "daily_label_no_minute");
        assert_eq!(stages[0].label_indices, vec![0, 2]);
        assert_eq!(stages[0].batch_ranges, vec![0..1, 1..2]);
        assert_eq!(stages[0].max_lookahead, 6);
        assert_eq!(stages[1].name, "minute_label_postprocess");
        assert_eq!(stages[1].label_indices, vec![1]);
        assert_eq!(stages[1].batch_ranges, vec![0..1]);
        assert_eq!(stages[1].max_lookahead, 21);
    }

    #[test]
    fn stage_requests_keep_minute_data_out_of_daily_stage() {
        let specs = vec![
            label_spec(
                "future_vwap_return_1d",
                vec![DataRequest::new(
                    DatasetId::StockDailyPv,
                    &["amount", "vol"],
                )],
                2,
            ),
            label_spec(
                "future_open_10m_vwap_return_1d",
                vec![DataRequest::new(
                    DatasetId::StockMinute1m,
                    &["amount", "vol"],
                )],
                2,
            ),
        ];
        let raw_dependency_flags = vec![false, true];
        let stages = label_execution_stages_with_raw_flags(&specs, &raw_dependency_flags, 5);

        let daily_requests = merge_requests(
            stages[0]
                .label_indices
                .iter()
                .flat_map(|idx| specs[*idx].dependencies.clone()),
        );
        let minute_requests = merge_requests(
            stages[1]
                .label_indices
                .iter()
                .flat_map(|idx| specs[*idx].dependencies.clone()),
        );

        assert!(!daily_requests
            .iter()
            .any(|request| request.dataset == DatasetId::StockMinute1m));
        assert!(minute_requests
            .iter()
            .any(|request| request.dataset == DatasetId::StockMinute1m));
    }

    #[test]
    fn target_dates_without_required_future_horizon_are_skipped() {
        let calendar = TradingCalendar::from_open_dates(vec![
            20260105, 20260106, 20260107, 20260108, 20260109,
        ]);

        assert_eq!(
            eligible_label_target_dates(&[20260105, 20260106, 20260107], &calendar, 2),
            vec![20260105, 20260106, 20260107]
        );
        assert_eq!(
            eligible_label_target_dates(&[20260105, 20260106, 20260107], &calendar, 3),
            vec![20260105, 20260106]
        );
    }

    #[test]
    fn selection_matches_label_id_inside_asset_frequency() {
        let request = LabelRunRequest {
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            start_date: 20260105,
            end_date: 20260105,
            label_ids: Some(vec!["future_open_return_1d".to_string()]),
            tags: None,
            config_path: None,
            dry_run: false,
            label_batch_size: 10,
            date_batch_size: 1,
            threads: None,
            profile: false,
            refresh_label_cache: false,
        };
        let metadata = vec![
            metadata_row("future_open_return_1d", "stock", "daily"),
            metadata_row("future_open_return_1d", "future", "daily"),
        ];

        let SelectionResult::Selected(selected) = select_label_metadata(&request, &metadata) else {
            panic!("expected selected");
        };
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].asset_class, "stock");
        assert_eq!(selected[0].label_id, "future_open_return_1d");
    }

    fn metadata_row(label_id: &str, asset_class: &str, frequency: &str) -> LabelMetadata {
        LabelMetadata {
            label_id: label_id.to_string(),
            aliases: Vec::new(),
            aliases_json: String::new(),
            version: "0.1.0".to_string(),
            output_column: label_id.to_string(),
            name: label_id.to_string(),
            asset_class: asset_class.to_string(),
            frequency: frequency.to_string(),
            tags: Vec::new(),
            tags_json: String::new(),
            dependencies_json: String::new(),
            description: String::new(),
            updated_at: String::new(),
        }
    }

    fn label_spec(id: &str, dependencies: Vec<DataRequest>, lookahead: usize) -> LabelSpec {
        LabelSpec {
            id: id.to_string(),
            aliases: Vec::new(),
            name: id.to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: "0.1.0".to_string(),
            tags: Vec::new(),
            description: String::new(),
            dependencies,
            lookahead: Lookahead {
                trading_days: lookahead,
            },
        }
    }

    #[test]
    fn label_spec_output_column_is_short_id() {
        let spec = LabelSpec {
            id: "future_open_return_1d".to_string(),
            aliases: Vec::new(),
            name: String::new(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: "0.1.0".to_string(),
            tags: Vec::new(),
            description: String::new(),
            dependencies: vec![DataRequest::new(DatasetId::StockDailyPv, &["open"])],
            lookahead: Lookahead { trading_days: 2 },
        };
        assert_eq!(spec.output_column(), "future_open_return_1d");
    }
}

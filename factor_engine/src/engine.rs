use std::any::Any;
use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use crate::calendar::TradingCalendar;
use crate::config::EngineConfig;
use crate::core::{
    factor_registry_key, AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries,
    FactorSpec, Frequency, IntradayDailyRawAuxiliaryRequest, IntradayDailyRawRequest,
    IntradayDailyRawSpec,
};
use crate::data::{
    financial_disclosure_years_for_range, DataCatalog, DataPool, DisclosureTableCache,
    MarketDataLoader,
};
use crate::error::{err, Result};
use crate::factor::registry::{all_factors, factor_map};
use crate::factor::{Factor, IntradayRawMaterializeMode};
use crate::progress::ProgressBar;
use crate::storage::{FactorMetadata, FactorStorage, IntradayDailyRawStorage};
use rayon::prelude::*;

pub const DEFAULT_FACTOR_BATCH_SIZE: usize = 64;
pub const DEFAULT_DATE_BATCH_SIZE: usize = 1;

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
    pub date_batch_size: usize,
    pub threads: Option<usize>,
    pub profile: bool,
    pub refresh_minute_cache: bool,
}

#[derive(Clone, Debug)]
pub struct RunReport {
    pub factor_count: usize,
    pub output_file_count: usize,
    pub load_start_date: i32,
    pub target_dates: Vec<i32>,
    pub effective_start_date: Option<i32>,
    pub effective_end_date: Option<i32>,
    pub execution_stages: Vec<String>,
    pub date_batch_count: usize,
    pub factor_batch_count: usize,
    pub execution_batch_count: usize,
    pub selected_factor_ids: Vec<String>,
    pub loaded_requests: Vec<DataRequest>,
    pub loaded_intraday_raw_requests: Vec<IntradayDailyRawRequest>,
    pub profiles: Vec<BatchProfile>,
    pub status_message: Option<String>,
}

#[derive(Clone, Debug)]
pub struct BatchProfile {
    pub stage: String,
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
        let raw_providers = intraday_raw_provider_map()?;
        let raw_requirements = resolve_intraday_raw_requirements(&specs, &raw_providers)?;
        let max_lookback = specs
            .iter()
            .map(spec_calendar_lookback_days)
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

        let factor_batch_size = request.factor_batch_size.max(1);
        let raw_work = build_intraday_raw_work(&raw_requirements, &target_dates, &calendar)?;
        let execution_groups = execution_groups_for_specs(request.frequency, &specs);
        let execution_stages = execution_stage_names(&raw_work, &execution_groups);
        let date_batch_size = request.date_batch_size.max(1);
        let raw_date_batch_count = raw_work
            .iter()
            .map(|work| split_dates_by_chunk(&work.target_dates, DEFAULT_DATE_BATCH_SIZE).len())
            .sum::<usize>();
        let raw_factor_batch_count = raw_work.len();
        let raw_execution_batch_count = raw_date_batch_count;
        let execution_plans = execution_groups
            .iter()
            .map(|group| {
                let date_batch_count =
                    date_batches_for_stage(&group.stage, &target_dates, date_batch_size).len();
                let factor_batch_count =
                    provider_factor_batches(&factors, &group.factor_indices, factor_batch_size)
                        .len();
                (date_batch_count, factor_batch_count)
            })
            .collect::<Vec<_>>();
        let date_batch_count = raw_date_batch_count
            + execution_plans
                .iter()
                .map(|(date_batch_count, _)| *date_batch_count)
                .sum::<usize>();
        let factor_batch_count = raw_factor_batch_count
            + execution_plans
                .iter()
                .map(|(_, factor_batch_count)| *factor_batch_count)
                .sum::<usize>();
        let execution_batch_count = raw_execution_batch_count
            + execution_plans
                .iter()
                .map(|(date_batch_count, factor_batch_count)| date_batch_count * factor_batch_count)
                .sum::<usize>();
        let load_start_date = calendar.warmup_start(effective_start_date, max_lookback);

        let raw_source_specs_for_report = raw_requirements
            .iter()
            .map(|requirement| &requirement.spec)
            .collect::<Vec<_>>();
        let raw_auxiliary_requests_for_report =
            auxiliary_requests_for_raw_requirements(&raw_requirements, &raw_providers)?;
        let loaded_requests = merge_requests(
            specs
                .iter()
                .flat_map(|spec| spec.dependencies.clone())
                .chain(raw_source_specs_for_report.iter().map(|spec| DataRequest {
                    dataset: spec.source_dataset,
                    entity_id: None,
                    bar_size: spec.source_bar_size,
                    columns: spec.columns.clone(),
                    financial_quarters: None,
                    date_policy: Default::default(),
                }))
                .chain(
                    raw_auxiliary_requests_for_report
                        .into_iter()
                        .map(|request| request.request),
                ),
        );
        let loaded_intraday_raw_requests = raw_requirements
            .iter()
            .map(|requirement| {
                IntradayDailyRawRequest::new(&requirement.spec.raw_id, requirement.daily_lookback)
            })
            .collect::<Vec<_>>();
        if dry_run {
            return Ok(RunReport {
                factor_count: specs.len(),
                output_file_count: 0,
                load_start_date,
                target_dates,
                effective_start_date: Some(effective_start_date),
                effective_end_date: Some(effective_end_date),
                execution_stages,
                date_batch_count,
                factor_batch_count,
                execution_batch_count,
                selected_factor_ids: specs.iter().map(|spec| spec.id.clone()).collect(),
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
        let storage = FactorStorage::new(self.config.factor_root.clone());
        let raw_storage = IntradayDailyRawStorage::new(self.config.factor_root.clone());
        let thread_pool = build_thread_pool(request.threads)?;
        let mut output_paths = BTreeSet::new();
        let mut profiles = Vec::new();
        let mut materialized_intraday_raw_dates = BTreeSet::<(String, i32)>::new();
        let mut disclosure_cache = DisclosureTableCache::default();
        let mut compute_states = BTreeMap::<String, Box<dyn Any + Send>>::new();
        let progress = ProgressBar::new("run", execution_batch_count, true);

        for group in &execution_groups {
            let group_specs = group
                .factor_indices
                .iter()
                .map(|idx| specs[*idx].clone())
                .collect::<Vec<_>>();
            let group_requests = merge_requests(
                group_specs
                    .iter()
                    .flat_map(|spec| spec.dependencies.clone()),
            );
            let group_max_lookback = group_specs
                .iter()
                .map(spec_calendar_lookback_days)
                .max()
                .unwrap_or(0);
            let date_batches =
                date_batches_for_stage(&group.stage, &target_dates, request.date_batch_size);
            let provider_batches =
                provider_factor_batches(&factors, &group.factor_indices, factor_batch_size);
            let stage_name = group.stage.name();

            for (date_batch_index, date_batch) in date_batches.iter().enumerate() {
                let batch_start_date = *date_batch
                    .first()
                    .expect("date batches are never empty after split");
                let batch_end_date = *date_batch
                    .last()
                    .expect("date batches are never empty after split");
                let batch_load_start_date =
                    calendar.warmup_start(batch_start_date, group_max_lookback);
                let load_dates = calendar.open_dates_between(batch_load_start_date, batch_end_date);
                let context = FactorContext {
                    asset_class: request.asset_class,
                    frequency: request.frequency,
                    start_date: batch_start_date,
                    end_date: batch_end_date,
                    load_start_date: batch_load_start_date,
                    load_dates,
                    target_dates: date_batch.clone(),
                };

                for (factor_batch_index, batch_indices) in provider_batches.iter().enumerate() {
                    let batch_specs = batch_indices
                        .iter()
                        .map(|idx| specs[*idx].clone())
                        .collect::<Vec<_>>();
                    let batch_factors = batch_indices
                        .iter()
                        .map(|idx| factors[*idx].as_ref())
                        .collect::<Vec<_>>();
                    let contextual_requirements = contextual_requirements_for_factor_batch(
                        &batch_factors,
                        &batch_specs,
                        &context,
                        &calendar,
                    );
                    let batch_requests = merge_requests(contextual_requirements.all.clone());
                    let load_started = Instant::now();
                    let mut pool = DataPool::load_with_disclosure_cache(
                        &loader,
                        &batch_requests,
                        &context,
                        &mut disclosure_cache,
                    )?;
                    let load_ms = load_started.elapsed().as_millis();
                    let raw_ids = raw_ids_for_specs(&batch_specs);
                    if !raw_ids.is_empty() {
                        let (raw_table, mut raw_profiles) = materialize_intraday_raw_table(
                            &raw_ids,
                            &raw_requirements,
                            &raw_providers,
                            &loader,
                            &raw_storage,
                            &calendar,
                            request,
                            &context,
                            date_batch_index + 1,
                            factor_batch_index + 1,
                            thread_pool.as_ref(),
                            &mut materialized_intraday_raw_dates,
                            &progress,
                        )?;
                        profiles.append(&mut raw_profiles);
                        pool.set_intraday_daily_raw(raw_table, &context)?;
                    }
                    let compute_started = Instant::now();
                    let results = compute_factor_batch(
                        &batch_factors,
                        &context,
                        &pool,
                        &contextual_requirements.by_provider,
                        &mut compute_states,
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
                            stage: stage_name.clone(),
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
                    progress.tick(format!(
                        "stage={} date={} factors={}",
                        stage_name,
                        batch_end_date,
                        batch_specs.len()
                    ));
                }
                let keep_years =
                    financial_years_for_requests(&group_requests, batch_start_date, batch_end_date);
                if !keep_years.is_empty() {
                    disclosure_cache.retain_financial_years(&keep_years);
                }
            }
        }
        progress.finish();

        Ok(RunReport {
            factor_count: specs.len(),
            output_file_count: output_paths.len(),
            load_start_date,
            target_dates,
            effective_start_date: Some(effective_start_date),
            effective_end_date: Some(effective_end_date),
            execution_stages,
            date_batch_count,
            factor_batch_count,
            execution_batch_count,
            selected_factor_ids: specs.iter().map(|spec| spec.id.clone()).collect(),
            loaded_requests,
            loaded_intraday_raw_requests,
            profiles,
            status_message: None,
        })
    }
}

enum SelectionResult {
    Selected(Vec<FactorMetadata>),
    Empty(String),
}

#[derive(Clone, Debug)]
struct IntradayRawRequirement {
    spec: IntradayDailyRawSpec,
    daily_lookback: usize,
}

#[derive(Clone)]
struct RawProvider {
    spec: IntradayDailyRawSpec,
    provider_key: String,
    factor: Arc<dyn Factor>,
}

#[derive(Clone)]
struct RawJob {
    raw_ids: Vec<String>,
    factor: Arc<dyn Factor>,
    context: FactorContext,
}

#[derive(Clone, Debug)]
struct IntradayRawWork {
    spec: IntradayDailyRawSpec,
    target_dates: Vec<i32>,
}

fn intraday_raw_provider_map() -> Result<BTreeMap<String, RawProvider>> {
    let mut providers = BTreeMap::new();
    for factor in all_factors() {
        let factor: Arc<dyn Factor> = Arc::from(factor);
        for spec in factor.intraday_raw_specs() {
            if providers.contains_key(&spec.raw_id) {
                return Err(err(format!(
                    "duplicate intraday daily raw provider registered: {}",
                    spec.raw_id
                )));
            }
            let provider_key = factor.intraday_raw_provider_key(&spec.raw_id);
            providers.insert(
                spec.raw_id.clone(),
                RawProvider {
                    spec,
                    provider_key: provider_key.clone(),
                    factor: Arc::clone(&factor),
                },
            );
        }
    }
    Ok(providers)
}

fn resolve_intraday_raw_requirements(
    specs: &[FactorSpec],
    providers: &BTreeMap<String, RawProvider>,
) -> Result<Vec<IntradayRawRequirement>> {
    let mut grouped = BTreeMap::<String, usize>::new();
    for spec in specs {
        for dependency in &spec.intraday_raw_dependencies {
            grouped
                .entry(dependency.raw_id.clone())
                .and_modify(|lookback| *lookback = (*lookback).max(dependency.daily_lookback))
                .or_insert(dependency.daily_lookback);
        }
    }

    let mut requirements = Vec::new();
    for (raw_id, daily_lookback) in grouped {
        let provider = providers.get(&raw_id).ok_or_else(|| {
            err(format!(
                "intraday daily raw implementation not found: {raw_id}"
            ))
        })?;
        requirements.push(IntradayRawRequirement {
            spec: provider.spec.clone(),
            daily_lookback,
        });
    }
    Ok(requirements)
}

fn build_intraday_raw_work(
    requirements: &[IntradayRawRequirement],
    target_dates: &[i32],
    calendar: &TradingCalendar,
) -> Result<Vec<IntradayRawWork>> {
    let Some(first_target_date) = target_dates.first().copied() else {
        return Ok(Vec::new());
    };
    let Some(last_target_date) = target_dates.last().copied() else {
        return Ok(Vec::new());
    };
    let mut work = Vec::new();
    for requirement in requirements {
        let raw_start_date = calendar.warmup_start(first_target_date, requirement.daily_lookback);
        let target_dates = calendar.open_dates_between(raw_start_date, last_target_date);
        if target_dates.is_empty() {
            continue;
        }
        work.push(IntradayRawWork {
            spec: requirement.spec.clone(),
            target_dates,
        });
    }
    Ok(work)
}

fn materialize_intraday_raw_table(
    raw_ids: &[String],
    requirements: &[IntradayRawRequirement],
    providers: &BTreeMap<String, RawProvider>,
    loader: &MarketDataLoader,
    storage: &IntradayDailyRawStorage,
    calendar: &TradingCalendar,
    request: &RunRequest,
    context: &FactorContext,
    date_batch_index: usize,
    factor_batch_index: usize,
    thread_pool: Option<&rayon::ThreadPool>,
    materialized_intraday_raw_dates: &mut BTreeSet<(String, i32)>,
    progress: &ProgressBar,
) -> Result<(crate::data::Table, Vec<BatchProfile>)> {
    let raw_id_set =
        expand_raw_ids_to_selected_provider_siblings(raw_ids, requirements, providers)?;
    let mut stateless_requirements_by_dataset =
        BTreeMap::<(DatasetId, Option<usize>), Vec<(&IntradayRawRequirement, BTreeSet<i32>)>>::new(
        );
    let mut stateful_requirements_by_dataset =
        BTreeMap::<(DatasetId, Option<usize>), Vec<(&IntradayRawRequirement, BTreeSet<i32>)>>::new(
        );
    for requirement in requirements
        .iter()
        .filter(|requirement| raw_id_set.contains(&requirement.spec.raw_id))
    {
        let mut missing_dates = storage
            .missing_dates(
                &requirement.spec,
                &context.load_dates,
                request.refresh_minute_cache,
            )?
            .into_iter()
            .collect::<BTreeSet<_>>();
        let raw_id = requirement.spec.raw_id.clone();
        missing_dates
            .retain(|date| !materialized_intraday_raw_dates.contains(&(raw_id.clone(), *date)));
        if missing_dates.is_empty() {
            continue;
        }
        let provider = providers.get(&requirement.spec.raw_id).ok_or_else(|| {
            err(format!(
                "intraday daily raw implementation not found: {}",
                requirement.spec.raw_id
            ))
        })?;
        let mode = provider
            .factor
            .intraday_raw_materialize_mode(std::slice::from_ref(&requirement.spec.raw_id));
        let target_map = match mode {
            IntradayRawMaterializeMode::Stateless => &mut stateless_requirements_by_dataset,
            IntradayRawMaterializeMode::Stateful => &mut stateful_requirements_by_dataset,
        };
        target_map
            .entry((
                requirement.spec.source_dataset,
                requirement.spec.source_bar_size,
            ))
            .or_default()
            .push((requirement, missing_dates));
    }
    let mut profiles = Vec::new();
    let mut materialized_specs = Vec::new();

    for ((source_dataset, source_bar_size), plans) in
        ordered_source_groups(stateless_requirements_by_dataset)
    {
        let max_window_days = plans
            .iter()
            .map(|(requirement, _)| requirement.spec.window_days)
            .max()
            .unwrap_or(1);
        let columns = plans
            .iter()
            .flat_map(|(requirement, _)| requirement.spec.columns.iter().cloned())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        let missing_dates = plans
            .iter()
            .flat_map(|(_, dates)| dates.iter().copied())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        let date_batches = split_dates_by_chunk(&missing_dates, DEFAULT_DATE_BATCH_SIZE);
        let stage_name = if plans.len() == 1 {
            raw_materialize_stage_name(&plans[0].0.spec)
        } else {
            format!("intraday_raw_materialize_window_{max_window_days}")
        };
        for date_batch in date_batches {
            let batch_start_date = *date_batch
                .first()
                .expect("raw date batches are never empty after split");
            let batch_end_date = *date_batch
                .last()
                .expect("raw date batches are never empty after split");
            let minute_lookback = max_window_days.saturating_sub(1);
            let batch_load_start_date = calendar.warmup_start(batch_start_date, minute_lookback);
            let load_dates = calendar.open_dates_between(batch_load_start_date, batch_end_date);
            let raw_context = FactorContext {
                asset_class: request.asset_class,
                frequency: Frequency::Daily,
                start_date: batch_start_date,
                end_date: batch_end_date,
                load_start_date: batch_load_start_date,
                load_dates,
                target_dates: date_batch.clone(),
            };
            let batch_requests = vec![DataRequest {
                dataset: source_dataset,
                entity_id: None,
                bar_size: source_bar_size,
                columns: columns.clone(),
                financial_quarters: None,
                date_policy: Default::default(),
            }];

            let load_started = Instant::now();
            let mut raw_pool = DataPool::load(loader, &batch_requests, &raw_context)?;
            let auxiliary_requests =
                auxiliary_requests_for_raw_batch(&plans, providers, &date_batch)?;
            let auxiliary_target_dates =
                auxiliary_target_dates(source_dataset, source_bar_size, &raw_pool, &date_batch);
            if !auxiliary_requests.is_empty() && !auxiliary_target_dates.is_empty() {
                let max_auxiliary_lookback = auxiliary_requests
                    .iter()
                    .map(|request| request.daily_lookback)
                    .max()
                    .unwrap_or(0);
                let auxiliary_start_date = *auxiliary_target_dates
                    .first()
                    .expect("auxiliary target dates are not empty");
                let auxiliary_end_date = *auxiliary_target_dates
                    .last()
                    .expect("auxiliary target dates are not empty");
                let auxiliary_load_start_date =
                    calendar.warmup_start(auxiliary_start_date, max_auxiliary_lookback);
                let auxiliary_context = FactorContext {
                    asset_class: request.asset_class,
                    frequency: Frequency::Daily,
                    start_date: auxiliary_start_date,
                    end_date: auxiliary_end_date,
                    load_start_date: auxiliary_load_start_date,
                    load_dates: calendar
                        .open_dates_between(auxiliary_load_start_date, auxiliary_end_date),
                    target_dates: auxiliary_target_dates,
                };
                let auxiliary_pool = DataPool::load(
                    loader,
                    &merge_requests(
                        auxiliary_requests
                            .into_iter()
                            .map(|request| request.request),
                    ),
                    &auxiliary_context,
                )?;
                raw_pool.extend(auxiliary_pool);
            }
            let load_ms = load_started.elapsed().as_millis();
            let compute_started = Instant::now();
            let mut grouped_jobs =
                BTreeMap::<String, (Arc<dyn Factor>, BTreeSet<String>, BTreeSet<i32>, usize)>::new(
                );
            for (requirement, requirement_missing_dates) in &plans {
                let provider_target_dates = date_batch
                    .iter()
                    .filter(|date| requirement_missing_dates.contains(date))
                    .copied()
                    .collect::<Vec<_>>();
                if provider_target_dates.is_empty() {
                    continue;
                }
                let provider = providers.get(&requirement.spec.raw_id).ok_or_else(|| {
                    err(format!(
                        "intraday daily raw implementation not found: {}",
                        requirement.spec.raw_id
                    ))
                })?;
                let entry = grouped_jobs
                    .entry(provider.provider_key.clone())
                    .or_insert_with(|| {
                        (
                            Arc::clone(&provider.factor),
                            BTreeSet::new(),
                            BTreeSet::new(),
                            1,
                        )
                    });
                entry.1.insert(requirement.spec.raw_id.clone());
                entry.2.extend(provider_target_dates);
                entry.3 = entry.3.max(requirement.spec.window_days);
            }
            let jobs = grouped_jobs
                .into_values()
                .filter_map(|(factor, raw_ids, target_dates, window_days)| {
                    let provider_target_dates = target_dates.into_iter().collect::<Vec<_>>();
                    if provider_target_dates.is_empty() {
                        return None;
                    }
                    let provider_lookback = window_days.saturating_sub(1);
                    let provider_batch_start = *provider_target_dates
                        .first()
                        .expect("provider target dates are not empty");
                    let provider_batch_end = *provider_target_dates
                        .last()
                        .expect("provider target dates are not empty");
                    let provider_load_start_date =
                        calendar.warmup_start(provider_batch_start, provider_lookback);
                    let provider_context = FactorContext {
                        asset_class: raw_context.asset_class,
                        frequency: raw_context.frequency,
                        start_date: provider_batch_start,
                        end_date: provider_batch_end,
                        load_start_date: provider_load_start_date,
                        load_dates: calendar
                            .open_dates_between(provider_load_start_date, provider_batch_end),
                        target_dates: provider_target_dates,
                    };
                    Some(RawJob {
                        raw_ids: raw_ids.into_iter().collect(),
                        factor,
                        context: provider_context,
                    })
                })
                .collect::<Vec<_>>();

            let compute_raw = || {
                jobs.par_iter()
                    .map(|job| {
                        let requested = job.raw_ids.iter().cloned().collect::<BTreeSet<_>>();
                        let mut raw_series_list = job.factor.minute_compute_many(
                            &job.raw_ids,
                            &job.context,
                            &raw_pool,
                        )?;
                        raw_series_list.retain(|series| requested.contains(&series.spec.raw_id));
                        let returned = raw_series_list
                            .iter()
                            .map(|series| series.spec.raw_id.clone())
                            .collect::<BTreeSet<_>>();
                        let missing = requested.difference(&returned).cloned().collect::<Vec<_>>();
                        if !missing.is_empty() {
                            return Err(err(format!(
                                "intraday daily raw provider did not return requested raw(s): {}",
                                missing.join(",")
                            )));
                        }
                        let raw_profiles = raw_series_list
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
                        Ok((
                            raw_series_list,
                            raw_profiles,
                            job.context.target_dates.clone(),
                        ))
                    })
                    .collect::<Vec<_>>()
            };
            let computed = match thread_pool {
                Some(thread_pool) => thread_pool.install(compute_raw),
                None => compute_raw(),
            };
            let mut chunk_series = Vec::new();
            let mut raw_profiles = Vec::new();
            let mut chunk_materialized_dates = Vec::new();
            for item in computed {
                let (raw_series_list, profiles, target_dates) = item?;
                for raw_series in raw_series_list {
                    for date in &target_dates {
                        chunk_materialized_dates.push((raw_series.spec.raw_id.clone(), *date));
                    }
                    chunk_series.push(raw_series);
                }
                raw_profiles.extend(profiles);
            }
            let compute_ms = compute_started.elapsed().as_millis();
            let write_started = Instant::now();
            if !chunk_series.is_empty() {
                storage.write_results(&chunk_series)?;
            }
            materialized_intraday_raw_dates.extend(chunk_materialized_dates);
            let write_ms = write_started.elapsed().as_millis();
            if request.profile && !raw_profiles.is_empty() {
                profiles.push(BatchProfile {
                    stage: stage_name.clone(),
                    date_batch_index,
                    factor_batch_index,
                    start_date: batch_start_date,
                    end_date: batch_end_date,
                    factor_count: raw_profiles.len(),
                    load_ms,
                    compute_ms,
                    write_ms,
                    factors: raw_profiles,
                });
            }
            progress.tick(format!(
                "stage={} date={} raw={}",
                stage_name,
                batch_end_date,
                chunk_series.len()
            ));
        }
        materialized_specs.extend(
            plans
                .iter()
                .map(|(requirement, _)| requirement.spec.clone()),
        );
    }
    for ((source_dataset, source_bar_size), plans) in
        ordered_source_groups(stateful_requirements_by_dataset)
    {
        let columns = plans
            .iter()
            .flat_map(|(requirement, _)| requirement.spec.columns.iter().cloned())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        let batch_requests = vec![DataRequest {
            dataset: source_dataset,
            entity_id: None,
            bar_size: source_bar_size,
            columns,
            financial_quarters: None,
            date_policy: Default::default(),
        }];

        let mut grouped_jobs = BTreeMap::<
            String,
            (
                Arc<dyn Factor>,
                BTreeSet<String>,
                BTreeSet<i32>,
                usize,
                Vec<IntradayDailyRawSpec>,
            ),
        >::new();
        for (requirement, requirement_missing_dates) in &plans {
            let provider = providers.get(&requirement.spec.raw_id).ok_or_else(|| {
                err(format!(
                    "intraday daily raw implementation not found: {}",
                    requirement.spec.raw_id
                ))
            })?;
            let entry = grouped_jobs
                .entry(provider.provider_key.clone())
                .or_insert_with(|| {
                    (
                        Arc::clone(&provider.factor),
                        BTreeSet::new(),
                        BTreeSet::new(),
                        1,
                        Vec::new(),
                    )
                });
            entry.1.insert(requirement.spec.raw_id.clone());
            entry.2.extend(requirement_missing_dates.iter().copied());
            entry.3 = entry.3.max(requirement.spec.window_days);
            entry.4.push(requirement.spec.clone());
        }

        for (_provider_key, (factor, raw_ids, target_dates, window_days, specs)) in grouped_jobs {
            if target_dates.is_empty() {
                continue;
            }
            let raw_ids_vec = raw_ids.iter().cloned().collect::<Vec<_>>();
            let first_target = *target_dates
                .iter()
                .next()
                .expect("target dates are not empty");
            let last_target = *target_dates
                .iter()
                .next_back()
                .expect("target dates are not empty");
            let warmup_days = window_days.saturating_sub(1);
            let stream_start = calendar.warmup_start(first_target, warmup_days);
            let stream_dates = calendar.open_dates_between(stream_start, last_target);
            let target_set = target_dates.iter().copied().collect::<BTreeSet<_>>();
            let auxiliary_requirements = factor.intraday_raw_auxiliary_requirements(&raw_ids_vec);
            let auxiliary_max_lookback = auxiliary_requirements
                .iter()
                .map(|request| request.daily_lookback)
                .max()
                .unwrap_or(0);
            let mut state = factor.initial_intraday_raw_state(&raw_ids_vec);
            let stage_name = format!("intraday_raw_materialize_window_{window_days}");
            let mut total_load_ms = 0u128;
            let mut total_compute_ms = 0u128;
            let mut total_write_ms = 0u128;
            let mut raw_profiles = BTreeMap::<String, (usize, usize)>::new();

            for trade_date in stream_dates {
                let raw_context = FactorContext {
                    asset_class: request.asset_class,
                    frequency: Frequency::Daily,
                    start_date: trade_date,
                    end_date: trade_date,
                    load_start_date: trade_date,
                    load_dates: vec![trade_date],
                    target_dates: vec![trade_date],
                };

                let load_started = Instant::now();
                let mut raw_pool = DataPool::load(loader, &batch_requests, &raw_context)?;
                let auxiliary_target_dates = auxiliary_target_dates(
                    source_dataset,
                    source_bar_size,
                    &raw_pool,
                    &[trade_date],
                );
                if !auxiliary_requirements.is_empty() && !auxiliary_target_dates.is_empty() {
                    let auxiliary_load_start_date =
                        calendar.warmup_start(trade_date, auxiliary_max_lookback);
                    let auxiliary_context = FactorContext {
                        asset_class: request.asset_class,
                        frequency: Frequency::Daily,
                        start_date: trade_date,
                        end_date: trade_date,
                        load_start_date: auxiliary_load_start_date,
                        load_dates: calendar
                            .open_dates_between(auxiliary_load_start_date, trade_date),
                        target_dates: auxiliary_target_dates,
                    };
                    let auxiliary_pool = DataPool::load(
                        loader,
                        &merge_requests(
                            auxiliary_requirements
                                .iter()
                                .map(|request| request.request.clone()),
                        ),
                        &auxiliary_context,
                    )?;
                    raw_pool.extend(auxiliary_pool);
                }
                total_load_ms += load_started.elapsed().as_millis();

                let compute_started = Instant::now();
                let mut raw_series_list = factor.minute_compute_stateful_many(
                    &raw_ids_vec,
                    &raw_context,
                    &raw_pool,
                    state.as_mut(),
                )?;
                raw_series_list.retain(|series| raw_ids.contains(&series.spec.raw_id));
                for series in &mut raw_series_list {
                    series
                        .values
                        .retain(|value| target_set.contains(&value.key.trade_date()));
                }
                total_compute_ms += compute_started.elapsed().as_millis();

                if !target_set.contains(&trade_date) {
                    continue;
                }

                let returned = raw_series_list
                    .iter()
                    .map(|series| series.spec.raw_id.clone())
                    .collect::<BTreeSet<_>>();
                let missing = raw_ids.difference(&returned).cloned().collect::<Vec<_>>();
                if !missing.is_empty() {
                    return Err(err(format!(
                        "stateful intraday daily raw provider did not return requested raw(s): {}",
                        missing.join(",")
                    )));
                }

                for raw_series in &raw_series_list {
                    let entry = raw_profiles
                        .entry(raw_series.spec.raw_id.clone())
                        .or_insert((0, 0));
                    entry.0 += raw_series.values.len();
                    entry.1 += raw_series
                        .values
                        .iter()
                        .filter(|item| item.value.is_some())
                        .count();
                }

                let write_started = Instant::now();
                if !raw_series_list.is_empty() {
                    storage.write_results(&raw_series_list)?;
                }
                total_write_ms += write_started.elapsed().as_millis();

                for raw_series in &raw_series_list {
                    materialized_intraday_raw_dates
                        .insert((raw_series.spec.raw_id.clone(), trade_date));
                }
                progress.tick(format!(
                    "stage={} date={} raw={}",
                    stage_name,
                    trade_date,
                    raw_series_list.len()
                ));
            }

            if request.profile && !raw_profiles.is_empty() {
                profiles.push(BatchProfile {
                    stage: stage_name,
                    date_batch_index,
                    factor_batch_index,
                    start_date: first_target,
                    end_date: last_target,
                    factor_count: raw_profiles.len(),
                    load_ms: total_load_ms,
                    compute_ms: total_compute_ms,
                    write_ms: total_write_ms,
                    factors: raw_profiles
                        .into_iter()
                        .map(|(factor_id, (row_count, non_null_count))| FactorProfile {
                            factor_id,
                            row_count,
                            non_null_count,
                        })
                        .collect(),
                });
            }
            materialized_specs.extend(specs);
        }
    }
    if !materialized_specs.is_empty() {
        storage.write_metadata(&materialized_specs)?;
    }

    Ok((
        storage.load_raw_by_dates(request.asset_class, raw_ids, &context.load_dates)?,
        profiles,
    ))
}

fn expand_raw_ids_to_selected_provider_siblings(
    raw_ids: &[String],
    requirements: &[IntradayRawRequirement],
    providers: &BTreeMap<String, RawProvider>,
) -> Result<BTreeSet<String>> {
    let mut raw_id_set = raw_ids.iter().cloned().collect::<BTreeSet<_>>();
    let mut provider_keys = BTreeSet::new();
    for raw_id in raw_ids {
        let provider = providers.get(raw_id).ok_or_else(|| {
            err(format!(
                "intraday daily raw implementation not found: {raw_id}"
            ))
        })?;
        provider_keys.insert(provider.provider_key.clone());
    }
    for requirement in requirements {
        let provider = providers.get(&requirement.spec.raw_id).ok_or_else(|| {
            err(format!(
                "intraday daily raw implementation not found: {}",
                requirement.spec.raw_id
            ))
        })?;
        if provider_keys.contains(&provider.provider_key) {
            raw_id_set.insert(requirement.spec.raw_id.clone());
        }
    }
    Ok(raw_id_set)
}

fn raw_ids_for_specs(specs: &[FactorSpec]) -> Vec<String> {
    let mut raw_ids = BTreeSet::new();
    for spec in specs {
        for dependency in &spec.intraday_raw_dependencies {
            raw_ids.insert(dependency.raw_id.clone());
        }
    }
    raw_ids.into_iter().collect()
}

fn raw_materialize_stage_name(spec: &IntradayDailyRawSpec) -> String {
    format!("intraday_raw_materialize_window_{}", spec.window_days)
}

fn auxiliary_requests_for_raw_requirements(
    requirements: &[IntradayRawRequirement],
    providers: &BTreeMap<String, RawProvider>,
) -> Result<Vec<IntradayDailyRawAuxiliaryRequest>> {
    let mut raw_ids_by_provider = BTreeMap::<String, BTreeSet<String>>::new();
    for requirement in requirements {
        let provider = providers.get(&requirement.spec.raw_id).ok_or_else(|| {
            err(format!(
                "intraday daily raw implementation not found: {}",
                requirement.spec.raw_id
            ))
        })?;
        raw_ids_by_provider
            .entry(provider.provider_key.clone())
            .or_default()
            .insert(requirement.spec.raw_id.clone());
    }
    auxiliary_requests_from_provider_groups(raw_ids_by_provider, providers)
}

fn auxiliary_requests_for_raw_batch(
    plans: &[(&IntradayRawRequirement, BTreeSet<i32>)],
    providers: &BTreeMap<String, RawProvider>,
    date_batch: &[i32],
) -> Result<Vec<IntradayDailyRawAuxiliaryRequest>> {
    let date_set = date_batch.iter().copied().collect::<BTreeSet<_>>();
    let mut raw_ids_by_provider = BTreeMap::<String, BTreeSet<String>>::new();
    for (requirement, missing_dates) in plans {
        if missing_dates.is_disjoint(&date_set) {
            continue;
        }
        let provider = providers.get(&requirement.spec.raw_id).ok_or_else(|| {
            err(format!(
                "intraday daily raw implementation not found: {}",
                requirement.spec.raw_id
            ))
        })?;
        raw_ids_by_provider
            .entry(provider.provider_key.clone())
            .or_default()
            .insert(requirement.spec.raw_id.clone());
    }

    auxiliary_requests_from_provider_groups(raw_ids_by_provider, providers)
}

fn auxiliary_requests_from_provider_groups(
    raw_ids_by_provider: BTreeMap<String, BTreeSet<String>>,
    providers: &BTreeMap<String, RawProvider>,
) -> Result<Vec<IntradayDailyRawAuxiliaryRequest>> {
    let mut output = Vec::new();
    for (provider_key, raw_ids) in raw_ids_by_provider {
        let factor = providers
            .values()
            .find(|provider| provider.provider_key == provider_key)
            .map(|provider| Arc::clone(&provider.factor))
            .ok_or_else(|| {
                err(format!(
                    "intraday daily raw provider not found: {provider_key}"
                ))
            })?;
        let raw_ids = raw_ids.into_iter().collect::<Vec<_>>();
        output.extend(factor.intraday_raw_auxiliary_requirements(&raw_ids));
    }
    Ok(output)
}

fn auxiliary_target_dates(
    source_dataset: DatasetId,
    source_bar_size: Option<usize>,
    pool: &DataPool,
    target_dates: &[i32],
) -> Vec<i32> {
    if source_dataset != DatasetId::StockDerivedBar {
        return target_dates.to_vec();
    }
    let Some(bar_size) = source_bar_size else {
        return target_dates.to_vec();
    };
    target_dates
        .iter()
        .copied()
        .filter(|trade_date| pool.derived_bar(bar_size, *trade_date).is_none())
        .collect()
}

fn ordered_source_groups<T>(
    groups: BTreeMap<(DatasetId, Option<usize>), T>,
) -> Vec<((DatasetId, Option<usize>), T)> {
    let mut groups = groups.into_iter().collect::<Vec<_>>();
    groups.sort_by_key(|((dataset, bar_size), _)| source_group_order_key(*dataset, *bar_size));
    groups
}

fn source_group_order_key(
    dataset: DatasetId,
    bar_size: Option<usize>,
) -> (u8, Reverse<usize>, DatasetId, Option<usize>) {
    match dataset {
        DatasetId::StockDerivedBar => (0, Reverse(bar_size.unwrap_or(0)), dataset, bar_size),
        DatasetId::StockMinute1m | DatasetId::FutureMinute1m => (1, Reverse(0), dataset, bar_size),
        _ => (2, Reverse(0), dataset, bar_size),
    }
}

fn execution_stage_names(
    raw_work: &[IntradayRawWork],
    execution_groups: &[ExecutionGroup],
) -> Vec<String> {
    let mut seen = BTreeSet::new();
    let mut stages = Vec::new();
    for work in raw_work {
        let stage = raw_materialize_stage_name(&work.spec);
        if seen.insert(stage.clone()) {
            stages.push(stage);
        }
    }
    for group in execution_groups {
        let stage = group.stage.name();
        if seen.insert(stage.clone()) {
            stages.push(stage);
        }
    }
    stages
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
            .filter(|row| !is_deprecated_factor(row))
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
        let mut deprecated = Vec::new();
        for factor_id_or_name in factor_ids {
            if let Some(row) = base.iter().find(|row| {
                row.factor_id == *factor_id_or_name
                    || row.name == *factor_id_or_name
                    || row.aliases.iter().any(|alias| alias == factor_id_or_name)
            }) {
                if is_deprecated_factor(row) {
                    deprecated.push(row.factor_id.clone());
                } else {
                    selected.push(row.clone());
                }
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
        if !deprecated.is_empty() {
            deprecated.sort();
            deprecated.dedup();
            return SelectionResult::Empty(format!(
                "Deprecated factors are excluded from run: {}",
                deprecated.join(",")
            ));
        }
        return SelectionResult::Selected(selected);
    }

    let active = base
        .into_iter()
        .filter(|row| !is_deprecated_factor(row))
        .collect::<Vec<_>>();
    if active.is_empty() {
        return SelectionResult::Empty(format!(
            "No factors found in metadata for asset={} frequency={}.",
            request.asset_class, request.frequency
        ));
    }
    SelectionResult::Selected(active)
}

fn is_deprecated_factor(row: &FactorMetadata) -> bool {
    row.tags.iter().any(|tag| tag == "deprecated")
}

fn empty_report(request: &RunRequest, message: String) -> RunReport {
    RunReport {
        factor_count: 0,
        output_file_count: 0,
        load_start_date: request.start_date,
        target_dates: Vec::new(),
        effective_start_date: None,
        effective_end_date: None,
        execution_stages: Vec::new(),
        date_batch_count: 0,
        factor_batch_count: 0,
        execution_batch_count: 0,
        selected_factor_ids: Vec::new(),
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
    use std::collections::{BTreeSet, HashMap};

    let mut grouped: HashMap<_, (BTreeSet<String>, Option<usize>)> = HashMap::new();
    for request in requests {
        let key = (
            request.dataset,
            request.entity_id.clone(),
            request.bar_size,
            request.date_policy.clone(),
        );
        let entry = grouped.entry(key).or_default();
        entry.0.extend(request.columns.into_iter());
        entry.1 = match (entry.1, request.financial_quarters) {
            (Some(left), Some(right)) => Some(left.max(right)),
            (None, Some(right)) => Some(right),
            (left, None) => left,
        };
    }
    let mut merged = grouped
        .into_iter()
        .map(
            |((dataset, entity_id, bar_size, date_policy), (columns, financial_quarters))| {
                DataRequest {
                    dataset,
                    entity_id,
                    bar_size,
                    columns: columns.into_iter().collect(),
                    financial_quarters,
                    date_policy,
                }
            },
        )
        .collect::<Vec<_>>();
    merged.sort_by(|left, right| {
        left.dataset
            .cmp(&right.dataset)
            .then_with(|| left.entity_id.cmp(&right.entity_id))
            .then_with(|| left.bar_size.cmp(&right.bar_size))
            .then_with(|| left.date_policy.cmp(&right.date_policy))
    });
    merged
}

fn spec_calendar_lookback_days(spec: &crate::core::FactorSpec) -> usize {
    spec.dependencies
        .iter()
        .map(DataRequest::calendar_lookback_days)
        .max()
        .unwrap_or(0)
        .max(spec.lookback.trading_days)
}

#[derive(Clone, Debug, Default)]
struct ContextualRequirements {
    all: Vec<DataRequest>,
    by_provider: BTreeMap<String, Vec<DataRequest>>,
}

fn contextual_requirements_for_factor_batch<'a>(
    factors: &[&'a dyn Factor],
    specs: &[FactorSpec],
    context: &FactorContext,
    calendar: &TradingCalendar,
) -> ContextualRequirements {
    let mut output = ContextualRequirements::default();
    for (factor, spec) in factors.iter().zip(specs.iter()) {
        let lookback = spec_calendar_lookback_days(spec);
        let load_start_date = calendar.warmup_start(context.start_date, lookback);
        let factor_context = FactorContext {
            asset_class: context.asset_class,
            frequency: context.frequency,
            start_date: context.start_date,
            end_date: context.end_date,
            load_start_date,
            load_dates: calendar.open_dates_between(load_start_date, context.end_date),
            target_dates: context.target_dates.clone(),
        };
        let requirements = factor.requirements_for_context(&factor_context);
        output
            .by_provider
            .entry(factor.compute_provider_key())
            .or_default()
            .extend(requirements.iter().cloned());
        output.all.extend(requirements);
    }
    output
}

fn financial_years_for_requests(
    requests: &[DataRequest],
    start_date: i32,
    end_date: i32,
) -> BTreeSet<i32> {
    let mut years = BTreeSet::new();
    for request in requests {
        if !matches!(
            request.dataset,
            DatasetId::StockIncome | DatasetId::StockBalanceSheet | DatasetId::StockCashFlow
        ) {
            continue;
        }
        years.extend(financial_disclosure_years_for_range(
            start_date,
            end_date,
            request.financial_quarters.unwrap_or(0),
        ));
    }
    years
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

fn split_dates_by_chunk(dates: &[i32], chunk_size: usize) -> Vec<Vec<i32>> {
    let chunk_size = chunk_size.max(1);
    dates
        .chunks(chunk_size)
        .map(|chunk| chunk.to_vec())
        .collect()
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ExecutionGroup {
    stage: ExecutionStage,
    factor_indices: Vec<usize>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum ExecutionStage {
    DailyNoMinute,
    IntradayDaily { lookback: usize },
    IntradayDailyPostprocess { lookback: usize },
}

impl ExecutionStage {
    fn name(&self) -> String {
        match self {
            Self::DailyNoMinute => "daily_no_minute".to_string(),
            Self::IntradayDaily { lookback } => format!("intraday_daily_lookback_{lookback}"),
            Self::IntradayDailyPostprocess { lookback } => {
                format!("intraday_daily_postprocess_lookback_{lookback}")
            }
        }
    }
}

fn execution_groups_for_specs(frequency: Frequency, specs: &[FactorSpec]) -> Vec<ExecutionGroup> {
    let mut daily_no_minute = Vec::new();
    let mut intraday_daily: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    let mut intraday_daily_postprocess: BTreeMap<usize, Vec<usize>> = BTreeMap::new();

    for (idx, spec) in specs.iter().enumerate() {
        if frequency == Frequency::Daily && spec_has_intraday_raw_dependency(spec) {
            intraday_daily_postprocess
                .entry(spec.lookback.trading_days)
                .or_default()
                .push(idx);
        } else if frequency == Frequency::Daily && spec_has_minute_dependency(spec) {
            intraday_daily
                .entry(spec.lookback.trading_days)
                .or_default()
                .push(idx);
        } else {
            daily_no_minute.push(idx);
        }
    }

    let mut groups = Vec::new();
    if !daily_no_minute.is_empty() {
        groups.push(ExecutionGroup {
            stage: ExecutionStage::DailyNoMinute,
            factor_indices: daily_no_minute,
        });
    }
    groups.extend(
        intraday_daily_postprocess
            .into_iter()
            .map(|(lookback, factor_indices)| ExecutionGroup {
                stage: ExecutionStage::IntradayDailyPostprocess { lookback },
                factor_indices,
            }),
    );
    groups.extend(
        intraday_daily
            .into_iter()
            .map(|(lookback, factor_indices)| ExecutionGroup {
                stage: ExecutionStage::IntradayDaily { lookback },
                factor_indices,
            }),
    );
    groups
}

fn date_batches_for_stage(
    stage: &ExecutionStage,
    target_dates: &[i32],
    date_batch_size: usize,
) -> Vec<Vec<i32>> {
    match stage {
        ExecutionStage::DailyNoMinute | ExecutionStage::IntradayDailyPostprocess { .. } => {
            split_dates_by_chunk(target_dates, date_batch_size.max(1))
        }
        ExecutionStage::IntradayDaily { .. } => {
            split_dates_by_chunk(target_dates, DEFAULT_DATE_BATCH_SIZE)
        }
    }
}

fn spec_has_minute_dependency(spec: &FactorSpec) -> bool {
    spec.dependencies.iter().any(|request| {
        matches!(
            request.dataset,
            DatasetId::StockMinute1m | DatasetId::FutureMinute1m
        )
    })
}

fn spec_has_intraday_raw_dependency(spec: &FactorSpec) -> bool {
    !spec.intraday_raw_dependencies.is_empty()
}

fn provider_factor_batches(
    factors: &[Box<dyn Factor>],
    factor_indices: &[usize],
    factor_batch_size: usize,
) -> Vec<Vec<usize>> {
    if factor_indices.is_empty() {
        return Vec::new();
    }
    let mut provider_positions = HashMap::<String, usize>::new();
    let mut providers = Vec::<(String, Vec<usize>)>::new();
    for factor_idx in factor_indices {
        let key = factors[*factor_idx].compute_provider_key();
        if let Some(position) = provider_positions.get(&key).copied() {
            providers[position].1.push(*factor_idx);
        } else {
            provider_positions.insert(key.clone(), providers.len());
            providers.push((key, vec![*factor_idx]));
        }
    }

    let batch_size = factor_batch_size.max(1);
    let mut batches = Vec::new();
    let mut current = Vec::new();
    let mut current_count = 0usize;
    for (_, indices) in providers {
        if !current.is_empty() && current_count + indices.len() > batch_size {
            batches.push(std::mem::take(&mut current));
            current_count = 0;
        }
        current_count += indices.len();
        current.extend(indices);
    }
    if !current.is_empty() {
        batches.push(current);
    }
    batches
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

fn request_has_daily_panel_dates(request: &DataRequest) -> bool {
    matches!(
        request.dataset,
        DatasetId::StockDailyPv
            | DatasetId::StockDailyBasic
            | DatasetId::StockDailyLimit
            | DatasetId::StockAdjFactor
            | DatasetId::StockMoneyflow
            | DatasetId::StockBarraDaily
            | DatasetId::IndexDaily
            | DatasetId::FutureDaily
    )
}

fn context_for_provider_requests(
    context: &FactorContext,
    requirements: &[DataRequest],
) -> FactorContext {
    let mut dates = BTreeSet::new();
    for request in requirements {
        if request_has_daily_panel_dates(request) {
            dates.extend(request.resolved_dates(context));
        }
    }
    let load_dates = if dates.is_empty() {
        context.load_dates.clone()
    } else {
        dates.into_iter().collect::<Vec<_>>()
    };
    FactorContext {
        asset_class: context.asset_class,
        frequency: context.frequency,
        start_date: context.start_date,
        end_date: context.end_date,
        load_start_date: load_dates
            .first()
            .copied()
            .unwrap_or(context.load_start_date),
        load_dates,
        target_dates: context.target_dates.clone(),
    }
}

fn compute_factor_batch(
    factors: &[&dyn Factor],
    context: &FactorContext,
    pool: &DataPool,
    provider_requirements: &BTreeMap<String, Vec<DataRequest>>,
    states: &mut BTreeMap<String, Box<dyn Any + Send>>,
    thread_pool: Option<&rayon::ThreadPool>,
) -> Result<Vec<FactorSeries>> {
    let mut requested_order = BTreeMap::new();
    let mut groups = BTreeMap::<String, ComputeProviderGroup>::new();
    for (idx, factor) in factors.iter().enumerate() {
        let spec = factor.spec();
        requested_order.insert(spec.id.clone(), idx);
        let key = factor.compute_provider_key();
        let group = groups
            .entry(key.clone())
            .or_insert_with(|| ComputeProviderGroup {
                factor: *factor,
                requested_ids: Vec::new(),
                requirements: provider_requirements.get(&key).cloned().unwrap_or_default(),
            });
        if !group.requested_ids.iter().any(|id| id == &spec.id) {
            group.requested_ids.push(spec.id);
        }
    }

    let jobs = groups
        .into_iter()
        .map(|(key, group)| {
            let state = states
                .remove(&key)
                .unwrap_or_else(|| group.factor.initial_compute_state(&group.requested_ids));
            ComputeProviderJob {
                key,
                factor: group.factor,
                requested_ids: group.requested_ids,
                requirements: group.requirements,
                state,
            }
        })
        .collect::<Vec<_>>();
    let compute = || {
        jobs.into_par_iter()
            .map(|mut job| {
                let provider_context = context_for_provider_requests(context, &job.requirements);
                let provider_pool = pool.view_for_requests(&job.requirements, &provider_context);
                let result = job.factor.compute_many_stateful(
                    &job.requested_ids,
                    &provider_context,
                    &provider_pool,
                    job.state.as_mut(),
                );
                (job.key, job.state, result)
            })
            .collect::<Vec<_>>()
    };
    let results = match thread_pool {
        Some(thread_pool) => thread_pool.install(compute),
        None => compute(),
    };
    let mut output = Vec::new();
    for (key, state, result) in results {
        states.insert(key, state);
        output.extend(result?);
    }
    let mut returned = BTreeSet::new();
    for series in &output {
        if !requested_order.contains_key(&series.spec.id) {
            return Err(err(format!(
                "compute provider returned unrequested factor {}",
                series.spec.id
            )));
        }
        if !returned.insert(series.spec.id.clone()) {
            return Err(err(format!(
                "compute provider returned duplicate factor {}",
                series.spec.id
            )));
        }
    }
    let missing = requested_order
        .keys()
        .filter(|id| !returned.contains(*id))
        .cloned()
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        return Err(err(format!(
            "compute provider did not return requested factor(s): {}",
            missing.join(",")
        )));
    }
    output.sort_by_key(|series| {
        requested_order
            .get(&series.spec.id)
            .copied()
            .unwrap_or(usize::MAX)
    });
    Ok(output)
}

pub fn available_specs() -> Vec<FactorSpec> {
    all_factors()
        .into_iter()
        .flat_map(|factor| factor.provided_specs())
        .collect()
}

struct ComputeProviderGroup<'a> {
    factor: &'a dyn Factor,
    requested_ids: Vec<String>,
    requirements: Vec<DataRequest>,
}

struct ComputeProviderJob<'a> {
    key: String,
    factor: &'a dyn Factor,
    requested_ids: Vec<String>,
    requirements: Vec<DataRequest>,
    state: Box<dyn Any + Send>,
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    use crate::core::{
        AssetClass, DataRequest, DatasetId, DateLoadPolicy, FactorContext, FactorSeries,
        FactorSpec, Frequency, IntradayDailyRawSpec, Lookback,
    };
    use crate::data::table::Table;
    use crate::data::DataPool;
    use crate::error::Result;
    use crate::factor::Factor;
    use crate::storage::FactorMetadata;

    use super::{
        date_batches_for_stage, execution_groups_for_specs, provider_factor_batches,
        select_metadata, split_dates_by_chunk, validate_date_value, ExecutionStage,
        IntradayRawRequirement, RawProvider, RunRequest, SelectionResult, DEFAULT_DATE_BATCH_SIZE,
    };

    #[test]
    fn validates_eight_digit_dates() {
        assert!(validate_date_value(20260424, "end-date").is_ok());
        assert!(validate_date_value(2026424, "end-date").is_err());
    }

    #[test]
    fn factor_batches_keep_provider_group_together() {
        let counter = Arc::new(AtomicUsize::new(0));
        let factors: Vec<Box<dyn Factor>> = vec![
            Box::new(MultiOutputFactor::new("left", "provider_a", &counter)),
            Box::new(MultiOutputFactor::new("right", "provider_a", &counter)),
            Box::new(MultiOutputFactor::new("solo", "provider_b", &counter)),
        ];

        assert_eq!(
            provider_factor_batches(&factors, &[0, 1, 2], 1),
            vec![vec![0, 1], vec![2]]
        );
        assert_eq!(
            provider_factor_batches(&factors, &[], 1),
            Vec::<Vec<usize>>::new()
        );
    }

    #[test]
    fn chunks_dates_by_configured_batch_size() {
        assert_eq!(
            split_dates_by_chunk(
                &[20260101, 20260102, 20260103, 20260104, 20260105, 20260106],
                1
            ),
            vec![
                vec![20260101],
                vec![20260102],
                vec![20260103],
                vec![20260104],
                vec![20260105],
                vec![20260106]
            ]
        );
        assert_eq!(
            split_dates_by_chunk(&[20260101, 20260102], 0),
            vec![vec![20260101], vec![20260102]]
        );
    }

    #[test]
    fn merge_requests_keeps_derived_bar_sizes_separate() {
        let merged = super::merge_requests(vec![
            DataRequest::stock_derived_bar(5, &["close"]),
            DataRequest::stock_derived_bar(15, &["close"]),
            DataRequest::stock_derived_bar(5, &["volume"]),
        ]);

        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0].dataset, DatasetId::StockDerivedBar);
        assert_eq!(merged[0].bar_size, Some(5));
        assert_eq!(merged[0].columns, vec!["close", "volume"]);
        assert_eq!(merged[1].bar_size, Some(15));
        assert_eq!(merged[1].columns, vec!["close"]);
    }

    #[test]
    fn merge_requests_keeps_explicit_date_policies_separate() {
        let merged = super::merge_requests(vec![
            DataRequest::new(DatasetId::StockDailyPv, &["close"]),
            DataRequest::explicit_dates(DatasetId::StockDailyPv, &["close"], vec![20260102]),
            DataRequest::explicit_dates(DatasetId::StockDailyPv, &["pre_close"], vec![20260102]),
        ]);

        assert_eq!(merged.len(), 2);
        assert!(merged
            .iter()
            .any(|request| request.date_policy == DateLoadPolicy::ContextLoadDates));
        let sparse = merged
            .iter()
            .find(|request| request.date_policy == DateLoadPolicy::ExplicitDates(vec![20260102]))
            .expect("sparse request");
        assert_eq!(sparse.columns, vec!["close", "pre_close"]);
    }

    #[test]
    fn derived_bar_auxiliary_dates_are_lazy_missing_dates_only() {
        let mut pool = DataPool::default();
        pool.insert_minute_table_for_test(
            DatasetId::StockDerivedBar,
            Some(5),
            20260424,
            Table::empty(),
        );

        assert_eq!(
            super::auxiliary_target_dates(
                DatasetId::StockDerivedBar,
                Some(5),
                &pool,
                &[20260424, 20260425]
            ),
            vec![20260425]
        );
        assert_eq!(
            super::auxiliary_target_dates(
                DatasetId::StockMinute1m,
                None,
                &pool,
                &[20260424, 20260425]
            ),
            vec![20260424, 20260425]
        );
    }

    #[test]
    fn source_groups_order_derived_bars_before_raw_1m_by_descending_bar_size() {
        let groups = BTreeMap::from([
            ((DatasetId::StockMinute1m, None), "1m"),
            ((DatasetId::StockDerivedBar, Some(5)), "5m"),
            ((DatasetId::StockDerivedBar, Some(15)), "15m"),
        ]);
        let ordered = super::ordered_source_groups(groups)
            .into_iter()
            .map(|(_, value)| value)
            .collect::<Vec<_>>();

        assert_eq!(ordered, vec!["15m", "5m", "1m"]);
    }

    #[test]
    fn daily_and_postprocess_stages_use_configured_date_batch_size() {
        let target_dates = vec![20260101, 20260102, 20260105, 20260106, 20260107];
        let expected = vec![
            vec![20260101, 20260102],
            vec![20260105, 20260106],
            vec![20260107],
        ];
        assert_eq!(
            date_batches_for_stage(&ExecutionStage::DailyNoMinute, &target_dates, 2),
            expected
        );
        assert_eq!(
            date_batches_for_stage(
                &ExecutionStage::IntradayDailyPostprocess { lookback: 19 },
                &target_dates,
                2,
            ),
            expected
        );
    }

    #[test]
    fn direct_intraday_stage_ignores_configured_date_batch_size() {
        let target_dates = vec![20260101, 20260102, 20260105];
        assert_eq!(
            date_batches_for_stage(
                &ExecutionStage::IntradayDaily { lookback: 19 },
                &target_dates,
                20,
            ),
            vec![vec![20260101], vec![20260102], vec![20260105]]
        );
    }

    #[test]
    fn execution_groups_split_daily_and_intraday_daily_by_lookback() {
        let specs = vec![
            spec_with_dataset("daily_factor", DatasetId::StockDailyPv, 0),
            spec_with_dataset("intraday_factor_lookback_0", DatasetId::StockMinute1m, 0),
            spec_with_dataset("intraday_factor_lookback_19", DatasetId::StockMinute1m, 19),
            spec_with_dataset("another_intraday_lookback_19", DatasetId::StockMinute1m, 19),
        ];

        let groups = execution_groups_for_specs(Frequency::Daily, &specs);

        assert_eq!(groups.len(), 3);
        assert_eq!(groups[0].stage, ExecutionStage::DailyNoMinute);
        assert_eq!(groups[0].factor_indices, vec![0]);
        assert_eq!(
            groups[1].stage,
            ExecutionStage::IntradayDaily { lookback: 0 }
        );
        assert_eq!(groups[1].factor_indices, vec![1]);
        assert_eq!(
            groups[2].stage,
            ExecutionStage::IntradayDaily { lookback: 19 }
        );
        assert_eq!(groups[2].factor_indices, vec![2, 3]);
    }

    #[test]
    fn execution_groups_route_intraday_raw_factors_to_postprocess_stage() {
        let specs = vec![
            spec_with_dataset("daily_factor", DatasetId::StockDailyPv, 0),
            spec_with_raw("intraday_factor_lookback_0", "intraday_raw_lookback_0", 0),
            spec_with_raw(
                "intraday_factor_lookback_19",
                "intraday_raw_lookback_19",
                19,
            ),
        ];

        let groups = execution_groups_for_specs(Frequency::Daily, &specs);

        assert_eq!(groups.len(), 3);
        assert_eq!(groups[0].stage, ExecutionStage::DailyNoMinute);
        assert_eq!(
            groups[1].stage,
            ExecutionStage::IntradayDailyPostprocess { lookback: 0 }
        );
        assert_eq!(groups[1].factor_indices, vec![1]);
        assert_eq!(
            groups[2].stage,
            ExecutionStage::IntradayDailyPostprocess { lookback: 19 }
        );
        assert_eq!(groups[2].factor_indices, vec![2]);
    }

    #[test]
    fn execution_groups_run_postprocess_before_direct_intraday() {
        let specs = vec![
            spec_with_dataset("daily_factor", DatasetId::StockDailyPv, 0),
            spec_with_dataset("direct_intraday_factor", DatasetId::StockMinute1m, 0),
            spec_with_raw("postprocess_intraday_factor", "intraday_raw", 0),
        ];

        let groups = execution_groups_for_specs(Frequency::Daily, &specs);

        assert_eq!(groups.len(), 3);
        assert_eq!(groups[0].stage, ExecutionStage::DailyNoMinute);
        assert_eq!(
            groups[1].stage,
            ExecutionStage::IntradayDailyPostprocess { lookback: 0 }
        );
        assert_eq!(groups[1].factor_indices, vec![2]);
        assert_eq!(
            groups[2].stage,
            ExecutionStage::IntradayDaily { lookback: 0 }
        );
        assert_eq!(groups[2].factor_indices, vec![1]);
    }

    #[test]
    fn raw_materialization_expands_to_selected_sibling_raws_by_provider_key() {
        let factor: Arc<dyn Factor> = Arc::new(DummyFactor);
        let providers = BTreeMap::from([
            (
                "raw_a".to_string(),
                raw_provider("raw_a", "provider_one", Arc::clone(&factor)),
            ),
            (
                "raw_b".to_string(),
                raw_provider("raw_b", "provider_one", Arc::clone(&factor)),
            ),
            (
                "raw_unselected".to_string(),
                raw_provider("raw_unselected", "provider_one", Arc::clone(&factor)),
            ),
            (
                "raw_c".to_string(),
                raw_provider("raw_c", "provider_two", Arc::clone(&factor)),
            ),
        ]);
        let requirements = vec![
            raw_requirement("raw_a"),
            raw_requirement("raw_b"),
            raw_requirement("raw_c"),
        ];
        let raw_ids = vec!["raw_a".to_string()];

        let expanded = super::expand_raw_ids_to_selected_provider_siblings(
            &raw_ids,
            &requirements,
            &providers,
        )
        .expect("expand raw ids");

        assert!(expanded.contains("raw_a"));
        assert!(expanded.contains("raw_b"));
        assert!(!expanded.contains("raw_c"));
        assert!(!expanded.contains("raw_unselected"));
    }

    #[test]
    fn compute_factor_batch_calls_shared_provider_once() {
        let shared_counter = Arc::new(AtomicUsize::new(0));
        let solo_counter = Arc::new(AtomicUsize::new(0));
        let left = MultiOutputFactor::new("shared_left", "shared_provider", &shared_counter);
        let right = MultiOutputFactor::new("shared_right", "shared_provider", &shared_counter);
        let solo = MultiOutputFactor::new("solo", "solo_provider", &solo_counter);
        let factors: Vec<&dyn Factor> = vec![&left, &right, &solo];
        let context = FactorContext {
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            start_date: 20260105,
            end_date: 20260105,
            load_start_date: 20260105,
            load_dates: vec![20260105],
            target_dates: vec![20260105],
        };

        let mut states = BTreeMap::new();
        let results = super::compute_factor_batch(
            &factors,
            &context,
            &DataPool::default(),
            &BTreeMap::new(),
            &mut states,
            None,
        )
        .unwrap();
        let ids = results
            .iter()
            .map(|series| series.spec.id.as_str())
            .collect::<Vec<_>>();

        assert_eq!(ids, vec!["shared_left", "shared_right", "solo"]);
        assert_eq!(shared_counter.load(Ordering::SeqCst), 1);
        assert_eq!(solo_counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn selection_matches_short_id_and_legacy_alias_inside_asset_frequency() {
        let request = RunRequest {
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            start_date: 20260105,
            end_date: 20260105,
            factor_ids: Some(vec!["stock.daily.pv.daily_factor".to_string()]),
            tags: None,
            config_path: None,
            dry_run: false,
            factor_batch_size: 64,
            date_batch_size: DEFAULT_DATE_BATCH_SIZE,
            threads: None,
            profile: false,
            refresh_minute_cache: false,
        };
        let metadata = vec![
            metadata_row(
                "daily_factor",
                "stock",
                "daily",
                &["stock.daily.pv.daily_factor"],
            ),
            metadata_row(
                "daily_factor",
                "future",
                "daily",
                &["future.daily.pv.daily_factor"],
            ),
        ];

        let SelectionResult::Selected(selected) = select_metadata(&request, &metadata) else {
            panic!("expected selected");
        };
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].asset_class, "stock");
        assert_eq!(selected[0].factor_id, "daily_factor");
    }

    #[test]
    fn selection_excludes_deprecated_for_tag_requests() {
        let request = RunRequest {
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            start_date: 20260105,
            end_date: 20260105,
            factor_ids: None,
            tags: Some(vec!["worldquant101alpha".to_string()]),
            config_path: None,
            dry_run: false,
            factor_batch_size: 64,
            date_batch_size: DEFAULT_DATE_BATCH_SIZE,
            threads: None,
            profile: false,
            refresh_minute_cache: false,
        };
        let mut active = metadata_row("WQAlpha002", "stock", "daily", &[]);
        active.tags = vec!["worldquant101alpha".to_string()];
        let mut deprecated = metadata_row("WQAlpha001", "stock", "daily", &[]);
        deprecated.tags = vec!["worldquant101alpha".to_string(), "deprecated".to_string()];

        let SelectionResult::Selected(selected) = select_metadata(&request, &[active, deprecated])
        else {
            panic!("expected selected");
        };
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].factor_id, "WQAlpha002");
    }

    #[test]
    fn selection_rejects_explicit_deprecated_factor() {
        let request = RunRequest {
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            start_date: 20260105,
            end_date: 20260105,
            factor_ids: Some(vec!["WQAlpha001".to_string()]),
            tags: None,
            config_path: None,
            dry_run: false,
            factor_batch_size: 64,
            date_batch_size: DEFAULT_DATE_BATCH_SIZE,
            threads: None,
            profile: false,
            refresh_minute_cache: false,
        };
        let mut deprecated = metadata_row("WQAlpha001", "stock", "daily", &[]);
        deprecated.tags = vec!["deprecated".to_string()];

        let SelectionResult::Empty(message) = select_metadata(&request, &[deprecated]) else {
            panic!("expected empty");
        };
        assert!(message.contains("Deprecated factors are excluded from run"));
        assert!(message.contains("WQAlpha001"));
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

    struct DummyFactor;

    impl Factor for DummyFactor {
        fn spec(&self) -> FactorSpec {
            spec_with_dataset("dummy", DatasetId::StockDailyPv, 0)
        }

        fn compute(&self, _context: &FactorContext, _data: &DataPool) -> Result<FactorSeries> {
            unimplemented!("dummy factor is only used as a raw provider handle in engine tests")
        }
    }

    struct MultiOutputFactor {
        id: String,
        provider_key: String,
        counter: Arc<AtomicUsize>,
    }

    impl MultiOutputFactor {
        fn new(id: &str, provider_key: &str, counter: &Arc<AtomicUsize>) -> Self {
            Self {
                id: id.to_string(),
                provider_key: provider_key.to_string(),
                counter: Arc::clone(counter),
            }
        }
    }

    impl Factor for MultiOutputFactor {
        fn spec(&self) -> FactorSpec {
            spec_with_dataset(&self.id, DatasetId::StockDailyPv, 0)
        }

        fn compute_provider_key(&self) -> String {
            self.provider_key.clone()
        }

        fn compute(&self, _context: &FactorContext, _data: &DataPool) -> Result<FactorSeries> {
            self.counter.fetch_add(1, Ordering::SeqCst);
            Ok(FactorSeries {
                spec: self.spec(),
                values: Vec::new(),
            })
        }

        fn compute_many(
            &self,
            requested_ids: &[String],
            _context: &FactorContext,
            _data: &DataPool,
        ) -> Result<Vec<FactorSeries>> {
            self.counter.fetch_add(1, Ordering::SeqCst);
            Ok(requested_ids
                .iter()
                .map(|id| FactorSeries {
                    spec: spec_with_dataset(id, DatasetId::StockDailyPv, 0),
                    values: Vec::new(),
                })
                .collect())
        }
    }

    fn raw_provider(raw_id: &str, provider_key: &str, factor: Arc<dyn Factor>) -> RawProvider {
        RawProvider {
            spec: raw_spec(raw_id),
            provider_key: provider_key.to_string(),
            factor,
        }
    }

    fn raw_requirement(raw_id: &str) -> IntradayRawRequirement {
        IntradayRawRequirement {
            spec: raw_spec(raw_id),
            daily_lookback: 0,
        }
    }

    fn raw_spec(raw_id: &str) -> IntradayDailyRawSpec {
        IntradayDailyRawSpec {
            raw_id: raw_id.to_string(),
            version: "0.1.0".to_string(),
            asset_class: AssetClass::Stock,
            source_dataset: DatasetId::StockMinute1m,
            source_bar_size: None,
            columns: vec!["close".to_string()],
            window_days: 1,
        }
    }

    fn spec_with_dataset(id: &str, dataset: DatasetId, lookback: usize) -> FactorSpec {
        FactorSpec {
            id: id.to_string(),
            aliases: Vec::new(),
            name: id.to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: "0.1.0".to_string(),
            tags: Vec::new(),
            description: String::new(),
            dependencies: vec![DataRequest::new(dataset, &["close"])],
            intraday_raw_dependencies: Vec::new(),
            lookback: Lookback {
                trading_days: lookback,
            },
        }
    }

    fn spec_with_raw(id: &str, raw_id: &str, lookback: usize) -> FactorSpec {
        FactorSpec {
            id: id.to_string(),
            aliases: Vec::new(),
            name: id.to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: "0.1.0".to_string(),
            tags: Vec::new(),
            description: String::new(),
            dependencies: Vec::new(),
            intraday_raw_dependencies: vec![crate::core::IntradayDailyRawRequest::new(
                raw_id, lookback,
            )],
            lookback: Lookback {
                trading_days: lookback,
            },
        }
    }
}

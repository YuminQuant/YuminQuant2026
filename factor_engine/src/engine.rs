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
    financial_disclosure_years_for_range, DataCatalog, DataPool, DisclosureCacheProfile,
    DisclosureTableCache, FinancialBatchContext, FinancialBatchProfile, MarketDataLoader,
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
    pub include_deprecated: bool,
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
    pub provider_count: usize,
    pub loaded_table_rows: usize,
    pub indexed_rows: usize,
    pub result_rows: usize,
    pub disclosure_cache: Vec<DisclosureCacheProfile>,
    pub financial_context: Option<FinancialBatchProfile>,
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

#[derive(Clone, Copy, PartialEq)]
enum RawMode {
    Inline,
    Prepare,
    Owned,
    Prepared,
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
        self.execute(request, true, RawMode::Inline)
    }

    pub fn run(&self, request: &RunRequest) -> Result<RunReport> {
        self.execute(request, request.dry_run, RawMode::Inline)
    }

    pub fn prepare_raw(&self, request: &RunRequest) -> Result<RunReport> {
        self.execute(request, false, RawMode::Prepare)
    }

    pub fn prepare_owned_raw(&self, request: &RunRequest) -> Result<RunReport> {
        self.execute(request, false, RawMode::Owned)
    }

    pub fn run_with_prepared_raw(&self, request: &RunRequest) -> Result<RunReport> {
        if request.refresh_minute_cache {
            return Err(err("prepared raw is read-only; refresh it in a preparation task"));
        }
        self.execute(request, request.dry_run, RawMode::Prepared)
    }

    pub fn write_raw_metadata(&self, request: &RunRequest) -> Result<usize> {
        let report = self.plan(request)?;
        let providers = intraday_raw_provider_map()?;
        let specs = report.loaded_intraday_raw_requests.iter()
            .map(|raw| providers[&raw.raw_id].spec.clone()).collect::<Vec<_>>();
        self.raw_storage().write_metadata(&specs)?;
        Ok(specs.len())
    }

    fn raw_storage(&self) -> IntradayDailyRawStorage {
        match &self.config.intraday_raw_root {
            Some(root) => IntradayDailyRawStorage::with_cache_root(root.clone()),
            None => IntradayDailyRawStorage::new(self.config.factor_root.clone()),
        }
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

    fn execute(&self, request: &RunRequest, dry_run: bool, raw_mode: RawMode) -> Result<RunReport> {
        validate_date_value(request.start_date, "start-date")?;
        validate_date_value(request.end_date, "end-date")?;

        let metadata = available_specs().iter().map(FactorMetadata::from_spec).collect::<Vec<_>>();
        let selected_metadata = select_metadata(request, &metadata);
        if let SelectionResult::Empty(message) = selected_metadata {
            return Ok(empty_report(request, message));
        }
        let SelectionResult::Selected(selected_metadata) = selected_metadata else {
            unreachable!("selection result handled");
        };

        let model_ids = selected_metadata
            .iter()
            .filter(|row| is_model_generated_factor(row))
            .map(|row| row.factor_id.as_str())
            .collect::<Vec<_>>();
        if !model_ids.is_empty() {
            return Err(err(format!(
                "model-generated factors cannot be produced by Rust run/plan: {}. Use the ml_alpha prediction workflow; their stored outputs remain available for backtest.",
                model_ids.join(",")
            )));
        }

        let mut registry = factor_map();
        let mut factors = Vec::new();
        for metadata in &selected_metadata {
            let key = factor_registry_key(
                &metadata.asset_class,
                &metadata.frequency,
                &metadata.factor_id,
            );
            if let Some(factor) = registry.remove(&key) {
                factors.push(factor);
            } else {
                return Err(err(format!("registry has no factor implementation for {}",metadata.factor_id)));
            }
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
        let factor_calendar = Arc::new(calendar.clone());
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
        let execution_groups = execution_groups_for_factors(request.frequency, &specs, &factors);
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
                let date_batch_count = date_batches_for_stage(
                    &group.stage,
                    &stage_dates(&group.stage, &target_dates, &calendar),
                    date_batch_size,
                )
                .len();
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
                    panel_only: false,
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
        let raw_storage = self.raw_storage();
        let thread_pool = build_thread_pool(request.threads)?;
        let mut output_paths = BTreeSet::new();
        let mut profiles = Vec::new();
        let mut materialized_intraday_raw_dates = BTreeSet::<(String, i32)>::new();
        let mut disclosure_cache = DisclosureTableCache::default();
        let mut compute_states = BTreeMap::<String, Box<dyn Any + Send>>::new();
        let progress = ProgressBar::new("run", execution_batch_count, true);

        if matches!(raw_mode, RawMode::Prepare | RawMode::Owned) {
            let ids = raw_requirements.iter().map(|r|r.spec.raw_id.clone()).collect::<Vec<_>>();
            let raw_lookback = raw_requirements.iter().map(|r|r.daily_lookback).max().unwrap_or(0);
            if !ids.is_empty() {
                for (index, dates) in split_dates_by_chunk(&target_dates,request.date_batch_size.max(1)).into_iter().enumerate() {
                    let start = dates[0]; let end = *dates.last().unwrap();
                    let load_start = calendar.warmup_start(start,
                        if raw_mode == RawMode::Owned { 0 } else { raw_lookback });
                    let context = FactorContext { asset_class:request.asset_class, frequency:request.frequency,
                        start_date:start,end_date:end,load_start_date:load_start,
                        load_dates:calendar.open_dates_between(load_start,end),target_dates:dates };
                    let (_, mut batch_profiles) = materialize_intraday_raw_table(&ids,&raw_requirements,&raw_providers,
                        &loader,&raw_storage,&calendar,request,&context,index+1,1,thread_pool.as_ref(),
                        &mut materialized_intraday_raw_dates,&progress,raw_mode == RawMode::Owned)?;
                    profiles.append(&mut batch_profiles);
                    if raw_mode == RawMode::Owned {
                        raw_storage.complete_prepared_dates(request.asset_class, &ids, &context.target_dates)?;
                    }
                }
            }
            progress.finish();
            return Ok(RunReport { factor_count:0,output_file_count:0,load_start_date,target_dates,
                effective_start_date:Some(effective_start_date),effective_end_date:Some(effective_end_date),
                execution_stages,date_batch_count,factor_batch_count,execution_batch_count,
                selected_factor_ids:specs.iter().map(|s|s.id.clone()).collect(),loaded_requests,
                loaded_intraday_raw_requests,profiles,status_message:Some("Raw preparation complete".into()) });
        }

        for group in &execution_groups {
            let group_specs = group
                .factor_indices
                .iter()
                .map(|idx| specs[*idx].clone())
                .collect::<Vec<_>>();
            let group_factors = group
                .factor_indices
                .iter()
                .map(|idx| factors[*idx].as_ref())
                .collect::<Vec<_>>();
            let group_max_lookback = group_specs
                .iter()
                .map(spec_calendar_lookback_days)
                .max()
                .unwrap_or(0);
            let streaming = matches!(group.stage, ExecutionStage::IntradayDailyStreaming { .. });
            let group_dates = stage_dates(&group.stage, &target_dates, &calendar);
            let date_batches =
                date_batches_for_stage(&group.stage, &group_dates, request.date_batch_size);
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
                let batch_load_start_date = calendar.warmup_start(
                    batch_start_date,
                    if streaming { 0 } else { group_max_lookback },
                );
                let load_dates = calendar.open_dates_between(batch_load_start_date, batch_end_date);
                let context = FactorContext {
                    asset_class: request.asset_class,
                    frequency: request.frequency,
                    start_date: batch_start_date,
                    end_date: batch_end_date,
                    load_start_date: batch_load_start_date,
                    load_dates,
                    target_dates: date_batch
                        .iter()
                        .copied()
                        .filter(|date| *date >= effective_start_date)
                        .collect(),
                };
                let group_contextual_requirements = contextual_requirements_for_factor_batch(
                    &group_factors,
                    &group_specs,
                    &context,
                    &calendar,
                );
                let group_requests = merge_requests(group_contextual_requirements.all.clone());
                let group_disclosure_windows = disclosure_windows_for_requests(
                    &group_requests,
                    batch_start_date,
                    batch_load_start_date,
                    batch_end_date,
                );
                let group_disclosure_columns = disclosure_columns_for_requests(&group_requests);
                disclosure_cache.retain_disclosure_windows_and_columns(
                    &group_disclosure_windows,
                    &group_disclosure_columns,
                )?;
                let financial_context = FinancialBatchContext::build(
                    &loader,
                    &group_requests,
                    &context,
                    &mut disclosure_cache,
                )?;
                let shared_load_started = Instant::now();
                let batch_inputs = if provider_batches.len() > 1 {
                    let request_batches = provider_batches.iter().map(|batch| batch.iter()
                        .flat_map(|index| specs[*index].dependencies.clone()).collect()).collect::<Vec<_>>();
                    Some(DataPool::load_reused_batch_inputs(&loader, &group_requests, &request_batches, &context,
                        &mut disclosure_cache, financial_context.clone())?)
                } else { None };
                let shared_load_ms = shared_load_started.elapsed().as_millis();
                let mut batch_outputs = crate::storage::OutputBatch::default();

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
                    progress.update(format!(
                        "stage={} phase=load dates={}..{} factors={}",
                        stage_name,
                        batch_start_date,
                        batch_end_date,
                        batch_specs.len()
                    ));
                    let mut pool = if let Some(inputs) = &batch_inputs {
                        inputs.load_from_batch_inputs(&loader, &batch_requests, &context, &mut disclosure_cache)?
                    } else {
                        DataPool::load_with_financial_context(&loader, &batch_requests, &context,
                            &mut disclosure_cache, financial_context.clone())?
                    };
                    let load_ms = load_started.elapsed().as_millis()
                        + if factor_batch_index == 0 { shared_load_ms } else { 0 };
                    pool.set_trading_calendar(Arc::clone(&factor_calendar));
                    let raw_ids = raw_ids_for_specs(&batch_specs);
                    if !raw_ids.is_empty() {
                        let (raw_table, mut raw_profiles) = if raw_mode == RawMode::Prepared {
                            let specs = raw_ids.iter().map(|id| &raw_providers[id].spec).collect::<Vec<_>>();
                            let missing = raw_storage.missing_dates_many(&specs, &context.load_dates, false)?;
                            if let Some((id, dates)) = missing.iter().find(|(_, dates)| !dates.is_empty()) {
                                return Err(err(format!("prepared raw cache is incomplete: {id} at {}; run its owner preparation first", dates[0])));
                            }
                            (raw_storage.load_raw_by_dates(request.asset_class, &raw_ids, &context.load_dates)?, Vec::new())
                        } else { materialize_intraday_raw_table(
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
                            false,
                        )? };
                        profiles.append(&mut raw_profiles);
                        progress.update(format!(
                            "stage={} phase=raw_panel dates={}..{}",
                            stage_name, batch_start_date, batch_end_date
                        ));
                        pool.set_intraday_daily_raw_panel(raw_table, &context)?;
                    }
                    progress.update(format!(
                        "stage={} phase=compute dates={}..{} factors={}",
                        stage_name,
                        batch_start_date,
                        batch_end_date,
                        batch_specs.len()
                    ));
                    let compute_started = Instant::now();
                    let results = compute_factor_batch(
                        &batch_factors,
                        &context,
                        &pool,
                        &contextual_requirements,
                        &mut compute_states,
                        thread_pool.as_ref(),
                    )?;
                    validate_factor_series_target_dates(&results, &context.target_dates)?;
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
                    progress.update(format!(
                        "stage={} phase=write_prepare dates={}..{}",
                        stage_name, batch_start_date, batch_end_date
                    ));
                    let result_rows: usize = results.iter().map(|series| series.values.len()).sum();
                    let written_paths = if provider_batches.len() == 1 {
                        storage.write_results(&results)?
                    } else {
                        batch_outputs.push_factors(results);
                        if batch_outputs.full()? { batch_outputs.write_factors(&storage)? } else { Vec::new() }
                    };
                    let write_ms = write_started.elapsed().as_millis();
                    output_paths.extend(written_paths);
                    if request.profile {
                        let provider_count = provider_count_for_factors(&batch_factors);
                        let disclosure_cache = disclosure_cache.profiles();
                        let financial_context_profile = if factor_batch_index == 0 {
                            pool.financial_context_profile()
                        } else {
                            None
                        };
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
                            provider_count,
                            loaded_table_rows: pool.loaded_table_row_count(),
                            indexed_rows: pool.indexed_row_count(),
                            result_rows,
                            disclosure_cache,
                            financial_context: financial_context_profile,
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
                let write_started = Instant::now();
                output_paths.extend(batch_outputs.write_factors(&storage)?);
                if request.profile {
                    if let Some(profile) = profiles.last_mut() { profile.write_ms += write_started.elapsed().as_millis(); }
                }
                disclosure_cache.retain_disclosure_windows(&group_disclosure_windows);
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
    owned_targets: bool,
) -> Result<(crate::data::Table, Vec<BatchProfile>)> {
    let raw_id_set =
        expand_raw_ids_to_selected_provider_siblings(raw_ids, requirements, providers)?;
    let mut stateless_requirements_by_dataset =
        BTreeMap::<(DatasetId, Option<usize>), Vec<(&IntradayRawRequirement, BTreeSet<i32>)>>::new(
        );
    let mut stateful_requirements_by_dataset =
        BTreeMap::<(DatasetId, Option<usize>), Vec<(&IntradayRawRequirement, BTreeSet<i32>)>>::new(
        );
    let selected_requirements = requirements
        .iter()
        .filter(|requirement| raw_id_set.contains(&requirement.spec.raw_id))
        .collect::<Vec<_>>();
    progress.update(format!(
        "stage=intraday_raw_cache_check raws={} dates={}",
        selected_requirements.len(),
        context.load_dates.len()
    ));
    let mut missing_by_raw = storage.missing_dates_many(
        &selected_requirements
            .iter()
            .map(|requirement| &requirement.spec)
            .collect::<Vec<_>>(),
        &context.load_dates,
        request.refresh_minute_cache,
    )?;
    for requirement in selected_requirements {
        let mut missing_dates = missing_by_raw
            .remove(&requirement.spec.raw_id)
            .expect("cache check includes every selected raw")
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
                panel_only: false,
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
                        if owned_targets {
                            let owned = job.context.target_dates.iter().copied().collect::<BTreeSet<_>>();
                            for series in &mut raw_series_list {
                                series.values.retain(|value| owned.contains(&value.key.trade_date()));
                            }
                        }
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
                    provider_count: raw_profiles.len(),
                    loaded_table_rows: 0,
                    indexed_rows: 0,
                    result_rows: chunk_series.iter().map(|series| series.values.len()).sum(),
                    disclosure_cache: Vec::new(),
                    financial_context: None,
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
            panel_only: false,
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
                let result_rows = raw_profiles.values().map(|(row_count, _)| *row_count).sum();
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
                    provider_count: raw_profiles.len(),
                    loaded_table_rows: 0,
                    indexed_rows: 0,
                    result_rows,
                    disclosure_cache: Vec::new(),
                    financial_context: None,
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
    if !owned_targets && !materialized_specs.is_empty() {
        storage.write_metadata(&materialized_specs)?;
    }

    progress.update(format!(
        "stage=intraday_raw_read dates={} raw={}",
        context.load_dates.len(),
        raw_ids.len()
    ));
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
    if request.include_deprecated
        && (request.factor_ids.as_ref().is_none_or(Vec::is_empty) || request.tags.is_some())
    {
        return SelectionResult::Empty(
            "--include-deprecated requires explicit --factors and cannot be used with --tags"
                .to_string(),
        );
    }
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
            .filter(|row| !is_model_generated_factor(row))
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
                if is_deprecated_factor(row) && !request.include_deprecated {
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
                "No factors found in the code registry for: {}",
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
        .filter(|row| !is_model_generated_factor(row))
        .collect::<Vec<_>>();
    if active.is_empty() {
        return SelectionResult::Empty(format!(
            "No factors found in the code registry for asset={} frequency={}.",
            request.asset_class, request.frequency
        ));
    }
    SelectionResult::Selected(active)
}

fn is_deprecated_factor(row: &FactorMetadata) -> bool {
    row.tags.iter().any(|tag| tag == "deprecated")
}

fn is_model_generated_factor(row: &FactorMetadata) -> bool {
    row.tags.iter().any(|tag| tag == "model_generated")
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
            request.panel_only,
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
            |((dataset, entity_id, bar_size, date_policy, panel_only), (columns, financial_quarters))| {
                DataRequest {
                    dataset,
                    entity_id,
                    bar_size,
                    columns: columns.into_iter().collect(),
                    financial_quarters,
                    date_policy,
                    panel_only,
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
            .then_with(|| left.panel_only.cmp(&right.panel_only))
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
        .max(
            spec.intraday_raw_dependencies
                .iter()
                .map(|request| request.daily_lookback)
                .max()
                .unwrap_or(0),
        )
}

#[derive(Clone, Debug, Default)]
struct ContextualRequirements {
    all: Vec<DataRequest>,
    by_provider: BTreeMap<String, Vec<DataRequest>>,
    load_dates_by_provider: BTreeMap<String, BTreeSet<i32>>,
}

fn contextual_requirements_for_factor_batch<'a>(
    factors: &[&'a dyn Factor],
    specs: &[FactorSpec],
    context: &FactorContext,
    calendar: &TradingCalendar,
) -> ContextualRequirements {
    let mut output = ContextualRequirements::default();
    for (factor, spec) in factors.iter().zip(specs.iter()) {
        let lookback = if factor.streams_minute_state() {
            0
        } else {
            spec_calendar_lookback_days(spec)
        };
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
            .load_dates_by_provider
            .entry(factor.compute_provider_key())
            .or_default()
            .extend(factor_context.load_dates.iter().copied());
        output
            .by_provider
            .entry(factor.compute_provider_key())
            .or_default()
            .extend(requirements.iter().cloned());
        output.all.extend(requirements);
    }
    output
}

fn disclosure_windows_for_requests(
    requests: &[DataRequest],
    start_date: i32,
    load_start_date: i32,
    end_date: i32,
) -> HashMap<DatasetId, BTreeSet<i32>> {
    let mut windows = HashMap::<DatasetId, BTreeSet<i32>>::new();
    for request in requests {
        match request.dataset {
            DatasetId::FundPortfolio => {
                windows.entry(request.dataset).or_default().extend(load_start_date / 10000..=end_date / 10000);
            }
            DatasetId::StockIncome | DatasetId::StockBalanceSheet | DatasetId::StockCashFlow => {
                windows.entry(request.dataset).or_default().extend(
                    financial_disclosure_years_for_range(
                        start_date,
                        end_date,
                        request.financial_quarters.unwrap_or(0),
                    ),
                );
            }
            DatasetId::StockMainBusiness => {
                windows.entry(request.dataset).or_default().extend(
                    financial_disclosure_years_for_range(
                        start_date,
                        end_date,
                        request.financial_quarters.unwrap_or(8),
                    ),
                );
            }
            DatasetId::StockAnalystReport => {
                windows
                    .entry(request.dataset)
                    .or_default()
                    .extend(analyst_report_years_for_range(load_start_date, end_date));
            }
            _ => {}
        }
    }
    windows
}

fn analyst_report_years_for_range(start_date: i32, end_date: i32) -> BTreeSet<i32> {
    let start_year = (start_date / 10_000 - 1).max(0);
    let end_year = end_date / 10_000;
    (start_year..=end_year).collect()
}

fn disclosure_columns_for_requests(
    requests: &[DataRequest],
) -> HashMap<DatasetId, BTreeSet<String>> {
    let mut columns = HashMap::<DatasetId, BTreeSet<String>>::new();
    for request in requests {
        match request.dataset {
            DatasetId::FundPortfolio => {
                let entry = columns.entry(request.dataset).or_default();
                entry.extend(crate::data::FUND_PORTFOLIO_COLUMNS.iter().map(|c| c.to_string()));
                entry.extend(request.columns.iter().cloned());
            }
            DatasetId::StockIncome | DatasetId::StockBalanceSheet | DatasetId::StockCashFlow => {
                let entry = columns.entry(request.dataset).or_default();
                entry.extend(
                    [
                        "ts_code",
                        "ann_date",
                        "f_ann_date",
                        "end_date",
                        "report_type",
                        "update_flag",
                    ]
                    .into_iter()
                    .map(str::to_string),
                );
                entry.extend(request.columns.iter().cloned());
            }
            DatasetId::StockMainBusiness => {
                let entry = columns.entry(request.dataset).or_default();
                entry.extend(
                    [
                        "ts_code",
                        "end_date",
                        "bz_type",
                        "bz_item",
                        "bz_sales",
                        "update_flag",
                    ]
                    .into_iter()
                    .map(str::to_string),
                );
                entry.extend(request.columns.iter().cloned());
            }
            DatasetId::StockAnalystReport => {
                let entry = columns.entry(request.dataset).or_default();
                entry.extend(
                    ["ts_code", "report_date", "quarter", "rd"]
                        .into_iter()
                        .map(str::to_string),
                );
                entry.extend(request.columns.iter().cloned());
            }
            _ => {}
        }
    }
    columns
}

fn provider_count_for_factors(factors: &[&dyn Factor]) -> usize {
    factors
        .iter()
        .map(|factor| factor.compute_provider_key())
        .collect::<BTreeSet<_>>()
        .len()
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
    IntradayDailyStreaming { lookback: usize },
}

impl ExecutionStage {
    fn name(&self) -> String {
        match self {
            Self::DailyNoMinute => "daily_no_minute".to_string(),
            Self::IntradayDaily { lookback } => format!("intraday_daily_lookback_{lookback}"),
            Self::IntradayDailyPostprocess { lookback } => {
                format!("intraday_daily_postprocess_lookback_{lookback}")
            }
            Self::IntradayDailyStreaming { lookback } => {
                format!("intraday_daily_streaming_lookback_{lookback}")
            }
        }
    }
}

fn stage_dates(
    stage: &ExecutionStage,
    target_dates: &[i32],
    calendar: &TradingCalendar,
) -> Vec<i32> {
    if let ExecutionStage::IntradayDailyStreaming { lookback } = stage {
        if let (Some(first), Some(last)) = (target_dates.first(), target_dates.last()) {
            return calendar.open_dates_between(calendar.warmup_start(*first, *lookback), *last);
        }
    }
    target_dates.to_vec()
}

fn execution_groups_for_factors(
    frequency: Frequency,
    specs: &[FactorSpec],
    factors: &[Box<dyn Factor>],
) -> Vec<ExecutionGroup> {
    let mut groups = execution_groups_for_specs(frequency, specs);
    let mut streaming = BTreeMap::<usize, Vec<usize>>::new();
    for group in &mut groups {
        group.factor_indices.retain(|idx| {
            if frequency == Frequency::Daily && factors[*idx].streams_minute_state() {
                streaming
                    .entry(spec_calendar_lookback_days(&specs[*idx]))
                    .or_default()
                    .push(*idx);
                false
            } else {
                true
            }
        });
    }
    groups.retain(|group| !group.factor_indices.is_empty());
    groups.extend(
        streaming
            .into_iter()
            .map(|(lookback, factor_indices)| ExecutionGroup {
                stage: ExecutionStage::IntradayDailyStreaming { lookback },
                factor_indices,
            }),
    );
    groups
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
        ExecutionStage::IntradayDaily { .. } | ExecutionStage::IntradayDailyStreaming { .. } => {
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
    Ok(Some(crate::resources::thread_pool(threads)?))
}

fn request_has_daily_panel_dates(request: &DataRequest) -> bool {
    matches!(
        request.dataset,
        DatasetId::StockDailyPv
            | DatasetId::StockDailyBasic
            | DatasetId::StockDailyLimit
            | DatasetId::StockAdjFactor
            | DatasetId::StockMoneyflow
            | DatasetId::StockConsensus
            | DatasetId::StockBarraDaily
            | DatasetId::IndexDaily
            | DatasetId::FutureDaily
    )
}

fn context_for_provider_requests(
    context: &FactorContext,
    requirements: &[DataRequest],
    computation_dates: &BTreeSet<i32>,
) -> FactorContext {
    // Preserve declared warmup and every output date, including when daily sources
    // are absent or only requested at month ends. Never inherit another provider's history.
    let mut dates = computation_dates.clone();
    dates.extend(context.target_dates.iter().copied());
    for request in requirements {
        if request_has_daily_panel_dates(request) {
            dates.extend(request.resolved_dates(context));
        }
    }
    let load_dates = dates.into_iter().collect::<Vec<_>>();
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
    provider_requirements: &ContextualRequirements,
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
                requirements: provider_requirements
                    .by_provider
                    .get(&key)
                    .cloned()
                    .unwrap_or_default(),
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
                let computation_dates = provider_requirements
                    .load_dates_by_provider
                    .get(&job.key)
                    .expect("provider computation dates resolved with its requirements");
                let provider_context =
                    context_for_provider_requests(context, &job.requirements, computation_dates);
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

fn validate_factor_series_target_dates(
    results: &[FactorSeries],
    target_dates: &[i32],
) -> Result<()> {
    let target_dates = target_dates.iter().copied().collect::<BTreeSet<_>>();
    for series in results {
        for value in &series.values {
            let trade_date = value.key.trade_date();
            if !target_dates.contains(&trade_date) {
                return Err(err(format!(
                    "factor {} returned date {} outside current target date batch",
                    series.spec.id, trade_date
                )));
            }
        }
    }
    Ok(())
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

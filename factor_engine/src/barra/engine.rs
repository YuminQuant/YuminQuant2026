use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::ops::Range;
use std::path::PathBuf;
use std::time::Instant;

use rayon::prelude::*;

use crate::barra::registry::all_barra_exposures;
use crate::barra::BarraExposure;
use crate::calendar::TradingCalendar;
use crate::config::EngineConfig;
use crate::core::{
    barra_registry_key, AssetClass, BarraSeries, BarraSpec, DataRequest, FactorContext, Frequency,
};
use crate::data::{DataCatalog, DataPool, DisclosureTableCache, MarketDataLoader};
use crate::engine::{BatchProfile, FactorProfile};
use crate::error::{err, Result};
use crate::progress::ProgressBar;
use crate::storage::{BarraMetadata, BarraStorage};

pub const DEFAULT_BARRA_MODEL: &str = "CNE6";

#[derive(Clone, Debug)]
pub struct BarraRunRequest {
    pub asset_class: AssetClass,
    pub frequency: Frequency,
    pub model: String,
    pub start_date: i32,
    pub end_date: i32,
    pub exposure_ids: Option<Vec<String>>,
    pub tags: Option<Vec<String>>,
    pub families: Option<Vec<String>>,
    pub config_path: Option<PathBuf>,
    pub dry_run: bool,
    pub exposure_batch_size: usize,
    pub date_batch_size: usize,
    pub threads: Option<usize>,
    pub profile: bool,
}

#[derive(Clone, Debug)]
pub struct BarraRunReport {
    pub model: String,
    pub exposure_count: usize,
    pub output_file_count: usize,
    pub load_start_date: i32,
    pub target_dates: Vec<i32>,
    pub effective_start_date: Option<i32>,
    pub effective_end_date: Option<i32>,
    pub date_batch_count: usize,
    pub exposure_batch_count: usize,
    pub execution_batch_count: usize,
    pub selected_exposure_ids: Vec<String>,
    pub loaded_requests: Vec<DataRequest>,
    pub profiles: Vec<BatchProfile>,
    pub status_message: Option<String>,
}

pub struct BarraEngine {
    config: EngineConfig,
}

struct BarraProvider {
    family_id: String,
    exposure: Box<dyn BarraExposure>,
    specs: Vec<BarraSpec>,
}

impl BarraEngine {
    pub fn new(config: EngineConfig) -> Self {
        Self { config }
    }

    pub fn from_request(request: &BarraRunRequest) -> Result<Self> {
        Ok(Self::new(EngineConfig::discover(
            request.config_path.clone(),
        )?))
    }

    pub fn write_metadata(&self) -> Result<usize> {
        let specs = available_barra_specs();
        let storage = BarraStorage::new(self.config.barra_root.clone());
        storage.write_metadata(&specs)?;
        Ok(specs.len())
    }

    pub fn read_metadata(&self) -> Result<Vec<BarraMetadata>> {
        BarraStorage::new(self.config.barra_root.clone()).read_metadata()
    }

    pub fn plan(&self, request: &BarraRunRequest) -> Result<BarraRunReport> {
        self.execute(request, true)
    }

    pub fn run(&self, request: &BarraRunRequest) -> Result<BarraRunReport> {
        self.execute(request, request.dry_run)
    }

    fn execute(&self, request: &BarraRunRequest, dry_run: bool) -> Result<BarraRunReport> {
        validate_date_value(request.start_date, "start-date")?;
        validate_date_value(request.end_date, "end-date")?;
        if !request.model.eq_ignore_ascii_case(DEFAULT_BARRA_MODEL) {
            return Err(err(format!(
                "unsupported Barra model {}, currently only {} is available",
                request.model, DEFAULT_BARRA_MODEL
            )));
        }
        if request.frequency != Frequency::Daily {
            return Ok(empty_barra_report(
                request,
                "Barra engine currently supports daily exposures only.".to_string(),
            ));
        }

        let metadata = self.read_metadata()?;
        let selected_metadata = select_barra_metadata(request, &metadata);
        if let SelectionResult::Empty(message) = selected_metadata {
            return Ok(empty_barra_report(request, message));
        }
        let SelectionResult::Selected(selected_metadata) = selected_metadata else {
            unreachable!("selection result handled");
        };

        let providers = barra_providers();
        let mut exposure_to_provider = HashMap::new();
        for (provider_idx, provider) in providers.iter().enumerate() {
            for spec in &provider.specs {
                exposure_to_provider.insert(spec.registry_key(), provider_idx);
            }
        }
        let mut selected_provider_ids = BTreeMap::<usize, BTreeSet<String>>::new();
        let mut stale_ids = Vec::new();
        for metadata in &selected_metadata {
            let key = barra_registry_key(
                &metadata.asset_class,
                &metadata.frequency,
                &metadata.model,
                &metadata.exposure_id,
            );
            if let Some(provider_idx) = exposure_to_provider.get(&key).copied() {
                selected_provider_ids
                    .entry(provider_idx)
                    .or_default()
                    .insert(metadata.exposure_id.clone());
            } else {
                stale_ids.push(metadata.exposure_id.clone());
            }
        }
        if !stale_ids.is_empty() {
            return Err(err(format!(
                "barra_metadata.parquet is stale; missing registered implementation(s): {}. Run `barra-metadata` again.",
                stale_ids.join(",")
            )));
        }
        if let Some(families) = &request.families {
            let requested_families = families
                .iter()
                .map(|value| normalize_family_id(value))
                .collect::<BTreeSet<_>>();
            selected_provider_ids.retain(|provider_idx, _| {
                requested_families
                    .contains(&normalize_family_id(&providers[*provider_idx].family_id))
            });
        }
        if selected_provider_ids.is_empty() {
            return Ok(empty_barra_report(
                request,
                match &request.families {
                    Some(families) => format!(
                        "No Barra families selected for asset={} frequency={} families={}.",
                        request.asset_class,
                        request.frequency,
                        families.join(",")
                    ),
                    None => format!(
                        "No Barra exposures selected for asset={} frequency={}.",
                        request.asset_class, request.frequency
                    ),
                },
            ));
        }

        let selected_provider_entries = selected_provider_ids
            .iter()
            .map(|(provider_idx, selected_ids)| (*provider_idx, selected_ids.clone()))
            .collect::<Vec<_>>();
        let selected_providers = selected_provider_entries
            .iter()
            .map(|(provider_idx, _)| &providers[*provider_idx])
            .collect::<Vec<_>>();
        let specs = selected_provider_entries
            .iter()
            .flat_map(|(provider_idx, selected_ids)| {
                providers[*provider_idx]
                    .specs
                    .iter()
                    .filter(|spec| selected_ids.contains(&spec.id))
                    .cloned()
                    .collect::<Vec<_>>()
            })
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
            return Ok(empty_barra_report(
                request,
                format!(
                    "No open trading dates found on or after start_date {}.",
                    request.start_date
                ),
            ));
        };
        let Some(effective_end_date) = calendar.last_open_on_or_before(request.end_date) else {
            return Ok(empty_barra_report(
                request,
                format!(
                    "No open trading dates found on or before end_date {}.",
                    request.end_date
                ),
            ));
        };
        if effective_start_date > effective_end_date {
            return Ok(empty_barra_report(
                request,
                format!(
                    "No open trading dates found between {} and {} after calendar alignment.",
                    request.start_date, request.end_date
                ),
            ));
        }

        let target_dates = calendar.open_dates_between(effective_start_date, effective_end_date);
        if target_dates.is_empty() {
            return Ok(empty_barra_report(
                request,
                format!(
                    "No open trading dates found between {} and {}.",
                    request.start_date, request.end_date
                ),
            ));
        }

        let exposure_batch_size = request.exposure_batch_size.max(1);
        let date_batch_size = request.date_batch_size.max(1);
        let exposure_ranges = exposure_batch_ranges(selected_providers.len(), exposure_batch_size);
        let date_batches = split_dates_by_chunk(&target_dates, date_batch_size);
        let execution_batch_count = date_batches.len() * exposure_ranges.len();
        let load_start_date = calendar.warmup_start(effective_start_date, max_lookback);
        let loaded_requests =
            merge_requests(specs.iter().flat_map(|spec| spec.dependencies.clone()));
        if dry_run {
            return Ok(BarraRunReport {
                exposure_count: specs.len(),
                output_file_count: 0,
                load_start_date,
                target_dates,
                effective_start_date: Some(effective_start_date),
                effective_end_date: Some(effective_end_date),
                date_batch_count: date_batches.len(),
                exposure_batch_count: exposure_ranges.len(),
                execution_batch_count,
                selected_exposure_ids: specs.iter().map(|spec| spec.id.clone()).collect(),
                loaded_requests,
                profiles: Vec::new(),
                status_message: None,
                model: request.model.clone(),
            });
        }

        let catalog = DataCatalog::new(self.config.data_root.clone())
            .with_stock_sw_classification_path(self.config.stock_sw_classification_path.clone())
            .with_stock_ci_classification_path(self.config.stock_ci_classification_path.clone());
        let loader = MarketDataLoader::new(catalog);
        let storage = BarraStorage::with_model(self.config.barra_root.clone(), &request.model);
        let thread_pool = build_thread_pool(request.threads)?;
        let progress = ProgressBar::new("barra-run", execution_batch_count, true);
        let mut output_paths = BTreeSet::new();
        let mut profiles = Vec::new();
        let mut disclosure_cache = DisclosureTableCache::default();

        for (date_batch_index, date_batch) in date_batches.iter().enumerate() {
            let batch_start_date = *date_batch
                .first()
                .expect("date batches are never empty after split");
            let batch_end_date = *date_batch
                .last()
                .expect("date batches are never empty after split");
            let batch_load_start_date = calendar.warmup_start(batch_start_date, max_lookback);
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

            for (exposure_batch_index, range) in exposure_ranges.iter().enumerate() {
                let batch_entries = &selected_provider_entries[range.clone()];
                let batch_providers = batch_entries
                    .iter()
                    .map(|(provider_idx, _)| &providers[*provider_idx])
                    .collect::<Vec<_>>();
                let batch_selected_ids = batch_entries
                    .iter()
                    .map(|(_, selected_ids)| selected_ids.clone())
                    .collect::<Vec<_>>();
                let batch_specs = batch_entries
                    .iter()
                    .flat_map(|(provider_idx, selected_ids)| {
                        providers[*provider_idx]
                            .specs
                            .iter()
                            .filter(|spec| selected_ids.contains(&spec.id))
                            .cloned()
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
                let batch_requests = merge_requests(
                    batch_specs
                        .iter()
                        .flat_map(|spec| spec.dependencies.clone()),
                );
                let load_started = Instant::now();
                let pool = DataPool::load_with_disclosure_cache(
                    &loader,
                    &batch_requests,
                    &context,
                    &mut disclosure_cache,
                )?;
                let load_ms = load_started.elapsed().as_millis();
                let compute_started = Instant::now();
                let results = compute_exposure_batch(
                    &batch_providers,
                    &batch_selected_ids,
                    &context,
                    &pool,
                    thread_pool.as_ref(),
                )?;
                let compute_ms = compute_started.elapsed().as_millis();
                let exposure_profiles = results
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
                let stage_name = batch_stage_name(&batch_providers);
                if request.profile {
                    profiles.push(BatchProfile {
                        stage: stage_name.clone(),
                        date_batch_index: date_batch_index + 1,
                        factor_batch_index: exposure_batch_index + 1,
                        start_date: batch_start_date,
                        end_date: batch_end_date,
                        factor_count: batch_specs.len(),
                        load_ms,
                        compute_ms,
                        write_ms,
                        factors: exposure_profiles,
                    });
                }
                progress.tick(format!(
                    "stage={} dates={}..{} exposures={}",
                    stage_name,
                    batch_start_date,
                    batch_end_date,
                    batch_specs.len()
                ));
            }
        }
        progress.finish();

        Ok(BarraRunReport {
            exposure_count: specs.len(),
            output_file_count: output_paths.len(),
            load_start_date,
            target_dates,
            effective_start_date: Some(effective_start_date),
            effective_end_date: Some(effective_end_date),
            date_batch_count: date_batches.len(),
            exposure_batch_count: exposure_ranges.len(),
            execution_batch_count,
            selected_exposure_ids: specs.iter().map(|spec| spec.id.clone()).collect(),
            loaded_requests,
            profiles,
            status_message: None,
            model: request.model.clone(),
        })
    }
}

enum SelectionResult {
    Selected(Vec<BarraMetadata>),
    Empty(String),
}

fn select_barra_metadata(request: &BarraRunRequest, metadata: &[BarraMetadata]) -> SelectionResult {
    let base = metadata
        .iter()
        .filter(|row| {
            row.asset_class == request.asset_class.as_str()
                && row.frequency == request.frequency.as_str()
                && row.model.eq_ignore_ascii_case(&request.model)
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
                "No Barra exposures found for tag(s): {}",
                tags.join(",")
            ));
        }
        return SelectionResult::Selected(selected);
    }

    if let Some(exposure_ids) = &request.exposure_ids {
        let mut selected = Vec::new();
        let mut missing = Vec::new();
        for exposure_id_or_name in exposure_ids {
            if let Some(row) = base.iter().find(|row| {
                row.exposure_id == *exposure_id_or_name
                    || row.name == *exposure_id_or_name
                    || row.aliases.iter().any(|alias| alias == exposure_id_or_name)
            }) {
                selected.push(row.clone());
            } else {
                missing.push(exposure_id_or_name.clone());
            }
        }
        if !missing.is_empty() {
            return SelectionResult::Empty(format!(
                "No Barra exposures found in metadata for: {}",
                missing.join(",")
            ));
        }
        return SelectionResult::Selected(selected);
    }

    if base.is_empty() {
        return SelectionResult::Empty(format!(
            "No Barra exposures found in metadata for asset={} frequency={}.",
            request.asset_class, request.frequency
        ));
    }
    SelectionResult::Selected(base)
}

fn empty_barra_report(request: &BarraRunRequest, message: String) -> BarraRunReport {
    BarraRunReport {
        model: request.model.clone(),
        exposure_count: 0,
        output_file_count: 0,
        load_start_date: request.start_date,
        target_dates: Vec::new(),
        effective_start_date: None,
        effective_end_date: None,
        date_batch_count: 0,
        exposure_batch_count: 0,
        execution_batch_count: 0,
        selected_exposure_ids: Vec::new(),
        loaded_requests: Vec::new(),
        profiles: Vec::new(),
        status_message: Some(message),
    }
}

fn merge_requests<I>(requests: I) -> Vec<DataRequest>
where
    I: IntoIterator<Item = DataRequest>,
{
    let mut grouped: HashMap<_, (BTreeSet<String>, Option<usize>)> = HashMap::new();
    for request in requests {
        let key = (request.dataset, request.entity_id.clone(), request.bar_size);
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
            |((dataset, entity_id, bar_size), (columns, financial_quarters))| DataRequest {
                dataset,
                entity_id,
                bar_size,
                columns: columns.into_iter().collect(),
                financial_quarters,
            },
        )
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

fn exposure_batch_ranges(exposure_count: usize, exposure_batch_size: usize) -> Vec<Range<usize>> {
    if exposure_count == 0 {
        return Vec::new();
    }
    let batch_size = exposure_batch_size.max(1);
    (0..exposure_count)
        .step_by(batch_size)
        .map(|start| start..(start + batch_size).min(exposure_count))
        .collect()
}

fn split_dates_by_chunk(target_dates: &[i32], chunk_size: usize) -> Vec<Vec<i32>> {
    if target_dates.is_empty() {
        return Vec::new();
    }
    target_dates
        .chunks(chunk_size.max(1))
        .map(|chunk| chunk.to_vec())
        .collect()
}

fn normalize_family_id(value: &str) -> String {
    value
        .chars()
        .filter(|ch| *ch != '-' && *ch != '_' && !ch.is_whitespace())
        .flat_map(char::to_uppercase)
        .collect()
}

fn batch_stage_name(providers: &[&BarraProvider]) -> String {
    if providers.len() == 1 {
        format!(
            "barra_family_{}",
            providers[0].family_id.to_ascii_lowercase()
        )
    } else {
        "barra_family_multi".to_string()
    }
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

fn compute_exposure_batch(
    providers: &[&BarraProvider],
    selected_ids: &[BTreeSet<String>],
    context: &FactorContext,
    pool: &DataPool,
    thread_pool: Option<&rayon::ThreadPool>,
) -> Result<Vec<BarraSeries>> {
    let compute = || {
        providers
            .par_iter()
            .zip(selected_ids.par_iter())
            .map(|(provider, selected_ids)| {
                let series = provider.exposure.compute(context, pool)?;
                Ok(series
                    .into_iter()
                    .filter(|series| selected_ids.contains(&series.spec.id))
                    .collect::<Vec<_>>())
            })
            .collect::<Vec<_>>()
    };
    let results = match thread_pool {
        Some(thread_pool) => thread_pool.install(compute),
        None => compute(),
    };
    let series = results
        .into_iter()
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    Ok(series)
}

pub fn available_barra_specs() -> Vec<BarraSpec> {
    all_barra_exposures()
        .into_iter()
        .flat_map(|exposure| exposure.specs())
        .collect()
}

fn barra_providers() -> Vec<BarraProvider> {
    all_barra_exposures()
        .into_iter()
        .map(|exposure| {
            let family_id = exposure.family_id().to_string();
            let specs = exposure.specs();
            BarraProvider {
                family_id,
                exposure,
                specs,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use crate::core::{AssetClass, DataRequest, DatasetId, Frequency};
    use crate::storage::BarraMetadata;

    use super::{
        exposure_batch_ranges, select_barra_metadata, split_dates_by_chunk, BarraRunRequest,
        SelectionResult,
    };

    #[test]
    fn chunks_exposures_by_configured_batch_size() {
        assert_eq!(
            exposure_batch_ranges(0, 10),
            Vec::<std::ops::Range<usize>>::new()
        );
        assert_eq!(exposure_batch_ranges(3, 10), vec![0..3]);
        assert_eq!(exposure_batch_ranges(11, 10), vec![0..10, 10..11]);
    }

    #[test]
    fn chunks_barra_dates_by_configured_batch_size() {
        assert_eq!(split_dates_by_chunk(&[], 10), Vec::<Vec<i32>>::new());
        assert_eq!(
            split_dates_by_chunk(&[20260101, 20260102, 20260103, 20260104, 20260105], 2),
            vec![
                vec![20260101, 20260102],
                vec![20260103, 20260104],
                vec![20260105],
            ]
        );
        assert_eq!(
            split_dates_by_chunk(&[20260101, 20260102], 0),
            vec![vec![20260101], vec![20260102]]
        );
    }

    #[test]
    fn selection_matches_exposure_id_inside_asset_frequency() {
        let request = BarraRunRequest {
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            model: super::DEFAULT_BARRA_MODEL.to_string(),
            start_date: 20260105,
            end_date: 20260105,
            exposure_ids: Some(vec!["SIZE".to_string()]),
            tags: None,
            families: None,
            config_path: None,
            dry_run: false,
            exposure_batch_size: 10,
            date_batch_size: 1,
            threads: None,
            profile: false,
        };
        let metadata = vec![
            metadata_row("SIZE", "stock", "daily"),
            metadata_row("SIZE", "future", "daily"),
        ];

        let SelectionResult::Selected(selected) = select_barra_metadata(&request, &metadata) else {
            panic!("expected selected");
        };
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].asset_class, "stock");
        assert_eq!(selected[0].exposure_id, "SIZE");
    }

    fn metadata_row(exposure_id: &str, asset_class: &str, frequency: &str) -> BarraMetadata {
        BarraMetadata {
            exposure_id: exposure_id.to_string(),
            aliases: Vec::new(),
            aliases_json: String::new(),
            version: "0.1.0".to_string(),
            output_column: exposure_id.to_string(),
            name: exposure_id.to_string(),
            model: super::DEFAULT_BARRA_MODEL.to_string(),
            asset_class: asset_class.to_string(),
            frequency: frequency.to_string(),
            tags: Vec::new(),
            tags_json: String::new(),
            dependencies_json: String::new(),
            description: String::new(),
            updated_at: String::new(),
        }
    }

    #[test]
    fn merge_requests_keeps_bara_dependencies_unique() {
        let merged = super::merge_requests(vec![
            DataRequest::new(DatasetId::StockDailyBasic, &["total_mv"]),
            DataRequest::new(DatasetId::StockDailyBasic, &["total_mv"]),
        ]);
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].columns, vec!["total_mv".to_string()]);
    }
}

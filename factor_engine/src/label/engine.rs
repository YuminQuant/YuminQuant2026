use std::collections::{BTreeSet, HashMap};
use std::ops::Range;
use std::path::PathBuf;
use std::time::Instant;

use rayon::prelude::*;

use crate::calendar::TradingCalendar;
use crate::config::EngineConfig;
use crate::core::{
    label_registry_key, AssetClass, DataRequest, DatasetId, FactorContext, Frequency, LabelSeries,
    LabelSpec,
};
use crate::data::{DataCatalog, DataPool, MarketDataLoader};
use crate::engine::{BatchProfile, FactorProfile};
use crate::error::{err, Result};
use crate::label::registry::{all_labels, label_map};
use crate::label::Label;
use crate::progress::ProgressBar;
use crate::storage::{LabelMetadata, LabelStorage};

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
    pub threads: Option<usize>,
    pub profile: bool,
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
        let label_ranges = label_batch_ranges(labels.len(), label_batch_size);
        let loaded_requests =
            merge_requests(specs.iter().flat_map(|spec| spec.dependencies.clone()));
        let execution_batch_count = target_dates.len() * label_ranges.len();
        let date_batch_count = target_dates.len();
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
                label_batch_count: label_ranges.len(),
                execution_batch_count,
                selected_label_ids: specs.iter().map(|spec| spec.id.clone()).collect(),
                loaded_requests,
                profiles: Vec::new(),
                status_message: None,
            });
        }

        let catalog = DataCatalog::new(self.config.data_root.clone())
            .with_stock_sw_classification_path(self.config.stock_sw_classification_path.clone())
            .with_stock_ci_classification_path(self.config.stock_ci_classification_path.clone());
        let loader = MarketDataLoader::new(catalog);
        let storage = LabelStorage::new(self.config.label_root.clone());
        let thread_pool = build_thread_pool(request.threads)?;
        let progress = ProgressBar::new("label-run", execution_batch_count, true);
        let mut output_paths = BTreeSet::new();
        let mut profiles = Vec::new();
        let mut skipped_dates = Vec::new();

        for (date_idx, trade_date) in target_dates.iter().copied().enumerate() {
            let load_end_date = calendar
                .open_date_after(trade_date, max_lookahead)
                .expect("target dates are filtered by lookahead");
            let load_dates = calendar.open_dates_between(trade_date, load_end_date);
            let context = FactorContext {
                asset_class: request.asset_class,
                frequency: request.frequency,
                start_date: trade_date,
                end_date: load_end_date,
                load_start_date: trade_date,
                load_dates,
                target_dates: vec![trade_date],
            };

            if requires_stock_daily_pv(&specs)
                && !stock_daily_pv_has_dates(&loader, &context.load_dates)?
            {
                skipped_dates.push(trade_date);
                for _ in &label_ranges {
                    progress.tick(format!(
                        "stage=daily_label date={} skipped=missing_future_data",
                        trade_date
                    ));
                }
                continue;
            }

            for (label_batch_index, range) in label_ranges.iter().enumerate() {
                let batch_specs = specs[range.clone()].to_vec();
                let batch_labels = labels[range.clone()]
                    .iter()
                    .map(|label| label.as_ref())
                    .collect::<Vec<_>>();
                let batch_requests = merge_requests(
                    batch_specs
                        .iter()
                        .flat_map(|spec| spec.dependencies.clone()),
                );
                let load_started = Instant::now();
                let pool = DataPool::load(&loader, &batch_requests, &context)?;
                let load_ms = load_started.elapsed().as_millis();
                let compute_started = Instant::now();
                let results =
                    compute_label_batch(&batch_labels, &context, &pool, thread_pool.as_ref())?;
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
                        stage: "daily_label".to_string(),
                        date_batch_index: date_idx + 1,
                        factor_batch_index: label_batch_index + 1,
                        start_date: trade_date,
                        end_date: trade_date,
                        factor_count: batch_specs.len(),
                        load_ms,
                        compute_ms,
                        write_ms,
                        factors: label_profiles,
                    });
                }
                progress.tick(format!(
                    "stage=daily_label date={} labels={}",
                    trade_date,
                    batch_specs.len()
                ));
            }
        }
        progress.finish();

        Ok(LabelRunReport {
            label_count: specs.len(),
            output_file_count: output_paths.len(),
            target_dates,
            skipped_dates,
            effective_start_date: Some(effective_start_date),
            effective_end_date: Some(effective_end_date),
            max_lookahead,
            date_batch_count,
            label_batch_count: label_ranges.len(),
            execution_batch_count,
            selected_label_ids: specs.iter().map(|spec| spec.id.clone()).collect(),
            loaded_requests,
            profiles,
            status_message: None,
        })
    }
}

enum SelectionResult {
    Selected(Vec<LabelMetadata>),
    Empty(String),
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
        let key = (request.dataset, request.entity_id.clone());
        grouped
            .entry(key)
            .or_default()
            .extend(request.columns.into_iter());
    }
    let mut merged = grouped
        .into_iter()
        .map(|((dataset, entity_id), columns)| DataRequest {
            dataset,
            entity_id,
            columns: columns.into_iter().collect(),
            financial_quarters: None,
        })
        .collect::<Vec<_>>();
    merged.sort_by(|left, right| {
        left.dataset
            .cmp(&right.dataset)
            .then_with(|| left.entity_id.cmp(&right.entity_id))
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

fn requires_stock_daily_pv(specs: &[LabelSpec]) -> bool {
    specs.iter().any(|spec| {
        spec.dependencies
            .iter()
            .any(|request| request.dataset == DatasetId::StockDailyPv)
    })
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

fn stock_daily_pv_has_dates(loader: &MarketDataLoader, trade_dates: &[i32]) -> Result<bool> {
    if trade_dates.is_empty() {
        return Ok(false);
    }
    let table =
        loader.load_daily_by_dates(DatasetId::StockDailyPv, &["open".to_string()], trade_dates)?;
    let actual_dates = table
        .required_i32("trade_date")?
        .iter()
        .flatten()
        .copied()
        .collect::<BTreeSet<_>>();
    Ok(trade_dates
        .iter()
        .all(|trade_date| actual_dates.contains(trade_date)))
}

pub fn available_label_specs() -> Vec<LabelSpec> {
    all_labels().into_iter().map(|label| label.spec()).collect()
}

#[cfg(test)]
mod tests {
    use crate::core::{AssetClass, DataRequest, DatasetId, Frequency, LabelSpec, Lookahead};
    use crate::storage::LabelMetadata;

    use super::{
        eligible_label_target_dates, label_batch_ranges, select_label_metadata, LabelRunRequest,
        SelectionResult,
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
            threads: None,
            profile: false,
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

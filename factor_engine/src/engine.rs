use std::path::PathBuf;

use crate::calendar::TradingCalendar;
use crate::config::EngineConfig;
use crate::core::{AssetClass, DataRequest, FactorContext, FactorSpec, Frequency};
use crate::data::{DataCatalog, DataPool, MarketDataLoader};
use crate::error::{err, Result};
use crate::factor::registry::{all_factors, factor_map};
use crate::storage::{FactorMetadata, FactorStorage};

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
}

#[derive(Clone, Debug)]
pub struct RunReport {
    pub factor_count: usize,
    pub output_file_count: usize,
    pub load_start_date: i32,
    pub target_dates: Vec<i32>,
    pub selected_factor_ids: Vec<String>,
    pub loaded_requests: Vec<DataRequest>,
    pub status_message: Option<String>,
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
            if let Some(factor) = registry.remove(&metadata.factor_id) {
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
        let target_dates = calendar.open_dates_between(request.start_date, request.end_date);
        let load_start_date = calendar.warmup_start(request.start_date, max_lookback);
        let context = FactorContext {
            asset_class: request.asset_class,
            frequency: request.frequency,
            start_date: request.start_date,
            end_date: request.end_date,
            load_start_date,
            target_dates: target_dates.clone(),
        };

        let loaded_requests =
            merge_requests(specs.iter().flat_map(|spec| spec.dependencies.clone()));
        if dry_run {
            return Ok(RunReport {
                factor_count: specs.len(),
                output_file_count: 0,
                load_start_date,
                target_dates,
                selected_factor_ids: specs.iter().map(|spec| spec.id.clone()).collect(),
                loaded_requests,
                status_message: None,
            });
        }

        let catalog = DataCatalog::new(self.config.data_root.clone());
        let loader = MarketDataLoader::new(catalog);
        let pool = DataPool::load(&loader, &loaded_requests, &context)?;
        let mut results = Vec::new();
        for factor in factors {
            results.push(factor.compute(&context, &pool)?);
        }

        let storage = FactorStorage::new(self.config.factor_root.clone());
        let output_file_count = storage.write_results(&results)?;
        let result_specs = results
            .iter()
            .map(|series| series.spec.clone())
            .collect::<Vec<_>>();

        Ok(RunReport {
            factor_count: result_specs.len(),
            output_file_count,
            load_start_date,
            target_dates,
            selected_factor_ids: result_specs.iter().map(|spec| spec.id.clone()).collect(),
            loaded_requests,
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
            if let Some(row) = base
                .iter()
                .find(|row| row.factor_id == *factor_id_or_name || row.name == *factor_id_or_name)
            {
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
        selected_factor_ids: Vec::new(),
        loaded_requests: Vec::new(),
        status_message: Some(message),
    }
}

fn merge_requests<I>(requests: I) -> Vec<DataRequest>
where
    I: IntoIterator<Item = DataRequest>,
{
    use std::collections::{BTreeSet, HashMap};

    let mut grouped: HashMap<_, BTreeSet<String>> = HashMap::new();
    for request in requests {
        grouped
            .entry(request.dataset)
            .or_default()
            .extend(request.columns.into_iter());
    }
    let mut merged = grouped
        .into_iter()
        .map(|(dataset, columns)| DataRequest {
            dataset,
            columns: columns.into_iter().collect(),
        })
        .collect::<Vec<_>>();
    merged.sort_by_key(|request| request.dataset);
    merged
}

pub fn available_specs() -> Vec<FactorSpec> {
    all_factors()
        .into_iter()
        .map(|factor| factor.spec())
        .collect()
}

use std::path::PathBuf;

use rayon::prelude::*;

use crate::calendar::TradingCalendar;
use crate::config::EngineConfig;
use crate::core::{AssetClass, DatasetId};
use crate::data::parquet_io::read_parquet;
use crate::data::{DataCatalog, MarketDataLoader, Table};
use crate::derive::bar::{derive_stock_minute_bars, validate_stock_minute_bar_size};
use crate::derive::logsig::derive_logsig_volume_signature_rows;
use crate::derive::request::{BarSource, DeriveBarRequest, DeriveLogsigVolumeSignatureRequest};
use crate::derive::storage::{
    derived_logsig_volume_signature_path, derived_stock_bar_path, write_bar_rows,
    write_logsig_signature_rows,
};
use crate::error::{err, Result};
use crate::progress::ProgressBar;

#[derive(Clone, Debug)]
pub struct DeriveEngine {
    config: EngineConfig,
}

#[derive(Clone, Debug, Default)]
pub struct DeriveBarReport {
    pub output_files: Vec<PathBuf>,
    pub processed_dates: usize,
    pub missing_input_dates: Vec<i32>,
    pub skipped_existing_dates: Vec<i32>,
    pub total_rows: usize,
}

#[derive(Clone, Debug, Default)]
pub struct DeriveLogsigVolumeSignatureReport {
    pub output_files: Vec<PathBuf>,
    pub processed_dates: usize,
    pub missing_input_dates: Vec<i32>,
    pub skipped_existing_dates: Vec<i32>,
    pub total_rows: usize,
}

impl DeriveEngine {
    pub fn from_request(request: &DeriveBarRequest) -> Result<Self> {
        Ok(Self {
            config: EngineConfig::discover(request.project_config_path.clone())?,
        })
    }

    pub fn from_logsig_volume_signature_request(
        request: &DeriveLogsigVolumeSignatureRequest,
    ) -> Result<Self> {
        Ok(Self {
            config: EngineConfig::discover(request.project_config_path.clone())?,
        })
    }

    pub fn run_bar(&self, request: &DeriveBarRequest) -> Result<DeriveBarReport> {
        ensure_supported_request(request)?;
        let calendar =
            TradingCalendar::load(&self.config.data_root, &self.config.stock_calendar_exchange)?;
        let dates = calendar.open_dates_between(request.start_date, request.end_date);
        let catalog = DataCatalog::new(self.config.data_root.clone())
            .with_stock_sw_classification_path(self.config.stock_sw_classification_path.clone())
            .with_stock_ci_classification_path(self.config.stock_ci_classification_path.clone());
        let progress = ProgressBar::new("derive-bar", dates.len(), true);

        let columns = vec![
            "open".to_string(),
            "high".to_string(),
            "low".to_string(),
            "close".to_string(),
            "vol".to_string(),
            "amount".to_string(),
        ];
        let mut report = DeriveBarReport::default();
        let thread_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(request.date_batch_size)
            .build()?;
        for date_batch in dates.chunks(request.date_batch_size) {
            let outcomes = thread_pool.install(|| {
                date_batch
                    .par_iter()
                    .map(|trade_date| {
                        derive_one_stock_minute_bar_date(
                            *trade_date,
                            request,
                            &self.config.data_root,
                            &catalog,
                            &columns,
                        )
                    })
                    .collect::<Vec<_>>()
            });
            for outcome in outcomes {
                match outcome? {
                    DeriveDateOutcome::Written {
                        trade_date,
                        output_path,
                        rows,
                    } => {
                        report.output_files.push(output_path);
                        report.processed_dates += 1;
                        report.total_rows += rows;
                        progress.tick(format!("date={trade_date} rows={rows}"));
                    }
                    DeriveDateOutcome::MissingInput { trade_date } => {
                        report.missing_input_dates.push(trade_date);
                        progress.tick(format!("date={trade_date} missing input"));
                    }
                    DeriveDateOutcome::SkippedExisting { trade_date } => {
                        report.skipped_existing_dates.push(trade_date);
                        progress.tick(format!("date={trade_date} skipped existing"));
                    }
                }
            }
        }
        progress.finish();
        Ok(report)
    }

    pub fn run_logsig_volume_signature(
        &self,
        request: &DeriveLogsigVolumeSignatureRequest,
    ) -> Result<DeriveLogsigVolumeSignatureReport> {
        ensure_supported_logsig_request(request)?;
        let calendar =
            TradingCalendar::load(&self.config.data_root, &self.config.stock_calendar_exchange)?;
        let target_dates = calendar.open_dates_between(request.start_date, request.end_date);
        let progress = ProgressBar::new("derive-logsig-volume-signature", target_dates.len(), true);
        let mut report = DeriveLogsigVolumeSignatureReport::default();
        let thread_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(request.date_batch_size)
            .build()?;
        for date_batch in target_dates.chunks(request.date_batch_size) {
            let outcomes = thread_pool.install(|| {
                date_batch
                    .par_iter()
                    .map(|trade_date| {
                        derive_one_logsig_volume_signature_date(
                            *trade_date,
                            request,
                            &self.config.data_root,
                            &calendar,
                        )
                    })
                    .collect::<Vec<_>>()
            });
            for outcome in outcomes {
                match outcome? {
                    DeriveDateOutcome::Written {
                        trade_date,
                        output_path,
                        rows,
                    } => {
                        report.output_files.push(output_path);
                        report.processed_dates += 1;
                        report.total_rows += rows;
                        progress.tick(format!("date={trade_date} rows={rows}"));
                    }
                    DeriveDateOutcome::MissingInput { trade_date } => {
                        report.missing_input_dates.push(trade_date);
                        progress.tick(format!("date={trade_date} missing input"));
                    }
                    DeriveDateOutcome::SkippedExisting { trade_date } => {
                        report.skipped_existing_dates.push(trade_date);
                        progress.tick(format!("date={trade_date} skipped existing"));
                    }
                }
            }
        }
        progress.finish();
        Ok(report)
    }
}

#[derive(Debug)]
enum DeriveDateOutcome {
    Written {
        trade_date: i32,
        output_path: PathBuf,
        rows: usize,
    },
    MissingInput {
        trade_date: i32,
    },
    SkippedExisting {
        trade_date: i32,
    },
}

fn derive_one_stock_minute_bar_date(
    trade_date: i32,
    request: &DeriveBarRequest,
    data_root: &std::path::Path,
    catalog: &DataCatalog,
    columns: &[String],
) -> Result<DeriveDateOutcome> {
    let output_path = derived_stock_bar_path(data_root, request.bar_size, trade_date);
    if output_path.exists() && !request.overwrite {
        return Ok(DeriveDateOutcome::SkippedExisting { trade_date });
    }

    let loader = MarketDataLoader::new(catalog.clone());
    let table = load_stock_minute_table_for_date(&loader, columns, trade_date)?;
    let Some(table) = table else {
        return Ok(DeriveDateOutcome::MissingInput { trade_date });
    };
    let rows = derive_stock_minute_bars(&table, trade_date, request.bar_size)?;
    let row_count = rows.len();
    write_bar_rows(&output_path, &rows)?;
    Ok(DeriveDateOutcome::Written {
        trade_date,
        output_path,
        rows: row_count,
    })
}

fn derive_one_logsig_volume_signature_date(
    trade_date: i32,
    request: &DeriveLogsigVolumeSignatureRequest,
    data_root: &std::path::Path,
    calendar: &TradingCalendar,
) -> Result<DeriveDateOutcome> {
    let output_path = derived_logsig_volume_signature_path(data_root, trade_date);
    if output_path.exists() && !request.overwrite {
        return Ok(DeriveDateOutcome::SkippedExisting { trade_date });
    }
    let history = calendar.open_dates_between(0, trade_date);
    if history.len() < request.lookback_days {
        return Ok(DeriveDateOutcome::MissingInput { trade_date });
    }
    let source_dates = &history[history.len() - request.lookback_days..];
    let mut tables = Vec::with_capacity(source_dates.len());
    let columns = vec![
        "trade_date".to_string(),
        "bar_index".to_string(),
        "ts_code".to_string(),
        "volume".to_string(),
    ];
    for source_date in source_dates {
        let path = derived_stock_bar_path(data_root, request.bar_size, *source_date);
        if !path.exists() {
            return Ok(DeriveDateOutcome::MissingInput { trade_date });
        }
        tables.push(read_parquet(&path, Some(&columns))?);
    }
    let rows = derive_logsig_volume_signature_rows(
        trade_date,
        &tables,
        request.lookback_days,
        request.bar_size,
        request.order,
    )?;
    let row_count = rows.len();
    write_logsig_signature_rows(&output_path, &rows)?;
    Ok(DeriveDateOutcome::Written {
        trade_date,
        output_path,
        rows: row_count,
    })
}

fn load_stock_minute_table_for_date(
    loader: &MarketDataLoader,
    columns: &[String],
    trade_date: i32,
) -> Result<Option<Table>> {
    let mut minute_tables =
        loader.load_minute_by_date(DatasetId::StockMinute1m, columns, &[trade_date])?;
    Ok(minute_tables.remove(&trade_date))
}

fn ensure_supported_request(request: &DeriveBarRequest) -> Result<()> {
    if request.asset_class != AssetClass::Stock {
        return Err(err("derive-bar v1 only supports --asset stock"));
    }
    if request.source != BarSource::Minute {
        return Err(err("derive-bar v1 only supports --source minute"));
    }
    validate_stock_minute_bar_size(request.bar_size)?;
    if request.start_date > request.end_date {
        return Err(err("--start-date must be <= --end-date"));
    }
    if request.date_batch_size == 0 {
        return Err(err("--date-batch-size must be greater than 0"));
    }
    Ok(())
}

fn ensure_supported_logsig_request(request: &DeriveLogsigVolumeSignatureRequest) -> Result<()> {
    if request.asset_class != AssetClass::Stock {
        return Err(err(
            "derive-logsig-volume-signature only supports --asset stock",
        ));
    }
    validate_stock_minute_bar_size(request.bar_size)?;
    if request.bar_size != 5 {
        return Err(err(
            "derive-logsig-volume-signature v1 expects --bar-size 5",
        ));
    }
    if request.lookback_days == 0 {
        return Err(err("--lookback-days must be greater than 0"));
    }
    if request.lookback_days != 20 {
        return Err(err(
            "derive-logsig-volume-signature v1 expects --lookback-days 20",
        ));
    }
    if request.order == 0 {
        return Err(err("--order must be greater than 0"));
    }
    if request.order != 10 {
        return Err(err("derive-logsig-volume-signature v1 expects --order 10"));
    }
    if request.start_date > request.end_date {
        return Err(err("--start-date must be <= --end-date"));
    }
    if request.date_batch_size == 0 {
        return Err(err("--date-batch-size must be greater than 0"));
    }
    Ok(())
}

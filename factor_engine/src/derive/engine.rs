use std::path::PathBuf;

use rayon::prelude::*;

use crate::calendar::TradingCalendar;
use crate::config::EngineConfig;
use crate::core::{AssetClass, DatasetId};
use crate::data::{DataCatalog, MarketDataLoader, Table};
use crate::derive::bar::{derive_stock_minute_bars, validate_stock_minute_bar_size};
use crate::derive::request::{BarSource, DeriveBarRequest};
use crate::derive::storage::{derived_stock_bar_path, write_bar_rows};
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

impl DeriveEngine {
    pub fn from_request(request: &DeriveBarRequest) -> Result<Self> {
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

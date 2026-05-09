pub mod cross_section;
pub mod data;
pub mod ic;
pub mod metrics;
pub mod preprocess;
pub mod request;
pub mod schedule;
pub mod storage;
pub mod time_series;

use std::path::PathBuf;

use crate::backtest::cross_section::{ensure_backtest_inputs, run_cross_section_backtest};
use crate::backtest::data::load_backtest_input;
use crate::backtest::metrics::summarize_factor_stats;
use crate::backtest::request::BacktestRunRequest;
use crate::backtest::schedule::rebalance_dates;
use crate::backtest::storage::write_backtest_outputs;
use crate::config::EngineConfig;
use crate::error::Result;
use crate::progress::ProgressBar;

#[derive(Clone, Debug)]
pub struct BacktestEngine {
    config: EngineConfig,
}

#[derive(Clone, Debug)]
pub struct BacktestRunReport {
    pub factor_count: usize,
    pub selected_factor_ids: Vec<String>,
    pub output_dir: PathBuf,
    pub output_files: Vec<PathBuf>,
    pub rebalance_count: usize,
}

impl BacktestEngine {
    pub fn new(config: EngineConfig) -> Self {
        Self { config }
    }

    pub fn from_request(request: &BacktestRunRequest) -> Result<Self> {
        Ok(Self::new(EngineConfig::discover(
            request.config_path.clone(),
        )?))
    }

    pub fn run(&self, request: &BacktestRunRequest) -> Result<BacktestRunReport> {
        ensure_backtest_inputs(request)?;
        let input = load_backtest_input(&self.config, request)?;
        let rebalance_dates = rebalance_dates(&input.target_dates, &request.rebalance);
        let progress = ProgressBar::new("backtest", input.factor_metadata.len(), true);
        let output = run_cross_section_backtest(request, &input, &rebalance_dates, &progress)?;
        progress.finish();
        let factor_stats_summary = summarize_factor_stats(&output.factor_stats);
        let output_dir = request
            .output_dir
            .clone()
            .unwrap_or_else(|| default_output_dir(&self.config, request));
        let output_files = write_backtest_outputs(
            &output_dir,
            &output.returns,
            &output.daily_ic,
            &factor_stats_summary,
        )?;
        Ok(BacktestRunReport {
            factor_count: input.factor_metadata.len(),
            selected_factor_ids: input
                .factor_metadata
                .iter()
                .map(|row| row.factor_id.clone())
                .collect(),
            output_dir,
            output_files,
            rebalance_count: rebalance_dates.len(),
        })
    }
}

fn default_output_dir(config: &EngineConfig, request: &BacktestRunRequest) -> PathBuf {
    config
        .data_root
        .join("backtest")
        .join(request.asset_class.as_str())
        .join(request.frequency.as_str())
}

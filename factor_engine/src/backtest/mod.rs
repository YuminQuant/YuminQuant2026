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

use crate::backtest::cross_section::{
    ensure_backtest_inputs, finalize_cross_section_backtest, init_cross_section_states,
    update_cross_section_backtest_states,
};
use crate::backtest::data::{
    load_backtest_input_batch, prepare_backtest_data_plan, FactorFillState,
};
use crate::backtest::request::BacktestRunRequest;
use crate::backtest::schedule::rebalance_dates;
use crate::backtest::storage::write_backtest_outputs;
use crate::config::EngineConfig;
use crate::error::{err, Result};
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
        let plan = prepare_backtest_data_plan(&self.config, request)?;
        let selected_factor_ids = plan
            .factor_metadata
            .iter()
            .map(|row| row.factor_id.clone())
            .collect::<Vec<_>>();
        let rebalance_dates = rebalance_dates(&plan.target_dates, &request.rebalance);
        let factor_batches =
            factor_batch_ranges(plan.factor_metadata.len(), request.factor_batch_size);
        let date_batches = date_batch_ranges(plan.target_dates.len(), request.date_batch_size);
        let progress = ProgressBar::new("backtest", plan.factor_metadata.len(), true);
        let thread_pool = build_thread_pool(request.threads)?;
        let output_dir = request
            .output_dir
            .clone()
            .unwrap_or_else(|| default_output_dir(&self.config, request));
        let mut output_files = Vec::new();
        for (batch_idx, range) in factor_batches.iter().enumerate() {
            let batch_factors = plan.factor_metadata[range.clone()].to_vec();
            let factor_columns = batch_factors
                .iter()
                .map(|row| row.output_column.clone())
                .collect::<Vec<_>>();
            let mut factor_fill_state =
                FactorFillState::new(&factor_columns, plan.instruments.len());
            let mut states = init_cross_section_states(&batch_factors);
            for date_range in &date_batches {
                let target_dates = &plan.target_dates[date_range.clone()];
                let input = load_backtest_input_batch(
                    &self.config,
                    request,
                    &plan,
                    &batch_factors,
                    target_dates,
                    &mut factor_fill_state,
                )?;
                update_cross_section_backtest_states(
                    request,
                    &input,
                    &rebalance_dates,
                    &mut states,
                    thread_pool.as_ref(),
                )?;
            }
            let output = finalize_cross_section_backtest(
                states,
                &batch_factors,
                &progress,
                batch_idx + 1,
                factor_batches.len(),
                plan.target_dates.len(),
                rebalance_dates.len(),
                request.groups,
            )?;
            output_files.extend(write_backtest_outputs(
                &output_dir,
                &output.returns,
                &output.daily_ic,
                &output.factor_stats,
                &output.holdings,
                &output.industry_weights,
                &output.barra_exposure,
            )?);
        }
        progress.finish();
        Ok(BacktestRunReport {
            factor_count: selected_factor_ids.len(),
            selected_factor_ids,
            output_dir,
            output_files,
            rebalance_count: rebalance_dates.len(),
        })
    }
}

fn factor_batch_ranges(
    factor_count: usize,
    factor_batch_size: usize,
) -> Vec<std::ops::Range<usize>> {
    if factor_count == 0 {
        return Vec::new();
    }
    let batch_size = factor_batch_size.max(1);
    (0..factor_count)
        .step_by(batch_size)
        .map(|start| start..(start + batch_size).min(factor_count))
        .collect()
}

fn date_batch_ranges(date_count: usize, date_batch_size: usize) -> Vec<std::ops::Range<usize>> {
    if date_count == 0 {
        return Vec::new();
    }
    let batch_size = date_batch_size.max(1);
    (0..date_count)
        .step_by(batch_size)
        .map(|start| start..(start + batch_size).min(date_count))
        .collect()
}

fn build_thread_pool(threads: Option<usize>) -> Result<Option<rayon::ThreadPool>> {
    match threads {
        Some(0) => Err(err("--threads must be greater than 0")),
        Some(threads) => Ok(Some(
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()?,
        )),
        None => Ok(None),
    }
}

fn default_output_dir(config: &EngineConfig, request: &BacktestRunRequest) -> PathBuf {
    config
        .data_root
        .join("backtest")
        .join(request.asset_class.as_str())
        .join(request.frequency.as_str())
}

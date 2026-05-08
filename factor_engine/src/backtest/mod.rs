pub mod cross_section;
pub mod data;
pub mod ic;
pub mod metrics;
pub mod preprocess;
pub mod request;
pub mod schedule;
pub mod storage;
pub mod time_series;

use std::collections::BTreeMap;
use std::path::PathBuf;

use crate::backtest::cross_section::{ensure_backtest_inputs, run_cross_section_backtest};
use crate::backtest::data::load_backtest_input;
use crate::backtest::ic::IcObservation;
use crate::backtest::metrics::{
    summarize_factor_stats, summarize_ic, summarize_performance, FactorStatsSummary, IcSummary,
    PerformanceSummary,
};
use crate::backtest::request::BacktestRunRequest;
use crate::backtest::schedule::rebalance_dates;
use crate::backtest::storage::{write_detail_outputs, write_summary_outputs};
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
    pub summary_files: Vec<PathBuf>,
    pub detail_files: Vec<PathBuf>,
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
        let performance_summary = summarize_performance(&output.returns);
        let factor_stats_summary = summarize_factor_stats(&output.factor_stats);
        let ic_summary = summarize_daily_ic(&output.daily_ic);
        let ic_decay_summary = summarize_decay_ic(&output.ic_decay);
        let output_dir = request.output_dir.clone().unwrap_or_else(|| {
            default_output_dir(
                &self.config,
                request,
                input
                    .factor_metadata
                    .first()
                    .map(|row| row.factor_id.as_str())
                    .unwrap_or("factors"),
            )
        });
        let summary_files = write_summary_outputs(
            &output_dir,
            &performance_summary,
            &ic_summary,
            &ic_decay_summary,
            &factor_stats_summary,
        )?;
        let detail_files = if request.write_detail {
            write_detail_outputs(
                &output_dir,
                &output.returns,
                &output.daily_ic,
                &output.ic_decay,
            )?
        } else {
            Vec::new()
        };
        Ok(BacktestRunReport {
            factor_count: input.factor_metadata.len(),
            selected_factor_ids: input
                .factor_metadata
                .iter()
                .map(|row| row.factor_id.clone())
                .collect(),
            output_dir,
            summary_files,
            detail_files,
            rebalance_count: rebalance_dates.len(),
        })
    }
}

fn summarize_daily_ic(rows: &[IcObservation]) -> Vec<IcSummary> {
    let mut grouped = BTreeMap::<(String, Option<i32>), Vec<&IcObservation>>::new();
    for row in rows {
        grouped
            .entry((row.factor_id.clone(), None))
            .or_default()
            .push(row);
        grouped
            .entry((row.factor_id.clone(), Some(row.factor_date / 10_000)))
            .or_default()
            .push(row);
    }
    grouped
        .into_iter()
        .map(|((factor_id, year), rows)| {
            let ic = rows.iter().map(|row| row.ic).collect::<Vec<_>>();
            let rank_ic = rows.iter().map(|row| row.rank_ic).collect::<Vec<_>>();
            let coverage = rows.iter().map(|row| row.coverage).collect::<Vec<_>>();
            let inf_rate = rows.iter().map(|row| row.inf_rate).collect::<Vec<_>>();
            summarize_ic(&factor_id, year, None, &ic, &rank_ic, &coverage, &inf_rate)
        })
        .collect()
}

fn summarize_decay_ic(rows: &[IcObservation]) -> Vec<IcSummary> {
    let mut grouped = BTreeMap::<(String, usize, Option<i32>), Vec<&IcObservation>>::new();
    for row in rows {
        let Some(horizon) = row.horizon else {
            continue;
        };
        grouped
            .entry((row.factor_id.clone(), horizon, None))
            .or_default()
            .push(row);
        grouped
            .entry((
                row.factor_id.clone(),
                horizon,
                Some(row.factor_date / 10_000),
            ))
            .or_default()
            .push(row);
    }
    grouped
        .into_iter()
        .map(|((factor_id, horizon, year), rows)| {
            let ic = rows.iter().map(|row| row.ic).collect::<Vec<_>>();
            let rank_ic = rows.iter().map(|row| row.rank_ic).collect::<Vec<_>>();
            let coverage = rows.iter().map(|row| row.coverage).collect::<Vec<_>>();
            let inf_rate = rows.iter().map(|row| row.inf_rate).collect::<Vec<_>>();
            summarize_ic(
                &factor_id,
                year,
                Some(horizon),
                &ic,
                &rank_ic,
                &coverage,
                &inf_rate,
            )
        })
        .collect()
}

fn default_output_dir(
    config: &EngineConfig,
    request: &BacktestRunRequest,
    first_factor_id: &str,
) -> PathBuf {
    let factor_label = match (&request.factor_ids, &request.tags) {
        _ if request.all_factors => "all_factors".to_string(),
        (Some(ids), _) if ids.len() == 1 => ids[0].clone(),
        (Some(ids), _) => format!("{}_factors", ids.len()),
        (_, Some(tags)) => format!("tags_{}", tags.join("_")),
        _ => first_factor_id.to_string(),
    };
    config
        .data_root
        .join("backtest")
        .join(request.asset_class.as_str())
        .join(request.frequency.as_str())
        .join(format!(
            "{}_{}_{}_{}_g{}_{}",
            request.start_date,
            request.end_date,
            factor_label,
            request.rebalance.label(),
            request.groups,
            request.neutralize.label()
        ))
}

#[allow(dead_code)]
fn _keep_types(_: &[PerformanceSummary], _: &[FactorStatsSummary]) {}

use std::collections::BTreeMap;

use crate::backtest::data::{BacktestInput, BacktestPanel, BenchmarkKind};
use crate::backtest::ic::{daily_ic_observation_with_universe, IcObservation};
use crate::backtest::metrics::{daily_factor_stats, FactorStatsDaily, PerformancePoint};
use crate::backtest::preprocess::{
    coverage_stats_with_universe, equal_group_weights, group_assignments, keyed_values,
    long_short_weights, maybe_neutralize, portfolio_return, portfolio_scores, turnover,
};
use crate::backtest::request::BacktestRunRequest;
use crate::backtest::schedule::date_after;
use crate::error::{err, Result};
use crate::progress::ProgressBar;
use crate::storage::FactorMetadata;
use rayon::prelude::*;

#[derive(Clone, Debug, Default)]
pub struct CrossSectionBacktestOutput {
    pub returns: Vec<PerformancePoint>,
    pub daily_ic: Vec<IcObservation>,
    pub factor_stats: Vec<FactorStatsDaily>,
}

#[derive(Clone, Debug)]
struct PortfolioState {
    weights: BTreeMap<String, Vec<f64>>,
    previous_weights: BTreeMap<String, Vec<f64>>,
}

impl PortfolioState {
    fn new() -> Self {
        Self {
            weights: BTreeMap::new(),
            previous_weights: BTreeMap::new(),
        }
    }

    fn update_weights(
        &mut self,
        weights: BTreeMap<String, Vec<f64>>,
    ) -> BTreeMap<String, Option<f64>> {
        let mut turnovers = BTreeMap::new();
        for (name, current) in &weights {
            turnovers.insert(
                name.clone(),
                turnover(self.previous_weights.get(name).map(Vec::as_slice), current),
            );
        }
        self.previous_weights = weights.clone();
        self.weights = weights;
        turnovers
    }
}

#[derive(Clone, Debug)]
pub struct CrossSectionBacktestState {
    portfolio: PortfolioState,
    latest_turnovers: BTreeMap<String, Option<f64>>,
    returns: Vec<PerformancePoint>,
    daily_ic: Vec<IcObservation>,
    factor_stats: Vec<FactorStatsDaily>,
}

impl CrossSectionBacktestState {
    fn new() -> Self {
        Self {
            portfolio: PortfolioState::new(),
            latest_turnovers: BTreeMap::new(),
            returns: Vec::new(),
            daily_ic: Vec::new(),
            factor_stats: Vec::new(),
        }
    }
}

pub fn init_cross_section_states(factors: &[FactorMetadata]) -> Vec<CrossSectionBacktestState> {
    factors
        .iter()
        .map(|_| CrossSectionBacktestState::new())
        .collect()
}

pub fn run_cross_section_backtest(
    request: &BacktestRunRequest,
    input: &BacktestInput,
    rebalance_dates: &[i32],
    progress: &ProgressBar,
    thread_pool: Option<&rayon::ThreadPool>,
    batch_index: usize,
    batch_count: usize,
) -> Result<CrossSectionBacktestOutput> {
    let mut states = init_cross_section_states(&input.factor_metadata);
    update_cross_section_backtest_states(
        request,
        input,
        rebalance_dates,
        &mut states,
        thread_pool,
    )?;
    finalize_cross_section_backtest(
        states,
        &input.factor_metadata,
        progress,
        batch_index,
        batch_count,
        input.target_dates.len(),
        rebalance_dates.len(),
        request.groups,
    )
}

pub fn update_cross_section_backtest_states(
    request: &BacktestRunRequest,
    input: &BacktestInput,
    rebalance_dates: &[i32],
    states: &mut [CrossSectionBacktestState],
    thread_pool: Option<&rayon::ThreadPool>,
) -> Result<()> {
    if states.len() != input.factor_metadata.len() {
        return Err(err(format!(
            "backtest state count {} does not match factor count {}",
            states.len(),
            input.factor_metadata.len()
        )));
    }
    let rebalance_lookup = rebalance_dates
        .iter()
        .enumerate()
        .map(|(idx, date)| (*date, idx))
        .collect::<BTreeMap<_, _>>();

    let mut compute_batch = || {
        states
            .par_iter_mut()
            .zip(input.factor_metadata.par_iter())
            .map(|(state, factor)| {
                update_factor_cross_section_state(request, input, &rebalance_lookup, factor, state)
            })
            .collect::<Vec<_>>()
    };
    let results = if let Some(pool) = thread_pool {
        pool.install(compute_batch)
    } else {
        compute_batch()
    };
    for result in results {
        result?;
    }
    Ok(())
}

pub fn finalize_cross_section_backtest(
    states: Vec<CrossSectionBacktestState>,
    factors: &[FactorMetadata],
    progress: &ProgressBar,
    batch_index: usize,
    batch_count: usize,
    target_date_count: usize,
    rebalance_count: usize,
    group_count: usize,
) -> Result<CrossSectionBacktestOutput> {
    let mut output = CrossSectionBacktestOutput::default();
    for (mut state, factor) in states.into_iter().zip(factors) {
        finalize_factor_returns(&mut state, group_count);
        output.returns.extend(state.returns);
        output.daily_ic.extend(state.daily_ic);
        output.factor_stats.extend(state.factor_stats);
        progress.tick(format!(
            "batch={}/{} factor={} dates={} rebalance={}",
            batch_index, batch_count, factor.factor_id, target_date_count, rebalance_count
        ));
    }
    Ok(output)
}

fn update_factor_cross_section_state(
    request: &BacktestRunRequest,
    input: &BacktestInput,
    rebalance_lookup: &BTreeMap<i32, usize>,
    factor: &FactorMetadata,
    state: &mut CrossSectionBacktestState,
) -> Result<()> {
    for date in &input.target_dates {
        let Some(date_idx) = input.panel.date_index(*date) else {
            continue;
        };
        let raw = input.panel.cross_section(&factor.output_column, date_idx)?;
        let universe_mask = input.universe.mask_for(*date);
        let masked_raw = apply_universe_mask(&raw, universe_mask);
        let stats = coverage_stats_with_universe(&raw, universe_mask);
        state.factor_stats.push(daily_factor_stats(
            factor.factor_id.clone(),
            *date,
            &masked_raw,
            stats.coverage,
            stats.inf_rate,
        ));
        let barra = barra_cross_sections(request, input, date_idx)?;
        let groups = input
            .sectors
            .as_ref()
            .and_then(|by_date| by_date.get(date))
            .map(Vec::as_slice);
        let processed = maybe_neutralize(&masked_raw, &request.neutralize, &barra, groups);

        let label = input
            .panel
            .cross_section(&input.label_metadata.output_column, date_idx)?;
        let benchmark_return = benchmark_return(&input.benchmark.kind, *date, &label);
        let settle_date = date_after(input.panel.dates(), *date, input.label_metadata.lookahead);
        state.daily_ic.push(daily_ic_observation_with_universe(
            &factor.factor_id,
            *date,
            *date,
            settle_date,
            None,
            &processed,
            &label,
            universe_mask,
        ));

        if rebalance_lookup.contains_key(date) {
            let weights = build_portfolio_weights(&processed, request.groups);
            state.latest_turnovers = state.portfolio.update_weights(weights);
        }
        if state.portfolio.weights.is_empty() {
            continue;
        }
        let trade_date = date_after(input.panel.dates(), *date, 1);
        let settle_date = date_after(input.panel.dates(), *date, input.label_metadata.lookahead);
        for (portfolio, weights) in &state.portfolio.weights {
            let return_value = portfolio_return(weights, &label);
            let turnover = state.latest_turnovers.remove(portfolio).flatten();
            state.returns.push(PerformancePoint {
                factor_id: factor.factor_id.clone(),
                factor_date: *date,
                trade_date,
                settle_date,
                portfolio: portfolio.clone(),
                return_value,
                benchmark_return,
                excess_return: None,
                turnover,
            });
        }
    }
    Ok(())
}

fn finalize_factor_returns(state: &mut CrossSectionBacktestState, group_count: usize) {
    let long_short_sign = ic_mean_sign(&state.daily_ic);
    let selected_long_group = if long_short_sign < 0.0 {
        group_name(0)
    } else {
        group_name(group_count.saturating_sub(1))
    };
    for row in &mut state.returns {
        if row.portfolio == "long_short" {
            row.return_value = row.return_value.map(|value| value * long_short_sign);
        }
        if row.portfolio == selected_long_group {
            row.excess_return = match (row.return_value, row.benchmark_return) {
                (Some(value), Some(benchmark)) if value.is_finite() && benchmark.is_finite() => {
                    Some(value - benchmark)
                }
                _ => None,
            };
        }
    }
}

fn apply_universe_mask(values: &[Option<f64>], universe: Option<&[bool]>) -> Vec<Option<f64>> {
    let Some(universe) = universe else {
        return values.to_vec();
    };
    values
        .iter()
        .enumerate()
        .map(|(idx, value)| {
            universe
                .get(idx)
                .copied()
                .unwrap_or(false)
                .then_some(*value)
                .flatten()
        })
        .collect()
}

fn benchmark_return(kind: &BenchmarkKind, date: i32, label: &[Option<f64>]) -> Option<f64> {
    match kind {
        BenchmarkKind::MarketMean => {
            let mut sum = 0.0;
            let mut count = 0usize;
            for value in label
                .iter()
                .filter_map(|value| value.filter(|value| value.is_finite()))
            {
                sum += value;
                count += 1;
            }
            (count > 0).then_some(sum / count as f64)
        }
        BenchmarkKind::Weighted(weights_by_date) => {
            let weights = weights_by_date.get(&date)?;
            if weights.len() != label.len() {
                return None;
            }
            let mut numerator = 0.0;
            let mut denominator = 0.0;
            for (weight, value) in weights.iter().zip(label) {
                let (Some(weight), Some(value)) = (
                    weight.filter(|value| value.is_finite() && *value > 0.0),
                    value.filter(|value| value.is_finite()),
                ) else {
                    continue;
                };
                numerator += weight * value;
                denominator += weight;
            }
            (denominator > f64::EPSILON).then_some(numerator / denominator)
        }
    }
}

fn build_portfolio_weights(
    values: &[Option<f64>],
    group_count: usize,
) -> BTreeMap<String, Vec<f64>> {
    let mut output = BTreeMap::new();
    let assignments = group_assignments(values, group_count);
    let group_weights = equal_group_weights(&assignments, group_count);
    for (idx, weights) in group_weights.into_iter().enumerate() {
        output.insert(group_name(idx), weights);
    }
    let scores = portfolio_scores(values);
    output.insert("long_short".to_string(), long_short_weights(&scores));
    output
}

fn ic_mean_sign(rows: &[IcObservation]) -> f64 {
    let mut sum = 0.0;
    let mut count = 0usize;
    for row in rows {
        if let Some(value) = row.ic.filter(|value| value.is_finite()) {
            sum += value;
            count += 1;
        }
    }
    if count > 0 && sum / (count as f64) < 0.0 {
        -1.0
    } else {
        1.0
    }
}

fn group_name(group_idx: usize) -> String {
    format!("group_{}", group_idx + 1)
}

fn barra_cross_sections(
    request: &BacktestRunRequest,
    input: &BacktestInput,
    date_idx: usize,
) -> Result<Vec<Vec<Option<f64>>>> {
    request
        .neutralize
        .barra_columns()
        .iter()
        .map(|column| input.panel.cross_section(column, date_idx))
        .collect()
}

#[allow(dead_code)]
fn _weights_by_code(panel: &BacktestPanel, weights: &[f64]) -> BTreeMap<String, f64> {
    keyed_values(panel.instruments(), weights)
}

pub fn ensure_backtest_inputs(request: &BacktestRunRequest) -> Result<()> {
    if request.groups == 0 {
        return Err(err("--groups must be greater than 0"));
    }
    if request.factor_batch_size == 0 {
        return Err(err("--factor-batch-size must be greater than 0"));
    }
    if request.date_batch_size == 0 {
        return Err(err("--date-batch-size must be greater than 0"));
    }
    if matches!(request.threads, Some(0)) {
        return Err(err("--threads must be greater than 0"));
    }
    if request.universe.trim().is_empty() {
        return Err(err("--universe cannot be empty"));
    }
    if request.benchmark.trim().is_empty() {
        return Err(err("--benchmark cannot be empty"));
    }
    Ok(())
}

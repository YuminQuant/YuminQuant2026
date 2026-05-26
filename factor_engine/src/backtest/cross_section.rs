use std::collections::BTreeMap;

use crate::backtest::data::{
    BacktestInput, BacktestPanel, BenchmarkKind, IndexGroupBatch, CNE6_PRIMARY_BARRA_COLUMNS,
};
use crate::backtest::ic::{daily_ic_observation_with_universe, IcObservation};
use crate::backtest::metrics::{
    daily_factor_stats, BarraExposureRecord, FactorStatsDaily, HoldingWeight,
    IndexGroupReturnPoint, IndustryWeight, PerformancePoint,
};
use crate::backtest::preprocess::{
    coverage_stats_with_universe, equal_group_weights, group_assignments, keyed_values,
    long_short_weights, maybe_neutralize, portfolio_return, portfolio_scores, turnover,
};
use crate::backtest::request::{BacktestRunRequest, NeutralizeSpec};
use crate::backtest::schedule::date_after;
use crate::error::{err, Result};
use crate::progress::ProgressBar;
use crate::storage::FactorMetadata;
use rayon::prelude::*;

#[derive(Clone, Debug, Default)]
pub struct CrossSectionBacktestOutput {
    pub returns: Vec<PerformancePoint>,
    pub index_group_returns: Vec<IndexGroupReturnPoint>,
    pub daily_ic: Vec<IcObservation>,
    pub factor_stats: Vec<FactorStatsDaily>,
    pub holdings: Vec<HoldingWeight>,
    pub industry_weights: Vec<IndustryWeight>,
    pub barra_exposure: Vec<BarraExposureRecord>,
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
struct IndexPortfolioState {
    portfolio: PortfolioState,
    latest_turnovers: BTreeMap<String, Option<f64>>,
    factor_date: Option<i32>,
    member_counts: BTreeMap<String, i64>,
}

impl IndexPortfolioState {
    fn new() -> Self {
        Self {
            portfolio: PortfolioState::new(),
            latest_turnovers: BTreeMap::new(),
            factor_date: None,
            member_counts: BTreeMap::new(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct CrossSectionBacktestState {
    portfolio: PortfolioState,
    index_portfolios: BTreeMap<String, IndexPortfolioState>,
    latest_turnovers: BTreeMap<String, Option<f64>>,
    returns: Vec<PerformancePoint>,
    index_group_returns: Vec<IndexGroupReturnPoint>,
    daily_ic: Vec<IcObservation>,
    factor_stats: Vec<FactorStatsDaily>,
    holdings: Vec<HoldingWeight>,
    industry_weights: Vec<IndustryWeight>,
    barra_exposure: Vec<BarraExposureRecord>,
}

impl CrossSectionBacktestState {
    fn new() -> Self {
        Self {
            portfolio: PortfolioState::new(),
            index_portfolios: BTreeMap::new(),
            latest_turnovers: BTreeMap::new(),
            returns: Vec::new(),
            index_group_returns: Vec::new(),
            daily_ic: Vec::new(),
            factor_stats: Vec::new(),
            holdings: Vec::new(),
            industry_weights: Vec::new(),
            barra_exposure: Vec::new(),
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
        output.index_group_returns.extend(state.index_group_returns);
        output.daily_ic.extend(state.daily_ic);
        output.factor_stats.extend(state.factor_stats);
        output.holdings.extend(state.holdings);
        output.industry_weights.extend(state.industry_weights);
        output.barra_exposure.extend(state.barra_exposure);
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
        let trade_filter_mask = input.trade_filter.mask_for(*date);
        let eligible_mask = combined_mask(universe_mask, trade_filter_mask);
        let eligible_mask_ref = eligible_mask.as_deref();
        let masked_raw = apply_universe_mask(&raw, eligible_mask_ref);
        let stats = coverage_stats_with_universe(&raw, eligible_mask_ref);
        state.factor_stats.push(daily_factor_stats(
            factor.factor_id.clone(),
            *date,
            &masked_raw,
            stats.coverage,
            stats.inf_rate,
        ));
        let barra_exposure = named_barra_cross_sections(input, date_idx)?;
        let neutralize_barra = neutralize_barra_cross_sections(
            &request.neutralize,
            &barra_exposure,
            &factor.output_column,
        );
        let sector_groups = input
            .sectors
            .as_ref()
            .and_then(|by_date| by_date.get(date))
            .map(Vec::as_slice);
        let detail_sector_groups = input
            .detail_sectors
            .as_ref()
            .and_then(|by_date| by_date.get(date))
            .map(Vec::as_slice);
        let processed = maybe_neutralize(
            &masked_raw,
            &request.neutralize,
            &neutralize_barra,
            sector_groups,
        );
        record_barra_ic(
            &mut state.barra_exposure,
            &factor.factor_id,
            *date,
            &processed,
            &barra_exposure,
        );

        let label = input
            .panel
            .cross_section(&input.label_metadata.output_column, date_idx)?;
        let benchmark_return =
            benchmark_return(&input.benchmark.kind, *date, &label, trade_filter_mask);
        for ic_label in &input.ic_label_metadata {
            let ic_values = input
                .panel
                .cross_section(&ic_label.label.output_column, date_idx)?;
            let settle_date = date_after(input.panel.dates(), *date, ic_label.label.lookahead);
            state.daily_ic.push(daily_ic_observation_with_universe(
                &factor.factor_id,
                *date,
                *date,
                settle_date,
                Some(ic_label.horizon),
                &processed,
                &ic_values,
                eligible_mask_ref,
            ));
        }

        if rebalance_lookup.contains_key(date) {
            let weights = build_portfolio_weights(&processed, request.groups);
            record_barra_group_exposure(
                &mut state.barra_exposure,
                &factor.factor_id,
                *date,
                &weights,
                &barra_exposure,
                request.groups,
            );
            record_rebalance_detail(
                request,
                input,
                factor,
                *date,
                &weights,
                detail_sector_groups,
                input.detail_sector_source.as_deref(),
                state,
            )?;
            state.latest_turnovers = state.portfolio.update_weights(weights);
            rebalance_index_group_weights(
                state,
                input,
                factor,
                *date,
                &processed,
                &label,
                eligible_mask_ref,
            );
        }
        if state.portfolio.weights.is_empty() {
            record_index_group_returns(state, input, factor, *date, &label);
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
        record_index_group_returns(state, input, factor, *date, &label);
    }
    Ok(())
}

fn finalize_factor_returns(state: &mut CrossSectionBacktestState, _group_count: usize) {
    let long_short_sign = rank_ic_mean_sign(&state.daily_ic);
    let selected_portfolio = selected_long_group(long_short_sign, _group_count);
    for row in &mut state.returns {
        if row.portfolio == "long_short" {
            row.return_value = row.return_value.map(|value| value * long_short_sign);
        }
        if row.portfolio.starts_with("group_") {
            row.excess_return = match (row.return_value, row.benchmark_return) {
                (Some(value), Some(benchmark)) if value.is_finite() && benchmark.is_finite() => {
                    Some(value - benchmark)
                }
                _ => None,
            };
        }
    }
    state
        .holdings
        .retain(|row| row.portfolio == selected_portfolio);
    for row in &mut state.holdings {
        row.rank_ic_sign = long_short_sign;
    }
    state
        .industry_weights
        .retain(|row| row.portfolio == selected_portfolio);
    for row in &mut state.industry_weights {
        row.rank_ic_sign = long_short_sign;
    }
    finalize_barra_exposure(state, &selected_portfolio, long_short_sign);
}

fn selected_long_group(rank_ic_sign: f64, group_count: usize) -> String {
    if rank_ic_sign < 0.0 {
        group_name(0)
    } else {
        group_name(group_count.saturating_sub(1))
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

fn combined_mask(left: Option<&[bool]>, right: Option<&[bool]>) -> Option<Vec<bool>> {
    match (left, right) {
        (Some(left), Some(right)) => Some(
            left.iter()
                .enumerate()
                .map(|(idx, value)| *value && right.get(idx).copied().unwrap_or(false))
                .collect(),
        ),
        (Some(values), None) | (None, Some(values)) => Some(values.to_vec()),
        (None, None) => None,
    }
}

fn benchmark_return(
    kind: &BenchmarkKind,
    date: i32,
    label: &[Option<f64>],
    trade_filter: Option<&[bool]>,
) -> Option<f64> {
    match kind {
        BenchmarkKind::MarketMean => {
            let mut sum = 0.0;
            let mut count = 0usize;
            for (idx, value) in label.iter().enumerate() {
                if trade_filter.is_some_and(|mask| !mask.get(idx).copied().unwrap_or(false)) {
                    continue;
                }
                let Some(value) = value.filter(|value| value.is_finite()) else {
                    continue;
                };
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

fn rebalance_index_group_weights(
    state: &mut CrossSectionBacktestState,
    input: &BacktestInput,
    _factor: &FactorMetadata,
    rebalance_date: i32,
    processed: &[Option<f64>],
    label: &[Option<f64>],
    eligible_mask: Option<&[bool]>,
) {
    for index in &input.index_groups {
        let weights =
            build_index_group_weights(index, rebalance_date, processed, label, eligible_mask);
        let member_counts = group_member_counts(&weights);
        let entry = state
            .index_portfolios
            .entry(index.id.clone())
            .or_insert_with(IndexPortfolioState::new);
        entry.factor_date = Some(rebalance_date);
        entry.member_counts = member_counts;
        entry.latest_turnovers = entry.portfolio.update_weights(weights);
    }
}

fn build_index_group_weights(
    index: &IndexGroupBatch,
    date: i32,
    processed: &[Option<f64>],
    label: &[Option<f64>],
    eligible_mask: Option<&[bool]>,
) -> BTreeMap<String, Vec<f64>> {
    let mut values = vec![None; processed.len()];
    let index_weights = index.weights_for(date);
    for idx in 0..processed.len() {
        if eligible_mask.is_some_and(|mask| !mask.get(idx).copied().unwrap_or(false)) {
            continue;
        }
        let in_index = index_weights
            .and_then(|weights| weights.get(idx))
            .copied()
            .flatten()
            .is_some_and(|value| value.is_finite() && value > 0.0);
        if !in_index {
            continue;
        }
        if !label
            .get(idx)
            .copied()
            .flatten()
            .is_some_and(f64::is_finite)
        {
            continue;
        }
        values[idx] = processed[idx].filter(|value| value.is_finite());
    }
    let assignments = group_assignments(&values, INDEX_GROUP_COUNT);
    let group_weights = equal_group_weights(&assignments, INDEX_GROUP_COUNT);
    group_weights
        .into_iter()
        .enumerate()
        .map(|(idx, weights)| (group_name(idx), weights))
        .collect()
}

fn group_member_counts(weights: &BTreeMap<String, Vec<f64>>) -> BTreeMap<String, i64> {
    weights
        .iter()
        .map(|(portfolio, weights)| {
            (
                portfolio.clone(),
                weights
                    .iter()
                    .filter(|weight| weight.abs() > f64::EPSILON)
                    .count() as i64,
            )
        })
        .collect()
}

fn record_index_group_returns(
    state: &mut CrossSectionBacktestState,
    input: &BacktestInput,
    factor: &FactorMetadata,
    date: i32,
    label: &[Option<f64>],
) {
    let trade_date = date_after(input.panel.dates(), date, 1);
    let settle_date = date_after(input.panel.dates(), date, input.label_metadata.lookahead);
    for index in &input.index_groups {
        let Some(portfolio_state) = state.index_portfolios.get_mut(&index.id) else {
            continue;
        };
        let Some(factor_date) = portfolio_state.factor_date else {
            continue;
        };
        let (benchmark_return, benchmark_count) = index_benchmark_return(index, date, label);
        for (portfolio, weights) in &portfolio_state.portfolio.weights {
            let member_count = portfolio_state
                .member_counts
                .get(portfolio)
                .copied()
                .unwrap_or(0);
            let return_value = (member_count > 0)
                .then(|| portfolio_return(weights, label))
                .flatten();
            let benchmark_value = (member_count > 0).then_some(benchmark_return).flatten();
            let excess_return = match (return_value, benchmark_value) {
                (Some(value), Some(benchmark)) if value.is_finite() && benchmark.is_finite() => {
                    Some(value - benchmark)
                }
                _ => None,
            };
            let turnover = if date == factor_date {
                portfolio_state.latest_turnovers.remove(portfolio).flatten()
            } else {
                Some(0.0)
            };
            state.index_group_returns.push(IndexGroupReturnPoint {
                factor_id: factor.factor_id.clone(),
                index_id: index.id.clone(),
                factor_date,
                trade_date,
                settle_date,
                portfolio: portfolio.clone(),
                return_value,
                benchmark_return: benchmark_value,
                excess_return,
                turnover,
                member_count,
                benchmark_count,
            });
        }
    }
}

fn index_benchmark_return(
    index: &IndexGroupBatch,
    date: i32,
    label: &[Option<f64>],
) -> (Option<f64>, i64) {
    let Some(weights) = index.weights_for(date) else {
        return (None, 0);
    };
    if weights.len() != label.len() {
        return (None, 0);
    }
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    let mut pair_count = 0i64;
    for (weight, value) in weights.iter().zip(label) {
        let (Some(weight), Some(value)) = (
            weight.filter(|value| value.is_finite() && *value > 0.0),
            value.filter(|value| value.is_finite()),
        ) else {
            continue;
        };
        numerator += weight * value;
        denominator += weight;
        pair_count += 1;
    }
    (
        (denominator > f64::EPSILON).then_some(numerator / denominator),
        pair_count,
    )
}

fn record_rebalance_detail(
    request: &BacktestRunRequest,
    input: &BacktestInput,
    factor: &FactorMetadata,
    rebalance_date: i32,
    weights: &BTreeMap<String, Vec<f64>>,
    detail_sector_groups: Option<&[Option<String>]>,
    detail_sector_source: Option<&str>,
    state: &mut CrossSectionBacktestState,
) -> Result<()> {
    if !request.detail.any() {
        return Ok(());
    }
    let mut endpoint_groups = vec![group_name(0), group_name(request.groups.saturating_sub(1))];
    endpoint_groups.sort();
    endpoint_groups.dedup();
    for portfolio in endpoint_groups {
        let Some(group_weights) = weights.get(&portfolio) else {
            continue;
        };
        if request.detail.holdings {
            record_holding_weights(
                &mut state.holdings,
                &factor.factor_id,
                rebalance_date,
                &portfolio,
                input.panel.instruments(),
                group_weights,
            );
        }
        if request.detail.industry_weights {
            let sectors = detail_sector_groups.ok_or_else(|| {
                err("--detail industry_weights requires configured level-1 sector data")
            })?;
            let sector_source = detail_sector_source.unwrap_or(request.detail_sector.label());
            record_industry_weights(
                &mut state.industry_weights,
                &factor.factor_id,
                rebalance_date,
                &portfolio,
                sector_source,
                sectors,
                group_weights,
            );
        }
    }
    Ok(())
}

fn record_holding_weights(
    output: &mut Vec<HoldingWeight>,
    factor_id: &str,
    rebalance_date: i32,
    portfolio: &str,
    instruments: &[String],
    weights: &[f64],
) {
    for (idx, weight) in weights.iter().enumerate() {
        if weight.abs() <= f64::EPSILON {
            continue;
        }
        let Some(ts_code) = instruments.get(idx) else {
            continue;
        };
        output.push(HoldingWeight {
            factor_id: factor_id.to_string(),
            rebalance_date,
            portfolio: portfolio.to_string(),
            rank_ic_sign: 0.0,
            ts_code: ts_code.clone(),
            weight: *weight,
        });
    }
}

fn record_industry_weights(
    output: &mut Vec<IndustryWeight>,
    factor_id: &str,
    rebalance_date: i32,
    portfolio: &str,
    sector_source: &str,
    sectors: &[Option<String>],
    weights: &[f64],
) {
    let mut grouped = BTreeMap::<String, (f64, i64)>::new();
    for (idx, weight) in weights.iter().enumerate() {
        if weight.abs() <= f64::EPSILON {
            continue;
        }
        let sector = sectors
            .get(idx)
            .and_then(|value| value.as_ref())
            .filter(|value| !value.trim().is_empty())
            .cloned()
            .unwrap_or_else(|| "__MISSING__".to_string());
        let entry = grouped.entry(sector).or_insert((0.0, 0));
        entry.0 += *weight;
        entry.1 += 1;
    }
    for (sector_code, (weight, stock_count)) in grouped {
        output.push(IndustryWeight {
            factor_id: factor_id.to_string(),
            rebalance_date,
            portfolio: portfolio.to_string(),
            rank_ic_sign: 0.0,
            sector_source: sector_source.to_string(),
            sector_code,
            weight,
            stock_count,
        });
    }
}

const BARRA_IC_METRIC: &str = "barra_ic";
const BARRA_IC_MEAN_METRIC: &str = "barra_ic_mean";
const LONG_GROUP_EXPOSURE_METRIC: &str = "long_group_exposure";
const INDEX_GROUP_COUNT: usize = 5;

fn record_barra_ic(
    output: &mut Vec<BarraExposureRecord>,
    factor_id: &str,
    trade_date: i32,
    factor_values: &[Option<f64>],
    barra: &[(String, Vec<Option<f64>>)],
) {
    for (barra_factor, exposure) in barra {
        if !is_cne6_primary_barra_factor(barra_factor) {
            continue;
        }
        let (value, pair_count) = pearson_corr_with_count(factor_values, exposure);
        output.push(BarraExposureRecord {
            factor_id: factor_id.to_string(),
            trade_date: Some(trade_date),
            metric: BARRA_IC_METRIC.to_string(),
            barra_factor: barra_factor.clone(),
            selected_group: None,
            rank_ic_sign: None,
            value,
            pair_count: Some(pair_count as i64),
        });
    }
}

fn record_barra_group_exposure(
    output: &mut Vec<BarraExposureRecord>,
    factor_id: &str,
    trade_date: i32,
    weights: &BTreeMap<String, Vec<f64>>,
    barra: &[(String, Vec<Option<f64>>)],
    group_count: usize,
) {
    let mut endpoint_groups = vec![group_name(0), group_name(group_count.saturating_sub(1))];
    endpoint_groups.sort();
    endpoint_groups.dedup();
    for portfolio in endpoint_groups {
        let Some(group_weights) = weights.get(&portfolio) else {
            continue;
        };
        for (barra_factor, exposure) in barra {
            if !is_cne6_primary_barra_factor(barra_factor) {
                continue;
            }
            let (value, pair_count) = weighted_exposure_mean(group_weights, exposure);
            output.push(BarraExposureRecord {
                factor_id: factor_id.to_string(),
                trade_date: Some(trade_date),
                metric: LONG_GROUP_EXPOSURE_METRIC.to_string(),
                barra_factor: barra_factor.clone(),
                selected_group: Some(portfolio.clone()),
                rank_ic_sign: None,
                value,
                pair_count: Some(pair_count as i64),
            });
        }
    }
}

fn finalize_barra_exposure(
    state: &mut CrossSectionBacktestState,
    selected_portfolio: &str,
    rank_ic_sign: f64,
) {
    let mut ic_sums = BTreeMap::<String, (f64, usize)>::new();
    for row in &state.barra_exposure {
        if row.metric != BARRA_IC_METRIC {
            continue;
        }
        let Some(value) = row.value.filter(|value| value.is_finite()) else {
            continue;
        };
        let entry = ic_sums.entry(row.barra_factor.clone()).or_insert((0.0, 0));
        entry.0 += value;
        entry.1 += 1;
    }
    let factor_id = state
        .barra_exposure
        .first()
        .map(|row| row.factor_id.clone());
    if let Some(factor_id) = factor_id {
        for (barra_factor, (sum, count)) in ic_sums {
            state.barra_exposure.push(BarraExposureRecord {
                factor_id: factor_id.clone(),
                trade_date: None,
                metric: BARRA_IC_MEAN_METRIC.to_string(),
                barra_factor,
                selected_group: None,
                rank_ic_sign: None,
                value: (count > 0).then_some(sum / count as f64),
                pair_count: Some(count as i64),
            });
        }
    }
    state.barra_exposure.retain(|row| {
        row.metric != LONG_GROUP_EXPOSURE_METRIC
            || row.selected_group.as_deref() == Some(selected_portfolio)
    });
    for row in &mut state.barra_exposure {
        if row.metric == LONG_GROUP_EXPOSURE_METRIC {
            row.rank_ic_sign = Some(rank_ic_sign);
        }
    }
}

fn is_cne6_primary_barra_factor(column: &str) -> bool {
    CNE6_PRIMARY_BARRA_COLUMNS.contains(&column)
}

fn pearson_corr_with_count(x: &[Option<f64>], y: &[Option<f64>]) -> (Option<f64>, usize) {
    if x.len() != y.len() {
        return (None, 0);
    }
    let pairs = x
        .iter()
        .zip(y)
        .filter_map(|(x, y)| {
            match (
                x.filter(|value| value.is_finite()),
                y.filter(|value| value.is_finite()),
            ) {
                (Some(x), Some(y)) => Some((x, y)),
                _ => None,
            }
        })
        .collect::<Vec<_>>();
    let pair_count = pairs.len();
    if pair_count < 2 {
        return (None, pair_count);
    }
    let mean_x = pairs.iter().map(|(x, _)| *x).sum::<f64>() / pair_count as f64;
    let mean_y = pairs.iter().map(|(_, y)| *y).sum::<f64>() / pair_count as f64;
    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;
    for (x, y) in pairs {
        let dx = x - mean_x;
        let dy = y - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }
    if var_x <= f64::EPSILON || var_y <= f64::EPSILON {
        return (None, pair_count);
    }
    (Some(cov / (var_x.sqrt() * var_y.sqrt())), pair_count)
}

fn weighted_exposure_mean(weights: &[f64], exposure: &[Option<f64>]) -> (Option<f64>, usize) {
    if weights.len() != exposure.len() {
        return (None, 0);
    }
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    let mut pair_count = 0usize;
    for (weight, value) in weights.iter().zip(exposure) {
        if weight.abs() <= f64::EPSILON {
            continue;
        }
        let Some(value) = value.filter(|value| value.is_finite()) else {
            continue;
        };
        numerator += *weight * value;
        denominator += *weight;
        pair_count += 1;
    }
    if denominator.abs() <= f64::EPSILON {
        return (None, pair_count);
    }
    (Some(numerator / denominator), pair_count)
}

fn rank_ic_mean_sign(rows: &[IcObservation]) -> f64 {
    let mut sum = 0.0;
    let mut count = 0usize;
    for row in rows {
        if !matches!(row.horizon, None | Some(1)) {
            continue;
        }
        if let Some(value) = row.rank_ic.filter(|value| value.is_finite()) {
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

fn named_barra_cross_sections(
    input: &BacktestInput,
    date_idx: usize,
) -> Result<Vec<(String, Vec<Option<f64>>)>> {
    input
        .barra_columns
        .iter()
        .map(|column| Ok((column.clone(), input.panel.cross_section(column, date_idx)?)))
        .collect()
}

fn neutralize_barra_cross_sections(
    spec: &NeutralizeSpec,
    barra: &[(String, Vec<Option<f64>>)],
    target_column: &str,
) -> Vec<Vec<Option<f64>>> {
    match spec {
        NeutralizeSpec::Barra { .. } => barra
            .iter()
            .filter(|(column, _)| column.as_str() != target_column)
            .map(|(_, values)| values.clone())
            .collect(),
        _ => Vec::new(),
    }
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

#[cfg(test)]
mod tests {
    use super::{
        finalize_factor_returns, pearson_corr_with_count, record_barra_group_exposure,
        record_industry_weights, weighted_exposure_mean, CrossSectionBacktestState,
        BARRA_IC_MEAN_METRIC, BARRA_IC_METRIC, LONG_GROUP_EXPOSURE_METRIC,
    };
    use crate::backtest::ic::IcObservation;
    use crate::backtest::metrics::{
        BarraExposureRecord, HoldingWeight, IndustryWeight, PerformancePoint,
    };
    use std::collections::BTreeMap;

    fn point(portfolio: &str, return_value: f64, benchmark_return: f64) -> PerformancePoint {
        PerformancePoint {
            factor_id: "factor_a".to_string(),
            factor_date: 20260424,
            trade_date: Some(20260427),
            settle_date: Some(20260428),
            portfolio: portfolio.to_string(),
            return_value: Some(return_value),
            benchmark_return: Some(benchmark_return),
            excess_return: None,
            turnover: None,
        }
    }

    #[test]
    fn finalize_writes_excess_return_for_every_group() {
        let mut state = CrossSectionBacktestState::new();
        state.daily_ic.push(IcObservation {
            factor_id: "factor_a".to_string(),
            factor_date: 20260424,
            label_date: 20260424,
            settle_date: Some(20260428),
            horizon: None,
            ic: Some(0.1),
            rank_ic: Some(-0.1),
            pair_count: 10,
            coverage: 1.0,
            inf_rate: 0.0,
        });
        state.returns = vec![
            point("group_1", 0.01, 0.005),
            point("group_2", -0.02, 0.005),
            point("long_short", 0.03, 0.005),
        ];

        finalize_factor_returns(&mut state, 2);

        assert_eq!(state.returns[0].excess_return, Some(0.005));
        assert_eq!(state.returns[1].excess_return, Some(-0.025));
        assert_eq!(state.returns[2].excess_return, None);
        assert_eq!(state.returns[2].return_value, Some(-0.03));
    }

    #[test]
    fn finalize_keeps_rank_ic_selected_long_side_detail_only() {
        let mut state = CrossSectionBacktestState::new();
        state.daily_ic.push(IcObservation {
            factor_id: "factor_a".to_string(),
            factor_date: 20260424,
            label_date: 20260424,
            settle_date: Some(20260428),
            horizon: None,
            ic: Some(0.1),
            rank_ic: Some(-0.1),
            pair_count: 10,
            coverage: 1.0,
            inf_rate: 0.0,
        });
        state.holdings = vec![
            HoldingWeight {
                factor_id: "factor_a".to_string(),
                rebalance_date: 20260424,
                portfolio: "group_1".to_string(),
                rank_ic_sign: 0.0,
                ts_code: "000001.SZ".to_string(),
                weight: 1.0,
            },
            HoldingWeight {
                factor_id: "factor_a".to_string(),
                rebalance_date: 20260424,
                portfolio: "group_2".to_string(),
                rank_ic_sign: 0.0,
                ts_code: "000002.SZ".to_string(),
                weight: 1.0,
            },
        ];
        state.industry_weights = vec![
            IndustryWeight {
                factor_id: "factor_a".to_string(),
                rebalance_date: 20260424,
                portfolio: "group_1".to_string(),
                rank_ic_sign: 0.0,
                sector_source: "sw_l1".to_string(),
                sector_code: "801010".to_string(),
                weight: 1.0,
                stock_count: 1,
            },
            IndustryWeight {
                factor_id: "factor_a".to_string(),
                rebalance_date: 20260424,
                portfolio: "group_2".to_string(),
                rank_ic_sign: 0.0,
                sector_source: "sw_l1".to_string(),
                sector_code: "801020".to_string(),
                weight: 1.0,
                stock_count: 1,
            },
        ];

        finalize_factor_returns(&mut state, 2);

        assert_eq!(state.holdings.len(), 1);
        assert_eq!(state.holdings[0].portfolio, "group_1");
        assert_eq!(state.holdings[0].rank_ic_sign, -1.0);
        assert_eq!(state.industry_weights.len(), 1);
        assert_eq!(state.industry_weights[0].portfolio, "group_1");
        assert_eq!(state.industry_weights[0].rank_ic_sign, -1.0);
    }

    #[test]
    fn industry_weights_keep_missing_sector_bucket() {
        let mut rows = Vec::new();
        let sectors = vec![Some("801010".to_string()), None, Some("801010".to_string())];
        let weights = vec![0.25, 0.5, 0.25];

        record_industry_weights(
            &mut rows, "factor_a", 20260424, "group_2", "ci_l1", &sectors, &weights,
        );

        assert_eq!(rows.len(), 2);
        let total_weight = rows.iter().map(|row| row.weight).sum::<f64>();
        assert!((total_weight - 1.0).abs() < 1e-12);
        let missing = rows
            .iter()
            .find(|row| row.sector_code == "__MISSING__")
            .expect("missing sector row");
        assert_eq!(missing.sector_source, "ci_l1");
        assert_eq!(missing.weight, 0.5);
        assert_eq!(missing.stock_count, 1);
    }

    #[test]
    fn barra_pearson_ic_skips_missing_and_constant_values() {
        let factor = vec![Some(1.0), Some(2.0), None, Some(4.0)];
        let barra = vec![Some(2.0), Some(4.0), Some(10.0), Some(8.0)];
        let (ic, pair_count) = pearson_corr_with_count(&factor, &barra);

        assert_eq!(pair_count, 3);
        assert!(ic.unwrap() > 0.99);

        let constant = vec![Some(1.0), Some(1.0), Some(1.0)];
        let label = vec![Some(1.0), Some(2.0), Some(3.0)];
        let (ic, pair_count) = pearson_corr_with_count(&constant, &label);
        assert_eq!(pair_count, 3);
        assert_eq!(ic, None);
    }

    #[test]
    fn weighted_exposure_mean_renormalizes_finite_contributors() {
        let weights = vec![0.5, 0.5, 0.0];
        let exposure = vec![Some(1.0), None, Some(100.0)];
        let (value, pair_count) = weighted_exposure_mean(&weights, &exposure);

        assert_eq!(pair_count, 1);
        assert_eq!(value, Some(1.0));
    }

    #[test]
    fn finalize_barra_exposure_keeps_rank_ic_selected_long_side_and_adds_means() {
        let mut state = CrossSectionBacktestState::new();
        state.daily_ic.push(IcObservation {
            factor_id: "factor_a".to_string(),
            factor_date: 20260424,
            label_date: 20260424,
            settle_date: Some(20260428),
            horizon: None,
            ic: Some(0.1),
            rank_ic: Some(-0.1),
            pair_count: 10,
            coverage: 1.0,
            inf_rate: 0.0,
        });
        state.barra_exposure = vec![
            BarraExposureRecord {
                factor_id: "factor_a".to_string(),
                trade_date: Some(20260424),
                metric: BARRA_IC_METRIC.to_string(),
                barra_factor: "SIZE".to_string(),
                selected_group: None,
                rank_ic_sign: None,
                value: Some(0.2),
                pair_count: Some(100),
            },
            BarraExposureRecord {
                factor_id: "factor_a".to_string(),
                trade_date: Some(20260425),
                metric: BARRA_IC_METRIC.to_string(),
                barra_factor: "SIZE".to_string(),
                selected_group: None,
                rank_ic_sign: None,
                value: Some(0.4),
                pair_count: Some(100),
            },
            BarraExposureRecord {
                factor_id: "factor_a".to_string(),
                trade_date: Some(20260424),
                metric: LONG_GROUP_EXPOSURE_METRIC.to_string(),
                barra_factor: "SIZE".to_string(),
                selected_group: Some("group_1".to_string()),
                rank_ic_sign: None,
                value: Some(-0.3),
                pair_count: Some(10),
            },
            BarraExposureRecord {
                factor_id: "factor_a".to_string(),
                trade_date: Some(20260424),
                metric: LONG_GROUP_EXPOSURE_METRIC.to_string(),
                barra_factor: "SIZE".to_string(),
                selected_group: Some("group_2".to_string()),
                rank_ic_sign: None,
                value: Some(0.3),
                pair_count: Some(10),
            },
        ];

        finalize_factor_returns(&mut state, 2);

        let selected = state
            .barra_exposure
            .iter()
            .filter(|row| row.metric == LONG_GROUP_EXPOSURE_METRIC)
            .collect::<Vec<_>>();
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].selected_group.as_deref(), Some("group_1"));
        assert_eq!(selected[0].rank_ic_sign, Some(-1.0));

        let mean = state
            .barra_exposure
            .iter()
            .find(|row| row.metric == BARRA_IC_MEAN_METRIC && row.barra_factor == "SIZE")
            .expect("mean row");
        assert!((mean.value.unwrap() - 0.3).abs() < 1e-12);
        assert_eq!(mean.pair_count, Some(2));
    }

    #[test]
    fn record_barra_group_exposure_records_only_endpoint_groups() {
        let mut weights = BTreeMap::new();
        weights.insert("group_1".to_string(), vec![1.0, 0.0]);
        weights.insert("group_2".to_string(), vec![0.0, 1.0]);
        weights.insert("long_short".to_string(), vec![-0.5, 0.5]);
        let barra = vec![("SIZE".to_string(), vec![Some(-1.0), Some(2.0)])];
        let mut rows = Vec::new();

        record_barra_group_exposure(&mut rows, "factor_a", 20260424, &weights, &barra, 2);

        assert_eq!(rows.len(), 2);
        assert!(rows
            .iter()
            .all(|row| row.metric == LONG_GROUP_EXPOSURE_METRIC));
        assert!(rows
            .iter()
            .all(|row| row.selected_group.as_deref() != Some("long_short")));
    }
}

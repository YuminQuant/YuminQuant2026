use std::collections::BTreeMap;

use crate::backtest::data::{BacktestInput, BacktestPanel};
use crate::backtest::ic::{daily_ic_observation_with_universe, IcObservation};
use crate::backtest::metrics::{FactorStatsDaily, PerformancePoint};
use crate::backtest::preprocess::{
    coverage_stats_with_universe, equal_group_weights, group_assignments, keyed_values,
    long_short_weights, maybe_neutralize, portfolio_return, portfolio_scores, turnover,
};
use crate::backtest::request::BacktestRunRequest;
use crate::backtest::schedule::date_after;
use crate::error::{err, Result};
use crate::progress::ProgressBar;

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
    nav: BTreeMap<String, f64>,
}

impl PortfolioState {
    fn new(group_count: usize) -> Self {
        let mut nav = BTreeMap::new();
        for group_idx in 0..group_count {
            nav.insert(group_name(group_idx), 1.0);
        }
        nav.insert("long_short".to_string(), 1.0);
        Self {
            weights: BTreeMap::new(),
            previous_weights: BTreeMap::new(),
            nav,
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

pub fn run_cross_section_backtest(
    request: &BacktestRunRequest,
    input: &BacktestInput,
    rebalance_dates: &[i32],
    progress: &ProgressBar,
) -> Result<CrossSectionBacktestOutput> {
    let mut output = CrossSectionBacktestOutput::default();
    let rebalance_lookup = rebalance_dates
        .iter()
        .enumerate()
        .map(|(idx, date)| (*date, idx))
        .collect::<BTreeMap<_, _>>();

    for factor in &input.factor_metadata {
        let mut processed_by_date = BTreeMap::<i32, Vec<Option<f64>>>::new();
        let daily_ic_start = output.daily_ic.len();
        for date in &input.target_dates {
            let Some(date_idx) = input.panel.date_index(*date) else {
                continue;
            };
            let raw = input.panel.cross_section(&factor.output_column, date_idx)?;
            let factor_universe = input
                .panel
                .cross_section_presence(&factor.output_column, date_idx)?;
            let stats = coverage_stats_with_universe(&raw, Some(&factor_universe));
            output.factor_stats.push(FactorStatsDaily {
                factor_id: factor.factor_id.clone(),
                trade_date: *date,
                values: raw.clone(),
                coverage: stats.coverage,
                inf_rate: stats.inf_rate,
            });
            let barra = barra_cross_sections(request, input, date_idx)?;
            let groups = input
                .sectors
                .as_ref()
                .and_then(|by_date| by_date.get(date))
                .map(Vec::as_slice);
            let processed = maybe_neutralize(&raw, &request.neutralize, &barra, groups);
            processed_by_date.insert(*date, processed.clone());

            let label = input
                .panel
                .cross_section(&input.label_metadata.output_column, date_idx)?;
            let settle_date =
                date_after(input.panel.dates(), *date, input.label_metadata.lookahead);
            output.daily_ic.push(daily_ic_observation_with_universe(
                &factor.factor_id,
                *date,
                *date,
                settle_date,
                None,
                &processed,
                &label,
                Some(&factor_universe),
            ));
        }
        let long_short_sign = ic_mean_sign(&output.daily_ic[daily_ic_start..]);

        let mut state = PortfolioState::new(request.groups);
        let mut latest_turnovers = BTreeMap::<String, Option<f64>>::new();
        for date in &input.target_dates {
            if rebalance_lookup.contains_key(date) {
                if let Some(scores) = processed_by_date.get(date) {
                    let weights = build_portfolio_weights(scores, request.groups);
                    latest_turnovers = state.update_weights(weights);
                }
            }
            if state.weights.is_empty() {
                continue;
            }
            let Some(date_idx) = input.panel.date_index(*date) else {
                continue;
            };
            let label = input
                .panel
                .cross_section(&input.label_metadata.output_column, date_idx)?;
            let trade_date = date_after(input.panel.dates(), *date, 1);
            let settle_date =
                date_after(input.panel.dates(), *date, input.label_metadata.lookahead);
            for (portfolio, weights) in &state.weights {
                let mut return_value = portfolio_return(weights, &label);
                if portfolio == "long_short" {
                    return_value = return_value.map(|value| value * long_short_sign);
                }
                let nav = state.nav.entry(portfolio.clone()).or_insert(1.0);
                if let Some(value) = return_value {
                    *nav *= 1.0 + value;
                }
                let turnover = latest_turnovers.remove(portfolio).flatten();
                output.returns.push(PerformancePoint {
                    factor_id: factor.factor_id.clone(),
                    factor_date: *date,
                    trade_date,
                    settle_date,
                    portfolio: portfolio.clone(),
                    return_value,
                    nav: Some(*nav),
                    turnover,
                });
            }
        }
        progress.tick(format!(
            "factor={} dates={} rebalance={}",
            factor.factor_id,
            input.target_dates.len(),
            rebalance_dates.len()
        ));
    }

    Ok(output)
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
    Ok(())
}

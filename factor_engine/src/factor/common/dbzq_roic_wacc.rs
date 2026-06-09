use std::collections::{BTreeMap, VecDeque};

use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::financial::previous_quarter_end_date;
use crate::factor::common::stock_daily_ops::{is_bj_stock, neutralize_size_sector};
use crate::factor::common::{
    ClassificationLevel, ClassificationMap, DailyPanel, EventDrivenCrossSectionCache,
    FinancialPitReader, PanelColumn, ReportTypePreference,
};

pub const PROVIDER_KEY: &str = "stock|daily|dbzq_roic_wacc";
pub const UNEXPECTED_ROIC_WACC_ID: &str = "unexpected_roic_wacc";
pub const ROIC_WACC_STATE_GROWTH_ID: &str = "roic_wacc_state_growth";

const VERSION: &str = "0.1.0";
const DAILY_LOOKBACK: usize = 820;
const CAPM_LOOKBACK_WEEKS: usize = 156;
const CAPM_MIN_PERIODS: usize = 104;
const HISTORY_KEEP_QUARTERS: usize = 13;
const UNEXPECTED_HISTORY_QUARTERS: usize = 8;
const EPS: f64 = 1e-12;

const INCOME_COLUMNS: [&str; 7] = [
    "operate_profit",
    "sell_exp",
    "admin_exp",
    "fin_exp",
    "income_tax",
    "n_income",
    "int_exp",
];
const BALANCE_COLUMNS: [&str; 6] = [
    "total_hldr_eqy_exc_min_int",
    "st_borr",
    "non_cur_liab_due_1y",
    "lt_borr",
    "bond_payable",
    "lease_liab",
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RoicWaccOutput {
    UnexpectedRoicWacc,
    StateGrowth,
}

#[derive(Default)]
pub struct RoicWaccComputeState {
    stock_states: BTreeMap<String, StockRoicWaccState>,
    final_cache: EventDrivenCrossSectionCache,
    last_processed_trade_date: Option<i32>,
}

#[derive(Clone, Copy, Debug, Default)]
struct RoicWaccNeeds {
    unexpected: bool,
    state_growth: bool,
}

#[derive(Clone, Debug, Default)]
struct StockRoicWaccState {
    quarters: VecDeque<QuarterMetrics>,
}

#[derive(Clone, Copy, Debug)]
struct QuarterMetrics {
    end_date: i32,
    noplat: f64,
    roic: f64,
    spread: f64,
}

#[derive(Clone, Copy, Debug)]
struct QuarterPrelim {
    tax: f64,
    debt: f64,
    equity: f64,
    noplat: f64,
    roic: f64,
    rd: f64,
    wd: f64,
    we: f64,
    re_accounting: Option<f64>,
}

#[derive(Clone, Copy, Debug)]
struct CapmStats {
    alpha: f64,
    beta: f64,
    mean_market_week: f64,
    re_self: f64,
}

#[derive(Clone, Debug)]
struct EventInputs {
    adj_close: PanelColumn,
    dv_ratio: PanelColumn,
    subindustry_map: ClassificationMap,
}

pub fn spec(output: RoicWaccOutput) -> FactorSpec {
    let (id, aliases, description) = match output {
        RoicWaccOutput::UnexpectedRoicWacc => (
            UNEXPECTED_ROIC_WACC_ID,
            vec![
                "Unexpected ROIC-WACC".to_string(),
                "Unexpected Spread".to_string(),
            ],
            "DBZQ unexpected ROIC-WACC factor. It recomputes on fixed disclosure checkpoints 0430/0831/1031, uses PIT quarterly ROIC-WACC, standardizes current spread against the previous eight quarterly spreads, replays between checkpoints, and neutralizes by Barra SIZE and SW sector.",
        ),
        RoicWaccOutput::StateGrowth => (
            ROIC_WACC_STATE_GROWTH_ID,
            vec!["ROIC-WACC State Growth".to_string()],
            "DBZQ ROIC-WACC state-domain growth factor. It recomputes on fixed disclosure checkpoints 0430/0831/1031, uses NOPLAT growth when ROIC-WACC is positive and ROIC growth otherwise, replays between checkpoints, and neutralizes by Barra SIZE and SW sector.",
        ),
    };
    FactorSpec {
        id: id.to_string(),
        aliases,
        name: id.to_string(),
        asset_class: AssetClass::Stock,
        frequency: Frequency::Daily,
        version: VERSION.to_string(),
        tags: tags(),
        description: description.to_string(),
        dependencies: dependencies(),
        intraday_raw_dependencies: Vec::new(),
        lookback: Lookback {
            trading_days: DAILY_LOOKBACK,
        },
    }
}

pub fn compute_requested(
    requested_ids: &[String],
    context: &FactorContext,
    data: &DataPool,
) -> Result<Vec<FactorSeries>> {
    let mut state = RoicWaccComputeState::default();
    compute_requested_stateful(requested_ids, context, data, &mut state)
}

pub fn compute_requested_stateful(
    requested_ids: &[String],
    context: &FactorContext,
    data: &DataPool,
    state: &mut RoicWaccComputeState,
) -> Result<Vec<FactorSeries>> {
    let needs = needs_from_requested(requested_ids);
    if !needs.unexpected && !needs.state_growth {
        return Ok(Vec::new());
    }

    let panel = data.daily_panel(DatasetId::StockDailyPv)?;
    let income = data.financial_reader(
        DatasetId::StockIncome,
        ReportTypePreference::income_single_quarter(),
    )?;
    let balance = data.financial_reader(
        DatasetId::StockBalanceSheet,
        ReportTypePreference::balance_sheet_consolidated(),
    )?;
    let inputs = event_inputs(panel, data)?;
    let requested_specs = requested_specs(needs);
    let mut output_by_id = requested_specs
        .iter()
        .map(|spec| {
            (
                spec.id.clone(),
                FactorSeries {
                    spec: spec.clone(),
                    values: Vec::new(),
                },
            )
        })
        .collect::<BTreeMap<_, _>>();

    for trade_date in context.target_dates.iter().copied() {
        if should_recompute(state.last_processed_trade_date, trade_date) {
            let event_series = compute_event_series(
                trade_date, needs, panel, data, &income, &balance, &inputs, state,
            )?;
            for series in event_series {
                state.final_cache.update_series(&series, panel);
                append_series_values(&mut output_by_id, series);
            }
        } else {
            for spec in &requested_specs {
                let series = state
                    .final_cache
                    .replay_series(spec.clone(), panel, trade_date);
                append_series_values(&mut output_by_id, series);
            }
        }
        state.last_processed_trade_date = Some(trade_date);
    }

    Ok(requested_specs
        .iter()
        .filter_map(|spec| output_by_id.remove(&spec.id))
        .collect())
}

fn compute_event_series(
    trade_date: i32,
    needs: RoicWaccNeeds,
    panel: &DailyPanel,
    data: &DataPool,
    income: &FinancialPitReader<'_>,
    balance: &FinancialPitReader<'_>,
    inputs: &EventInputs,
    state: &mut RoicWaccComputeState,
) -> Result<Vec<FactorSeries>> {
    let Some(date_idx) = panel.dates().iter().position(|date| *date == trade_date) else {
        return Ok(Vec::new());
    };
    let instrument_count = panel.instruments().len();
    let date_offset = date_idx * instrument_count;
    let capm_stats = capm_stats_by_stock(panel, &inputs.adj_close, date_idx);
    let all_present_indices = panel
        .instruments()
        .iter()
        .enumerate()
        .filter_map(|(instrument_idx, ts_code)| {
            let offset = date_offset + instrument_idx;
            (!is_bj_stock(ts_code) && panel.is_present_offset(offset)).then_some(instrument_idx)
        })
        .collect::<Vec<_>>();
    let mut required_by_end = BTreeMap::<i32, Vec<usize>>::new();
    let mut end_dates_by_stock = vec![Vec::<i32>::new(); instrument_count];

    for (instrument_idx, ts_code) in panel.instruments().iter().enumerate() {
        let offset = date_offset + instrument_idx;
        if is_bj_stock(ts_code) || !panel.is_present_offset(offset) {
            continue;
        }
        let Some(latest_end) = income.latest_quarter_end_date(ts_code, trade_date) else {
            continue;
        };
        let entry = state.stock_states.entry(ts_code.clone()).or_default();
        let ends = needed_quarter_end_dates(entry, latest_end);
        for end_date in &ends {
            required_by_end
                .entry(*end_date)
                .or_default()
                .push(instrument_idx);
        }
        end_dates_by_stock[instrument_idx] = ends;
    }

    for (end_date, stock_indices) in required_by_end {
        let prelim_by_stock = prelims_for_end_date(
            trade_date,
            end_date,
            panel,
            &inputs.dv_ratio,
            date_offset,
            income,
            balance,
            &all_present_indices,
        );
        let peer_re = peer_re_by_stock(
            trade_date,
            panel,
            &inputs.subindustry_map,
            &prelim_by_stock,
            &capm_stats,
            &all_present_indices,
        );
        for instrument_idx in stock_indices {
            if !end_dates_by_stock[instrument_idx].contains(&end_date) {
                continue;
            }
            let Some(prelim) = prelim_by_stock[instrument_idx] else {
                continue;
            };
            let re = max_available([
                prelim.re_accounting,
                capm_stats[instrument_idx].map(|stats| stats.re_self),
                peer_re[instrument_idx],
            ]);
            let Some(re) = re else {
                continue;
            };
            let wacc = prelim.wd * prelim.rd * (1.0 - prelim.tax) + prelim.we * re;
            if !wacc.is_finite() {
                continue;
            }
            let metrics = QuarterMetrics {
                end_date,
                noplat: prelim.noplat,
                roic: prelim.roic,
                spread: prelim.roic - wacc,
            };
            let ts_code = &panel.instruments()[instrument_idx];
            state
                .stock_states
                .entry(ts_code.clone())
                .or_default()
                .push(metrics);
        }
    }

    let mut output = Vec::new();
    if needs.unexpected {
        let values = event_raw_values(panel, trade_date, |ts_code| {
            state.stock_states.get(ts_code).and_then(unexpected_value)
        })?;
        let neutralized = neutralize_event_values(trade_date, values, panel, data)?;
        output.push(neutralized.to_factor_series(spec(RoicWaccOutput::UnexpectedRoicWacc)));
    }
    if needs.state_growth {
        let values = event_raw_values(panel, trade_date, |ts_code| {
            state.stock_states.get(ts_code).and_then(state_growth_value)
        })?;
        let neutralized = neutralize_event_values(trade_date, values, panel, data)?;
        output.push(neutralized.to_factor_series(spec(RoicWaccOutput::StateGrowth)));
    }
    Ok(output)
}

fn event_inputs(panel: &DailyPanel, data: &DataPool) -> Result<EventInputs> {
    let close = panel.column_from_table(data.daily(DatasetId::StockDailyPv)?, "close")?;
    let adj_factor =
        panel.column_from_table(data.daily(DatasetId::StockAdjFactor)?, "adj_factor")?;
    let adj_close = close.zip_binary(&adj_factor, |close, adj_factor| {
        Some(clean(close)? * clean(adj_factor)?)
    })?;
    let dv_ratio = panel.column_from_table(data.daily(DatasetId::StockDailyBasic)?, "dv_ratio")?;
    let sw = data.daily(DatasetId::StockSwClassification)?;
    let subindustry_map = ClassificationMap::from_table(sw, ClassificationLevel::Subindustry)?;
    Ok(EventInputs {
        adj_close,
        dv_ratio,
        subindustry_map,
    })
}

fn needed_quarter_end_dates(state: &StockRoicWaccState, latest_end: i32) -> Vec<i32> {
    let mut ends = if let Some(last) = state.quarters.back().map(|quarter| quarter.end_date) {
        quarter_ends_after_until(last, latest_end)
    } else {
        quarter_chain_ascending(latest_end, HISTORY_KEEP_QUARTERS)
    };
    ends.sort_unstable();
    ends.dedup();
    ends
}

fn prelims_for_end_date(
    trade_date: i32,
    end_date: i32,
    panel: &DailyPanel,
    dv_ratio: &PanelColumn,
    date_offset: usize,
    income: &FinancialPitReader<'_>,
    balance: &FinancialPitReader<'_>,
    stock_indices: &[usize],
) -> Vec<Option<QuarterPrelim>> {
    let mut output = vec![None; panel.instruments().len()];
    for &instrument_idx in stock_indices {
        let ts_code = &panel.instruments()[instrument_idx];
        let offset = date_offset + instrument_idx;
        output[instrument_idx] = quarter_prelim(
            ts_code,
            trade_date,
            end_date,
            income,
            balance,
            dv_ratio.values()[offset],
        );
    }
    output
}

fn quarter_prelim(
    ts_code: &str,
    trade_date: i32,
    end_date: i32,
    income: &FinancialPitReader<'_>,
    balance: &FinancialPitReader<'_>,
    dv_ratio: Option<f64>,
) -> Option<QuarterPrelim> {
    let income_record = income.record_for_end_date(ts_code, trade_date, end_date)?;
    let balance_record = balance.record_for_end_date(ts_code, trade_date, end_date)?;
    let operate_profit = clean(income_record.column("operate_profit"))?;
    let sell_exp = clean(income_record.column("sell_exp")).unwrap_or(0.0);
    let admin_exp = clean(income_record.column("admin_exp")).unwrap_or(0.0);
    let fin_exp = clean(income_record.column("fin_exp")).unwrap_or(0.0);
    let tax = tax_rate(
        income_record.column("income_tax"),
        income_record.column("n_income"),
    );
    let noplat = (operate_profit - sell_exp - admin_exp - fin_exp) * (1.0 - tax);
    if !noplat.is_finite() {
        return None;
    }
    let equity = clean(balance_record.column("total_hldr_eqy_exc_min_int")).filter(|v| *v > EPS)?;
    let debt = interest_bearing_debt(&balance_record);
    let invested_capital = equity + debt;
    if invested_capital <= EPS || !invested_capital.is_finite() {
        return None;
    }
    let roic = 4.0 * noplat / invested_capital;
    if !roic.is_finite() {
        return None;
    }
    let rd = if debt <= EPS {
        0.0
    } else {
        let prev_end = previous_quarter_end_date(end_date)?;
        let prev_balance = balance.record_for_end_date(ts_code, trade_date, prev_end)?;
        let prev_debt = interest_bearing_debt(&prev_balance);
        let avg_debt = 0.5 * (prev_debt + debt);
        if avg_debt <= EPS || !avg_debt.is_finite() {
            return None;
        }
        let interest_exp = clean(income_record.column("int_exp"))
            .unwrap_or(0.0)
            .max(0.0);
        4.0 * interest_exp / avg_debt
    };
    if !rd.is_finite() {
        return None;
    }
    let re_accounting = accounting_re(dv_ratio);
    Some(QuarterPrelim {
        tax,
        debt,
        equity,
        noplat,
        roic,
        rd,
        wd: debt / invested_capital,
        we: equity / invested_capital,
        re_accounting,
    })
}

fn interest_bearing_debt(record: &crate::factor::common::PitFinancialRecordView<'_>) -> f64 {
    [
        "st_borr",
        "non_cur_liab_due_1y",
        "lt_borr",
        "bond_payable",
        "lease_liab",
    ]
    .iter()
    .map(|column| clean(record.column(column)).unwrap_or(0.0))
    .sum::<f64>()
}

fn tax_rate(income_tax: Option<f64>, n_income: Option<f64>) -> f64 {
    match (clean(income_tax), clean(n_income)) {
        (Some(tax), Some(income)) if income.abs() > EPS => (tax / income).clamp(0.0, 0.25),
        _ => 0.0,
    }
}

fn accounting_re(dv_ratio: Option<f64>) -> Option<f64> {
    let dividend_yield = clean(dv_ratio)?.max(0.0) / 100.0;
    Some(dividend_yield + 0.02)
}

fn peer_re_by_stock(
    trade_date: i32,
    panel: &DailyPanel,
    subindustry_map: &ClassificationMap,
    prelim_by_stock: &[Option<QuarterPrelim>],
    capm_stats: &[Option<CapmStats>],
    stock_indices: &[usize],
) -> Vec<Option<f64>> {
    let mut grouped = BTreeMap::<String, (Vec<f64>, Vec<f64>)>::new();
    for &instrument_idx in stock_indices {
        let ts_code = &panel.instruments()[instrument_idx];
        let (Some(prelim), Some(capm), Some(group)) = (
            prelim_by_stock[instrument_idx],
            capm_stats[instrument_idx],
            subindustry_map.group_for(trade_date, ts_code),
        ) else {
            continue;
        };
        let beta_u = unlever_beta(capm.beta, prelim.tax, prelim.debt, prelim.equity);
        if !beta_u.is_finite() || !capm.alpha.is_finite() {
            continue;
        }
        let entry = grouped.entry(group.to_string()).or_default();
        entry.0.push(capm.alpha);
        entry.1.push(beta_u);
    }

    let medians = grouped
        .into_iter()
        .filter_map(|(group, (alphas, betas))| Some((group, (median(&alphas)?, median(&betas)?))))
        .collect::<BTreeMap<_, _>>();
    let mut output = vec![None; panel.instruments().len()];
    for &instrument_idx in stock_indices {
        let ts_code = &panel.instruments()[instrument_idx];
        let (Some(prelim), Some(capm), Some(group)) = (
            prelim_by_stock[instrument_idx],
            capm_stats[instrument_idx],
            subindustry_map.group_for(trade_date, ts_code),
        ) else {
            continue;
        };
        let Some((median_alpha, median_beta_u)) = medians.get(group).copied() else {
            continue;
        };
        let relevered_beta = lever_beta(median_beta_u, prelim.tax, prelim.debt, prelim.equity);
        let re = 52.0 * (median_alpha + relevered_beta * capm.mean_market_week);
        if re.is_finite() {
            output[instrument_idx] = Some(re);
        }
    }
    output
}

fn unlever_beta(beta_l: f64, tax: f64, debt: f64, equity: f64) -> f64 {
    beta_l / (1.0 + (1.0 - tax) * debt / equity)
}

fn lever_beta(beta_u: f64, tax: f64, debt: f64, equity: f64) -> f64 {
    beta_u * (1.0 + (1.0 - tax) * debt / equity)
}

fn capm_stats_by_stock(
    panel: &DailyPanel,
    adj_close: &PanelColumn,
    date_idx: usize,
) -> Vec<Option<CapmStats>> {
    let instrument_count = panel.instruments().len();
    let week_ends = week_end_indices(panel.dates());
    let Some(week_end_cutoff) = week_ends.iter().rposition(|idx| *idx <= date_idx) else {
        return vec![None; instrument_count];
    };
    let start_week = week_end_cutoff.saturating_sub(CAPM_LOOKBACK_WEEKS);
    let selected_week_ends = &week_ends[start_week..=week_end_cutoff];
    if selected_week_ends.len() < 2 {
        return vec![None; instrument_count];
    }
    let week_count = selected_week_ends.len() - 1;
    let mut stock_weekly = vec![vec![None; week_count]; instrument_count];
    let mut market_weekly = vec![None; week_count];

    for week_idx in 0..week_count {
        let prev_date_idx = selected_week_ends[week_idx];
        let curr_date_idx = selected_week_ends[week_idx + 1];
        let mut market_sum = 0.0;
        let mut market_count = 0usize;
        for (instrument_idx, ts_code) in panel.instruments().iter().enumerate() {
            let prev = adj_close.values()[prev_date_idx * instrument_count + instrument_idx];
            let curr = adj_close.values()[curr_date_idx * instrument_count + instrument_idx];
            let ret = price_return(prev, curr);
            stock_weekly[instrument_idx][week_idx] = ret;
            if !is_bj_stock(ts_code) {
                if let Some(value) = ret {
                    market_sum += value;
                    market_count += 1;
                }
            }
        }
        if market_count > 0 {
            market_weekly[week_idx] = Some(market_sum / market_count as f64);
        }
    }

    stock_weekly
        .iter()
        .map(|returns| capm_stats(returns, &market_weekly))
        .collect()
}

fn capm_stats(stock_returns: &[Option<f64>], market_returns: &[Option<f64>]) -> Option<CapmStats> {
    let mut pairs = Vec::new();
    for (stock, market) in stock_returns.iter().zip(market_returns.iter()) {
        if let (Some(stock), Some(market)) = (clean(*stock), clean(*market)) {
            pairs.push((stock, market));
        }
    }
    if pairs.len() < CAPM_MIN_PERIODS {
        return None;
    }
    let (alpha, beta) = ols_intercept_slope(&pairs)?;
    let mean_market_week =
        pairs.iter().map(|(_, market)| *market).sum::<f64>() / pairs.len() as f64;
    let re_self = 52.0 * (alpha + beta * mean_market_week);
    (alpha.is_finite() && beta.is_finite() && re_self.is_finite()).then_some(CapmStats {
        alpha,
        beta,
        mean_market_week,
        re_self,
    })
}

fn price_return(prev: Option<f64>, curr: Option<f64>) -> Option<f64> {
    let prev = clean(prev).filter(|value| value.abs() > EPS)?;
    let curr = clean(curr)?;
    let ret = curr / prev - 1.0;
    ret.is_finite().then_some(ret)
}

fn ols_intercept_slope(pairs: &[(f64, f64)]) -> Option<(f64, f64)> {
    if pairs.len() < 2 {
        return None;
    }
    let n = pairs.len() as f64;
    let mean_y = pairs.iter().map(|(y, _)| *y).sum::<f64>() / n;
    let mean_x = pairs.iter().map(|(_, x)| *x).sum::<f64>() / n;
    let mut cov = 0.0;
    let mut var = 0.0;
    for (y, x) in pairs {
        cov += (x - mean_x) * (y - mean_y);
        var += (x - mean_x) * (x - mean_x);
    }
    if var.abs() <= EPS {
        return None;
    }
    let beta = cov / var;
    let alpha = mean_y - beta * mean_x;
    Some((alpha, beta))
}

fn unexpected_value(state: &StockRoicWaccState) -> Option<f64> {
    let current = state.quarters.back()?;
    let mut history = Vec::with_capacity(UNEXPECTED_HISTORY_QUARTERS);
    let mut end_date = current.end_date;
    for _ in 0..UNEXPECTED_HISTORY_QUARTERS {
        end_date = previous_quarter_end_date(end_date)?;
        let quarter = state
            .quarters
            .iter()
            .find(|quarter| quarter.end_date == end_date)?;
        history.push(quarter.spread);
    }
    let mean = history.iter().sum::<f64>() / history.len() as f64;
    let std = sample_std(&history)?;
    let value = (current.spread - mean) / std;
    value.is_finite().then_some(value)
}

fn state_growth_value(state: &StockRoicWaccState) -> Option<f64> {
    let current = state.quarters.back()?;
    let target_end = nth_previous_quarter_end_date(current.end_date, 4)?;
    let previous = state
        .quarters
        .iter()
        .find(|quarter| quarter.end_date == target_end)?;
    let value = if current.spread > 0.0 {
        growth(current.noplat, previous.noplat)?
    } else {
        growth(current.roic, previous.roic)?
    };
    value.is_finite().then_some(value)
}

fn growth(current: f64, previous: f64) -> Option<f64> {
    (previous.abs() > EPS).then_some((current - previous) / previous.abs())
}

fn event_raw_values<F>(
    panel: &DailyPanel,
    trade_date: i32,
    mut value_fn: F,
) -> Result<Vec<Option<f64>>>
where
    F: FnMut(&str) -> Option<f64>,
{
    let Some(date_idx) = panel.dates().iter().position(|date| *date == trade_date) else {
        return Ok(vec![None; panel.instruments().len()]);
    };
    let instrument_count = panel.instruments().len();
    let date_offset = date_idx * instrument_count;
    let mut values = vec![None; instrument_count];
    for (instrument_idx, ts_code) in panel.instruments().iter().enumerate() {
        let offset = date_offset + instrument_idx;
        if is_bj_stock(ts_code) || !panel.is_present_offset(offset) {
            continue;
        }
        values[instrument_idx] = value_fn(ts_code);
    }
    Ok(values)
}

fn neutralize_event_values(
    trade_date: i32,
    values: Vec<Option<f64>>,
    panel: &DailyPanel,
    data: &DataPool,
) -> Result<PanelColumn> {
    let event_panel = panel.slice_dates(&[trade_date]);
    let raw = event_panel.column_from_values(values)?;
    neutralize_size_sector(&raw, &event_panel, data)
}

fn append_series_values(output_by_id: &mut BTreeMap<String, FactorSeries>, series: FactorSeries) {
    if let Some(output) = output_by_id.get_mut(&series.spec.id) {
        output.values.extend(series.values);
    }
}

fn requested_specs(needs: RoicWaccNeeds) -> Vec<FactorSpec> {
    let mut output = Vec::new();
    if needs.unexpected {
        output.push(spec(RoicWaccOutput::UnexpectedRoicWacc));
    }
    if needs.state_growth {
        output.push(spec(RoicWaccOutput::StateGrowth));
    }
    output
}

fn needs_from_requested(requested_ids: &[String]) -> RoicWaccNeeds {
    RoicWaccNeeds {
        unexpected: requested_ids.iter().any(|id| id == UNEXPECTED_ROIC_WACC_ID),
        state_growth: requested_ids
            .iter()
            .any(|id| id == ROIC_WACC_STATE_GROWTH_ID),
    }
}

fn should_recompute(last_processed: Option<i32>, trade_date: i32) -> bool {
    last_processed.is_none() || fixed_checkpoint_after_until(last_processed, trade_date)
}

fn fixed_checkpoint_after_until(after_exclusive: Option<i32>, until_inclusive: i32) -> bool {
    let lower = after_exclusive.unwrap_or(i32::MIN);
    let start_year = (lower / 10_000).saturating_sub(1);
    let end_year = until_inclusive / 10_000 + 1;
    for year in start_year..=end_year {
        for month_day in [430, 831, 1031] {
            let checkpoint = year * 10_000 + month_day;
            if checkpoint > lower && checkpoint <= until_inclusive {
                return true;
            }
        }
    }
    false
}

fn quarter_chain_ascending(latest_end: i32, count: usize) -> Vec<i32> {
    let mut dates = Vec::new();
    let mut current = Some(latest_end);
    for _ in 0..count {
        let Some(end_date) = current else {
            break;
        };
        dates.push(end_date);
        current = previous_quarter_end_date(end_date);
    }
    dates.reverse();
    dates
}

fn quarter_ends_after_until(after_end: i32, latest_end: i32) -> Vec<i32> {
    let mut reversed = quarter_chain_ascending(latest_end, HISTORY_KEEP_QUARTERS + 8);
    reversed.retain(|end_date| *end_date > after_end && *end_date <= latest_end);
    reversed
}

fn nth_previous_quarter_end_date(mut end_date: i32, count: usize) -> Option<i32> {
    for _ in 0..count {
        end_date = previous_quarter_end_date(end_date)?;
    }
    Some(end_date)
}

impl StockRoicWaccState {
    fn push(&mut self, metrics: QuarterMetrics) {
        if self
            .quarters
            .back()
            .is_some_and(|latest| latest.end_date >= metrics.end_date)
        {
            return;
        }
        self.quarters.push_back(metrics);
        while self.quarters.len() > HISTORY_KEEP_QUARTERS {
            self.quarters.pop_front();
        }
    }
}

fn week_end_indices(dates: &[i32]) -> Vec<usize> {
    let mut output = Vec::new();
    for (idx, date) in dates.iter().copied().enumerate() {
        let current = week_index(date);
        let next = dates.get(idx + 1).copied().map(week_index);
        if next != Some(current) {
            output.push(idx);
        }
    }
    output
}

fn week_index(date: i32) -> i32 {
    let (year, month, day) = split_yyyymmdd(date);
    (days_from_civil(year, month, day) + 2).div_euclid(7)
}

fn split_yyyymmdd(date: i32) -> (i32, i32, i32) {
    let year = date / 10_000;
    let month = (date / 100) % 100;
    let day = date % 100;
    (year, month, day)
}

fn days_from_civil(year: i32, month: i32, day: i32) -> i32 {
    let y = year - i32::from(month <= 2);
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = month + if month > 2 { -3 } else { 9 };
    let doy = (153 * mp + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe
}

fn sample_std(values: &[f64]) -> Option<f64> {
    if values.len() < 2 {
        return None;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let var = values
        .iter()
        .map(|value| {
            let diff = value - mean;
            diff * diff
        })
        .sum::<f64>()
        / (values.len() as f64 - 1.0);
    let std = var.sqrt();
    (std > EPS && std.is_finite()).then_some(std)
}

fn median(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut values = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    if values.is_empty() {
        return None;
    }
    values.sort_by(|left, right| left.total_cmp(right));
    let mid = values.len() / 2;
    if values.len() % 2 == 1 {
        Some(values[mid])
    } else {
        Some(0.5 * (values[mid - 1] + values[mid]))
    }
}

fn max_available(values: [Option<f64>; 3]) -> Option<f64> {
    values
        .into_iter()
        .flatten()
        .filter(|value| value.is_finite())
        .reduce(f64::max)
}

fn clean(value: Option<f64>) -> Option<f64> {
    value.filter(|value| value.is_finite())
}

fn dependencies() -> Vec<DataRequest> {
    vec![
        DataRequest::new(DatasetId::StockDailyPv, &["close"]),
        DataRequest::new(DatasetId::StockAdjFactor, &["adj_factor"]),
        DataRequest::new(DatasetId::StockDailyBasic, &["dv_ratio"]),
        DataRequest::new(DatasetId::StockBarraDaily, &["SIZE"]),
        DataRequest::new(DatasetId::StockSwClassification, &["l1_code", "l3_code"]),
        DataRequest::financial_quarters(DatasetId::StockIncome, &INCOME_COLUMNS, 13),
        DataRequest::financial_quarters(DatasetId::StockBalanceSheet, &BALANCE_COLUMNS, 13),
    ]
}

pub fn requirements_for_context(context: &FactorContext) -> Vec<DataRequest> {
    let capm_dates = target_dates_plus_recent_week_ends(context, CAPM_LOOKBACK_WEEKS);
    let checkpoint_dates = fixed_checkpoint_target_dates(context, &[430, 831, 1031], true);
    vec![
        DataRequest::explicit_dates(DatasetId::StockDailyPv, &["close"], capm_dates.clone()),
        DataRequest::explicit_dates(DatasetId::StockAdjFactor, &["adj_factor"], capm_dates),
        DataRequest::explicit_dates(
            DatasetId::StockDailyBasic,
            &["dv_ratio"],
            checkpoint_dates.clone(),
        ),
        DataRequest::explicit_dates(DatasetId::StockBarraDaily, &["SIZE"], checkpoint_dates),
        DataRequest::new(DatasetId::StockSwClassification, &["l1_code", "l3_code"])
            .with_explicit_dates(context.load_dates.clone()),
        DataRequest::financial_quarters(DatasetId::StockIncome, &INCOME_COLUMNS, 13)
            .with_explicit_dates(context.load_dates.clone()),
        DataRequest::financial_quarters(DatasetId::StockBalanceSheet, &BALANCE_COLUMNS, 13)
            .with_explicit_dates(context.load_dates.clone()),
    ]
}

fn target_dates_plus_recent_week_ends(context: &FactorContext, weeks: usize) -> Vec<i32> {
    let mut dates = context
        .target_dates
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<_>>();
    let week_ends = week_end_dates(context.load_dates.as_slice());
    let needed = weeks.saturating_add(1);
    let start = week_ends.len().saturating_sub(needed);
    dates.extend(week_ends[start..].iter().copied());
    dates.into_iter().collect()
}

fn fixed_checkpoint_target_dates(
    context: &FactorContext,
    month_days: &[i32],
    include_first_target: bool,
) -> Vec<i32> {
    let mut dates = std::collections::BTreeSet::new();
    let mut previous = None;
    for (idx, trade_date) in context.target_dates.iter().copied().enumerate() {
        if idx == 0 && include_first_target {
            dates.insert(trade_date);
        } else if fixed_checkpoint_after_until_dates(previous, trade_date, month_days) {
            dates.insert(trade_date);
        }
        previous = Some(trade_date);
    }
    dates.into_iter().collect()
}

fn fixed_checkpoint_after_until_dates(
    after_exclusive: Option<i32>,
    until_inclusive: i32,
    month_days: &[i32],
) -> bool {
    let lower = after_exclusive.unwrap_or(i32::MIN);
    let start_year = (lower / 10_000).saturating_sub(1);
    let end_year = until_inclusive / 10_000 + 1;
    for year in start_year..=end_year {
        for month_day in month_days {
            let checkpoint = year * 10_000 + month_day;
            if checkpoint > lower && checkpoint <= until_inclusive {
                return true;
            }
        }
    }
    false
}

fn week_end_dates(dates: &[i32]) -> Vec<i32> {
    let mut output = Vec::new();
    for (idx, date) in dates.iter().copied().enumerate() {
        let current = week_index(date);
        let next = dates.get(idx + 1).copied().map(week_index);
        if next != Some(current) {
            output.push(date);
        }
    }
    output
}

fn tags() -> Vec<String> {
    [
        "DBZQ",
        "financial",
        "fundamental",
        "pit",
        "roic",
        "wacc",
        "neutralize",
        "barra",
        "size",
        "sector",
        "daily",
    ]
    .iter()
    .map(|tag| (*tag).to_string())
    .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: Option<f64>, expected: Option<f64>) {
        match (actual, expected) {
            (Some(actual), Some(expected)) => assert!(
                (actual - expected).abs() < 1e-10,
                "expected {expected}, got {actual}"
            ),
            (None, None) => {}
            _ => panic!("expected {:?}, got {:?}", expected, actual),
        }
    }

    #[test]
    fn fixed_checkpoint_schedule_maps_to_next_trading_target() {
        assert!(fixed_checkpoint_after_until(Some(20260429), 20260506));
        assert!(!fixed_checkpoint_after_until(Some(20260506), 20260507));
        assert!(should_recompute(None, 20260105));
    }

    #[test]
    fn tax_rate_clips_and_defaults_to_zero() {
        assert_close(Some(tax_rate(Some(10.0), Some(100.0))), Some(0.10));
        assert_close(Some(tax_rate(Some(50.0), Some(100.0))), Some(0.25));
        assert_close(Some(tax_rate(Some(1.0), Some(0.0))), Some(0.0));
    }

    #[test]
    fn capm_ols_uses_intercept_and_beta() {
        let pairs = vec![(0.03, 0.01), (0.05, 0.02), (0.07, 0.03)];
        let (alpha, beta) = ols_intercept_slope(&pairs).expect("ols");
        assert!((alpha - 0.01).abs() < 1e-10);
        assert!((beta - 2.0).abs() < 1e-10);
    }

    #[test]
    fn unexpected_uses_previous_eight_spreads_only() {
        let mut state = StockRoicWaccState::default();
        for (idx, end_date) in [
            20200331, 20200630, 20200930, 20201231, 20210331, 20210630, 20210930, 20211231,
            20220331,
        ]
        .iter()
        .copied()
        .enumerate()
        {
            state.push(QuarterMetrics {
                end_date,
                noplat: 1.0,
                roic: 1.0,
                spread: idx as f64,
            });
        }
        assert!(unexpected_value(&state).is_some());
    }

    #[test]
    fn state_growth_switches_on_current_spread_sign() {
        let mut state = StockRoicWaccState::default();
        for (idx, end_date) in [20200331, 20200630, 20200930, 20201231, 20210331]
            .iter()
            .copied()
            .enumerate()
        {
            state.push(QuarterMetrics {
                end_date,
                noplat: (idx + 1) as f64,
                roic: (idx + 2) as f64,
                spread: 1.0,
            });
        }
        assert_close(state_growth_value(&state), Some((5.0 - 1.0) / 1.0));
        state.quarters.back_mut().unwrap().spread = -1.0;
        assert_close(state_growth_value(&state), Some((6.0 - 2.0) / 2.0));
    }

    #[test]
    fn specs_have_required_tags_and_dependencies() {
        let spec = spec(RoicWaccOutput::UnexpectedRoicWacc);
        assert!(spec.tags.contains(&"DBZQ".to_string()));
        assert!(spec.tags.contains(&"wacc".to_string()));
        assert_eq!(spec.lookback.trading_days, DAILY_LOOKBACK);
        assert!(spec
            .dependencies
            .iter()
            .any(|request| request.dataset == DatasetId::StockIncome));
    }

    #[test]
    fn context_requirements_pass_explicit_dates() {
        let context = FactorContext {
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            start_date: 20260115,
            end_date: 20260116,
            load_start_date: 20260102,
            load_dates: vec![
                20260102, 20260105, 20260106, 20260107, 20260108, 20260109, 20260112, 20260113,
                20260114, 20260115, 20260116,
            ],
            target_dates: vec![20260115, 20260116],
        };

        let requests = requirements_for_context(&context);
        let pv = requests
            .iter()
            .find(|request| request.dataset == DatasetId::StockDailyPv)
            .expect("pv request");

        assert_eq!(
            pv.resolved_dates(&context),
            vec![20260102, 20260109, 20260115, 20260116]
        );
    }
}

use std::any::Any;
use std::collections::{BTreeMap, BTreeSet, HashMap};

use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::{DataPool, FundDisclosureView, FUND_PORTFOLIO_COLUMNS};
use crate::error::{err, Result};
use crate::factor::common::{ClassificationLevel, ClassificationMap, PanelColumn};
use crate::factor::{Factor, FactorUpdatePolicy};
use crate::operators::{cs_neutralize_regression, ts_mean, ts_prod};

const WINDOW: usize = 20;
const TOP_N: usize = 10;
// Include the prior month-end plus its 20-return window for a mid-month start.
const INITIAL_LOOKBACK: usize = 52;
// Tushare daily.amount is thousand CNY; fund_portfolio.mkv is CNY.
const AMOUNT_TO_CNY: f64 = 1000.0;

#[derive(Clone, Debug, PartialEq)]
struct Holding {
    code: String,
    market_value: f64,
}

#[derive(Clone, Debug)]
struct FundSnapshot {
    end_date: i32,
    ann_date: i32,
    holdings: Vec<Holding>,
}

#[derive(Default, Debug)]
struct Network {
    codes: Vec<String>,
    edges: Vec<(usize, usize, f64)>,
}

#[derive(Default)]
struct TractionState {
    funds: BTreeMap<String, FundSnapshot>,
    last_processed: Option<i32>,
    last_target_date: Option<i32>,
    final_values: HashMap<String, Option<f64>>,
    conflict_groups: usize,
}

#[derive(Clone, Copy, Default)]
pub struct StockDailyTraction20d;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyTraction20d)
}

impl Factor for StockDailyTraction20d {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "traction20d".into(), aliases: vec![], name: "traction20d".into(),
            asset_class: AssetClass::Stock, frequency: Frequency::Daily, version: "0.2.0".into(),
            tags: ["deprecated", "KYZQ", "alternative", "cs_network", "fund_holdings", "pit", "traction", "momentum", "monthly_update", "neutralize", "sector", "daily"].into_iter().map(str::to_string).collect(),
            description: "KYZQ fund top-ten A-share holdings traction computed at each completed trading month-end using PIT disclosures, current 20-day mean turnover and available compounded adjusted returns (min_periods=1). Neutralize own Alpha and SW L1 without SIZE; replay the final cross-section by stock code until next month-end. Mid-month starts reconstruct the previous month-end, not a start-date snapshot.".into(),
            dependencies: vec![
                DataRequest::new(DatasetId::FundPortfolio, FUND_PORTFOLIO_COLUMNS),
                DataRequest::new(DatasetId::StockDailyPv, &["close", "amount"]),
                DataRequest::new(DatasetId::StockAdjFactor, &["adj_factor"]),
                DataRequest::new(DatasetId::StockSwClassification, &["l1_code"]),
            ],
            intraday_raw_dependencies: vec![], lookback: Lookback { trading_days: INITIAL_LOOKBACK },
        }
    }
    fn update_policy(&self) -> FactorUpdatePolicy {
        FactorUpdatePolicy::FinancialEventSnapshot
    }
    fn initial_compute_state(&self, _: &[String]) -> Box<dyn Any + Send> {
        Box::<TractionState>::default()
    }
    fn compute(&self, context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        compute(context, data, &mut TractionState::default())
    }
    fn compute_many_stateful(
        &self,
        ids: &[String],
        context: &FactorContext,
        data: &DataPool,
        state: &mut (dyn Any + Send),
    ) -> Result<Vec<FactorSeries>> {
        if !ids.iter().any(|id| id == "traction20d") {
            return Ok(vec![]);
        }
        let state = state
            .downcast_mut::<TractionState>()
            .ok_or_else(|| err("traction20d state type mismatch"))?;
        Ok(vec![compute(context, data, state)?])
    }
}

fn standard_stock(code: &str) -> bool {
    let b = code.as_bytes();
    b.len() == 9
        && b[..6].iter().all(u8::is_ascii_digit)
        && b[6] == b'.'
        && matches!(&b[7..], b"SH" | b"SZ")
}

fn clean(value: Option<f64>) -> Option<f64> {
    value.filter(|v| v.is_finite())
}

fn same_value(a: Option<f64>, b: Option<f64>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(a), Some(b)) => a == b || (a.is_nan() && b.is_nan()),
        _ => false,
    }
}

fn decimal_precision(value: f64) -> usize {
    // Parquet stores numbers, not original formatting; use effective decimal precision.
    let text = value.to_string();
    let (mantissa, exponent) = text.split_once('e').map_or((text.as_str(), 0), |(m, e)| {
        (m, e.parse::<i32>().unwrap_or(0))
    });
    let decimals = mantissa
        .split_once('.')
        .map_or(0, |(_, fraction)| fraction.len());
    (decimals as i32 - exponent).max(0) as usize
}

/// Business policy belongs here, not in the shared data index.
fn select_holdings(group: FundDisclosureView<'_>) -> (Vec<Holding>, bool) {
    let mut rows = BTreeMap::<Option<&str>, (usize, Option<(f64, f64)>)>::new();
    for &row in group.rows() {
        let symbol = group.symbol(row);
        if let Some((previous, range)) = rows.get_mut(&symbol) {
            if FUND_PORTFOLIO_COLUMNS[5..]
                .iter()
                .any(|column| !same_value(group.value(*previous, column), group.value(row, column)))
            {
                return (Vec::new(), true);
            }
            let old = group.value(*previous, "mkv");
            let new = group.value(row, "mkv");
            if let (Some(a), Some(b)) = (clean(old), clean(new)) {
                let (min, max) = range.unwrap_or((a, a));
                let bounds = (min.min(b), max.max(b));
                // Check the entire group range, not a chain of pairwise tolerances.
                if bounds.1 - bounds.0 > 0.5 {
                    return (Vec::new(), true);
                }
                *range = Some(bounds);
                // Equal precision uses a stable numeric tie-break, independent of row order.
                if (decimal_precision(b), b.total_cmp(&a))
                    > (decimal_precision(a), std::cmp::Ordering::Equal)
                {
                    *previous = row;
                }
            } else if !same_value(old, new) {
                return (Vec::new(), true);
            }
        } else {
            rows.insert(
                symbol,
                (row, clean(group.value(row, "mkv")).map(|v| (v, v))),
            );
        }
    }
    let mut holdings = rows
        .into_iter()
        .filter_map(|(code, (row, _))| {
            let code = code.filter(|code| standard_stock(code))?;
            let market_value = clean(group.value(row, "mkv")).filter(|v| *v > 0.0)?;
            Some(Holding {
                code: code.into(),
                market_value,
            })
        })
        .collect::<Vec<_>>();
    holdings.sort_by(|a, b| {
        b.market_value
            .total_cmp(&a.market_value)
            .then_with(|| a.code.cmp(&b.code))
    });
    holdings.truncate(TOP_N);
    (holdings, false)
}

fn apply_disclosure(state: &mut TractionState, group: FundDisclosureView<'_>) -> bool {
    if state
        .funds
        .get(group.fund_code())
        .is_some_and(|old| (old.end_date, old.ann_date) > (group.end_date(), group.ann_date()))
    {
        return false;
    }
    let (holdings, conflict) = select_holdings(group);
    state.conflict_groups += usize::from(conflict);
    let changed = state.funds.get(group.fund_code()).is_none_or(|old| {
        (old.end_date, old.ann_date) != (group.end_date(), group.ann_date())
            || old.holdings != holdings
    });
    state.funds.insert(
        group.fund_code().into(),
        FundSnapshot {
            end_date: group.end_date(),
            ann_date: group.ann_date(),
            holdings,
        },
    );
    changed
}

fn build_network(funds: &BTreeMap<String, FundSnapshot>, amounts: &HashMap<&str, f64>) -> Network {
    let mut codes = funds
        .values()
        .flat_map(|f| f.holdings.iter().map(|h| h.code.clone()))
        .collect::<Vec<_>>();
    codes.sort();
    codes.dedup();
    codes.shrink_to_fit();
    let ids = codes
        .iter()
        .enumerate()
        .map(|(i, code)| (code.as_str(), i))
        .collect::<HashMap<_, _>>();
    let mut pairs = HashMap::<(usize, usize), f64>::new();
    for fund in funds.values() {
        let crowding = fund
            .holdings
            .iter()
            .filter_map(|holding| {
                let amount = amounts
                    .get(holding.code.as_str())
                    .copied()
                    .filter(|v| v.is_finite() && *v > 0.0)?;
                let weight = holding.market_value / amount;
                (weight.is_finite() && weight > 0.0).then_some((ids[holding.code.as_str()], weight))
            })
            .collect::<Vec<_>>();
        for a in 0..crowding.len() {
            for b in a + 1..crowding.len() {
                let (i, wi) = crowding[a];
                let (j, wj) = crowding[b];
                *pairs.entry((i.min(j), i.max(j))).or_default() += wi.min(wj);
            }
        }
    }
    let mut edges = pairs
        .into_iter()
        .filter_map(|((i, j), w)| (w.is_finite() && w > 0.0).then_some((i, j, w)))
        .collect::<Vec<_>>();
    edges.sort_by_key(|&(i, j, _)| (i, j));
    Network { codes, edges }
}

fn compounded_return(series: &[Option<f64>]) -> Vec<Option<f64>> {
    let mut gross = vec![None; series.len()];
    for i in 1..series.len() {
        if let (Some(today), Some(previous)) = (clean(series[i]), clean(series[i - 1])) {
            if today > 0.0 && previous > 0.0 {
                gross[i] = clean(Some(today / previous));
            }
        }
    }
    ts_prod(&gross, WINDOW, 1)
        .into_iter()
        .map(|v| clean(v.map(|v| v - 1.0)))
        .collect()
}

fn month_end_market_inputs(
    close: &PanelColumn,
    adjustment: &PanelColumn,
    amount: &PanelColumn,
    day: usize,
    stock: usize,
    n: usize,
) -> (Option<f64>, Option<f64>) {
    let prices = (day.saturating_sub(WINDOW)..=day)
        .map(|d| {
            let offset = d * n + stock;
            let price = clean(close.values()[offset]).filter(|v| *v > 0.0)?;
            let factor = clean(adjustment.values()[offset]).filter(|v| *v > 0.0)?;
            clean(Some(price * factor))
        })
        .collect::<Vec<_>>();
    let turnover = ((day + 1).saturating_sub(WINDOW)..=day)
        .map(|d| {
            clean(amount.values()[d * n + stock])
                .filter(|v| *v >= 0.0)
                .map(|v| v * AMOUNT_TO_CNY)
        })
        .collect::<Vec<_>>();
    (
        compounded_return(&prices).last().copied().flatten(),
        ts_mean(&turnover, WINDOW, 1).last().copied().flatten(),
    )
}

fn daily_raw(network: &Network, returns: &[Option<f64>]) -> (Vec<Option<f64>>, Vec<Option<f64>>) {
    let mut valid = returns.iter().filter_map(|v| clean(*v)).collect::<Vec<_>>();
    if valid.is_empty() {
        return (vec![None; returns.len()], vec![None; returns.len()]);
    }
    valid.sort_by(f64::total_cmp);
    let median = if valid.len() % 2 == 0 {
        (valid[valid.len() / 2 - 1] + valid[valid.len() / 2]) / 2.0
    } else {
        valid[valid.len() / 2]
    };
    let alpha = returns
        .iter()
        .map(|v| clean(v.map(|v| v - median)))
        .collect::<Vec<_>>();
    let mut sums = vec![0.0; returns.len()];
    let mut counts = vec![0usize; returns.len()];
    for &(i, j, weight) in &network.edges {
        if let Some(a) = alpha[j] {
            sums[i] += weight * a;
            counts[i] += 1;
        }
        if let Some(a) = alpha[i] {
            sums[j] += weight * a;
            counts[j] += 1;
        }
    }
    let raw = (0..returns.len())
        .map(|i| {
            if counts[i] == 0 || alpha[i].is_none() {
                None
            } else {
                clean(Some(sums[i] / counts[i] as f64))
            }
        })
        .collect();
    (raw, alpha)
}

fn compute(
    context: &FactorContext,
    data: &DataPool,
    state: &mut TractionState,
) -> Result<FactorSeries> {
    let panel = data.stock_universe_panel()?;
    let reader = data.fund_portfolio_reader()?;
    let calendar = data.trading_calendar()?;
    let mut market_columns = None;
    let mut sectors = None;
    let n = panel.instruments().len();
    let stock_index = panel
        .instruments()
        .iter()
        .enumerate()
        .map(|(i, code)| (code.as_str(), i))
        .collect::<HashMap<_, _>>();
    let mut output = vec![None; panel.shape_len()];
    let conflicts_before = state.conflict_groups;
    for (day, &date) in panel.dates().iter().enumerate() {
        if !panel.is_target_date(date) {
            continue;
        }
        if state.last_target_date.is_some_and(|last| date < last) {
            return Err(err(
                "traction20d provider state cannot move backwards in time",
            ));
        }
        if let Some(anchor) = calendar.month_end_on_or_before(date) {
            if state.last_processed != Some(anchor) {
                let anchor_day = panel
                    .dates()
                    .binary_search(&anchor)
                    .map_err(|_| err("traction20d month-end anchor missing from warmup panel"))?;
                let required_start = calendar.warmup_start(anchor, WINDOW);
                if required_start < context.load_start_date {
                    return Err(err(
                        "traction20d month-end return window missing from warmup panel",
                    ));
                }
                if state.last_processed.is_none() {
                    reader.bootstrap_until(anchor, |group| {
                        apply_disclosure(state, group);
                        Ok(())
                    })?;
                } else {
                    for group in reader.disclosures_between(state.last_processed, anchor) {
                        apply_disclosure(state, group);
                    }
                }
                if market_columns.is_none() {
                    market_columns = Some((
                        panel.column_from_table(data.daily(DatasetId::StockDailyPv)?, "close")?,
                        panel.column_from_table(
                            data.daily(DatasetId::StockAdjFactor)?,
                            "adj_factor",
                        )?,
                        panel.column_from_table(data.daily(DatasetId::StockDailyPv)?, "amount")?,
                    ));
                    sectors = Some(ClassificationMap::from_table(
                        data.daily(DatasetId::StockSwClassification)?,
                        ClassificationLevel::Sector,
                    )?);
                }
                let (close, adjustment, amount) = market_columns.as_ref().unwrap();
                let mut returns = vec![None; n];
                let mut amounts = HashMap::new();
                let held_codes = state
                    .funds
                    .values()
                    .flat_map(|f| f.holdings.iter().map(|h| h.code.as_str()))
                    .collect::<BTreeSet<_>>();
                for code in held_codes {
                    if let Some(&i) = stock_index
                        .get(code)
                        .filter(|&&i| panel.is_present_offset(anchor_day * n + i))
                    {
                        let (ret, amt) =
                            month_end_market_inputs(close, adjustment, amount, anchor_day, i, n);
                        returns[i] = ret;
                        if let Some(amt) = amt.filter(|v| v.is_finite() && *v > 0.0) {
                            amounts.insert(code, amt);
                        }
                    }
                }
                // Only one month-end network exists; drop it after finalization.
                let network = build_network(&state.funds, &amounts);
                let peer_returns = network
                    .codes
                    .iter()
                    .map(|code| stock_index.get(code.as_str()).and_then(|i| returns[*i]))
                    .collect::<Vec<_>>();
                let (raw, alpha) = daily_raw(&network, &peer_returns);
                let groups = sectors.as_ref().unwrap().groups_for(anchor, &network.codes);
                let final_values = cs_neutralize_regression(&raw, &[&alpha], Some(&groups), None);
                state.final_values.clear();
                state
                    .final_values
                    .extend(network.codes.into_iter().zip(final_values));
                state.last_processed = Some(anchor);
            }
        }
        for (i, code) in panel.instruments().iter().enumerate() {
            if panel.is_present_offset(day * n + i) {
                output[day * n + i] = state.final_values.get(code).copied().flatten();
            }
        }
        state.last_target_date = Some(date);
    }
    if state.conflict_groups > conflicts_before {
        eprintln!("warning: traction20d excluded {} conflicting fund disclosure groups in {}..{}; no prior-disclosure fallback", state.conflict_groups-conflicts_before, context.start_date, context.end_date);
    }
    Ok(panel
        .column_from_values(output)?
        .to_factor_series(StockDailyTraction20d.spec()))
}

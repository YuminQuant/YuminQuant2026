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
            tags: ["KYZQ", "alternative", "cs_network", "fund_holdings", "pit", "traction", "momentum", "monthly_update", "neutralize", "sector", "daily"].into_iter().map(str::to_string).collect(),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::fund_portfolio::tests::table;
    use crate::data::{ColumnData, FundPortfolioBatchIndex, FundPortfolioIndex, Table};
    use std::sync::Arc;

    fn index(rows: &[(&str, i32, i32, &str, f64)]) -> FundPortfolioBatchIndex {
        FundPortfolioBatchIndex::new(
            vec![Arc::new(
                FundPortfolioIndex::from_table(Arc::new(table(rows))).unwrap(),
            )],
            vec![],
            20261231,
        )
    }

    #[test]
    fn traction20d_mkv_rounding_keeps_precision_and_checks_full_range() {
        for (values, expected) in [
            (vec![21532., 21532.25], Some(21532.25)),
            (vec![21532.25, 21532.], Some(21532.25)),
            (vec![100., 100.5], Some(100.5)),
            (vec![100.5, 100.], Some(100.5)),
            (vec![100.1, 100.12], Some(100.12)),
            (vec![100.12, 100.1], Some(100.12)),
            (vec![100., 100.500001], None),
            (vec![100., 100.4, 100.8], None),
            (vec![100.4, 100.8, 100.], None),
        ] {
            let rows = values
                .iter()
                .map(|&v| ("F", 20260420, 20260331, "000001.SZ", v))
                .collect::<Vec<_>>();
            let mut source = table(&rows);
            for column in FUND_PORTFOLIO_COLUMNS[5..].iter() {
                source.columns.insert(
                    (*column).into(),
                    ColumnData::F64(vec![Some(1.); rows.len()]),
                );
            }
            let index = FundPortfolioBatchIndex::new(
                vec![Arc::new(
                    FundPortfolioIndex::from_table(Arc::new(source)).unwrap(),
                )],
                vec![],
                20260430,
            );
            let mut state = TractionState::default();
            let reader = index.reader();
            let group = reader.disclosures_between(None, 20260420)[0];
            apply_disclosure(&mut state, group);
            assert_eq!(
                state.funds["F"].holdings.first().map(|h| h.market_value),
                expected,
                "{values:?}"
            );
            assert_eq!(state.conflict_groups, usize::from(expected.is_none()));
        }
    }

    #[test]
    fn traction20d_rounding_requires_exact_other_fields() {
        for column in FUND_PORTFOLIO_COLUMNS[5..].iter() {
            let mut source = table(&[
                ("F", 20260420, 20260331, "000001.SZ", 100.),
                ("F", 20260420, 20260331, "000001.SZ", 100.25),
            ]);
            for other in FUND_PORTFOLIO_COLUMNS[5..].iter() {
                source
                    .columns
                    .insert((*other).into(), ColumnData::F64(vec![Some(1.); 2]));
            }
            source.columns.insert(
                (*column).into(),
                ColumnData::F64(vec![Some(1.), Some(1.000001)]),
            );
            let index = FundPortfolioBatchIndex::new(
                vec![Arc::new(
                    FundPortfolioIndex::from_table(Arc::new(source)).unwrap(),
                )],
                vec![],
                20260430,
            );
            assert!(
                select_holdings(index.reader().disclosures_between(None, 20260420)[0]).1,
                "{column}"
            );
        }
    }

    #[test]
    fn traction20d_factor_owns_conflict_policy_and_never_falls_back() {
        let index = index(&[
            ("F", 20260420, 20260331, "000001.SZ", 10.),
            ("F", 20260420, 20260331, "000001.SZ", 10.),
            ("F", 20260422, 20260331, "000001.SZ", 11.),
            ("F", 20260422, 20260331, "000001.SZ", 12.),
            ("F", 20260422, 20260331, "000002.SZ", 20.),
            ("F", 20260423, 20251231, "000001.SZ", 9.),
            ("F", 20260424, 20260331, "000001.SZ", 13.),
        ]);
        let reader = index.reader();
        let mut state = TractionState::default();
        for group in reader.disclosures_between(None, 20260420) {
            assert!(apply_disclosure(&mut state, group));
        }
        assert_eq!(state.funds["F"].holdings.len(), 1);
        for group in reader.disclosures_between(Some(20260420), 20260422) {
            assert!(apply_disclosure(&mut state, group));
        }
        assert!(state.funds["F"].holdings.is_empty());
        assert_eq!(state.conflict_groups, 1);
        for group in reader.disclosures_between(Some(20260422), 20260423) {
            assert!(!apply_disclosure(&mut state, group));
        }
        assert!(state.funds["F"].holdings.is_empty());
        for group in reader.disclosures_between(Some(20260423), 20260424) {
            assert!(apply_disclosure(&mut state, group));
        }
        assert_eq!(state.funds["F"].holdings[0].market_value, 13.);
    }

    #[test]
    fn traction20d_conflicts_in_non_mkv_original_fields_exclude_whole_group() {
        let mut source = table(&[
            ("F", 20260420, 20260331, "000001.SZ", 10.),
            ("F", 20260420, 20260331, "000001.SZ", 10.),
        ]);
        source.columns.insert(
            "stk_float_ratio".into(),
            ColumnData::F64(vec![Some(0.), Some(0.1)]),
        );
        let index = FundPortfolioBatchIndex::new(
            vec![Arc::new(
                FundPortfolioIndex::from_table(Arc::new(source)).unwrap(),
            )],
            vec![],
            20260430,
        );
        assert!(select_holdings(index.reader().disclosures_between(None, 20260420)[0]).1);
    }

    #[test]
    fn traction20d_top_ten_is_per_fund_with_stable_ties_and_valid_a_shares() {
        let codes = (1..=12).map(|i| format!("{i:06}.SZ")).collect::<Vec<_>>();
        let mut rows = codes
            .iter()
            .rev()
            .map(|code| ("F", 20260420, 20260331, code.as_str(), 10.))
            .collect::<Vec<_>>();
        rows.extend([
            ("F", 20260420, 20260331, "A26018.SZ", 1000.),
            ("F", 20260420, 20260331, "430001.BJ", 1000.),
            ("F", 20260420, 20260331, "600001.SH", 0.),
            ("F", 20260420, 20260331, "600002.SH", f64::INFINITY),
        ]);
        let index = index(&rows);
        let reader = index.reader();
        let (holdings, conflict) = select_holdings(reader.disclosures_between(None, 20260420)[0]);
        assert!(!conflict);
        assert_eq!(holdings.len(), 10);
        assert_eq!(holdings[0].code, "000001.SZ");
        assert_eq!(holdings[9].code, "000010.SZ");
    }

    #[test]
    fn traction20d_new_disclosure_updates_fund_marker_even_with_identical_values() {
        let index = index(&[
            ("F", 20260420, 20260331, "000001.SZ", 10.),
            ("F", 20260421, 20260331, "000001.SZ", 10.),
        ]);
        let reader = index.reader();
        let mut state = TractionState::default();
        for group in reader.disclosures_between(None, 20260421) {
            assert!(apply_disclosure(&mut state, group));
        }
        assert!(!apply_disclosure(
            &mut state,
            reader.disclosures_between(Some(20260420), 20260421)[0]
        ));
    }

    #[test]
    fn traction20d_weekend_disclosures_merge_on_next_trade_without_future_rows() {
        let index = index(&[
            ("F", 20260102, 20251231, "000001.SZ", 10.),
            ("F", 20260103, 20251231, "000001.SZ", 20.),
            ("G", 20260104, 20251231, "000002.SZ", 30.),
            ("H", 20260105, 20251231, "000003.SZ", 40.),
            ("F", 20260106, 20251231, "000001.SZ", 50.),
        ]);
        let reader = index.reader();
        let mut state = TractionState::default();
        for group in reader.disclosures_between(None, 20260102) {
            apply_disclosure(&mut state, group);
        }
        assert_eq!(state.funds.len(), 1);
        let groups = reader.disclosures_between(Some(20260102), 20260105);
        assert_eq!(groups.len(), 3);
        for group in groups {
            apply_disclosure(&mut state, group);
        }
        assert_eq!(state.funds.len(), 3);
        assert_eq!(state.funds["F"].holdings[0].market_value, 20.);
        assert_eq!(state.funds["H"].ann_date, 20260105);
    }

    #[test]
    fn traction20d_sparse_network_sums_shared_funds_and_counts_unique_peers() {
        let index = index(&[
            ("F", 20260420, 20260331, "000001.SZ", 200.),
            ("F", 20260420, 20260331, "000002.SZ", 300.),
            ("F", 20260420, 20260331, "000003.SZ", 800.),
            ("G", 20260420, 20260331, "000001.SZ", 100.),
            ("G", 20260420, 20260331, "000002.SZ", 600.),
        ]);
        let reader = index.reader();
        let mut state = TractionState::default();
        for group in reader.disclosures_between(None, 20260420) {
            apply_disclosure(&mut state, group);
        }
        let network = build_network(
            &state.funds,
            &HashMap::from([
                ("000001.SZ", 100.),
                ("000002.SZ", 100.),
                ("000003.SZ", 100.),
            ]),
        );
        assert_eq!(network.edges, vec![(0, 1, 3.), (0, 2, 2.), (1, 2, 3.)]);
        let (raw, alpha) = daily_raw(&network, &[Some(0.), Some(1.), Some(3.)]);
        assert_eq!(alpha, vec![Some(-1.), Some(0.), Some(2.)]);
        assert_eq!(raw, vec![Some(2.), Some(1.5), Some(-1.)]);
        let (raw, _) = daily_raw(&network, &[Some(0.), Some(1.), None]);
        assert_eq!(raw[0], Some(1.5));
        assert_eq!(raw[2], None);
    }

    #[test]
    fn traction20d_return_compounds_available_days_min_periods_one() {
        let r = compounded_return(&[Some(10.), Some(11.), None, Some(12.), Some(18.)]);
        assert_eq!(r[0], None);
        assert!((r[1].unwrap() - 0.1).abs() < 1e-12);
        assert!((r[4].unwrap() - 0.65).abs() < 1e-12);
        let prices = (0..=21).map(|i| Some(1.01f64.powi(i))).collect::<Vec<_>>();
        let r = compounded_return(&prices);
        assert!((r[21].unwrap() - (1.01f64.powi(20) - 1.)).abs() < 1e-12);
    }

    #[test]
    fn traction20d_regression_controls_alpha_and_sector_without_size() {
        let alpha = [Some(-2.), Some(-1.), Some(0.), Some(1.), Some(2.), Some(3.)];
        let raw = [
            Some(-3.),
            Some(-1.),
            Some(1.),
            Some(6.),
            Some(8.),
            Some(10.),
        ];
        let groups = [
            Some("A".into()),
            Some("A".into()),
            Some("A".into()),
            Some("B".into()),
            Some("B".into()),
            Some("B".into()),
        ];
        let residual = cs_neutralize_regression(&raw, &[&alpha], Some(&groups), None);
        assert!(residual
            .into_iter()
            .all(|v| v.is_some_and(|v| v.abs() < 1e-10)));
        let spec = StockDailyTraction20d.spec();
        assert!(spec.tags.iter().any(|v| v == "alternative"));
        assert!(!spec.tags.iter().any(|v| v == "fundamental"));
        assert!(!spec
            .dependencies
            .iter()
            .any(|v| v.dataset == DatasetId::StockBarraDaily));
        assert!(
            crate::factor::chn_stock::daily::mainbz_mom_resvol::spec_for()
                .tags
                .iter()
                .any(|v| v == "alternative")
        );
    }

    fn fixture(context: &FactorContext, codes: &[&str]) -> DataPool {
        let dates = &context.load_dates;
        let mut pv_codes = Vec::new();
        let mut pv_dates = Vec::new();
        let mut close = Vec::new();
        let mut amount = Vec::new();
        for date in dates {
            for code in codes.iter().rev() {
                let i = code[..6].parse::<usize>().unwrap();
                pv_codes.push(Some((*code).into()));
                pv_dates.push(Some(*date));
                close.push(Some(
                    (10. + i as f64) * (1. + 0.0005 * i as f64).powi((*date % 100) % 8),
                ));
                amount.push(Some(1. + (*date % 100) as f64));
            }
        }
        let basic = Table::new(BTreeMap::from([
            (
                "ts_code".into(),
                ColumnData::Utf8(codes.iter().map(|c| Some((*c).into())).collect()),
            ),
            (
                "list_date".into(),
                ColumnData::I32(vec![Some(20000101); codes.len()]),
            ),
            (
                "delist_date".into(),
                ColumnData::I32(vec![None; codes.len()]),
            ),
        ]))
        .unwrap();
        let pv = Table::new(BTreeMap::from([
            ("ts_code".into(), ColumnData::Utf8(pv_codes.clone())),
            ("trade_date".into(), ColumnData::I32(pv_dates.clone())),
            ("close".into(), ColumnData::F64(close)),
            ("amount".into(), ColumnData::F64(amount)),
        ]))
        .unwrap();
        let adjustment = Table::new(BTreeMap::from([
            ("ts_code".into(), ColumnData::Utf8(pv_codes)),
            ("trade_date".into(), ColumnData::I32(pv_dates)),
            (
                "adj_factor".into(),
                ColumnData::F64(vec![Some(1.); codes.len() * dates.len()]),
            ),
        ]))
        .unwrap();
        let sectors = Table::new(BTreeMap::from([
            (
                "ts_code".into(),
                ColumnData::Utf8(codes.iter().map(|c| Some((*c).into())).collect()),
            ),
            (
                "in_date".into(),
                ColumnData::I32(vec![Some(20000101); codes.len()]),
            ),
            ("out_date".into(), ColumnData::I32(vec![None; codes.len()])),
            (
                "l1_code".into(),
                ColumnData::Utf8(vec![Some("SW1".into()); codes.len()]),
            ),
        ]))
        .unwrap();
        let holding_rows = codes
            .iter()
            .filter(|c| **c != "000005.SZ")
            .flat_map(|code| {
                [
                    (
                        "F",
                        20251231,
                        20250930,
                        *code,
                        code[..6].parse::<f64>().unwrap() * 5000.,
                    ),
                    (
                        "F",
                        20260102,
                        20251231,
                        *code,
                        code[..6].parse::<f64>().unwrap() * 10000.,
                    ),
                    (
                        "F",
                        20260106,
                        20251231,
                        *code,
                        code[..6].parse::<f64>().unwrap() * 20000.,
                    ),
                    (
                        "F",
                        20260203,
                        20251231,
                        *code,
                        code[..6].parse::<f64>().unwrap() * 30000.,
                    ),
                ]
            })
            .collect::<Vec<_>>();
        let mut data = DataPool::from_daily_tables_for_test(
            HashMap::from([
                (DatasetId::StockBasic, basic),
                (DatasetId::StockDailyPv, pv),
                (DatasetId::StockAdjFactor, adjustment),
                (DatasetId::StockSwClassification, sectors),
                (DatasetId::FundPortfolio, table(&holding_rows)),
            ]),
            context,
        )
        .unwrap();
        data.set_trading_calendar(Arc::new(crate::calendar::TradingCalendar::from_open_dates(
            vec![
                20251230, 20251231, 20260102, 20260105, 20260106, 20260130, 20260202, 20260203,
                20260227, 20260302,
            ],
        )));
        data
    }

    fn context(dates: Vec<i32>, targets: Vec<i32>) -> FactorContext {
        FactorContext {
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            start_date: targets[0],
            end_date: *targets.last().unwrap(),
            load_start_date: dates[0],
            load_dates: dates,
            target_dates: targets,
        }
    }

    #[test]
    fn traction20d_month_end_final_replay_survives_batch_remap() {
        let codes = ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ"];
        let full_context = context(
            vec![
                20251230, 20251231, 20260102, 20260105, 20260106, 20260130, 20260202, 20260203,
                20260227,
            ],
            vec![
                20260102, 20260105, 20260106, 20260130, 20260202, 20260203, 20260227,
            ],
        );
        let data = fixture(&full_context, &codes);
        assert!(data.daily(DatasetId::FundPortfolio).is_err());
        let mut state = TractionState::default();
        let full = compute(&full_context, &data, &mut state).unwrap();
        assert_eq!(state.last_processed, Some(20260227));
        assert_eq!(full.values.len(), 28);
        assert!(full.values.iter().any(|v| v.value.is_some()));
        let on_date = |date| {
            full.values
                .iter()
                .filter_map(|v| match &v.key {
                    crate::core::FactorRowKey::Daily { trade_date, .. } if *trade_date == date => {
                        Some(v.value)
                    }
                    _ => None,
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(on_date(20260102), on_date(20260106));
        assert_eq!(on_date(20260130), on_date(20260203));
        assert_ne!(on_date(20260106), on_date(20260130));
        assert_ne!(on_date(20260203), on_date(20260227));
        let first_context = context(
            vec![20251230, 20251231, 20260102, 20260105, 20260106],
            vec![20260102, 20260105, 20260106],
        );
        let first_data = fixture(&first_context, &codes);
        let mut state = TractionState::default();
        let first = compute(&first_context, &first_data, &mut state).unwrap();
        assert_eq!(state.last_processed, Some(20251231));
        assert_eq!(state.funds["F"].ann_date, 20251231);
        let next_context = context(
            full_context.load_dates.clone(),
            vec![20260130, 20260202, 20260203, 20260227],
        );
        let next_data = fixture(
            &next_context,
            &[
                "000005.SZ",
                "000004.SZ",
                "000003.SZ",
                "000002.SZ",
                "000001.SZ",
            ],
        );
        let next = compute(&next_context, &next_data, &mut state).unwrap();
        assert_eq!(state.last_processed, Some(20260227));
        assert_eq!(state.funds["F"].ann_date, 20260203);
        let values = first
            .values
            .into_iter()
            .chain(next.values)
            .filter_map(|v| match &v.key {
                crate::core::FactorRowKey::Daily { ts_code, .. } if ts_code != "000005.SZ" => {
                    Some((v.key, v.value))
                }
                _ => None,
            })
            .collect::<HashMap<_, _>>();
        for v in full.values {
            assert_eq!(values.get(&v.key), Some(&v.value));
        }
    }

    #[test]
    fn traction20d_mid_month_start_uses_previous_month_not_start_date() {
        let codes = ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ"];
        let ctx = context(
            vec![20251230, 20251231, 20260102, 20260105, 20260106],
            vec![20260106],
        );
        let mut state = TractionState::default();
        let output = compute(&ctx, &fixture(&ctx, &codes), &mut state).unwrap();
        assert!(output.values.iter().any(|v| v.value.is_some()));
        assert_eq!(state.last_processed, Some(20251231));
        assert_eq!(state.funds["F"].ann_date, 20251231);
        let replay = context(vec![20260106], vec![20260106]);
        let again = compute(&replay, &fixture(&replay, &codes), &mut state).unwrap();
        for (a, b) in output.values.iter().zip(&again.values) {
            assert_eq!(a.key, b.key);
            assert_eq!(a.value, b.value);
        }
    }

    #[test]
    fn traction20d_unknown_month_boundary_does_not_force_initialization() {
        let codes = ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ"];
        let ctx = context(vec![20260102, 20260105], vec![20260102, 20260105]);
        let mut data = fixture(&ctx, &codes);
        data.set_trading_calendar(Arc::new(crate::calendar::TradingCalendar::from_open_dates(
            vec![20260102, 20260105, 20260106],
        )));
        let mut state = TractionState::default();
        let output = compute(&ctx, &data, &mut state).unwrap();
        assert!(output.values.iter().all(|v| v.value.is_none()));
        assert_eq!(state.last_processed, None);
        assert!(state.funds.is_empty());
    }

    #[test]
    fn traction20d_month_end_refreshes_even_without_new_disclosures() {
        let codes = ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ"];
        let ctx = context(
            vec![20251230, 20251231, 20260102, 20260105, 20260106, 20260130],
            vec![20260102, 20260130],
        );
        let source = fixture(&ctx, &codes);
        let mut daily = HashMap::from([
            (
                DatasetId::StockBasic,
                source.daily(DatasetId::StockBasic).unwrap().clone(),
            ),
            (
                DatasetId::StockDailyPv,
                source.daily(DatasetId::StockDailyPv).unwrap().clone(),
            ),
            (
                DatasetId::StockAdjFactor,
                source.daily(DatasetId::StockAdjFactor).unwrap().clone(),
            ),
            (
                DatasetId::StockSwClassification,
                source
                    .daily(DatasetId::StockSwClassification)
                    .unwrap()
                    .clone(),
            ),
        ]);
        let rows = codes
            .iter()
            .map(|code| {
                (
                    "F",
                    20251231,
                    20250930,
                    *code,
                    code[..6].parse::<f64>().unwrap() * 5000.,
                )
            })
            .collect::<Vec<_>>();
        daily.insert(DatasetId::FundPortfolio, table(&rows));
        let pv = daily.get_mut(&DatasetId::StockDailyPv).unwrap();
        let close = pv.required_f64_cast("close").unwrap();
        let dates = pv.required_i32("trade_date").unwrap();
        let symbols = pv.required_utf8("ts_code").unwrap();
        let changed_close = close
            .iter()
            .zip(dates)
            .zip(symbols)
            .map(|((price, date), code)| {
                if *date == Some(20260130) {
                    price.map(|p| {
                        p * (1. + code.as_ref().unwrap()[..6].parse::<f64>().unwrap() * 0.01)
                    })
                } else {
                    *price
                }
            })
            .collect();
        pv.columns
            .insert("close".into(), ColumnData::F64(changed_close));
        let mut data = DataPool::from_daily_tables_for_test(daily, &ctx).unwrap();
        data.set_trading_calendar(Arc::new(crate::calendar::TradingCalendar::from_open_dates(
            vec![
                20251230, 20251231, 20260102, 20260105, 20260106, 20260130, 20260202,
            ],
        )));
        let mut state = TractionState::default();
        let output = compute(&ctx, &data, &mut state).unwrap();
        assert_eq!(state.last_processed, Some(20260130));
        assert_eq!(state.funds["F"].ann_date, 20251231);
        let values = |date| {
            output
                .values
                .iter()
                .filter_map(|v| match v.key {
                    crate::core::FactorRowKey::Daily { trade_date, .. } if trade_date == date => {
                        Some(v.value)
                    }
                    _ => None,
                })
                .collect::<Vec<_>>()
        };
        assert!(values(20260102).iter().any(Option::is_some));
        assert!(values(20260130).iter().any(Option::is_some));
        assert_ne!(values(20260102), values(20260130));
    }
}

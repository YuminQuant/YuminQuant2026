use std::any::Any;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use crate::calendar::TradingCalendar;
use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::{DataPool, FundDisclosureView, Table, FUND_PORTFOLIO_COLUMNS};
use crate::error::{err, Result};
use crate::factor::{Factor, FactorUpdatePolicy};
use crate::operators::{cs_neutralize_regression, cs_zscore};

const INITIAL_LOOKBACK: usize = 32;
const EXPOSURES: [&str; 3] = ["VOLATILITY", "MOMENTUM", "SIZE"];
type Holdings = Vec<(String, f64)>;
type StockAmounts = BTreeMap<String, f64>;

#[derive(Clone, Debug)]
struct Snapshot {
    ann_date: i32,
    holdings: Option<Holdings>,
}

#[derive(Default)]
struct FoaState {
    eligible: Option<HashSet<String>>,
    quarterly_period: Option<i32>,
    quarterly: BTreeMap<String, Snapshot>,
    half_period: Option<i32>,
    half: BTreeMap<i32, BTreeMap<String, Snapshot>>,
    market_caps: BTreeMap<i32, StockAmounts>,
    last_anchor: Option<i32>,
    last_target: Option<i32>,
    final_values: HashMap<String, Option<f64>>,
    conflict_groups: usize,
}

pub struct StockDailyFoa;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyFoa)
}

impl Factor for StockDailyFoa {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "foa".into(), aliases: vec!["FOA".into()], name: "Fund holdings alpha".into(),
            asset_class: AssetClass::Stock, frequency: Frequency::Daily, version: "0.1.0".into(),
            tags: ["deprecated", "KYZQ", "alternative", "fund_holdings", "pit", "monthly_update", "neutralize", "barra", "daily"].into_iter().map(str::to_string).collect(),
            description: "Approximate KYZQ FOA: month-end PIT fund top-ten FOP residualized against VOLATILITY/MOMENTUM/SIZE (no industry), combined equally after cross-sectional zscore with common-fund half-year FOC. Both components required. Fund classification is a current snapshot; enhanced index funds included, passive/ETF/feeder/QDII excluded; share classes counted independently. Full disclosures approximated by Aug 1/Feb 1 cutoffs, with report-end market weights. Hold final values by stock code until next completed month-end.".into(),
            dependencies: vec![
                DataRequest::new(DatasetId::FundPortfolio, FUND_PORTFOLIO_COLUMNS),
                DataRequest::new(DatasetId::FundBasic, &["fund_type", "invest_type", "name"]),
                DataRequest::new(DatasetId::StockDailyBasic, &["total_mv"]),
                DataRequest::new(DatasetId::StockBarraDaily, &EXPOSURES),
            ],
            intraday_raw_dependencies: vec![], lookback: Lookback { trading_days: INITIAL_LOOKBACK },
        }
    }

    fn requirements_for_context(&self, context: &FactorContext) -> Vec<DataRequest> {
        self.spec()
            .dependencies
            .into_iter()
            .map(|request| {
                let dates = match request.dataset {
                    // Request only candidate report-end sessions, rather than daily market caps.
                    DatasetId::StockDailyBasic => report_end_dates(context),
                    DatasetId::StockBarraDaily => context
                        .load_dates
                        .iter()
                        .copied()
                        .filter(|d| d % 100 >= 20)
                        .collect(),
                    _ => context.load_dates.clone(),
                };
                request.with_explicit_dates(dates)
            })
            .collect()
    }

    fn update_policy(&self) -> FactorUpdatePolicy {
        FactorUpdatePolicy::FinancialEventSnapshot
    }

    fn initial_compute_state(&self, _: &[String]) -> Box<dyn Any + Send> {
        Box::<FoaState>::default()
    }

    fn compute(&self, context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        compute(context, data, &mut FoaState::default())
    }

    fn compute_many_stateful(
        &self,
        ids: &[String],
        context: &FactorContext,
        data: &DataPool,
        state: &mut (dyn Any + Send),
    ) -> Result<Vec<FactorSeries>> {
        if !ids.iter().any(|id| id == "foa") {
            return Ok(vec![]);
        }
        let state = state
            .downcast_mut::<FoaState>()
            .ok_or_else(|| err("foa state type mismatch"))?;
        Ok(vec![compute(context, data, state)?])
    }
}

fn report_end_dates(context: &FactorContext) -> Vec<i32> {
    let mut dates = Vec::new();
    for year in context.load_start_date / 10000 - 2..=context.end_date / 10000 {
        for (month, last) in [(3, 31), (6, 30), (9, 30), (12, 31)] {
            for day in 20..=last {
                let date = year * 10000 + month * 100 + day;
                if date <= context.end_date {
                    dates.push(date);
                }
            }
        }
    }
    dates
}

fn standard_stock(code: &str) -> bool {
    let b = code.as_bytes();
    b.len() == 9 && b[..6].iter().all(u8::is_ascii_digit) && matches!(&b[6..], b".SH" | b".SZ")
}

fn clean(value: Option<f64>) -> Option<f64> {
    value.filter(|v| v.is_finite())
}

fn active_equity(fund_type: Option<&str>, invest_type: Option<&str>, name: Option<&str>) -> bool {
    // Vendor classifications are a current snapshot, not historical PIT fund categories.
    let (Some(kind), Some(investment), Some(name)) = (fund_type, invest_type, name) else {
        return false;
    };
    let name = name.to_uppercase();
    matches!(kind, "股票型" | "混合型")
        && investment != "被动指数型"
        && !["ETF", "联接", "QDII"]
            .iter()
            .any(|token| name.contains(token))
}

fn eligible_funds(table: &Table) -> Result<HashSet<String>> {
    let codes = table.required_utf8("ts_code")?;
    let types = table.required_utf8("fund_type")?;
    let investments = table.required_utf8("invest_type")?;
    let names = table.required_utf8("name")?;
    Ok((0..table.len)
        .filter_map(|i| {
            active_equity(
                types[i].as_deref(),
                investments[i].as_deref(),
                names[i].as_deref(),
            )
            .then(|| codes[i].clone())
            .flatten()
        })
        .collect())
}

fn previous_half(period: i32) -> i32 {
    if period % 10000 == 630 {
        (period / 10000 - 1) * 10000 + 1231
    } else {
        period / 10000 * 10000 + 630
    }
}

fn full_disclosure(period: i32, announcement: i32) -> bool {
    let cutoff = match period % 10000 {
        630 => period / 10000 * 10000 + 801,
        1231 => (period / 10000 + 1) * 10000 + 201,
        _ => return false,
    };
    announcement >= cutoff
}

fn same_value(a: Option<f64>, b: Option<f64>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(a), Some(b)) => a == b || (a.is_nan() && b.is_nan()),
        _ => false,
    }
}

fn holdings(group: FundDisclosureView<'_>) -> Option<Holdings> {
    let mut rows = BTreeMap::<&str, (usize, f64, f64)>::new();
    for &row in group.rows() {
        let Some(code) = group.symbol(row).filter(|code| standard_stock(code)) else {
            continue;
        };
        let value = clean(group.value(row, "mkv")).filter(|v| *v >= 0.0)?;
        if let Some((previous, lo, hi)) = rows.get_mut(code) {
            if FUND_PORTFOLIO_COLUMNS[5..]
                .iter()
                .any(|c| !same_value(group.value(*previous, c), group.value(row, c)))
            {
                return None;
            }
            *lo = lo.min(value);
            *hi = hi.max(value);
            // Small vendor rounding differences must not duplicate a stock's contribution.
            if *hi - *lo > 0.5 {
                return None;
            }
        } else {
            rows.insert(code, (row, value, value));
        }
    }
    let mut result = rows
        .into_iter()
        .filter_map(|(code, (_, _, value))| (value > 0.0).then(|| (code.to_owned(), value)))
        .collect::<Vec<_>>();
    result.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    (!result.is_empty()).then_some(result)
}

fn apply_disclosure(state: &mut FoaState, group: FundDisclosureView<'_>) {
    if !state
        .eligible
        .as_ref()
        .is_some_and(|funds| funds.contains(group.fund_code()))
    {
        return;
    }
    let period = group.end_date();
    if !matches!(period % 10000, 331 | 630 | 930 | 1231) {
        return;
    }
    let full = full_disclosure(period, group.ann_date());
    let new_quarter = state.quarterly_period.is_none_or(|p| period > p);
    if new_quarter {
        state.quarterly_period = Some(period);
        state.quarterly.clear();
    }
    let quarter_current = state.quarterly_period == Some(period);
    let half_needed = full && state.half_period.is_none_or(|p| period >= previous_half(p));
    if !quarter_current && !half_needed {
        return;
    }
    let selected = holdings(group);
    if selected.is_none() {
        state.conflict_groups += 1;
    }
    if quarter_current
        && state
            .quarterly
            .get(group.fund_code())
            .is_none_or(|old| old.ann_date < group.ann_date())
    {
        let top = selected
            .as_ref()
            .map(|h| h.iter().take(10).cloned().collect());
        state.quarterly.insert(
            group.fund_code().into(),
            Snapshot {
                ann_date: group.ann_date(),
                holdings: top,
            },
        );
    }
    if half_needed {
        if state.half_period.is_none_or(|p| period > p) {
            state.half_period = Some(period);
            state.half.retain(|p, _| *p >= previous_half(period));
        }
        let funds = state.half.entry(period).or_default();
        if funds
            .get(group.fund_code())
            .is_none_or(|old| old.ann_date < group.ann_date())
        {
            funds.insert(
                group.fund_code().into(),
                Snapshot {
                    ann_date: group.ann_date(),
                    holdings: selected,
                },
            );
        }
    }
}

fn sum_holdings<'a>(snapshots: impl Iterator<Item = &'a Snapshot>) -> StockAmounts {
    let mut output = BTreeMap::new();
    for snapshot in snapshots {
        if let Some(holdings) = &snapshot.holdings {
            for (code, value) in holdings {
                *output.entry(code.clone()).or_default() += value;
            }
        }
    }
    output
}

fn common_holdings(state: &FoaState, current: i32) -> Option<(StockAmounts, StockAmounts)> {
    let now = state.half.get(&current)?;
    let old = state.half.get(&previous_half(current))?;
    let funds = now
        .keys()
        .filter(|code| {
            now[*code].holdings.is_some() && old.get(*code).is_some_and(|s| s.holdings.is_some())
        })
        .collect::<Vec<_>>();
    if funds.is_empty() {
        return None;
    }
    Some((
        sum_holdings(funds.iter().map(|code| &now[*code])),
        sum_holdings(funds.iter().map(|code| &old[*code])),
    ))
}

fn overweight(codes: &[String], held: &StockAmounts, caps: &StockAmounts) -> Vec<Option<f64>> {
    // Use the same valid market universe in both normalized weight denominators.
    let market_total: f64 = caps.values().sum();
    let held_total: f64 = held
        .iter()
        .filter(|(code, _)| caps.contains_key(*code))
        .map(|(_, value)| *value)
        .sum();
    if !market_total.is_finite()
        || market_total <= 0.0
        || !held_total.is_finite()
        || held_total <= 0.0
    {
        return vec![None; codes.len()];
    }
    codes
        .iter()
        .map(|code| {
            let cap = *caps.get(code)?;
            clean(Some(
                held.get(code).copied().unwrap_or(0.0) / held_total / (cap / market_total) - 1.0,
            ))
        })
        .collect()
}

fn half_change(now: Option<f64>, old: Option<f64>) -> Option<f64> {
    let denominator = 1.0 + clean(old)?;
    if denominator <= 0.0 {
        return None;
    }
    clean(Some((1.0 + clean(now)?) / denominator - 1.0))
}

fn combine(
    foc: &[Option<f64>],
    quarterly: &[Option<f64>],
    exposures: &[Vec<Option<f64>>; 3],
) -> Vec<Option<f64>> {
    let x = exposures.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let residual = cs_neutralize_regression(quarterly, &x, None, None);
    let foc = cs_zscore(foc);
    let residual = cs_zscore(&residual);
    foc.into_iter()
        .zip(residual)
        .map(|(a, b)| clean(a.zip(b).map(|(a, b)| (a + b) / 2.0)))
        .collect()
}

fn cache_caps(
    state: &mut FoaState,
    data: &DataPool,
    periods: &BTreeSet<i32>,
    legal: &HashSet<&str>,
    calendar: &TradingCalendar,
) -> Result<()> {
    state.market_caps.retain(|p, _| periods.contains(p));
    let missing = periods
        .iter()
        .filter(|p| !state.market_caps.contains_key(*p))
        .copied()
        .collect::<Vec<_>>();
    if missing.is_empty() {
        return Ok(());
    }
    let table = data.daily(DatasetId::StockDailyBasic)?;
    let codes = table.required_utf8("ts_code")?;
    let dates = table.required_i32_date_cast("trade_date")?;
    let values = table.required_f64_cast("total_mv")?;
    for period in missing {
        let Some(date) = calendar.last_open_on_or_before(period) else {
            continue;
        };
        let mut caps = BTreeMap::new();
        for i in 0..table.len {
            if dates[i] != Some(date) {
                continue;
            }
            if let (Some(code), Some(value)) = (
                codes[i]
                    .as_deref()
                    .filter(|c| standard_stock(c) && legal.contains(c)),
                clean(values[i]).filter(|v| *v > 0.0),
            ) {
                caps.insert(code.to_owned(), value);
            }
        }
        if !caps.is_empty() {
            state.market_caps.insert(period, caps);
        }
    }
    Ok(())
}

fn compute(context: &FactorContext, data: &DataPool, state: &mut FoaState) -> Result<FactorSeries> {
    let panel = data.stock_universe_panel()?;
    let reader = data.fund_portfolio_reader()?;
    let calendar = data.trading_calendar()?;
    let n = panel.instruments().len();
    let legal = panel
        .instruments()
        .iter()
        .map(String::as_str)
        .collect::<HashSet<_>>();
    let mut output = vec![None; panel.shape_len()];
    let mut exposure_columns = None;
    let conflicts_before = state.conflict_groups;
    for (day, &date) in panel.dates().iter().enumerate() {
        if !panel.is_target_date(date) {
            continue;
        }
        if state.last_target.is_some_and(|previous| date < previous) {
            return Err(err("foa provider state cannot move backwards"));
        }
        if let Some(anchor) = calendar.month_end_on_or_before(date) {
            if state.last_anchor != Some(anchor) {
                let anchor_day = panel
                    .dates()
                    .binary_search(&anchor)
                    .map_err(|_| err("foa prior month-end missing from warmup panel"))?;
                if state.eligible.is_none() {
                    state.eligible = Some(eligible_funds(data.daily(DatasetId::FundBasic)?)?);
                }
                if state.last_anchor.is_none() {
                    reader.bootstrap_until(anchor, |group| {
                        apply_disclosure(state, group);
                        Ok(())
                    })?;
                } else {
                    for group in reader.disclosures_between(state.last_anchor, anchor) {
                        apply_disclosure(state, group);
                    }
                }
                let mut periods = BTreeSet::new();
                if let Some(period) = state.quarterly_period {
                    periods.insert(period);
                }
                if let Some(period) = state.half_period {
                    periods.extend([period, previous_half(period)]);
                }
                cache_caps(state, data, &periods, &legal, calendar)?;
                let mut quarterly = vec![None; n];
                let mut foc = vec![None; n];
                if let Some(caps) = state
                    .quarterly_period
                    .and_then(|p| state.market_caps.get(&p))
                {
                    quarterly = overweight(
                        panel.instruments(),
                        &sum_holdings(state.quarterly.values()),
                        caps,
                    );
                }
                if let Some(period) = state.half_period {
                    if let (Some((held_now, held_old)), Some(caps_now), Some(caps_old)) = (
                        common_holdings(state, period),
                        state.market_caps.get(&period),
                        state.market_caps.get(&previous_half(period)),
                    ) {
                        let now = overweight(panel.instruments(), &held_now, caps_now);
                        let old = overweight(panel.instruments(), &held_old, caps_old);
                        foc = now
                            .into_iter()
                            .zip(old)
                            .map(|(a, b)| half_change(a, b))
                            .collect();
                    }
                }
                if exposure_columns.is_none() {
                    exposure_columns = Some(
                        EXPOSURES
                            .iter()
                            .map(|name| {
                                panel.column_from_table(
                                    data.daily(DatasetId::StockBarraDaily)?,
                                    name,
                                )
                            })
                            .collect::<Result<Vec<_>>>()?,
                    );
                }
                let exposures = std::array::from_fn(|j| {
                    (0..n)
                        .map(|i| exposure_columns.as_ref().unwrap()[j].values()[anchor_day * n + i])
                        .collect()
                });
                for (i, code) in panel.instruments().iter().enumerate() {
                    if !standard_stock(code) || !panel.is_present_offset(anchor_day * n + i) {
                        quarterly[i] = None;
                        foc[i] = None;
                    }
                }
                state.final_values = panel
                    .instruments()
                    .iter()
                    .cloned()
                    .zip(combine(&foc, &quarterly, &exposures))
                    .collect();
                state.last_anchor = Some(anchor);
            }
        }
        for (i, code) in panel.instruments().iter().enumerate() {
            if panel.is_present_offset(day * n + i) {
                output[day * n + i] = state.final_values.get(code).copied().flatten();
            }
        }
        state.last_target = Some(date);
    }
    if state.conflict_groups > conflicts_before {
        eprintln!(
            "warning: foa excluded {} invalid/conflicting fund disclosure groups in {}..{}",
            state.conflict_groups - conflicts_before,
            context.start_date,
            context.end_date
        );
    }
    Ok(panel
        .column_from_values(output)?
        .to_factor_series(StockDailyFoa.spec()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{fund_portfolio::tests::table, ColumnData};
    use std::sync::Arc;

    fn state() -> FoaState {
        FoaState {
            eligible: Some(HashSet::from(["F".into(), "G".into()])),
            ..Default::default()
        }
    }

    fn apply(state: &mut FoaState, rows: &[(&str, i32, i32, &str, f64)]) {
        let context = context(vec![20260101], vec![20260101]);
        let pool = DataPool::from_daily_tables_for_test(
            HashMap::from([(DatasetId::FundPortfolio, table(rows))]),
            &context,
        )
        .unwrap();
        let reader = pool.fund_portfolio_reader().unwrap();
        // Reader's batch cap must also be respected by consumers.
        for group in reader.disclosures_between(None, 20991231) {
            apply_disclosure(state, group);
        }
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
    fn foa_fund_scope_retains_enhanced_and_expired_classifications() {
        assert!(active_equity(
            Some("股票型"),
            Some("增强指数型"),
            Some("Example Enhanced")
        ));
        assert!(active_equity(
            Some("混合型"),
            Some("灵活配置型"),
            Some("Example")
        ));
        for (kind, investment, name) in [
            ("股票型", "被动指数型", "Index"),
            ("债券型", "债券型", "Bond"),
            ("股票型", "股票型", "Example ETF"),
            ("混合型", "混合型", "ETF联接"),
            ("股票型", "股票型", "Example QDII"),
        ] {
            assert!(!active_equity(Some(kind), Some(investment), Some(name)));
        }
        assert!(!active_equity(Some("股票型"), None, Some("Example")));
    }

    #[test]
    fn foa_full_disclosure_cutoffs_do_not_depend_on_holding_count() {
        assert!(!full_disclosure(20250630, 20250731));
        assert!(full_disclosure(20250630, 20250801));
        assert!(!full_disclosure(20251231, 20260131));
        assert!(full_disclosure(20251231, 20260201));
        assert!(!full_disclosure(20250331, 20250801));
        let mut state = state();
        apply(
            &mut state,
            &[
                ("F", 20250720, 20250630, "000001.SZ", 5.),
                ("F", 20250820, 20250630, "000001.SZ", 10.),
            ],
        );
        assert_eq!(
            state.half[&20250630]["F"].holdings.as_ref().unwrap().len(),
            1
        );
        assert_eq!(state.quarterly["F"].holdings.as_ref().unwrap()[0].1, 10.);
    }

    #[test]
    fn foa_holdings_deduplicate_rounding_and_reject_conflicts() {
        let source = Arc::new(table(&[
            ("F", 20250820, 20250630, "000001.SZ", 10.),
            ("F", 20250820, 20250630, "000001.SZ", 10.),
            ("F", 20250820, 20250630, "830001.BJ", 100.),
            ("F", 20250820, 20250630, "A00001.SZ", 200.),
        ]));
        let index = crate::data::FundPortfolioBatchIndex::new(
            vec![Arc::new(
                crate::data::FundPortfolioIndex::from_table(source).unwrap(),
            )],
            vec![],
            20250820,
        );
        let reader = index.reader();
        let groups = reader.disclosures_between(None, 20250820);
        assert_eq!(holdings(groups[0]), Some(vec![("000001.SZ".into(), 10.)]));
        let mut state = state();
        apply(
            &mut state,
            &[
                ("F", 20250820, 20250630, "000001.SZ", 10.),
                ("F", 20250821, 20250630, "000001.SZ", 20.),
                ("F", 20250821, 20250630, "000001.SZ", 30.),
            ],
        );
        assert!(state.half[&20250630]["F"].holdings.is_none());
        assert!(state.quarterly["F"].holdings.is_none());
        assert_eq!(state.conflict_groups, 1);
    }

    #[test]
    fn foa_common_funds_and_bounded_period_state() {
        let mut state = state();
        apply(
            &mut state,
            &[
                ("F", 20240320, 20231231, "000001.SZ", 5.),
                ("F", 20240820, 20240630, "000001.SZ", 6.),
                ("F", 20250320, 20241231, "000001.SZ", 10.),
                ("F", 20250820, 20250630, "000001.SZ", 20.),
                ("G", 20250820, 20250630, "000002.SZ", 1000.),
                ("IGNORED", 20250820, 20250630, "000001.SZ", 9000.),
                ("F", 20251020, 20250930, "000001.SZ", 30.),
                ("F", 20251201, 20240630, "000001.SZ", 100.),
            ],
        );
        assert_eq!(state.half.len(), 2);
        assert_eq!(state.quarterly_period, Some(20250930));
        assert_eq!(state.quarterly["F"].holdings.as_ref().unwrap()[0].1, 30.);
        let (now, old) = common_holdings(&state, 20250630).unwrap();
        assert_eq!(now, BTreeMap::from([("000001.SZ".into(), 20.)]));
        assert_eq!(old, BTreeMap::from([("000001.SZ".into(), 10.)]));
    }

    #[test]
    fn foa_weights_include_unheld_market_stocks_and_zero_base_is_null() {
        let codes = ["000001.SZ", "000002.SZ", "000003.SZ"].map(str::to_string);
        let held = BTreeMap::from([("000001.SZ".into(), 30.), ("000002.SZ".into(), 10.)]);
        let caps = BTreeMap::from([
            ("000001.SZ".into(), 20.),
            ("000002.SZ".into(), 30.),
            ("000003.SZ".into(), 50.),
        ]);
        let values = overweight(&codes, &held, &caps);
        assert!((values[0].unwrap() - 2.75).abs() < 1e-12);
        assert!((values[1].unwrap() + 1.0 / 6.0).abs() < 1e-12);
        assert_eq!(values[2], Some(-1.));
        assert_eq!(half_change(Some(1.), Some(0.)), Some(1.));
        assert_eq!(half_change(Some(1.), Some(-1.)), None);
        assert_eq!(half_change(Some(-1.), Some(1.)), Some(-1.));
    }

    #[test]
    fn foa_regression_removes_all_three_exposures_and_requires_both_components() {
        let x: [Vec<Option<f64>>; 3] = std::array::from_fn(|j| {
            (0..9)
                .map(|i| {
                    Some(match j {
                        0 => i as f64,
                        1 => (i * i) as f64,
                        _ => (i % 2) as f64,
                    })
                })
                .collect()
        });
        let y = (0..9)
            .map(|i| Some((i * i * i) as f64 + 1.))
            .collect::<Vec<_>>();
        let mut growth = (0..9).map(|i| Some((i % 3) as f64)).collect::<Vec<_>>();
        growth[8] = None;
        let original = combine(&growth, &y, &x);
        let shifted = y
            .iter()
            .enumerate()
            .map(|(i, y)| {
                y.map(|y| y + 4. + 2. * x[0][i].unwrap() - 3. * x[1][i].unwrap() + x[2][i].unwrap())
            })
            .collect::<Vec<_>>();
        let adjusted = combine(&growth, &shifted, &x);
        for (a, b) in original.iter().zip(adjusted) {
            assert!(match (a, b) {
                (Some(a), Some(b)) => (*a - b).abs() < 1e-9,
                (None, None) => true,
                _ => false,
            });
        }
        assert!(original[..8].iter().all(Option::is_some));
        assert!(original[8].is_none());
        let residual = cs_neutralize_regression(
            &y,
            &x.iter().map(Vec::as_slice).collect::<Vec<_>>(),
            None,
            None,
        );
        for exposure in &x {
            assert!(
                residual
                    .iter()
                    .zip(exposure)
                    .map(|(r, x)| r.unwrap() * x.unwrap())
                    .sum::<f64>()
                    .abs()
                    < 1e-8
            );
        }
    }

    fn fixture(context: &FactorContext) -> DataPool {
        let codes = (1..=8)
            .map(|i| format!("{i:06}.SZ"))
            .chain(["830001.BJ".into()])
            .collect::<Vec<_>>();
        let basic = Table::new(BTreeMap::from([
            (
                "ts_code".into(),
                ColumnData::Utf8(codes.iter().cloned().map(Some).collect()),
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
        let fund = Table::new(BTreeMap::from([
            ("ts_code".into(), ColumnData::Utf8(vec![Some("F".into())])),
            (
                "fund_type".into(),
                ColumnData::Utf8(vec![Some("股票型".into())]),
            ),
            (
                "invest_type".into(),
                ColumnData::Utf8(vec![Some("增强指数型".into())]),
            ),
            (
                "name".into(),
                ColumnData::Utf8(vec![Some("Enhanced".into())]),
            ),
        ]))
        .unwrap();
        let mut rows = Vec::new();
        let mut cap_codes = Vec::new();
        let mut cap_dates = Vec::new();
        let mut caps = Vec::new();
        for (i, code) in codes.iter().enumerate() {
            let k = i as f64 + 1.;
            for (period, announcement, value) in [
                (20241231, 20250320, 100. + k),
                (20250630, 20250820, 100. + k * k),
                (20250930, 20251020, 100. + k * k * k),
                (20251231, 20260120, 110. + k * k * k),
                (20251231, 20260220, 120. + k * k * k * k),
            ] {
                rows.push(("F", announcement, period, code.as_str(), value));
            }
            for date in [20241231, 20250630, 20250930, 20251231] {
                cap_codes.push(Some(code.clone()));
                cap_dates.push(Some(date));
                caps.push(Some(if code.ends_with(".BJ") {
                    1000000.
                } else {
                    100. + k * 10.
                }));
            }
        }
        let market = Table::new(BTreeMap::from([
            ("ts_code".into(), ColumnData::Utf8(cap_codes)),
            ("trade_date".into(), ColumnData::I32(cap_dates)),
            ("total_mv".into(), ColumnData::F64(caps)),
        ]))
        .unwrap();
        let mut exposure_codes = Vec::new();
        let mut exposure_dates = Vec::new();
        let mut exposure_values: [Vec<Option<f64>>; 3] = std::array::from_fn(|_| Vec::new());
        for date in [20251231, 20260130, 20260227] {
            for (i, code) in codes.iter().enumerate() {
                exposure_codes.push(Some(code.clone()));
                exposure_dates.push(Some(date));
                exposure_values[0].push(Some(i as f64));
                exposure_values[1].push(Some((i * i) as f64));
                exposure_values[2].push(Some((i % 2) as f64));
            }
        }
        let mut exposure = BTreeMap::from([
            ("ts_code".into(), ColumnData::Utf8(exposure_codes)),
            ("trade_date".into(), ColumnData::I32(exposure_dates)),
        ]);
        for (name, values) in EXPOSURES.into_iter().zip(exposure_values) {
            exposure.insert(name.into(), ColumnData::F64(values));
        }
        let mut pool = DataPool::from_daily_tables_for_test(
            HashMap::from([
                (DatasetId::StockBasic, basic),
                (DatasetId::FundBasic, fund),
                (DatasetId::FundPortfolio, table(&rows)),
                (DatasetId::StockDailyBasic, market),
                (DatasetId::StockBarraDaily, Table::new(exposure).unwrap()),
            ]),
            context,
        )
        .unwrap();
        pool.set_trading_calendar(Arc::new(TradingCalendar::from_open_dates(vec![
            20241231, 20250630, 20250930, 20251231, 20260102, 20260105, 20260130, 20260202,
            20260227, 20260302, 20260303,
        ])));
        pool
    }

    #[test]
    fn foa_month_end_pit_replay_and_batch_restart_are_identical() {
        let dates = vec![
            20251231, 20260102, 20260105, 20260130, 20260202, 20260227, 20260302,
        ];
        let ctx = context(dates.clone(), dates[1..].to_vec());
        let pool = fixture(&ctx);
        let mut state = FoaState::default();
        let result = compute(&ctx, &pool, &mut state).unwrap();
        let values = |date| daily_values(&result, date);
        assert_eq!(values(20260102), values(20260105));
        assert_eq!(values(20260130), values(20260202));
        assert_eq!(values(20260227), values(20260302));
        assert!(values(20260102).values().any(Option::is_some));
        assert_ne!(values(20260105), values(20260130));
        assert_ne!(values(20260202), values(20260227));
        assert_eq!(values(20260302)["830001.BJ"], None);
        assert_eq!(state.half.len(), 2);
        assert_eq!(state.market_caps.len(), 2);
        let restarted_ctx = context(vec![20260130, 20260202], vec![20260202]);
        let restarted = compute(
            &restarted_ctx,
            &fixture(&restarted_ctx),
            &mut FoaState::default(),
        )
        .unwrap();
        assert_eq!(daily_values(&restarted, 20260202), values(20260202));

        // Replace the cached map's insertion order and change the request batch.
        let mut reordered = state.final_values.drain().collect::<Vec<_>>();
        reordered.reverse();
        state.final_values.extend(reordered);
        let next_ctx = context(vec![20260227, 20260302], vec![20260302]);
        let next = compute(&next_ctx, &fixture(&next_ctx), &mut state).unwrap();
        assert_eq!(daily_values(&next, 20260302), values(20260302));
    }

    fn daily_values(series: &FactorSeries, date: i32) -> BTreeMap<String, Option<f64>> {
        series
            .values
            .iter()
            .filter_map(|v| match &v.key {
                crate::core::FactorRowKey::Daily {
                    trade_date,
                    ts_code,
                } if *trade_date == date => Some((ts_code.clone(), v.value)),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn foa_metadata_has_no_pv_or_sector_and_projected_requirements_include_prior_year() {
        let factor = StockDailyFoa;
        assert!(factor.spec().tags.iter().any(|t| t == "KYZQ"));
        assert!(factor.spec().tags.iter().any(|t| t == "alternative"));
        assert!(factor.spec().tags.iter().any(|t| t == "deprecated"));
        assert!(!factor.spec().dependencies.iter().any(|d| matches!(
            d.dataset,
            DatasetId::StockDailyPv | DatasetId::StockSwClassification
        )));
        let ctx = context(
            vec![20251231, 20260102, 20260130, 20260202],
            vec![20260102, 20260130, 20260202],
        );
        let requests = factor.requirements_for_context(&ctx);
        let market = requests
            .iter()
            .find(|r| r.dataset == DatasetId::StockDailyBasic)
            .unwrap();
        assert_eq!(market.columns, ["total_mv"]);
        assert!(market.resolved_dates(&ctx).contains(&20250630));
        assert!(!market.resolved_dates(&ctx).contains(&20260102));
        let exposure = requests
            .iter()
            .find(|r| r.dataset == DatasetId::StockBarraDaily)
            .unwrap();
        assert_eq!(exposure.columns, EXPOSURES);
        assert_eq!(exposure.resolved_dates(&ctx), [20251231, 20260130]);
    }
}

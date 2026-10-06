use rayon::prelude::*;
use std::collections::HashMap;

use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::stock_daily_ops::{is_bj_stock, neutralize_size_sector};
use crate::factor::Factor;

const WINDOW: usize = 60;
const MIN_PAIRS: usize = 30;
const MARKET: &str = "000985.CSI";
const MAX_THETA: f64 = 64.0;

pub struct StockDailyAbnormalUpperTailDependence;
pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyAbnormalUpperTailDependence)
}

impl Factor for StockDailyAbnormalUpperTailDependence {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "abnormal_upper_tail_dependence".into(),
            aliases: vec!["AbUTD".into(), "Abnormal Upper-tail Dependence".into()],
            name: "Abnormal Upper-tail Dependence".into(),
            asset_class: AssetClass::Stock, frequency: Frequency::Daily, version: "0.1.0".into(),
            tags: ["DFZQ", "price_volume", "return", "copula", "tail_dependence", "market", "neutralize", "size", "sector", "daily", "deprecated"].into_iter().map(str::to_string).collect(),
            description: "Daily rolling 60-session Gumbel/Clayton maximum pseudo-likelihood tail dependence on 000985.CSI. At least 30 paired valid adjusted daily returns, positive volume, no suspension zero fill. Pairwise average ranks/(n+1); constant margins null. Independent fits (not Kendall inversion), theta capped at 64 with upper-bound optima rejected; independence allowed. Cross-sectional upper-tail OLS residual on lower-tail with intercept, then SW L1/Barra SIZE neutralization. Excludes BJ. No monthly freeze, winsorization or final zscore; close-of-day information.".into(),
            dependencies: vec![
                DataRequest::new(DatasetId::StockDailyPv, &["close", "vol"]),
                DataRequest::new(DatasetId::StockAdjFactor, &["adj_factor"]),
                DataRequest::index_daily(MARKET, &["close", "pre_close"]),
                DataRequest::new(DatasetId::StockBarraDaily, &["SIZE"]),
                DataRequest::new(DatasetId::StockSwClassification, &["l1_code"]),
            ], intraday_raw_dependencies: vec![], lookback: Lookback { trading_days: WINDOW },
        }
    }
    fn compute(&self, _: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let panel = data.daily_panel(DatasetId::StockDailyPv)?;
        let close = panel.column("close")?;
        let volume = panel.column("vol")?;
        let adj = panel.column_from_table(data.daily(DatasetId::StockAdjFactor)?, "adj_factor")?;
        let index = data.index_daily_panel(MARKET)?;
        let ic = index.column("close")?;
        let ip = index.column("pre_close")?;
        let market: HashMap<_, _> = index
            .dates()
            .iter()
            .enumerate()
            .map(|(d, date)| {
                let offset = d * index.instruments().len();
                (
                    *date,
                    if index.instruments().is_empty() {
                        None
                    } else {
                        ratio_return(ic.values()[offset], ip.values()[offset])
                    },
                )
            })
            .collect();
        let market: Vec<_> = panel
            .dates()
            .iter()
            .map(|d| market.get(d).copied().flatten())
            .collect();
        let n = panel.instruments().len();
        let days = panel.dates().len();
        let mut upper = vec![None; panel.shape_len()];
        let mut lower = vec![None; panel.shape_len()];
        // Bound temporary results while sharing the engine's Rayon pool. Each fit
        // keeps its original summation order; only independent stocks run in parallel.
        for start in (0..n).step_by(64) {
            let results: Vec<_> = (start..(start + 64).min(n))
                .into_par_iter()
                .map_init(
                    || (vec![None; days], Vec::with_capacity(WINDOW)),
                    |(returns, pairs), stock| {
                        let code = &panel.instruments()[stock];
                        let mut output = Vec::new();
                        if is_bj_stock(code) {
                            return (stock, output);
                        }
                        returns.fill(None);
                        for day in 1..days {
                            let now = day * n + stock;
                            let prev = now - n;
                            if panel.is_present_offset(now)
                                && panel.is_present_offset(prev)
                                && positive(volume.values()[now]).is_some()
                            {
                                returns[day] = ratio_return(
                                    adjusted(close.values()[now], adj.values()[now]),
                                    adjusted(close.values()[prev], adj.values()[prev]),
                                );
                            }
                        }
                        for day in WINDOW..days {
                            if !panel.is_target_date(panel.dates()[day])
                                || !panel.is_present_offset(day * n + stock)
                            {
                                continue;
                            }
                            pairs.clear();
                            for d in day + 1 - WINDOW..=day {
                                if let (Some(x), Some(y)) = (returns[d], market[d]) {
                                    pairs.push((x, y));
                                }
                            }
                            if let Some((u, l)) = tail_coefficients(&pairs) {
                                output.push((day, u, l));
                            }
                        }
                        (stock, output)
                    },
                )
                .collect();
            for (stock, output) in results {
                for (day, u, l) in output {
                    upper[day * n + stock] = Some(u);
                    lower[day * n + stock] = Some(l);
                }
            }
        }
        let upper = panel.column_from_values(upper)?;
        let lower = panel.column_from_values(lower)?;
        let residual = upper.cs_neutralize_regression(&[&lower], None)?;
        Ok(neutralize_size_sector(&residual, panel, data)?.to_factor_series(self.spec()))
    }
}

fn positive(v: Option<f64>) -> Option<f64> {
    v.filter(|x| x.is_finite() && *x > 0.0)
}
fn adjusted(price: Option<f64>, adj: Option<f64>) -> Option<f64> {
    positive(Some(positive(price)? * positive(adj)?))
}
fn ratio_return(now: Option<f64>, prev: Option<f64>) -> Option<f64> {
    let v = positive(now)? / positive(prev)? - 1.0;
    v.is_finite().then_some(v)
}

fn pseudo_observations(values: &[f64]) -> Option<Vec<f64>> {
    if values.len() < 2 || values.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let mut order: Vec<_> = (0..values.len()).collect();
    order.sort_by(|&a, &b| values[a].total_cmp(&values[b]));
    if values[order[0]] == values[*order.last()?] {
        return None;
    }
    let mut ranks = vec![0.0; values.len()];
    let mut start = 0;
    while start < order.len() {
        let mut stop = start + 1;
        while stop < order.len() && values[order[stop]] == values[order[start]] {
            stop += 1;
        }
        let rank = (start + 1 + stop) as f64 / (2.0 * (values.len() + 1) as f64);
        for &i in &order[start..stop] {
            ranks[i] = rank;
        }
        start = stop;
    }
    Some(ranks)
}

#[derive(Clone, Copy)]
enum Family {
    Gumbel,
    Clayton,
}
#[derive(Clone, Copy)]
struct Point {
    x: f64,
    y: f64,
    lx: f64,
    ly: f64,
}
impl Point {
    fn new(u: f64, v: f64) -> Self {
        let x = -u.ln();
        let y = -v.ln();
        Self {
            x,
            y,
            lx: x.ln(),
            ly: y.ln(),
        }
    }
}
fn log_add(a: f64, b: f64) -> f64 {
    let m = a.max(b);
    m + ((a - m).exp() + (b - m).exp()).ln()
}
fn log_density(p: Point, theta: f64, family: Family) -> f64 {
    match family {
        Family::Gumbel => {
            if theta == 1.0 {
                return 0.0;
            }
            let log_s = log_add(theta * p.lx, theta * p.ly);
            let log_a = log_s / theta;
            -log_a.exp()
                + log_add(log_a, (theta - 1.0).ln())
                + (1.0 / theta - 2.0) * log_s
                + (theta - 1.0) * (p.lx + p.ly)
                + p.x
                + p.y
        }
        Family::Clayton => {
            if theta == 0.0 {
                return 0.0;
            }
            // log(exp(theta*x)+exp(theta*y)-1), stable near zero and in the tail.
            let a = theta * p.x;
            let b = theta * p.y;
            let m = a.max(b);
            let log_s = if m < 0.5 {
                (a.exp_m1() + b.exp_m1()).ln_1p()
            } else {
                m + ((a - m).exp() + (b - m).exp() - (-m).exp()).ln()
            };
            theta.ln_1p() + (1.0 + theta) * (p.x + p.y) - (2.0 + 1.0 / theta) * log_s
        }
    }
}

// Coarse log-parameter search brackets an optimum; fixed bounded refinement is
// deterministic across dates/batches. No previous-day warm-start dependence.
fn fit(points: &[Point], family: Family) -> Option<f64> {
    let base = match family {
        Family::Gumbel => 1.0,
        Family::Clayton => 0.0,
    };
    let upper = (MAX_THETA - base).ln_1p();
    let theta = |z: f64| base + z.exp_m1();
    let score = |z: f64| {
        points
            .iter()
            .map(|&p| log_density(p, theta(z), family))
            .sum::<f64>()
    };
    const GRID: usize = 24;
    let mut best = 0;
    let mut best_score = 0.0;
    for k in 1..=GRID {
        let s = score(upper * k as f64 / GRID as f64);
        if !s.is_finite() {
            return None;
        }
        if s > best_score {
            best = k;
            best_score = s;
        }
    }
    let mut a = upper * best.saturating_sub(1) as f64 / GRID as f64;
    let mut b = upper * (best + 1).min(GRID) as f64 / GRID as f64;
    let r = (5.0_f64.sqrt() - 1.0) / 2.0;
    let mut c = b - r * (b - a);
    let mut d = a + r * (b - a);
    let mut fc = score(c);
    let mut fd = score(d);
    for _ in 0..80 {
        if b - a < 1e-8 {
            break;
        }
        if fc > fd {
            b = d;
            d = c;
            fd = fc;
            c = b - r * (b - a);
            fc = score(c);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + r * (b - a);
            fd = score(d);
        }
    }
    if !fc.is_finite() || !fd.is_finite() {
        return None;
    }
    let z = if fc > fd { c } else { d };
    if score(upper) >= fc.max(fd) || theta(z) >= MAX_THETA - 1e-5 {
        return None;
    }
    if fc.max(fd) <= 1e-10 {
        Some(base)
    } else {
        Some(theta(z))
    }
}
fn tail_coefficients(pairs: &[(f64, f64)]) -> Option<(f64, f64)> {
    if pairs.len() < MIN_PAIRS {
        return None;
    }
    let u = pseudo_observations(&pairs.iter().map(|p| p.0).collect::<Vec<_>>())?;
    let v = pseudo_observations(&pairs.iter().map(|p| p.1).collect::<Vec<_>>())?;
    let points: Vec<_> = u
        .into_iter()
        .zip(v)
        .map(|(u, v)| Point::new(u, v))
        .collect();
    let gu = fit(&points, Family::Gumbel)?;
    let cl = fit(&points, Family::Clayton)?;
    Some((
        2.0 - 2.0_f64.powf(1.0 / gu),
        if cl == 0.0 {
            0.0
        } else {
            2.0_f64.powf(-1.0 / cl)
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{ColumnData, Table};
    use std::collections::BTreeMap;

    fn fixture(targets: Vec<i32>, bump: f64) -> (DataPool, FactorContext) {
        fixture_with_stocks(targets, bump, 32)
    }

    fn fixture_with_stocks(
        targets: Vec<i32>,
        bump: f64,
        stock_count: usize,
    ) -> (DataPool, FactorContext) {
        let dates: Vec<_> = (1..=25)
            .map(|d| 20250100 + d)
            .chain((1..=25).map(|d| 20250200 + d))
            .chain((1..=25).map(|d| 20250300 + d))
            .collect();
        let codes: Vec<_> = (1..=stock_count)
            .map(|i| {
                Some(if i == stock_count {
                    "430001.BJ".into()
                } else {
                    format!("{i:06}.SZ")
                })
            })
            .collect();
        let noise = |d: usize, s: usize| {
            (((d + 1) as f64 * 12.9898 + (s + 1) as f64 * 78.233).sin() * 43758.5453).fract()
        };
        let market: Vec<_> = (0..dates.len()).map(|d| noise(d, 97) * 0.02).collect();
        let mut prices = vec![20.0; dates.len() * stock_count];
        for s in 0..stock_count {
            for d in 1..dates.len() {
                let r = 0.45 * market[d]
                    + 0.02 * noise(d, s)
                    + if market[d] > 0.0 {
                        s as f64 * 0.0002 * noise(d, 101).abs()
                    } else {
                        0.0
                    };
                prices[d * stock_count + s] = prices[(d - 1) * stock_count + s] * (1.0 + r);
            }
        }
        prices[74 * stock_count] += bump;
        let daily = |fields: Vec<(&str, Vec<Option<f64>>)>| {
            let mut cols = BTreeMap::from([
                (
                    "ts_code".into(),
                    ColumnData::Utf8(dates.iter().flat_map(|_| codes.clone()).collect()),
                ),
                (
                    "trade_date".into(),
                    ColumnData::I32(
                        dates
                            .iter()
                            .flat_map(|d| vec![Some(*d); stock_count])
                            .collect(),
                    ),
                ),
            ]);
            for (name, values) in fields {
                cols.insert(name.into(), ColumnData::F64(values));
            }
            Table::new(cols).unwrap()
        };
        let pv = daily(vec![
            ("close", prices.into_iter().map(Some).collect()),
            (
                "vol",
                (0..dates.len() * stock_count)
                    .map(|i| Some(if i % stock_count == 30 { 0.0 } else { 100.0 }))
                    .collect(),
            ),
        ]);
        let adj = daily(vec![(
            "adj_factor",
            vec![Some(1.0); dates.len() * stock_count],
        )]);
        let size = daily(vec![(
            "SIZE",
            (0..dates.len() * stock_count)
                .map(|i| {
                    if i % stock_count == 29 {
                        None
                    } else {
                        Some((i % stock_count) as f64)
                    }
                })
                .collect(),
        )]);
        let sector = Table::new(BTreeMap::from([
            ("ts_code".into(), ColumnData::Utf8(codes)),
            (
                "in_date".into(),
                ColumnData::I32(vec![Some(20200101); stock_count]),
            ),
            ("out_date".into(), ColumnData::I32(vec![None; stock_count])),
            (
                "l1_code".into(),
                ColumnData::Utf8(vec![Some("801010".into()); stock_count]),
            ),
        ]))
        .unwrap();
        let index = Table::new(BTreeMap::from([
            (
                "ts_code".into(),
                ColumnData::Utf8(vec![Some(MARKET.into()); dates.len()]),
            ),
            (
                "trade_date".into(),
                ColumnData::I32(dates.iter().map(|d| Some(*d)).collect()),
            ),
            (
                "close".into(),
                ColumnData::F64(market.iter().map(|r| Some(100.0 * (1.0 + r))).collect()),
            ),
            (
                "pre_close".into(),
                ColumnData::F64(vec![Some(100.0); dates.len()]),
            ),
        ]))
        .unwrap();
        let context = FactorContext {
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            start_date: targets[0],
            end_date: *targets.last().unwrap(),
            load_start_date: dates[0],
            load_dates: dates,
            target_dates: targets,
        };
        let data = DataPool::from_daily_tables_for_test(
            HashMap::from([
                (DatasetId::StockDailyPv, pv),
                (DatasetId::StockAdjFactor, adj),
                (DatasetId::StockBarraDaily, size),
                (DatasetId::StockSwClassification, sector),
                (DatasetId::IndexDaily, index),
            ]),
            &context,
        )
        .unwrap();
        (data, context)
    }

    #[test]
    fn abnormal_upper_tail_daily_batch_future_and_masks() {
        let factor = StockDailyAbnormalUpperTailDependence;
        let targets = vec![20250311, 20250312, 20250325];
        let (data, ctx) = fixture(targets.clone(), 0.0);
        let all = factor.compute(&ctx, &data).unwrap();
        assert_eq!(all.values.len(), 96);
        assert!(all.values.iter().any(|v| v.value.is_some()));
        for v in &all.values {
            if let crate::core::FactorRowKey::Daily { ts_code, .. } = &v.key {
                if ["000030.SZ", "000031.SZ", "430001.BJ"].contains(&ts_code.as_str()) {
                    assert!(v.value.is_none());
                }
            }
        }
        for date in targets {
            let (data, ctx) = fixture(vec![date], 0.0);
            let single = factor.compute(&ctx, &data).unwrap();
            assert_eq!(
                single
                    .values
                    .iter()
                    .map(|v| (&v.key, v.value))
                    .collect::<Vec<_>>(),
                all.values
                    .iter()
                    .filter(|v| v.key.trade_date() == date)
                    .map(|v| (&v.key, v.value))
                    .collect::<Vec<_>>()
            );
        }
        let (data, ctx) = fixture(vec![20250311, 20250312, 20250325], 50.0);
        let future = factor.compute(&ctx, &data).unwrap();
        assert!(all
            .values
            .iter()
            .zip(&future.values)
            .filter(|(a, _)| a.key.trade_date() < 20250325)
            .all(|(a, b)| a.value == b.value));
        let first: Vec<_> = all
            .values
            .iter()
            .filter(|v| v.key.trade_date() == 20250311)
            .map(|v| v.value)
            .collect();
        let next: Vec<_> = all
            .values
            .iter()
            .filter(|v| v.key.trade_date() == 20250312)
            .map(|v| v.value)
            .collect();
        assert_ne!(first, next);
    }
    #[test]
    fn abnormal_upper_tail_thread_counts_are_identical() {
        let (data, ctx) = fixture_with_stocks(vec![20250311, 20250312, 20250325], 0.0, 96);
        let compute = |threads| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| {
                    // Match the engine: stock parallelism nested inside provider parallelism.
                    (0..1)
                        .into_par_iter()
                        .map(|_| {
                            assert_eq!(rayon::current_num_threads(), threads);
                            StockDailyAbnormalUpperTailDependence
                                .compute(&ctx, &data)
                                .unwrap()
                        })
                        .collect::<Vec<_>>()
                        .pop()
                        .unwrap()
                })
        };
        let serial = compute(1);
        let parallel = compute(8);
        assert_eq!(serial.values.len(), parallel.values.len());
        for (a, b) in serial.values.iter().zip(&parallel.values) {
            assert_eq!(a.key, b.key);
            assert_eq!(a.value.map(f64::to_bits), b.value.map(f64::to_bits));
        }
    }

    #[test]
    fn abnormal_upper_tail_ranks_and_missing() {
        assert_eq!(
            pseudo_observations(&[3.0, 1.0, 1.0, 2.0]).unwrap(),
            vec![0.8, 0.3, 0.3, 0.6]
        );
        assert!(pseudo_observations(&[1.0; 60]).is_none());
        assert!(tail_coefficients(&[(1.0, 2.0); 29]).is_none());
        assert_eq!(
            ratio_return(
                adjusted(Some(10.0), Some(2.0)),
                adjusted(Some(20.0), Some(1.0))
            ),
            Some(0.0)
        );
    }
    #[test]
    fn abnormal_upper_tail_density_matches_direct_formulas() {
        for (u, v) in [(0.1_f64, 0.8_f64), (0.4, 0.6), (0.99, 0.98)] {
            let p = Point::new(u, v);
            for t in [1.2, 2.0, 5.0] {
                let s = p.x.powf(t) + p.y.powf(t);
                let a = s.powf(1.0 / t);
                let g =
                    (-a).exp() * (a + t - 1.0) * s.powf(1.0 / t - 2.0) * (p.x * p.y).powf(t - 1.0)
                        / (u * v);
                let c = (1.0 + t)
                    * (u * v).powf(-1.0 - t)
                    * (u.powf(-t) + v.powf(-t) - 1.0).powf(-2.0 - 1.0 / t);
                assert!((log_density(p, t, Family::Gumbel) - g.ln()).abs() < 1e-10);
                assert!((log_density(p, t, Family::Clayton) - c.ln()).abs() < 1e-10);
            }
            assert_eq!(log_density(p, 1.0, Family::Gumbel), 0.0);
            assert_eq!(log_density(p, 0.0, Family::Clayton), 0.0);
            assert!(log_density(p, 1e-8, Family::Clayton).abs() < 1e-6);
        }
    }
    #[test]
    fn abnormal_upper_tail_fit_boundaries_and_optimum() {
        let negative: Vec<_> = (1..=60).map(|i| (i as f64, 61.0 - i as f64)).collect();
        assert_eq!(tail_coefficients(&negative), Some((0.0, 0.0)));
        let perfect: Vec<_> = (1..=60).map(|i| (i as f64, i as f64)).collect();
        assert!(tail_coefficients(&perfect).is_none());
        let points: Vec<_> = (1..=60)
            .map(|i| Point::new(i as f64 / 61.0, ((i * 17) % 60 + 1) as f64 / 61.0))
            .collect();
        for family in [Family::Gumbel, Family::Clayton] {
            let t = fit(&points, family).unwrap();
            let score = |t: f64| {
                points
                    .iter()
                    .map(|p| log_density(*p, t, family))
                    .sum::<f64>()
            };
            let base = match family {
                Family::Gumbel => 1.0,
                Family::Clayton => 0.0,
            };
            for i in 0..1000 {
                assert!(score(t) + 1e-6 >= score(base + (MAX_THETA - base) * i as f64 / 1000.0));
            }
        }
    }
    #[test]
    fn abnormal_upper_tail_metadata() {
        let s = StockDailyAbnormalUpperTailDependence.spec();
        assert!(s.tags.contains(&"DFZQ".into()));
        assert!(s.tags.contains(&"deprecated".into()));
        assert!(s.tags.contains(&"price_volume".into()));
        assert!(!s.tags.contains(&"fundamental".into()));
        assert_eq!(s.lookback.trading_days, 60);
    }
}

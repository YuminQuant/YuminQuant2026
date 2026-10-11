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

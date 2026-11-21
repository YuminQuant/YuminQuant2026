use crate::core::{AssetClass, FactorContext, FactorSeries, FactorSpec, Frequency, Lookback};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::future::minute_input::{
    compute_future_minute_many, future_minute_request, FutureMinuteField,
    FutureMinuteField::{ContinuousMinute, ContractDiff, ContractLevel},
};
use crate::factor::future::multiscale_volatility::Series;
use crate::factor::future::variance_timing::close_returns;
use crate::factor::Factor;

const FIRST_ID: usize = 673;
const WINDOWS: [usize; 3] = [5, 15, 30];
const C: FutureMinuteField = ContractLevel("close");
const V: FutureMinuteField = ContractLevel("vol");
const D: FutureMinuteField = ContractDiff("oi");
const COMMON: &[FutureMinuteField] = &[C, ContinuousMinute, V, D];
const RANGE: &[FutureMinuteField] = &[
    C,
    ContinuousMinute,
    V,
    D,
    ContractLevel("high"),
    ContractLevel("low"),
];
const AMOUNT: &[FutureMinuteField] = &[C, ContinuousMinute, V, D, ContractLevel("amount")];

const FACTORS: [(&str, &str); 5] = [
    ("liquidation_drift", "Follow recent price displacement when volume declines and net open interest contracts."),
    ("liquidation_attribution", "Follow returns on net open interest reduction minutes, weighting each by the reduction and conditioning on lower volume and overall net contraction."),
    ("quiet_breakout", "Follow a close outside the earlier high-low range when both traded volume and absolute net open interest activity slow."),
    ("thin_market_impact", "Fade the recent move when volume and net open interest decline but absolute price impact per unit of notional rises."),
    ("participation_restart", "Follow price after a low-volume, low-position-change middle segment gives way to higher volume and net open interest expansion."),
];

fn fields(kind: usize) -> &'static [FutureMinuteField] {
    match kind {
        2 => RANGE,
        3 => AMOUNT,
        _ => COMMON,
    }
}

#[derive(Clone, Copy)]
struct WaitAndSee {
    kind: usize,
    variant: usize,
}

pub(super) fn create(id: usize) -> Box<dyn Factor> {
    let offset = id - FIRST_ID;
    Box::new(WaitAndSee {
        kind: offset / WINDOWS.len(),
        variant: offset % WINDOWS.len(),
    })
}

impl Factor for WaitAndSee {
    fn spec(&self) -> FactorSpec {
        let (alias, description) = FACTORS[self.kind];
        let n = WINDOWS[self.variant];
        let id = format!(
            "yq_{:03}",
            FIRST_ID + self.kind * WINDOWS.len() + self.variant
        );
        FactorSpec {
            name: id.clone(),
            id,
            aliases: vec![format!("wait_{alias}_{n}m")],
            asset_class: AssetClass::Future,
            frequency: Frequency::Minute1,
            version: "0.1.0".into(),
            tags: ["minute", "pv", "yq", "wait_and_see"]
                .map(str::to_string)
                .to_vec(),
            description: format!(
                "{description} Uses {n} consecutive one-minute returns within a trading session."
            ),
            dependencies: vec![future_minute_request(fields(self.kind))],
            intraday_raw_dependencies: Vec::new(),
            lookback: Lookback { trading_days: 0 },
        }
    }

    fn compute_provider_key(&self) -> String {
        format!("future-minute-wait-and-see-{}", self.kind)
    }

    fn compute(&self, context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        Ok(self
            .compute_many(&[self.spec().id], context, data)?
            .remove(0))
    }

    fn compute_many(
        &self,
        ids: &[String],
        context: &FactorContext,
        data: &DataPool,
    ) -> Result<Vec<FactorSeries>> {
        let (specs, widths): (Vec<_>, Vec<_>) = WINDOWS
            .iter()
            .enumerate()
            .filter_map(|(variant, &n)| {
                let spec = Self { variant, ..*self }.spec();
                ids.contains(&spec.id).then_some((spec, n))
            })
            .unzip();
        if specs.is_empty() {
            return Ok(Vec::new());
        }
        compute_future_minute_many(&specs, context, data, fields(self.kind), 0, |inputs| {
            let returns = close_returns(&inputs[0], &inputs[1]);
            widths
                .iter()
                .map(|&n| {
                    (0..returns.len())
                        .map(|i| {
                            let end = i + 1;
                            let start = end.checked_sub(n)?;
                            score(self.kind, &returns[start..end], inputs, start, end)
                                .filter(|v| v.is_finite())
                        })
                        .collect::<Series>()
                })
                .collect()
        })
    }
}

#[derive(Default)]
struct Activity {
    volume: f64,
    net: f64,
    gross: f64,
    displacement: f64,
    power: f64,
    absolute_return: f64,
    reduction: f64,
    reduced_return: f64,
    reduced_power: f64,
}

impl Activity {
    fn measure(returns: &[Option<f64>], inputs: &[Series], first: usize) -> Option<Self> {
        let mut state = Self::default();
        for (j, &r) in returns.iter().enumerate() {
            let i = first + j;
            let r = r?;
            let v = inputs[2][i]?;
            let delta = inputs[3][i]?;
            if v < 0.0 {
                return None;
            }
            let weight = (-delta).max(0.0);
            state.volume += v;
            state.net += delta;
            state.gross += delta.abs();
            state.displacement += r;
            state.power += r * r;
            state.absolute_return += r.abs();
            state.reduction += weight;
            state.reduced_return += weight * r;
            state.reduced_power += weight * r * r;
        }
        Some(state)
    }

    fn direction(&self, count: usize) -> f64 {
        ratio(self.displacement, (count as f64 * self.power).sqrt())
    }

    fn reduction_share(&self) -> f64 {
        let reduction = (-self.net).max(0.0);
        ratio(reduction, self.volume + reduction)
    }
}

fn score(
    kind: usize,
    r: &[Option<f64>],
    inputs: &[Series],
    start: usize,
    end: usize,
) -> Option<f64> {
    let n = end - start;
    if kind == 4 {
        let latest = n / 3;
        let first = (n - latest).div_ceil(2);
        let middle = n - first - latest;
        let earlier = Activity::measure(&r[..first], inputs, start)?;
        let quiet = Activity::measure(&r[first..first + middle], inputs, start + first)?;
        let restart = Activity::measure(&r[first + middle..], inputs, start + first + middle)?;
        let prior_volume = earlier.volume / first as f64;
        let quiet_volume = quiet.volume / middle as f64;
        let new_volume = restart.volume / latest as f64;
        let inactivity = drop(prior_volume, quiet_volume).min(drop(
            earlier.gross / first as f64,
            quiet.gross / middle as f64,
        ));
        let added = restart.net.max(0.0);
        return Some(
            restart.direction(latest)
                * inactivity
                * drop(new_volume, quiet_volume)
                * ratio(added, added + restart.volume),
        );
    }
    let old = n.div_ceil(2);
    let recent = n - old;
    let mid = start + old;
    let earlier = Activity::measure(&r[..old], inputs, start)?;
    let current = Activity::measure(&r[old..], inputs, mid)?;
    if current.volume == 0.0 {
        return Some(0.0);
    }
    let dry = drop(earlier.volume / old as f64, current.volume / recent as f64);
    let shrink = current.reduction_share();
    let direction = current.direction(recent);
    let output = match kind {
        0 => direction * dry * shrink,
        1 => {
            let attribution = ratio(
                current.reduced_return,
                (current.reduction * current.reduced_power).sqrt(),
            );
            attribution * dry * shrink
        }
        2 => {
            let mut ceiling = f64::NEG_INFINITY;
            let mut floor = f64::INFINITY;
            for i in start..mid {
                let (h, l, c) = (inputs[4][i]?, inputs[5][i]?, inputs[0][i]?);
                if l <= 0.0 || c < l || c > h {
                    return None;
                }
                ceiling = ceiling.max(h);
                floor = floor.min(l);
            }
            let close = inputs[0][end - 1]?.ln();
            let up = (close - ceiling.ln()).max(0.0);
            let down = (floor.ln() - close).max(0.0);
            let scale = (ceiling.ln() - floor.ln()).max(earlier.power.sqrt());
            let quiet = drop(earlier.gross / old as f64, current.gross / recent as f64);
            dry * quiet * (ratio(up, up + scale) - ratio(down, down + scale))
        }
        3 => {
            let impact = |from: usize, to: usize, abs_return: f64| {
                let amount = inputs[4][from..to].iter().try_fold(0.0, |acc, &v| {
                    let v = v?;
                    (v >= 0.0).then_some(acc + v)
                })?;
                (amount > 0.0).then(|| abs_return / amount)
            };
            let before = impact(start, mid, earlier.absolute_return)?;
            let after = impact(mid, end, current.absolute_return)?;
            -direction * dry * shrink * drop(after, before)
        }
        _ => unreachable!("unknown wait-and-see factor"),
    };
    Some(output)
}

fn ratio(x: f64, d: f64) -> f64 {
    if d > 0.0 {
        x / d
    } else {
        0.0
    }
}

fn drop(before: f64, after: f64) -> f64 {
    ratio((before - after).max(0.0), before + after)
}

use crate::core::{AssetClass, FactorContext, FactorSeries, FactorSpec, Frequency, Lookback};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::intraday_raw::quantile_linear;
use crate::factor::future::minute_input::{
    compute_future_minute_many, future_minute_request, FutureMinuteField,
    FutureMinuteField::{ContinuousMinute, ContractDiff, ContractLevel},
};
use crate::factor::future::multiscale_volatility::{log_return, Series, Values};
use crate::factor::future::variance_amplitude::relative_volume;
use crate::factor::Factor;

const FIRST_ID: usize = 295;
const WINDOWS: [usize; 3] = [60, 120, 180];
const MAX_WINDOW: usize = 180;
const CHANNEL: usize = 30;
const AGGREGATION: usize = 5;
const VOLUME_BASE: usize = 5;
const CLOSE: &[FutureMinuteField] = &[ContractLevel("close"), ContinuousMinute];
const RANGE: &[FutureMinuteField] = &[
    ContractLevel("close"),
    ContinuousMinute,
    ContractLevel("high"),
    ContractLevel("low"),
];
const PARTICIPATION: &[FutureMinuteField] = &[
    ContractLevel("close"),
    ContinuousMinute,
    ContractLevel("vol"),
    ContractDiff("oi"),
];

const FACTORS: [(&str, &str); 6] = [
    ("vr_timing_breakout_confirmation", "Signed 30-minute channel breakout confirmed by positive five-step robust variance-ratio evidence."),
    ("vr_timing_short_reversal", "Current return reversed when the two-step robust variance ratio is negative and the ten-step statistic is near neutral."),
    ("vr_timing_lag_projection", "Weighted projection of lagged returns using the second to fifth autocorrelations, estimated before the most recent five minutes."),
    ("vr_timing_asymmetric_response", "Separate positive and negative return responses estimated from the first four lagged covariances and shrunk by sample count."),
    ("vr_timing_jump_contribution", "Signed excess return weighted by the variance-ratio difference before and after clipping historical return shocks."),
    ("vr_timing_participation_premium", "Signed return times the variance-ratio premium of blocks with sustained volume and rising open interest."),
];

fn fields(kind: usize) -> &'static [FutureMinuteField] {
    match kind {
        0 => RANGE,
        5 => PARTICIPATION,
        _ => CLOSE,
    }
}

#[derive(Clone, Copy)]
struct VarianceTiming {
    kind: usize,
    variant: usize,
}

pub(super) fn create(number: usize) -> Box<dyn Factor> {
    let offset = number - FIRST_ID;
    Box::new(VarianceTiming {
        kind: offset / WINDOWS.len(),
        variant: offset % WINDOWS.len(),
    })
}

impl Factor for VarianceTiming {
    fn spec(&self) -> FactorSpec {
        let n = WINDOWS[self.variant];
        let (alias, description) = FACTORS[self.kind];
        let id = format!(
            "yq_{:03}",
            FIRST_ID + self.kind * WINDOWS.len() + self.variant
        );
        FactorSpec {
            name: id.clone(),
            id,
            aliases: vec![format!("{alias}_{n}m")],
            asset_class: AssetClass::Future,
            frequency: Frequency::Minute1,
            version: "0.1.0".into(),
            tags: ["minute", "pv", "yq", "variance_ratio", "timing"]
                .map(str::to_string)
                .to_vec(),
            description: format!(
                "{description} The estimation window is {n} observed minute bars."
            ),
            dependencies: vec![future_minute_request(fields(self.kind))],
            intraday_raw_dependencies: Vec::new(),
            lookback: Lookback { trading_days: 3 },
        }
    }

    fn compute_provider_key(&self) -> String {
        format!("future-minute-vr-timing-{}", self.kind)
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
        let Some(&max_width) = widths.last() else {
            return Ok(Vec::new());
        };
        // The projection excludes the latest five returns. The participation
        // baseline uses five earlier levels plus the anchor for an OI difference.
        compute_future_minute_many(
            &specs,
            context,
            data,
            fields(self.kind),
            max_width + 6,
            |inputs| {
                let returns = close_returns(&inputs[0], &inputs[1]);
                let participation = (self.kind == 5)
                    .then(|| participation_weights(&inputs[2], &inputs[3], &inputs[1]));
                widths
                    .iter()
                    .map(|&n| {
                        (0..returns.len())
                            .map(|t| {
                                evaluate(
                                    self.kind,
                                    n,
                                    t,
                                    inputs,
                                    &returns,
                                    participation.as_deref(),
                                )
                                .filter(|value| value.is_finite())
                            })
                            .collect()
                    })
                    .collect()
            },
        )
    }
}

pub(super) fn close_returns(close: &Values, links: &Values) -> Series {
    (0..close.len())
        .map(|i| {
            (links[i] == Some(1.0)).then_some(())?;
            log_return(close[i]?, close[i.checked_sub(1)?]?)
        })
        .collect()
}

fn divide(value: f64, scale: f64) -> Option<f64> {
    (value.is_finite() && scale.is_finite() && scale > 0.0).then(|| value / scale)
}

fn bounded_ratio(vr: f64) -> f64 {
    (vr - 1.0) / (vr + 1.0)
}

pub(super) fn enough(valid: usize, expected: usize) -> bool {
    valid * 5 >= expected * 4
}

pub(super) struct ReturnWindow<'a> {
    returns: &'a Values,
    pub(super) count: usize,
    pub(super) mean: f64,
    pub(super) centered_squares: f64,
}

impl<'a> ReturnWindow<'a> {
    pub(super) fn new(returns: &'a Values) -> Option<Self> {
        let count = returns.iter().flatten().count();
        if count < 2 || !enough(count, returns.len()) {
            return None;
        }
        // Center around a reference return to preserve zero variance for constant windows.
        let reference = *returns.iter().flatten().next()?;
        let deviations = returns.iter().flatten().map(|x| x - reference).sum::<f64>();
        let mean = reference + deviations / count as f64;
        let centered_squares = returns
            .iter()
            .flatten()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>();
        (centered_squares.is_finite() && centered_squares > 0.0).then_some(Self {
            returns,
            count,
            mean,
            centered_squares,
        })
    }

    pub(super) fn std(&self) -> f64 {
        (self.centered_squares / self.count as f64).sqrt()
    }

    pub(super) fn lag_pairs(&self, lag: usize) -> impl Iterator<Item = (usize, f64, f64)> + '_ {
        self.returns
            .windows(lag + 1)
            .enumerate()
            .filter_map(move |(i, window)| {
                window.iter().all(Option::is_some).then_some(())?;
                Some((i, window[0]? - self.mean, window[lag]? - self.mean))
            })
    }

    fn variance_ratio(&self, q: usize) -> Option<f64> {
        let (blocks, squared) = self
            .returns
            .windows(q)
            .filter_map(|block| {
                block
                    .iter()
                    .try_fold(0.0, |sum, &r| Some(sum + r? - self.mean))
            })
            .fold((0, 0.0), |(n, sum), centered| {
                (n + 1, sum + centered.powi(2))
            });
        if self.count <= q || !enough(blocks, self.returns.len() - q + 1) {
            return None;
        }
        // The shared estimated mean removes q/count of each block's iid variance.
        // The number of eligible blocks counts only contiguous returns.
        let normalization = (q * blocks) as f64 * (1.0 - q as f64 / self.count as f64);
        divide(
            squared / normalization,
            self.centered_squares / (self.count - 1) as f64,
        )
    }

    fn robust_z(&self, q: usize) -> Option<f64> {
        let vr = self.variance_ratio(q)?;
        let theta = (1..q)
            .map(|lag| {
                let squared_pair_products = self
                    .lag_pairs(lag)
                    .map(|(_, x, y)| {
                        (x.powi(2) / self.centered_squares) * (y.powi(2) / self.centered_squares)
                    })
                    .sum::<f64>();
                4.0 * (1.0 - lag as f64 / q as f64).powi(2)
                    * self.count as f64
                    * squared_pair_products
            })
            .sum::<f64>();
        divide((self.count as f64).sqrt() * (vr - 1.0), theta.sqrt())
    }
}

fn channel_breakout(inputs: &[Series], t: usize, history: &ReturnWindow<'_>) -> Option<f64> {
    let high = &inputs[2][t - CHANNEL..t];
    let low = &inputs[3][t - CHANNEL..t];
    let (top, bottom) =
        high.iter()
            .zip(low)
            .try_fold((0.0_f64, f64::INFINITY), |(top, bottom), (&h, &l)| {
                let (h, l) = (h?, l?);
                (0.0 < l && l <= h).then(|| (top.max(h), bottom.min(l)))
            })?;
    let close = inputs[0][t]?;
    let up = log_return(close, top)?.max(0.0) / history.std();
    let down = log_return(bottom, close)?.max(0.0) / history.std();
    let regime = (history.robust_z(AGGREGATION)? / 2.0).tanh().max(0.0);
    Some((up.tanh() - down.tanh()) * regime)
}

fn lag_projection(history: &ReturnWindow<'_>, recent: &Values) -> Option<f64> {
    recent.iter().all(Option::is_some).then_some(())?;
    let mut projected = 0.0;
    for lag in 2..=5 {
        let (pairs, cross) = history
            .lag_pairs(lag)
            .fold((0, 0.0), |(n, sum), (_, x, y)| (n + 1, sum + x * y));
        if !enough(pairs, history.returns.len() - lag) {
            return None;
        }
        let rho = cross / history.centered_squares;
        let centered = recent[recent.len() - lag]? - history.mean;
        projected += (6 - lag) as f64 * rho * centered;
    }
    Some((projected / (10.0 * history.std())).tanh())
}

fn asymmetric_response(history: &ReturnWindow<'_>, current: f64) -> Option<f64> {
    let centered = current - history.mean;
    if centered == 0.0 {
        return Some(0.0);
    }
    let mut cross = 0.0;
    let mut energy = 0.0;
    let mut anchors = [false; MAX_WINDOW];
    for lag in 1..AGGREGATION {
        let weight = 1.0 - lag as f64 / AGGREGATION as f64;
        for (index, x, y) in history
            .lag_pairs(lag)
            .filter(|(_, x, _)| x * centered > 0.0)
        {
            cross += weight * x * y;
            energy += weight * x * x;
            anchors[index] = true;
        }
    }
    let count = anchors.iter().filter(|&&valid| valid).count() as f64;
    (count >= 12.0).then_some(())?;
    let strength = count / (count + 20.0) * divide(cross, energy)?;
    Some((strength * centered / history.std()).tanh())
}

fn jump_contribution(history: &ReturnWindow<'_>, current: f64) -> Option<f64> {
    let mut ordered = [0.0; MAX_WINDOW];
    for (slot, &r) in ordered.iter_mut().zip(history.returns.iter().flatten()) {
        *slot = r;
    }
    let sample = &mut ordered[..history.count];
    let center = quantile_linear(sample, 0.5)?;
    sample.iter_mut().for_each(|r| *r = (*r - center).abs());
    let scale = 1.4826 * quantile_linear(sample, 0.5)?;
    (scale > 0.0).then_some(())?;
    let bound = 3.0 * scale;
    let clip = |r: f64| r.clamp(center - bound, center + bound);
    let mut clipped = [None; MAX_WINDOW];
    for (slot, &r) in clipped.iter_mut().zip(history.returns) {
        *slot = r.map(clip);
    }
    let raw_vr = history.variance_ratio(AGGREGATION)?;
    let adjusted_vr =
        ReturnWindow::new(&clipped[..history.returns.len()])?.variance_ratio(AGGREGATION)?;
    let premium = (bounded_ratio(raw_vr) - bounded_ratio(adjusted_vr)) / 2.0;
    Some(((current - clip(current)) / scale).tanh() * premium)
}

fn participation_weights(volume: &Values, change: &Values, links: &Values) -> Series {
    let relative = relative_volume(volume);
    let same_session = |i: usize| {
        (links[i] == Some(1.0)).then_some(())?;
        change[i]
    };
    (0..volume.len())
        .map(|i| {
            let v = relative[i]?;
            let pos = same_session(i)?.max(0.0);
            let start = i.checked_sub(VOLUME_BASE)?;
            let (count, sum) = (start..i)
                .filter_map(same_session)
                .fold((0, 0.0), |(count, sum), value| {
                    (count + 1, sum + value.abs())
                });
            (count >= 3).then_some(())?;
            let oi = if pos == 0.0 {
                0.0
            } else {
                pos / (pos + sum / count as f64)
            };
            Some((v / (1.0 + v)) * (0.5 + 0.5 * oi))
        })
        .collect()
}

#[derive(Default)]
struct WeightedVariance {
    mass: f64,
    mean: f64,
    centered_squares: f64,
}

impl WeightedVariance {
    fn push(&mut self, x: f64, weight: f64) {
        if weight > 0.0 {
            self.mass += weight;
            let delta = x - self.mean;
            self.mean += weight / self.mass * delta;
            self.centered_squares += weight * delta * (x - self.mean);
        }
    }

    fn variance(&self) -> Option<f64> {
        Some(divide(self.centered_squares, self.mass)?.max(0.0))
    }
}

fn participation_premium(history: &ReturnWindow<'_>, weights: &Values) -> Option<f64> {
    let mut weighted_blocks = WeightedVariance::default();
    let mut weighted_singles = WeightedVariance::default();
    let mut equal_blocks = WeightedVariance::default();
    let mut equal_singles = WeightedVariance::default();
    let mut n = 0;
    let mut sum_squared_weights = 0.0;
    for (block, block_weights) in history
        .returns
        .windows(AGGREGATION)
        .zip(weights.windows(AGGREGATION))
    {
        let (Some(sum), Some(weight)) = (
            block.iter().copied().sum::<Option<f64>>(),
            block_weights
                .iter()
                .try_fold(1.0_f64, |lowest, &w| Some(lowest.min(w?))),
        ) else {
            continue;
        };
        n += 1;
        sum_squared_weights += weight * weight;
        weighted_blocks.push(sum, weight);
        equal_blocks.push(sum, 1.0);
        for &r in block.iter().flatten() {
            weighted_singles.push(r, weight);
            equal_singles.push(r, 1.0);
        }
    }
    if !enough(n, history.returns.len() - AGGREGATION + 1) {
        return None;
    }
    let effective = divide(weighted_blocks.mass.powi(2), sum_squared_weights)?;
    (effective >= 8.0).then_some(())?;
    let weighted = divide(
        weighted_blocks.variance()?,
        AGGREGATION as f64 * weighted_singles.variance()?,
    )?;
    let equal = divide(
        equal_blocks.variance()?,
        AGGREGATION as f64 * equal_singles.variance()?,
    )?;
    Some((bounded_ratio(weighted) - bounded_ratio(equal)) / 2.0)
}

fn evaluate(
    kind: usize,
    n: usize,
    t: usize,
    inputs: &[Series],
    returns: &Values,
    participation: Option<&Values>,
) -> Option<f64> {
    let current = returns[t]?;
    let end = if kind == 2 { t.checked_sub(4)? } else { t };
    let start = end.checked_sub(n)?;
    let history = ReturnWindow::new(&returns[start..end])?;
    let direction = (current / history.std()).tanh();
    match kind {
        0 => channel_breakout(inputs, t, &history),
        1 => {
            let reversal = (-history.robust_z(2)? / 2.0).tanh().max(0.0);
            let neutrality = 1.0 - (history.robust_z(10)? / 2.0).tanh().abs();
            Some(-direction * reversal * neutrality)
        }
        2 => lag_projection(&history, &returns[end..=t]),
        3 => asymmetric_response(&history, current),
        4 => jump_contribution(&history, current),
        5 => {
            let weights = participation?;
            Some(direction * weights[t]? * participation_premium(&history, &weights[start..end])?)
        }
        _ => unreachable!("unknown variance timing formula"),
    }
}

use std::ops::Range;

use crate::core::{AssetClass, FactorContext, FactorSeries, FactorSpec, Frequency, Lookback};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::future::minute_input::{
    compute_future_minute_many, future_minute_request, FutureMinuteField,
    FutureMinuteField::{ContractDiff, ContractLevel},
};
use crate::factor::future::multiscale_volatility::{
    divide, log_return, map, zip, Series, Values, HISTORY_BARS,
};
use crate::factor::future::soft_scoring::{scaled, sigmoid, softplus};
use crate::factor::Factor;
use crate::operators::cross_sectional::cs_neutralize_regression::solve_linear_system;
use crate::operators::{ts_delay, ts_mean, ts_median, ts_quantile};

const FIRST_ID: usize = 186;
const BASE: usize = 5;
const MAX_WINDOW: usize = 30;
const TAIL: f64 = 1.5;
const HIGH_VOLUME: f64 = 1.5;
const OI_SHOCK: f64 = 1.0;
const SHRINKAGE: f64 = 5.0;
const VARIANCE_RIDGE: f64 = 0.05;
const TAIL_TEMPERATURE: f64 = 0.5;

const CLOSE: &[FutureMinuteField] = &[ContractLevel("close")];
const OHLC: &[FutureMinuteField] = &[
    ContractLevel("close"),
    ContractLevel("open"),
    ContractLevel("high"),
    ContractLevel("low"),
];
const VOLUME: &[FutureMinuteField] = &[ContractLevel("close"), ContractLevel("vol")];
const OI: &[FutureMinuteField] = &[ContractLevel("close"), ContractDiff("oi")];
const VOLUME_OI: &[FutureMinuteField] = &[
    ContractLevel("close"),
    ContractLevel("vol"),
    ContractDiff("oi"),
];

const FACTORS: [(&str, &str); 24] = [
    ("skew_s01_short_return", "Population skewness of returns standardized by the prior five-minute RMS with a current-return-relative scale floor; flat valid windows are neutral."),
    ("skew_s02_short_mid_diff", "Recent return skewness minus skewness in the immediately preceding window."),
    ("skew_s03_acceleration", "Second difference of return skewness across three consecutive non-overlapping windows."),
    ("skew_s04_vol_adjusted_diff", "Short-minus-preceding return skewness dampened by recent volatility expansion."),
    ("skew_s05_quantile", "Bowley skewness of standardized returns, using linearly interpolated quartiles."),
    ("skew_s06_tail_exceedance", "Signed softplus tail imbalance centered on standardized returns of plus/minus 1.5, with temperature 0.5 and a neutral-tail denominator anchor."),
    ("skew_s07_extreme_contribution", "Return skewness minus skewness after winsorizing at the fifth and 95th percentiles."),
    ("skew_s08_tail_clustering", "Softplus tail imbalance weighted by the preceding minute's same-direction sigmoid tail strength. A neutral-tail denominator anchor keeps quiet valid windows finite."),
    ("skew_s09_candle_body_ratio", "Skewness of the signed candle body divided by the candle range."),
    ("skew_s10_closing_location", "Skewness of the close location within the candle range."),
    ("skew_s11_intrabar_third_moment", "Normalized imbalance of cubed log excursions above and below the candle open."),
    ("skew_s12_range_conditional_diff", "Regularized return-skew difference between sigmoid high-range weights centered at 1.5 and their complements, shrunk by the smaller effective sample size."),
    ("skew_s13_volume_weighted_premium", "Regularized log-one-plus-relative-volume-weighted return skewness minus unweighted skewness, shrunk by effective sample size."),
    ("skew_s14_high_low_volume_diff", "Regularized return-skew difference between sigmoid log-relative-volume weights centered at 1.5 and their complements, shrunk by the smaller effective sample size."),
    ("skew_s15_price_volume_coskew", "Standardized co-skewness of returns with squared log relative-volume deviations."),
    ("skew_s16_lagged_volume_shock", "Regularized return-skew premium weighted by the previous minute's sigmoid relative-volume score, shrunk by effective sample size."),
    ("skew_s17_oi_increase_conditioned", "Regularized return-skew premium weighted by a sigmoid standardized OI increase score, shrunk by effective sample size."),
    ("skew_s18_oi_increase_decrease_diff", "Regularized return-skew difference between sigmoid OI increase weights and their complements, shrunk by the smaller effective sample size."),
    ("skew_s19_price_oi_coskew", "Standardized co-skewness of returns with squared open-interest shock deviations."),
    ("skew_s20_lagged_oi_shock", "Regularized return-skew premium weighted by the previous minute's sigmoid OI shock score centered at one, shrunk by effective sample size."),
    ("skew_s21_volume_oi_conditioned", "Regularized return-skew premium weighted by current sigmoid volume and OI-increase participation, shrunk by effective sample size."),
    ("skew_s22_volume_oi_tail_asymmetry", "Softplus return-tail imbalance weighted by relative volume and sigmoid OI-increase strength, with a neutral-tail denominator anchor."),
    ("skew_s23_volume_oi_residual", "Skewness of out-of-sample return residuals from a prior-window regression on volume and open-interest shocks."),
    ("skew_s24_joint_participation_lagged", "Regularized return-skew premium weighted by the preceding minute's sigmoid volume and OI-increase participation, shrunk by effective sample size."),
];

type Windows = (usize, usize);

fn windows(kind: usize) -> &'static [Windows] {
    match kind {
        2 | 4 => &[(5, 10), (5, 15), (10, 15)],
        3 => &[(5, 0)],
        9..=11 => &[(5, 0), (10, 0), (15, 0), (30, 0)],
        12 | 14 | 18 => &[(10, 0), (15, 0)],
        23 => &[(5, 15), (10, 15), (15, 10)],
        _ => &[(5, 0), (10, 0), (15, 0)],
    }
}

fn fields(kind: usize) -> &'static [FutureMinuteField] {
    match kind {
        9..=12 => OHLC,
        13..=16 => VOLUME,
        17..=20 => OI,
        21..=24 => VOLUME_OI,
        _ => CLOSE,
    }
}

#[derive(Clone, Copy)]
struct SkewFactor {
    kind: usize,
    variant: usize,
    number: usize,
}

pub(super) fn create(number: usize) -> Box<dyn Factor> {
    let mut first = FIRST_ID;
    for kind in 1..=FACTORS.len() {
        let end = first + windows(kind).len();
        if (first..end).contains(&number) {
            return Box::new(SkewFactor {
                kind,
                variant: number - first,
                number,
            });
        }
        first = end;
    }
    unreachable!("unregistered skewness factor ID")
}

impl Factor for SkewFactor {
    fn spec(&self) -> FactorSpec {
        let (alias, description) = FACTORS[self.kind - 1];
        let id = format!("yq_{:03}", self.number);
        let (n, prior) = windows(self.kind)[self.variant];
        let suffix = if prior > 0 {
            format!("{n}m_{prior}m")
        } else {
            format!("{n}m")
        };
        let history = match self.kind {
            2 | 4 | 23 => format!(
                " The recent window is {n} minutes and the preceding window is {prior} minutes."
            ),
            3 => format!(" Each of the three windows is {n} minutes."),
            _ => format!(" The factor window is {n} minutes."),
        };
        let volume = if matches!(self.kind, 13..=16 | 21..=24) {
            " Relative volume uses the mean of the preceding five minutes."
        } else {
            ""
        };
        let regularization = if matches!(self.kind, 12..=14 | 16..=18 | 20..=21 | 24) {
            " Weighted and unweighted moments share a ridge of 5% of full-window variance; effective sample size shrinks scores with five pseudo-observations."
        } else {
            ""
        };
        let standardization = if !matches!(self.kind, 9..=11) {
            " Return and OI scales use a relative floor of 10% of the current observation, with flat complete inputs treated as neutral."
        } else {
            ""
        };
        FactorSpec {
            name: id.clone(),
            id,
            aliases: vec![format!("{alias}_{suffix}")],
            asset_class: AssetClass::Future,
            frequency: Frequency::Minute1,
            version: "0.2.0".into(),
            tags: ["minute", "pv", "yq", "skewness"]
                .map(str::to_string)
                .to_vec(),
            description: format!("{description}{history}{volume}{regularization}{standardization}"),
            dependencies: vec![future_minute_request(fields(self.kind))],
            intraday_raw_dependencies: Vec::new(),
            lookback: Lookback { trading_days: 3 },
        }
    }

    fn compute_provider_key(&self) -> String {
        format!("future-minute-skewness-s{:02}", self.kind)
    }

    fn compute(&self, context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        Ok(self
            .compute_many(&[self.spec().id], context, data)?
            .remove(0))
    }

    fn compute_many(
        &self,
        requested_ids: &[String],
        context: &FactorContext,
        data: &DataPool,
    ) -> Result<Vec<FactorSeries>> {
        let first = self.number - self.variant;
        let (specs, selected): (Vec<_>, Vec<_>) = windows(self.kind)
            .iter()
            .enumerate()
            .filter_map(|(variant, &config)| {
                let factor = Self {
                    kind: self.kind,
                    variant,
                    number: first + variant,
                };
                let spec = factor.spec();
                requested_ids.contains(&spec.id).then_some((spec, config))
            })
            .unzip();
        if specs.is_empty() {
            return Ok(Vec::new());
        }
        compute_future_minute_many(
            &specs,
            context,
            data,
            fields(self.kind),
            HISTORY_BARS,
            |inputs| evaluate(self.kind, &selected, inputs),
        )
    }
}

fn rolling(length: usize, n: usize, f: impl Fn(Range<usize>) -> Option<f64>) -> Series {
    let mut out = vec![None; length];
    for i in n..=length {
        out[i - 1] = f(i - n..i).filter(|value| value.is_finite());
    }
    out
}

fn full_sample(values: &Values, buffer: &mut [f64; MAX_WINDOW]) -> Option<()> {
    for (&value, out) in values.iter().zip(buffer) {
        *out = value?;
    }
    Some(())
}

fn mean(values: &[f64]) -> f64 {
    let anchor = values[0];
    anchor + values.iter().map(|value| value - anchor).sum::<f64>() / values.len() as f64
}

fn skew(values: &[f64]) -> Option<f64> {
    if values.len() < 3 {
        return None;
    }
    let avg = mean(values);
    let (m2, m3) = values.iter().fold((0.0, 0.0), |(m2, m3), &value| {
        let delta = value - avg;
        (m2 + delta * delta, m3 + delta.powi(3))
    });
    let n = values.len() as f64;
    if m2 == 0.0 {
        Some(0.0)
    } else {
        divide(m3 / n, (m2 / n).powf(1.5))
    }
}

fn rolling_skew(values: &Values, n: usize) -> Series {
    rolling(values.len(), n, |range| {
        let mut sample = [0.0; MAX_WINDOW];
        full_sample(&values[range], &mut sample)?;
        skew(&sample[..n])
    })
}

/// Weighted and unweighted skew use the same window-relative variance ridge.
/// Equal weights therefore give exactly zero premium, including flat windows.
fn weighted_shape(values: &Values, weights: &Values) -> Option<(f64, f64, f64)> {
    let n = values.len();
    let (mut x, mut w) = ([0.0; MAX_WINDOW], [0.0; MAX_WINDOW]);
    full_sample(values, &mut x)?;
    full_sample(weights, &mut w)?;
    let (x, w) = (&mut x[..n], &w[..n]);
    if x.iter().any(|x| !x.is_finite()) || w.iter().any(|w| !w.is_finite() || *w < 0.0) {
        return None;
    }
    let mass = w.iter().sum::<f64>();
    if mass == 0.0 {
        return Some((0.0, 0.0, 0.0));
    }
    let neff = mass * mass / w.iter().map(|w| w * w).sum::<f64>();
    let center = mean(x);
    let amplitude = x.iter().map(|x| (x - center).abs()).fold(0.0, f64::max);
    if amplitude == 0.0 {
        return Some((0.0, 0.0, neff));
    }
    x.iter_mut().for_each(|x| *x = (*x - center) / amplitude);
    let base_variance = x.iter().map(|x| x * x).sum::<f64>() / n as f64;
    let ridge = VARIANCE_RIDGE * base_variance;
    let weighted_center = x.iter().zip(w).map(|(x, w)| x * w).sum::<f64>() / mass;
    let (m2, m3) = x.iter().zip(w).fold((0.0, 0.0), |(v, s), (x, w)| {
        let d = x - weighted_center;
        (v + w * d * d, s + w * d.powi(3))
    });
    let weighted = (m3 / mass) / (m2 / mass + ridge).powf(1.5);
    let unweighted =
        (x.iter().map(|x| x.powi(3)).sum::<f64>() / n as f64) / (base_variance + ridge).powf(1.5);
    Some((weighted, unweighted, neff))
}

fn weighted_premium(values: &Values, weights: &Values, n: usize) -> Series {
    rolling(values.len(), n, |range| {
        let (weighted, raw, neff) = weighted_shape(&values[range.clone()], &weights[range])?;
        Some(((weighted - raw) * neff / (neff + SHRINKAGE)).tanh())
    })
}

fn conditional_difference(values: &Values, weights: &Values, n: usize) -> Series {
    let complement = map(weights, |w| 1.0 - w);
    rolling(values.len(), n, |range| {
        let (up, _, nu) = weighted_shape(&values[range.clone()], &weights[range.clone()])?;
        let (down, _, nd) = weighted_shape(&values[range.clone()], &complement[range])?;
        let neff = nu.min(nd);
        Some(((up - down) * neff / (neff + SHRINKAGE)).tanh())
    })
}

fn volume_weight(volume: &Values) -> Series {
    volume
        .iter()
        .map(|v| {
            let v = (*v)?;
            if v < 0.0 {
                None
            } else if v == 0.0 {
                Some(0.0)
            } else {
                Some(sigmoid((v.ln() - HIGH_VOLUME.ln()) / 0.5))
            }
        })
        .collect()
}

fn relative_volume(volume: &Values) -> Series {
    let valid = volume
        .iter()
        .map(|v| v.filter(|x| *x >= 0.0))
        .collect::<Series>();
    zip(
        &valid,
        &ts_delay(&ts_mean(&valid, BASE, BASE), 1),
        |v, base| {
            if base == 0.0 {
                scaled(v, base)
            } else {
                divide(v, base)
            }
        },
    )
}

fn covariance(left: &[f64], right: &[f64], center_l: f64, center_r: f64) -> f64 {
    left.iter()
        .zip(right)
        .map(|(l, r)| (l - center_l) * (r - center_r))
        .sum::<f64>()
        / left.len() as f64
}

fn coskew(values: &Values, signals: &Values, n: usize) -> Series {
    rolling(values.len(), n, |range| {
        let (mut x, mut y) = ([0.0; MAX_WINDOW], [0.0; MAX_WINDOW]);
        full_sample(&signals[range.clone()], &mut x)?;
        full_sample(&values[range], &mut y)?;
        let (x, y) = (&x[..n], &y[..n]);
        let (mx, my) = (mean(x), mean(y));
        let moment = y
            .iter()
            .zip(x)
            .map(|(y, x)| (y - my) * (x - mx).powi(2))
            .sum::<f64>()
            / n as f64;
        let scale = covariance(x, x, mx, mx) * covariance(y, y, my, my).sqrt();
        if scale == 0.0 {
            Some(0.0)
        } else {
            divide(moment, scale).map(f64::tanh)
        }
    })
}

fn imbalance(up: &Values, down: &Values, n: usize) -> Series {
    rolling(up.len(), n, |range| {
        let u = up[range.clone()].iter().copied().sum::<Option<f64>>()?;
        let d = down[range].iter().copied().sum::<Option<f64>>()?;
        if u + d == 0.0 {
            Some(0.0)
        } else {
            divide(u - d, u + d)
        }
    })
}

fn tail_exceedance(
    values: &Values,
    n: usize,
    previous: Option<&Values>,
    weights: Option<&Values>,
) -> Series {
    let tail = |sign: f64| -> Series {
        (0..values.len())
            .map(|i| {
                let excess =
                    TAIL_TEMPERATURE * softplus((sign * values[i]? - TAIL) / TAIL_TEMPERATURE);
                let persistence = match previous {
                    Some(lead) => sigmoid((sign * lead[i]? - TAIL) / TAIL_TEMPERATURE),
                    None => 1.0,
                };
                let weight = match weights {
                    Some(weights) => weights[i]?,
                    None => 1.0,
                };
                Some(persistence * weight * excess)
            })
            .collect()
    };
    let (up, down) = (tail(1.0), tail(-1.0));
    // Neutral-tail mass anchors the dimensionless denominator even without tail events.
    let neutral = 2.0 * TAIL_TEMPERATURE * softplus(-TAIL / TAIL_TEMPERATURE) * n as f64;
    rolling(values.len(), n, |range| {
        let u = up[range.clone()].iter().copied().sum::<Option<f64>>()?;
        let d = down[range].iter().copied().sum::<Option<f64>>()?;
        Some((u - d) / (u + d + neutral))
    })
}

fn rms(values: &Values, n: usize) -> Series {
    rolling(values.len(), n, |range| {
        let sum = values[range]
            .iter()
            .try_fold(0.0, |sum, &value| Some(sum + value?.powi(2)))?;
        Some((sum / n as f64).sqrt())
    })
}

fn position(series: &[Series], f: impl Fn(f64, f64, f64, f64) -> Option<f64>) -> Series {
    (0..series[0].len())
        .map(|i| {
            let (c, o, h, l) = (series[0][i]?, series[1][i]?, series[2][i]?, series[3][i]?);
            if l > 0.0 && l <= o.min(c) && h >= o.max(c) {
                f(c, o, h, l)
            } else {
                None
            }
        })
        .collect()
}

fn residual_skew(z: &Values, v: &Values, oi: &Values, n: usize, prior: usize) -> Series {
    rolling(z.len(), n + prior, |range| {
        let start = range.start;
        let split = start + prior;
        let (mut zb, mut vb, mut ob) = ([0.0; MAX_WINDOW], [0.0; MAX_WINDOW], [0.0; MAX_WINDOW]);
        full_sample(&z[start..split], &mut zb)?;
        full_sample(&v[start..split], &mut vb)?;
        full_sample(&oi[start..split], &mut ob)?;
        let (y, x1, x2) = (&zb[..prior], &vb[..prior], &ob[..prior]);
        let (my, m1, m2) = (mean(y), mean(x1), mean(x2));
        let s1 = covariance(x1, x1, m1, m1).sqrt();
        let s2 = covariance(x2, x2, m2, m2).sqrt();
        // Centering fits the intercept, and scaling makes the rank check independent of units.
        let corr = divide(covariance(x1, x2, m1, m2), s1 * s2)?;
        let rhs = vec![
            divide(covariance(x1, y, m1, my), s1)?,
            divide(covariance(x2, y, m2, my), s2)?,
        ];
        let beta = solve_linear_system(vec![vec![1.0, corr], vec![corr, 1.0]], rhs)?;
        for (i, output) in (split..range.end).zip(&mut zb) {
            *output = z[i]? - my - beta[0] * (v[i]? - m1) / s1 - beta[1] * (oi[i]? - m2) / s2;
        }
        skew(&zb[..n]).map(f64::tanh)
    })
}

fn each(configs: &[Windows], f: impl Fn(usize, usize) -> Series) -> Vec<Series> {
    configs.iter().map(|&(n, past)| f(n, past)).collect()
}

fn evaluate(kind: usize, configs: &[Windows], inputs: &[Series]) -> Vec<Series> {
    let close = &inputs[0];
    if kind == 9 || kind == 10 {
        let candle = position(inputs, |c, o, h, l| {
            let numerator = if kind == 9 { c - o } else { (c - l) - (h - c) };
            if h == l {
                Some(0.0)
            } else {
                divide(numerator, h - l)
            }
        });
        return each(configs, |n, _| map(&rolling_skew(&candle, n), f64::tanh));
    }
    if kind == 11 {
        let up = position(inputs, |_, o, h, _| log_return(h, o).map(|x| x.powi(3)));
        let down = position(inputs, |_, o, _, l| log_return(o, l).map(|x| x.powi(3)));
        return each(configs, |n, _| imbalance(&up, &down, n));
    }
    let returns = zip(close, &ts_delay(close, 1), log_return);
    let z = zip(&returns, &ts_delay(&rms(&returns, BASE), 1), scaled);
    match kind {
        1 => each(configs, |n, _| map(&rolling_skew(&z, n), f64::tanh)),
        2 | 4 => each(configs, |n, past| {
            let score = zip(
                &rolling_skew(&z, n),
                &ts_delay(&rolling_skew(&z, past), n),
                |a, b| Some((a - b).tanh()),
            );
            if kind == 2 {
                return score;
            }
            let baseline = ts_delay(&rms(&returns, past), n);
            let damp = zip(&rms(&returns, n), &baseline, |short, long| {
                if long == 0.0 {
                    Some(if short == 0.0 { 1.0 } else { 0.0 })
                } else if short == 0.0 {
                    Some(1.0)
                } else {
                    divide(short, long).map(|ratio| (-ratio.ln().max(0.0)).exp())
                }
            });
            zip(&score, &damp, |s, w| Some(s * w))
        }),
        3 => each(configs, |n, _| {
            let now = rolling_skew(&z, n);
            let prev = ts_delay(&now, n);
            let oldest = ts_delay(&now, 2 * n);
            (0..z.len())
                .map(|i| Some((now[i]? - 2.0 * prev[i]? + oldest[i]?).tanh()))
                .collect()
        }),
        5 => each(configs, |n, _| {
            let low = ts_quantile(&z, n, n, 0.25);
            let mid = ts_quantile(&z, n, n, 0.50);
            let high = ts_quantile(&z, n, n, 0.75);
            (0..z.len())
                .map(|i| {
                    let (l, m, h) = (low[i]?, mid[i]?, high[i]?);
                    if h == l {
                        Some(0.0)
                    } else {
                        divide((h - m) - (m - l), h - l)
                    }
                })
                .collect()
        }),
        6 => each(configs, |n, _| tail_exceedance(&z, n, None, None)),
        7 => each(configs, |n, _| {
            let low = ts_quantile(&z, n, n, 0.05);
            let high = ts_quantile(&z, n, n, 0.95);
            rolling(z.len(), n, |range| {
                let end = range.end - 1;
                let (lo, hi) = (low[end]?, high[end]?);
                let mut sample = [0.0; MAX_WINDOW];
                full_sample(&z[range], &mut sample)?;
                let sample = &mut sample[..n];
                let raw = skew(sample)?;
                sample.iter_mut().for_each(|x| *x = x.clamp(lo, hi));
                Some((raw - skew(sample)?).tanh())
            })
        }),
        8 => {
            let prev = ts_delay(&z, 1);
            each(configs, |n, _| tail_exceedance(&z, n, Some(&prev), None))
        }
        12 => {
            let range = position(inputs, |_, _, h, l| log_return(h, l));
            let relative_range = zip(&range, &ts_delay(&ts_median(&range, BASE, BASE), 1), scaled);
            let weight = map(&relative_range, |r| sigmoid((r - 1.5) / 0.5));
            each(configs, |n, _| conditional_difference(&z, &weight, n))
        }
        13..=16 => volume_factors(kind, configs, &z, &relative_volume(&inputs[1])),
        17..=20 => oi_factors(kind, configs, &z, &inputs[1]),
        21..=24 => joint_factors(kind, configs, &z, &relative_volume(&inputs[1]), &inputs[2]),
        _ => unreachable!("unknown skewness formula"),
    }
}

fn oi_shocks(changes: &Values) -> Series {
    let baseline = ts_delay(&ts_mean(&map(changes, f64::abs), BASE, BASE), 1);
    zip(changes, &baseline, |oi, scale| {
        scaled(oi, scale).map(f64::asinh)
    })
}

fn volume_factors(kind: usize, configs: &[Windows], z: &Values, volume: &Values) -> Vec<Series> {
    match kind {
        13 => {
            let weight = map(volume, f64::ln_1p);
            each(configs, |n, _| weighted_premium(z, &weight, n))
        }
        14 => {
            let weights = volume_weight(volume);
            each(configs, |n, _| conditional_difference(z, &weights, n))
        }
        15 => {
            let log_volume = map(volume, f64::ln);
            each(configs, |n, _| coskew(z, &log_volume, n))
        }
        16 => {
            let prev = ts_delay(&volume_weight(volume), 1);
            each(configs, |n, _| weighted_premium(z, &prev, n))
        }
        _ => unreachable!(),
    }
}

fn oi_factors(kind: usize, configs: &[Windows], z: &Values, oi: &Values) -> Vec<Series> {
    match kind {
        17 | 18 => {
            let weight = map(&oi_shocks(oi), |d| sigmoid(d / 0.5));
            each(configs, |n, _| {
                if kind == 17 {
                    weighted_premium(z, &weight, n)
                } else {
                    conditional_difference(z, &weight, n)
                }
            })
        }
        19 => {
            let shock = oi_shocks(oi);
            each(configs, |n, _| coskew(z, &shock, n))
        }
        20 => {
            let prev = ts_delay(&map(&oi_shocks(oi), |d| sigmoid((d - OI_SHOCK) / 0.5)), 1);
            each(configs, |n, _| weighted_premium(z, &prev, n))
        }
        _ => unreachable!(),
    }
}

fn joint_factors(
    kind: usize,
    configs: &[Windows],
    z: &Values,
    volume: &Values,
    oi: &Values,
) -> Vec<Series> {
    match kind {
        21 | 24 => {
            let joint = zip(&volume_weight(volume), &oi_shocks(oi), |v, d| {
                Some(v * sigmoid(d / 0.5))
            });
            let test = if kind == 24 {
                ts_delay(&joint, 1)
            } else {
                joint
            };
            each(configs, |n, _| weighted_premium(z, &test, n))
        }
        22 => {
            let weight = zip(volume, &oi_shocks(oi), |v, d| {
                Some(v.ln_1p() * sigmoid(d / 0.5))
            });
            each(configs, |n, _| tail_exceedance(z, n, None, Some(&weight)))
        }
        23 => {
            let log_volume = map(volume, f64::ln);
            let shock = oi_shocks(oi);
            each(configs, |n, past| {
                residual_skew(z, &log_volume, &shock, n, past)
            })
        }
        _ => unreachable!(),
    }
}

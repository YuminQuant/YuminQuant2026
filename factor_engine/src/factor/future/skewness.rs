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
use crate::factor::future::variance_amplitude::relative_volume;
use crate::factor::Factor;
use crate::operators::cross_sectional::cs_neutralize_regression::solve_linear_system;
use crate::operators::{ts_delay, ts_mean, ts_median, ts_quantile};

const FIRST_ID: usize = 186;
const BASE: usize = 5;
const MAX_WINDOW: usize = 30;
const TAIL: f64 = 1.5;
const HIGH_VOLUME: f64 = 1.5;
const OI_SHOCK: f64 = 1.0;

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
    ("skew_s01_short_return", "Population skewness of volatility-standardized returns."),
    ("skew_s02_short_mid_diff", "Recent return skewness minus skewness in the immediately preceding window."),
    ("skew_s03_acceleration", "Second difference of return skewness across three consecutive non-overlapping windows."),
    ("skew_s04_vol_adjusted_diff", "Short-minus-preceding return skewness dampened by recent volatility expansion."),
    ("skew_s05_quantile", "Bowley skewness of standardized returns, using linearly interpolated quartiles."),
    ("skew_s06_tail_exceedance", "Normalized imbalance of positive and negative standardized return exceedances beyond 1.5."),
    ("skew_s07_extreme_contribution", "Return skewness minus skewness after winsorizing at the fifth and 95th percentiles."),
    ("skew_s08_tail_clustering", "Normalized imbalance of tail exceedances that follow a same-direction exceedance."),
    ("skew_s09_candle_body_ratio", "Skewness of the signed candle body divided by the candle range."),
    ("skew_s10_closing_location", "Skewness of the close location within the candle range."),
    ("skew_s11_intrabar_third_moment", "Normalized imbalance of cubed log excursions above and below the candle open."),
    ("skew_s12_range_conditional_diff", "Return skewness in high-range candles minus skewness in normal-range candles."),
    ("skew_s13_volume_weighted_premium", "Return skewness weighted by log-one-plus-relative-volume minus unweighted skewness."),
    ("skew_s14_high_low_volume_diff", "Return skewness in high-volume minutes minus skewness in low-volume minutes."),
    ("skew_s15_price_volume_coskew", "Standardized co-skewness of returns with squared log relative-volume deviations."),
    ("skew_s16_lagged_volume_shock", "Return skewness following a high relative-volume minute."),
    ("skew_s17_oi_increase_conditioned", "Return skewness in minutes with rising open interest."),
    ("skew_s18_oi_increase_decrease_diff", "Return skewness during open-interest increases minus skewness during decreases."),
    ("skew_s19_price_oi_coskew", "Standardized co-skewness of returns with squared open-interest shock deviations."),
    ("skew_s20_lagged_oi_shock", "Return skewness following a positive standardized open-interest shock above one."),
    ("skew_s21_volume_oi_conditioned", "Return skewness in high-volume minutes with rising open interest."),
    ("skew_s22_volume_oi_tail_asymmetry", "Return tail imbalance weighted by relative volume and positive open-interest shocks."),
    ("skew_s23_volume_oi_residual", "Skewness of out-of-sample return residuals from a prior-window regression on volume and open-interest shocks."),
    ("skew_s24_joint_participation_lagged", "Return skewness following a minute with high volume and rising open interest."),
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
        FactorSpec {
            name: id.clone(),
            id,
            aliases: vec![format!("{alias}_{suffix}")],
            asset_class: AssetClass::Future,
            frequency: Frequency::Minute1,
            version: "0.1.0".into(),
            tags: ["minute", "pv", "yq", "skewness"]
                .map(str::to_string)
                .to_vec(),
            description: format!("{description}{history}{volume}"),
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
    divide(m3 / n, (m2 / n).powf(1.5))
}

fn rolling_skew(values: &Values, n: usize) -> Series {
    rolling(values.len(), n, |range| {
        let mut sample = [0.0; MAX_WINDOW];
        full_sample(&values[range], &mut sample)?;
        skew(&sample[..n])
    })
}

fn conditional_skew(
    values: &Values,
    condition: &Values,
    choose: impl Fn(f64) -> bool,
) -> Option<f64> {
    let mut sample = [0.0; MAX_WINDOW];
    let mut len = 0;
    for (&value, &test) in values.iter().zip(condition) {
        let value = value?;
        if choose(test?) {
            sample[len] = value;
            len += 1;
        }
    }
    skew(&sample[..len])
}

fn conditional(
    values: &Values,
    condition: &Values,
    n: usize,
    choose: impl Fn(f64) -> bool,
) -> Series {
    rolling(values.len(), n, |range| {
        conditional_skew(&values[range.clone()], &condition[range], &choose)
    })
}

fn conditional_difference(
    values: &Values,
    condition: &Values,
    n: usize,
    high: impl Fn(f64) -> bool,
    low: impl Fn(f64) -> bool,
) -> Series {
    zip(
        &conditional(values, condition, n, high),
        &conditional(values, condition, n, low),
        |a, b| Some((a - b).tanh()),
    )
}

fn weighted_skew(values: &Values, weights: &Values, n: usize) -> Series {
    rolling(values.len(), n, |range| {
        let anchor = values[range.start]?;
        let mut pairs = values[range.clone()].iter().zip(&weights[range]);
        let (mass, numerator) = pairs
            .clone()
            .try_fold((0.0, 0.0), |(mass, total), (&x, &w)| {
                Some((mass + w?, total + w? * (x? - anchor)))
            })?;
        let center = anchor + divide(numerator, mass)?;
        let (m2, m3) = pairs.try_fold((0.0, 0.0), |(m2, m3), (&x, &w)| {
            let delta = x? - center;
            Some((m2 + w? * delta.powi(2), m3 + w? * delta.powi(3)))
        })?;
        divide(m3 / mass, (m2 / mass).powf(1.5))
    })
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
        divide(
            moment,
            covariance(x, x, mx, mx) * covariance(y, y, my, my).sqrt(),
        )
        .map(f64::tanh)
    })
}

fn imbalance(up: &Values, down: &Values, n: usize) -> Series {
    rolling(up.len(), n, |range| {
        let u = up[range.clone()].iter().copied().sum::<Option<f64>>()?;
        let d = down[range].iter().copied().sum::<Option<f64>>()?;
        divide(u - d, u + d)
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
                let excess = (sign * values[i]? - TAIL).max(0.0);
                let selected = match previous {
                    Some(lead) => sign * lead[i]? > TAIL,
                    None => true,
                };
                let weight = match weights {
                    Some(weights) => weights[i]?,
                    None => 1.0,
                };
                Some(if selected { weight * excess } else { 0.0 })
            })
            .collect()
    };
    imbalance(&tail(1.0), &tail(-1.0), n)
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
            divide(numerator, h - l)
        });
        return each(configs, |n, _| map(&rolling_skew(&candle, n), f64::tanh));
    }
    if kind == 11 {
        let up = position(inputs, |_, o, h, _| log_return(h, o).map(|x| x.powi(3)));
        let down = position(inputs, |_, o, _, l| log_return(o, l).map(|x| x.powi(3)));
        return each(configs, |n, _| imbalance(&up, &down, n));
    }
    let returns = zip(close, &ts_delay(close, 1), log_return);
    let z = zip(&returns, &ts_delay(&rms(&returns, BASE), 1), divide);
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
                (short > 0.0).then_some(())?;
                divide(short, long).map(|ratio| (-ratio.ln().max(0.0)).exp())
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
                    divide((h - m) - (m - l), h - l)
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
            let scaled = zip(&range, &ts_delay(&ts_median(&range, BASE, BASE), 1), divide);
            each(configs, |n, _| {
                conditional_difference(&z, &scaled, n, |v| v > 1.5, |v| 0.0 < v && v <= 1.0)
            })
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
        divide(oi, scale).map(f64::asinh)
    })
}

fn volume_factors(kind: usize, configs: &[Windows], z: &Values, volume: &Values) -> Vec<Series> {
    match kind {
        13 => {
            let weight = map(volume, f64::ln_1p);
            each(configs, |n, _| {
                zip(
                    &weighted_skew(z, &weight, n),
                    &rolling_skew(z, n),
                    |a, b| Some((a - b).tanh()),
                )
            })
        }
        14 => each(configs, |n, _| {
            conditional_difference(z, volume, n, |v| v >= HIGH_VOLUME, |v| 0.0 < v && v <= 0.75)
        }),
        15 => {
            let log_volume = map(volume, f64::ln);
            each(configs, |n, _| coskew(z, &log_volume, n))
        }
        16 => {
            let prev = ts_delay(volume, 1);
            each(configs, |n, _| {
                map(&conditional(z, &prev, n, |v| v > HIGH_VOLUME), f64::tanh)
            })
        }
        _ => unreachable!(),
    }
}

fn oi_factors(kind: usize, configs: &[Windows], z: &Values, oi: &Values) -> Vec<Series> {
    match kind {
        17 => each(configs, |n, _| {
            map(&conditional(z, oi, n, |d| d > 0.0), f64::tanh)
        }),
        18 => each(configs, |n, _| {
            conditional_difference(z, oi, n, |d| d > 0.0, |d| d < 0.0)
        }),
        19 => {
            let shock = oi_shocks(oi);
            each(configs, |n, _| coskew(z, &shock, n))
        }
        20 => {
            let prev = ts_delay(&oi_shocks(oi), 1);
            each(configs, |n, _| {
                map(&conditional(z, &prev, n, |d| d > OI_SHOCK), f64::tanh)
            })
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
            let joint = zip(volume, oi, |v, oi| {
                Some(if v > HIGH_VOLUME && oi > 0.0 {
                    1.0
                } else {
                    0.0
                })
            });
            let test = if kind == 24 {
                ts_delay(&joint, 1)
            } else {
                joint
            };
            each(configs, |n, _| {
                map(&conditional(z, &test, n, |yes| yes > 0.0), f64::tanh)
            })
        }
        22 => {
            let weight = zip(volume, &oi_shocks(oi), |v, d| {
                Some(v.ln_1p() * d.max(0.0).tanh())
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

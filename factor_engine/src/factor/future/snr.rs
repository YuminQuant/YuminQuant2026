use crate::core::{AssetClass, FactorContext, FactorSeries, FactorSpec, Frequency, Lookback};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::intraday_raw::quantile_linear;
use crate::factor::future::minute_input::{
    compute_future_minute_many, future_minute_request, FutureMinuteField,
    FutureMinuteField::{ContractDiff, ContractLevel},
};
use crate::factor::future::multiscale_volatility::{divide, log_return, map, zip, Series, Values};
use crate::factor::future::rolling::rolling_exponential_mean;
use crate::factor::future::soft_scoring::{event_decay, scaled, sigmoid};
use crate::factor::future::variance_amplitude::relative_volume;
use crate::factor::Factor;
use crate::operators::cross_sectional::cs_neutralize_regression::solve_linear_system;
use crate::operators::{ts_delay, ts_mean};

const FIRST_ID: usize = 256;
const WINDOWS: [usize; 3] = [5, 15, 30];
const BASELINE_WINDOWS: [usize; 3] = [5, 15, 20];
const MAX_WINDOW: usize = 30;
const SCALE_WINDOW: usize = 5;
const EWMA_DECAY: f64 = 0.94;
const CLIP_SIGMA: f64 = 3.0;
const JUMP_DECAY_MINUTES: f64 = 5.0;
const OI_EVENT_THRESHOLD: f64 = 1.0;
const EVENT_SHRINKAGE: f64 = 5.0;

const CLOSE: &[FutureMinuteField] = &[ContractLevel("close")];
const HLC: &[FutureMinuteField] = &[
    ContractLevel("close"),
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

const FACTORS: [(&str, &str); 13] = [
    ("snr01_signed_drift_noise", "Signed mean log return scaled by its population standard deviation and the square root of the window size."),
    ("snr02_regression_trend_residual", "Log-close drift from a time-trend regression, scaled by its residual RMS."),
    ("snr03_robust_drift_noise", "Trimmed mean log return scaled by 1.4826 times the return median absolute deviation and the square root of the retained sample count."),
    ("snr04_multiscale_drift_noise_change", "Difference between recent and preceding non-overlapping mean-to-standard-deviation ratios."),
    ("snr05_range_based_noise", "Signed return drift scaled by Parkinson high-low volatility."),
    ("snr06_conditional_volatility_adjusted", "Signed sum of returns standardized by a prior five-minute finite EWMA variance with decay 0.94, divided by the square root of the window size."),
    ("snr07_jump_robust_diffusive", "Return drift clipped at three times the prior five-minute RMS and scaled by bipower volatility."),
    ("snr08_jump_dominated_shock", "Signed largest absolute return, scaled by prior five-minute RMS and weighted by jump share and five-minute exponential recency."),
    ("snr09_volume_conditioned_drift_premium", "Difference between relative-volume-weighted and unweighted return drift, scaled by return noise. Relative volume uses the preceding five-minute mean."),
    ("snr10_oi_event_forward_drift", "Soft-shock-weighted historical next-minute drift, scaled by regularized response noise and shrunk by effective sample size with five pseudo-observations. A single positive OI change updates the score, followed by finite exponential decay with a linear taper over the search window; absent events use the soft background."),
    ("snr11_volume_oi_predictive", "Next-minute return predicted from log relative volume, open-interest shock and their interaction, scaled by training residual noise. Relative volume and open-interest shocks use preceding five-minute baselines."),
    ("snr12_ar1_predictive", "Next-minute AR(1) return forecast scaled by the population standard deviation of training residuals."),
    ("snr13_ma_alignment_snr", "Continuous agreement of three price moving averages, scaled by prior ATR and weighted by unsigned return signal-to-noise."),
];

const DRIFT_WINDOWS: [(usize, usize); 3] = [(5, 10), (5, 15), (10, 20)];
const MA_WINDOWS: [(usize, usize, usize); 3] = [(5, 10, 15), (5, 15, 20), (10, 20, 30)];

fn window(kind: usize, variant: usize) -> usize {
    // These formulas use up to five additional minutes for an input baseline.
    if matches!(kind, 6..=11) {
        BASELINE_WINDOWS[variant]
    } else {
        WINDOWS[variant]
    }
}

fn fields(kind: usize) -> &'static [FutureMinuteField] {
    match kind {
        5 | 13 => HLC,
        9 => VOLUME,
        10 => OI,
        11 => VOLUME_OI,
        _ => CLOSE,
    }
}

#[derive(Clone, Copy)]
struct SnrFactor {
    kind: usize,
    variant: usize,
    number: usize,
}

pub(super) fn create(number: usize) -> Box<dyn Factor> {
    let offset = number - FIRST_ID;
    Box::new(SnrFactor {
        kind: offset / WINDOWS.len() + 1,
        variant: offset % WINDOWS.len(),
        number,
    })
}

impl Factor for SnrFactor {
    fn spec(&self) -> FactorSpec {
        let n = window(self.kind, self.variant);
        let (alias, description) = FACTORS[self.kind - 1];
        let (suffix, windows) = match self.kind {
            4 => {
                let (short, past) = DRIFT_WINDOWS[self.variant];
                (format!("{short}m_{past}m"), format!("The recent window is {short} minutes and the preceding window is {past} minutes."))
            }
            13 => {
                let (short, medium, long) = MA_WINDOWS[self.variant];
                (format!("ma{short}_{medium}_{long}_{n}m"), format!("The moving averages use {short}, {medium} and {long} minutes. ATR and return noise use {n} minutes."))
            }
            10 => (format!("{n}m"), format!("The search looks back {n} minutes and open-interest shocks use a prior five-minute scale.")),
            11 | 12 => (format!("{n}m"), format!("The regression uses {n} completed minute pairs.")),
            _ => (format!("{n}m"), format!("The factor window is {n} minutes.")),
        };
        let id = format!("yq_{:03}", self.number);
        FactorSpec {
            name: id.clone(),
            id,
            aliases: vec![format!("{alias}_{suffix}")],
            asset_class: AssetClass::Future,
            frequency: Frequency::Minute1,
            version: if self.kind == 10 { "0.2.0" } else { "0.1.0" }.into(),
            tags: ["minute", "pv", "yq", "snr"].map(str::to_string).to_vec(),
            description: format!("{description} {windows}"),
            dependencies: vec![future_minute_request(fields(self.kind))],
            intraday_raw_dependencies: Vec::new(),
            lookback: Lookback { trading_days: 3 },
        }
    }

    fn compute_provider_key(&self) -> String {
        format!("future-minute-snr-{:02}", self.kind)
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
        let (specs, selected): (Vec<_>, Vec<_>) = (0..WINDOWS.len())
            .filter_map(|variant| {
                let spec = Self {
                    kind: self.kind,
                    variant,
                    number: first + variant,
                }
                .spec();
                requested_ids.contains(&spec.id).then_some((spec, variant))
            })
            .unzip();
        if specs.is_empty() {
            return Ok(Vec::new());
        }
        // One extra close or OI level can anchor a lagged return or difference.
        compute_future_minute_many(
            &specs,
            context,
            data,
            fields(self.kind),
            MAX_WINDOW
                + 1
                + if self.kind == 10 {
                    BASELINE_WINDOWS[2]
                } else {
                    0
                },
            |inputs| evaluate(self.kind, &selected, inputs),
        )
    }
}

fn rolling(len: usize, first: usize, evaluate: impl Fn(usize) -> Option<f64>) -> Series {
    let mut out = vec![None; len];
    for (i, value) in out.iter_mut().enumerate().skip(first) {
        *value = evaluate(i).filter(|x| x.is_finite());
    }
    out
}

fn sample<'a>(values: &Values, buffer: &'a mut [f64; MAX_WINDOW]) -> Option<&'a [f64]> {
    for (&value, slot) in values.iter().zip(buffer.iter_mut()) {
        *slot = value.filter(|x| x.is_finite())?;
    }
    Some(&buffer[..values.len()])
}

fn rolling_sample(values: &Values, n: usize, f: impl Fn(&[f64]) -> Option<f64>) -> Series {
    rolling(values.len(), n - 1, |t| {
        f(sample(&values[t + 1 - n..=t], &mut [0.0; MAX_WINDOW])?)
    })
}

fn mean(values: &[f64]) -> f64 {
    let first = values[0];
    first + values.iter().map(|x| x - first).sum::<f64>() / values.len() as f64
}

fn mean_square(values: &[f64]) -> f64 {
    values.iter().map(|x| x * x).sum::<f64>() / values.len() as f64
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(x, y)| x * y).sum()
}

fn std_dev(values: &[f64]) -> f64 {
    let avg = mean(values);
    (values.iter().map(|x| (x - avg).powi(2)).sum::<f64>() / values.len() as f64).sqrt()
}

fn ratio(signal: f64, noise: f64) -> Option<f64> {
    if signal.is_finite() && noise.is_finite() {
        divide(signal, noise)
    } else {
        None
    }
}

fn score(signal: f64, noise: f64) -> Option<f64> {
    ratio(signal, noise).map(f64::tanh)
}

fn drift_noise(returns: &Values, n: usize, scale: f64) -> Series {
    rolling_sample(returns, n, |r| ratio(scale * mean(r), std_dev(r)))
}

fn trend_drift(log_close: &Values, n: usize) -> Series {
    rolling_sample(log_close, n, |values| {
        let center_t = (n - 1) as f64 / 2.0;
        let center_y = mean(values);
        let centered_t = (0..n).map(|i| i as f64 - center_t);
        let beta = centered_t
            .clone()
            .zip(values)
            .map(|(t, y)| t * (y - center_y))
            .sum::<f64>()
            / centered_t.clone().map(|t| t * t).sum::<f64>();
        let noise = (centered_t
            .zip(values)
            .map(|(t, y)| ((y - center_y) - beta * t).powi(2))
            .sum::<f64>()
            / n as f64)
            .sqrt();
        // Exact affine log prices can leave rounding-sized residuals after the regression.
        let magnitude = values.iter().map(|value| value.abs()).fold(0.0, f64::max);
        (noise > 4.0 * f64::EPSILON * magnitude).then_some(())?;
        score(beta * (n - 1) as f64, noise)
    })
}

fn robust_drift(values: &[f64]) -> Option<f64> {
    let mut buffer = [0.0; MAX_WINDOW];
    let sorted = &mut buffer[..values.len()];
    sorted.copy_from_slice(values);
    let lo = quantile_linear(sorted, 0.1)?;
    let hi = quantile_linear(sorted, 0.9)?;
    let center = quantile_linear(sorted, 0.5)?;
    let (count, sum) = values
        .iter()
        .filter(|&&r| lo <= r && r <= hi)
        .fold((0, 0.0), |(count, sum), r| (count + 1, sum + r));
    let signal = ratio(sum, (count as f64).sqrt())?;
    sorted.iter_mut().for_each(|r| *r = (*r - center).abs());
    score(signal, 1.4826 * quantile_linear(sorted, 0.5)?)
}

fn price_ranges(inputs: &[Series], true_range: bool) -> Series {
    let (close, high, low) = (&inputs[0], &inputs[1], &inputs[2]);
    (0..close.len())
        .map(|i| {
            let (c, h, l) = (close[i]?, high[i]?, low[i]?);
            (0.0 < l && l <= c && c <= h).then_some(())?;
            if true_range {
                let prior = close[i.checked_sub(1)?]?;
                (prior > 0.0).then(|| (h - l).max((h - prior).abs()).max((l - prior).abs()))
            } else {
                log_return(h, l)
            }
        })
        .collect()
}

fn lagged_rms(returns: &Values) -> Series {
    ts_delay(
        &rolling_sample(returns, SCALE_WINDOW, |r| Some(mean_square(r).sqrt())),
        1,
    )
}

fn bipower(returns: &[f64]) -> f64 {
    let sum = returns
        .windows(2)
        .map(|pair| (pair[0] * pair[1]).abs())
        .sum::<f64>();
    std::f64::consts::FRAC_PI_2 * sum / (returns.len() - 1) as f64
}

fn jumps(returns: &Values, n: usize, scales: &Values, clipped: Option<&Values>) -> Series {
    rolling(returns.len(), n - 1, |t| {
        let start = t + 1 - n;
        let mut buffer = [0.0; MAX_WINDOW];
        let r = sample(&returns[start..=t], &mut buffer)?;
        let continuous_variance = bipower(r);
        if let Some(clipped) = clipped {
            let sum = clipped[start..=t].iter().copied().sum::<Option<f64>>()?;
            return score(sum / (n as f64).sqrt(), continuous_variance.sqrt());
        }
        let realized = mean_square(r);
        let share = ratio((realized - continuous_variance).max(0.0), realized)?.min(1.0);
        let largest = r
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.abs().total_cmp(&b.abs()))?
            .0;
        let age = (n - 1 - largest) as f64;
        Some(
            share
                * (-age / JUMP_DECAY_MINUTES).exp()
                * score(r[largest], scales[start + largest]?)?,
        )
    })
}

fn oi_shocks(oi_change: &Values, soft: bool) -> Series {
    let typical = ts_delay(
        &ts_mean(&map(oi_change, f64::abs), SCALE_WINDOW, SCALE_WINDOW),
        1,
    );
    zip(oi_change, &typical, |change, base| {
        (if soft {
            scaled(change, base)
        } else {
            ratio(change, base)
        })
        .map(f64::asinh)
    })
}

fn oi_event_drift(returns: &Values, shocks: &Values, n: usize) -> Series {
    let scores = rolling(returns.len(), n, |t| {
        let current = shocks[t]?;
        let (mut mass, mut mass2, mut total) = (0.0, 0.0, 0.0);
        let mut square = 0.0;
        let mut baseline = 0.0;
        // Only realized responses strictly before t enter the training sample.
        for i in t - n..t - 1 {
            let next_return = returns[i + 1]?;
            let weight = sigmoid((shocks[i]? - OI_EVENT_THRESHOLD) / 0.5);
            mass += weight;
            mass2 += weight * weight;
            total += weight * next_return;
            square += weight * next_return * next_return;
            baseline += next_return * next_return;
        }
        if mass == 0.0 {
            return Some(0.0);
        }
        let response = total / mass;
        let variance = (square / mass - response * response).max(0.0);
        let noise = (variance + 0.05 * baseline / (n - 1) as f64).sqrt();
        let neff = mass * mass / mass2;
        let activation = sigmoid((current - OI_EVENT_THRESHOLD) / 0.5);
        Some(activation * neff / (neff + EVENT_SHRINKAGE) * scaled(response, noise)?.tanh())
    });
    // One positive OI change is enough to start/restart the finite event memory.
    let events = shocks.iter().map(|s| Some((*s)? > 0.0)).collect::<Vec<_>>();
    event_decay(&scores, &events, n)
}

fn forecast_score<const P: usize>(
    target: &[f64],
    predictors: &mut [[f64; MAX_WINDOW]; P],
    current: [f64; P],
) -> Option<f64> {
    let count = target.len();
    let target_mean = mean(target);
    let mut current_z = [0.0; P];
    // Standardize the predictors before solving the centered OLS normal equations.
    // The target mean supplies the intercept since the training features are centered.
    for (j, column) in predictors.iter_mut().enumerate() {
        let column = &mut column[..count];
        let center = mean(column);
        let scale = std_dev(column);
        current_z[j] = ratio(current[j] - center, scale)?;
        column.iter_mut().for_each(|x| *x = (*x - center) / scale);
    }
    let gram = predictors
        .iter()
        .map(|a| {
            predictors
                .iter()
                .map(|b| dot(&a[..count], &b[..count]))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let rhs = predictors
        .iter()
        .map(|a| {
            a[..count]
                .iter()
                .zip(target)
                .map(|(x, y)| x * (y - target_mean))
                .sum()
        })
        .collect::<Vec<_>>();
    let beta = solve_linear_system(gram, rhs)?;
    let mut residual = [0.0; MAX_WINDOW];
    for (i, &result) in target.iter().enumerate() {
        let adjustment = beta
            .iter()
            .zip(predictors.iter())
            .map(|(b, x)| b * x[i])
            .sum::<f64>();
        residual[i] = result - target_mean - adjustment;
    }
    let noise = std_dev(&residual[..count]);
    let magnitude = target.iter().map(|v| v.abs()).fold(0.0, f64::max);
    (noise > 4.0 * f64::EPSILON * magnitude).then_some(())?;
    score(target_mean + dot(&beta, &current_z), noise)
}

fn predictive<const P: usize>(returns: &Values, features: [&Values; P], n: usize) -> Series {
    rolling(returns.len(), n, |t| {
        let mut target = [0.0; MAX_WINDOW];
        // Features through t-1 are paired with realized returns through t.
        let target = sample(&returns[t + 1 - n..=t], &mut target)?;
        let mut training = [[0.0; MAX_WINDOW]; P];
        let mut current = [0.0; P];
        for (j, source) in features.iter().enumerate() {
            sample(&source[t - n..t], &mut training[j])?;
            current[j] = source[t]?;
        }
        forecast_score(target, &mut training, current)
    })
}

fn alignment(close: &Values, returns: &Values, tr: &Values, variant: usize) -> Series {
    let n = WINDOWS[variant];
    let (fast, medium, slow) = MA_WINDOWS[variant];
    let (a, b, c) = (
        ts_mean(close, fast, fast),
        ts_mean(close, medium, medium),
        ts_mean(close, slow, slow),
    );
    let atr = ts_delay(&ts_mean(tr, n, n), 1);
    let strength = map(&drift_noise(returns, n, (n as f64).sqrt()), |value| {
        value.abs().tanh()
    });
    (0..close.len())
        .map(|t| {
            let u = score(a[t]? - b[t]?, atr[t]?)?;
            let v = score(b[t]? - c[t]?, atr[t]?)?;
            Some((u.max(0.0) * v.max(0.0) - (-u).max(0.0) * (-v).max(0.0)) * strength[t]?)
        })
        .collect()
}

fn each(
    kind: usize,
    variants: &[usize],
    calculate: impl Fn(usize, usize) -> Series,
) -> Vec<Series> {
    variants
        .iter()
        .map(|&variant| calculate(window(kind, variant), variant))
        .collect()
}

fn evaluate(kind: usize, variants: &[usize], inputs: &[Series]) -> Vec<Series> {
    let close = &inputs[0];
    if kind == 2 {
        let log_close = map(close, f64::ln);
        return each(kind, variants, |n, _| trend_drift(&log_close, n));
    }
    let returns = zip(close, &ts_delay(close, 1), log_return);
    match kind {
        1 => each(kind, variants, |n, _| {
            map(&drift_noise(&returns, n, (n as f64).sqrt()), f64::tanh)
        }),
        3 => each(kind, variants, |n, _| {
            rolling_sample(&returns, n, robust_drift)
        }),
        4 => each(kind, variants, |_, v| {
            let (short, prior) = DRIFT_WINDOWS[v];
            let recent = drift_noise(&returns, short, 1.0);
            let history = ts_delay(&drift_noise(&returns, prior, 1.0), short);
            zip(&recent, &history, |a, b| Some((a - b).tanh()))
        }),
        5 => {
            let range = price_ranges(inputs, false);
            each(kind, variants, |n, _| {
                let drift = rolling_sample(&returns, n, |r| Some(mean(r) * (n as f64).sqrt()));
                let parkinson = rolling_sample(&range, n, |r| {
                    Some((mean_square(r) / (4.0 * std::f64::consts::LN_2)).sqrt())
                });
                zip(&drift, &parkinson, score)
            })
        }
        6 => {
            let variance =
                rolling_exponential_mean(&map(&returns, |r| r * r), SCALE_WINDOW, EWMA_DECAY);
            let standardized = zip(&returns, &ts_delay(&variance, 1), |r, var| {
                ratio(r, var.sqrt())
            });
            each(kind, variants, |n, _| {
                rolling_sample(&standardized, n, |r| {
                    Some((mean(r) * (n as f64).sqrt()).tanh())
                })
            })
        }
        7 | 8 => {
            let rms = lagged_rms(&returns);
            let clipped = (kind == 7).then(|| {
                zip(&returns, &rms, |r, scale| {
                    Some(r.clamp(-CLIP_SIGMA * scale, CLIP_SIGMA * scale))
                })
            });
            each(kind, variants, |n, _| {
                jumps(&returns, n, &rms, clipped.as_deref())
            })
        }
        9 => {
            let weights = map(&relative_volume(&inputs[1]), f64::ln_1p);
            each(kind, variants, |n, _| {
                rolling(returns.len(), n - 1, |t| {
                    let range = t + 1 - n..=t;
                    let mut r_buf = [0.0; MAX_WINDOW];
                    let mut w_buf = [0.0; MAX_WINDOW];
                    let r = sample(&returns[range.clone()], &mut r_buf)?;
                    let w = sample(&weights[range], &mut w_buf)?;
                    let premium = ratio(dot(w, r), w.iter().sum())? - mean(r);
                    score((n as f64).sqrt() * premium, std_dev(r))
                })
            })
        }
        10 => {
            let shock = oi_shocks(&inputs[1], true);
            each(kind, variants, |n, _| oi_event_drift(&returns, &shock, n))
        }
        11 => {
            let log_volume = map(&relative_volume(&inputs[1]), f64::ln);
            let shock = oi_shocks(&inputs[2], false);
            let joint = zip(&log_volume, &shock, |v, d| Some(v * d));
            each(kind, variants, |n, _| {
                predictive(&returns, [&log_volume, &shock, &joint], n)
            })
        }
        12 => each(kind, variants, |n, _| predictive(&returns, [&returns], n)),
        13 => {
            let valid_close = close
                .iter()
                .map(|v| v.filter(|value| *value > 0.0))
                .collect::<Series>();
            let tr = price_ranges(inputs, true);
            each(kind, variants, |_, v| {
                alignment(&valid_close, &returns, &tr, v)
            })
        }
        _ => unreachable!("unknown SNR formula"),
    }
}

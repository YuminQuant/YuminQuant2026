use crate::core::{AssetClass, FactorContext, FactorSeries, FactorSpec, Frequency, Lookback};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::future::minute_input::{
    compute_future_minute, future_minute_request, FutureMinuteField,
    FutureMinuteField::{ContractDiff, ContractLevel},
};
use crate::factor::future::multiscale_volatility::{
    divide, factor_id, log_return, map, sample_std, zip, Series, Values, HISTORY_BARS,
    VARIANT_SUFFIXES, WINDOWS,
};
use crate::factor::Factor;
use crate::operators::cross_sectional::cs_neutralize_regression::solve_linear_system;
use crate::operators::{ts_delay, ts_mean, ts_median, ts_sum};

const SHORT_VR: usize = 5;
const ARCH_LAGS: usize = 5;
const VOLUME_BASE: usize = 5;
const DUAL_VR_WINDOWS: [usize; 3] = [15, 20, 30];
const LONG_VR: [usize; 3] = [10, 15, 15];
const WEIGHTED_VR_WINDOWS: [usize; 3] = [10, 15, 20];
const VOLUME_ADJUSTED_WINDOWS: [usize; 3] = [5, 15, 25];

const CLOSE: &[FutureMinuteField] = &[ContractLevel("close")];
const CLOSE_OPEN: &[FutureMinuteField] = &[ContractLevel("close"), ContractLevel("open")];
const OHLC: &[FutureMinuteField] = &[
    ContractLevel("close"),
    ContractLevel("open"),
    ContractLevel("high"),
    ContractLevel("low"),
];
const OHLC_VOLUME: &[FutureMinuteField] =
    &[OHLC[0], OHLC[1], OHLC[2], OHLC[3], ContractLevel("vol")];
const OHLC_OI: &[FutureMinuteField] = &[OHLC[0], OHLC[1], OHLC[2], OHLC[3], ContractDiff("oi")];
const CLOSE_OPEN_VOLUME: &[FutureMinuteField] =
    &[CLOSE_OPEN[0], CLOSE_OPEN[1], ContractLevel("vol")];
const CLOSE_OPEN_OI: &[FutureMinuteField] = &[CLOSE_OPEN[0], CLOSE_OPEN[1], ContractDiff("oi")];
const OHLC_VOLUME_OI: &[FutureMinuteField] = &[
    OHLC[0],
    OHLC[1],
    OHLC[2],
    OHLC[3],
    ContractLevel("vol"),
    ContractDiff("oi"),
];

const FACTORS: [(&str, &str); 12] = [
    ("vr_v01_robust_direction_switch", "Candle-body direction weighted by the heteroskedasticity-robust variance ratio statistic."),
    ("vr_v02_multihorizon_divergence", "Candle-body direction weighted by opposite signs in short and long robust variance ratio statistics."),
    ("vr_v03_positive_negative_asymmetry", "Log variance ratio difference between positive and negative return magnitudes."),
    ("vr_v04_volume_conditioned_premium", "Candle-body direction weighted by the variance ratio premium of volume-weighted returns."),
    ("vr_v05_open_interest_conditioned_premium", "Candle-body direction weighted by the variance ratio premium of returns weighted by positive changes in open interest."),
    ("vr_v06_arch_closing_direction", "Close location within the candle multiplied by ARCH LM strength from a five-lag regression."),
    ("amp_a01_open_anchored_asymmetry", "Normalized imbalance of squared log amplitudes above and below the candle open."),
    ("amp_a02_volume_wick_rejection", "Normalized lower versus upper wick energy, weighted by relative volume capped at three."),
    ("amp_a03_volume_adjusted_range_impact", "Candle-body direction weighted by expansion in log range divided by the square root of relative volume."),
    ("amp_a04_oi_weighted_range_pressure", "Normalized imbalance of open-anchored amplitudes weighted by positive changes in the same contract's open interest."),
    ("amp_a05_rogers_satchell_disagreement", "Candle-body direction weighted by the normalized difference between squared body return and Rogers–Satchell variance."),
    ("amp_a06_participation_range_compression", "Candle-body direction weighted by relative volume, the fraction of volume that adds open interest, and compressed price range."),
];

struct VarianceAmplitudeFactor {
    index: usize,
    variant: usize,
}

pub(super) fn create(index: usize, variant: usize) -> Box<dyn Factor> {
    Box::new(VarianceAmplitudeFactor { index, variant })
}

fn fields(index: usize) -> &'static [FutureMinuteField] {
    match index {
        0 | 1 => CLOSE_OPEN,
        2 => CLOSE,
        3 => CLOSE_OPEN_VOLUME,
        4 => CLOSE_OPEN_OI,
        5 | 6 | 10 => OHLC,
        7 | 8 => OHLC_VOLUME,
        9 => OHLC_OI,
        11 => OHLC_VOLUME_OI,
        _ => unreachable!("variance or amplitude factor index"),
    }
}

impl Factor for VarianceAmplitudeFactor {
    fn spec(&self) -> FactorSpec {
        let (alias, description) = FACTORS[self.index];
        let id = factor_id(10 + self.index, self.variant);
        FactorSpec {
            name: id.clone(),
            id,
            aliases: if self.variant == 1 {
                vec![alias.into()]
            } else {
                vec![format!("{alias}_{}", VARIANT_SUFFIXES[self.variant])]
            },
            asset_class: AssetClass::Future,
            frequency: Frequency::Minute1,
            version: "0.1.0".into(),
            tags: ["minute", "pv", "yq", "volatility"]
                .map(str::to_string)
                .to_vec(),
            description: format!(
                "{description} {}",
                window_description(self.index, self.variant)
            ),
            dependencies: vec![future_minute_request(fields(self.index))],
            intraday_raw_dependencies: Vec::new(),
            lookback: Lookback { trading_days: 3 },
        }
    }

    fn compute(&self, context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        compute_future_minute(
            self.spec(),
            context,
            data,
            fields(self.index),
            HISTORY_BARS,
            |inputs| evaluate(self.index, self.variant, inputs),
        )
    }
}

fn variance_window(variant: usize) -> usize {
    // q = 5 must be strictly smaller than the variance-ratio window.
    [10, 15, 30][variant]
}

fn window_description(index: usize, variant: usize) -> String {
    let window = WINDOWS[variant];
    let volume = "Relative volume uses the preceding five minutes.";
    match index {
        0 => format!("Variance ratio uses {} minutes at order 5 and direction uses {} prior minutes.", variance_window(variant), window),
        1 => format!("Variance ratios use {} minutes at orders 5 and {} and direction uses {} prior minutes.", DUAL_VR_WINDOWS[variant], LONG_VR[variant], window),
        2 => format!("Variance ratios use {} minutes at order 5.", variance_window(variant)),
        3 => format!("Variance ratios use {} minutes at order 5 and direction uses {} prior minutes. {volume}", WEIGHTED_VR_WINDOWS[variant], window),
        4 => format!("Variance ratios use {} minutes at order 5 and direction uses {} prior minutes. The open-interest scale uses the preceding five minutes.", WEIGHTED_VR_WINDOWS[variant], window),
        5 => format!("The regression uses {} minutes of returns.", DUAL_VR_WINDOWS[variant]),
        6 | 9 => format!("The amplitude window is {window} minutes."),
        7 | 8 => format!("The amplitude window is {} minutes. {volume}", VOLUME_ADJUSTED_WINDOWS[variant]),
        10 => format!("Direction uses {window} prior minutes."),
        11 => format!("Direction and the range baseline use {window} prior minutes. {volume}"),
        _ => unreachable!(),
    }
}

fn with_windows(values: &Values, n: usize, f: impl Fn(&[f64]) -> Option<f64>) -> Series {
    let mut output = vec![None; values.len()];
    if let Some(tail) = output.get_mut(n - 1..) {
        for (result, window) in tail.iter_mut().zip(values.windows(n)) {
            let mut sample = [0.0; 30];
            let sample = &mut sample[..n];
            if window.iter().zip(sample.iter_mut()).all(|(value, slot)| {
                value.is_some_and(|v| {
                    *slot = v;
                    v.is_finite()
                })
            }) {
                *result = f(sample).filter(|value| value.is_finite());
            }
        }
    }
    output
}

fn previous_median(values: &Values, n: usize) -> Series {
    ts_delay(&ts_median(values, n, n), 1)
}

fn previous_mean(values: &Values, n: usize) -> Series {
    ts_delay(&ts_mean(values, n, n), 1)
}

pub(super) fn relative_volume(volume: &Values) -> Series {
    let volume = volume
        .iter()
        .map(|v| v.filter(|x| *x >= 0.0))
        .collect::<Series>();
    zip(&volume, &previous_mean(&volume, VOLUME_BASE), divide)
}

fn ratio_stats(sample: &[f64], q: usize, robust: bool) -> Option<f64> {
    let n = sample.len();
    let mean = sample.iter().sum::<f64>() / n as f64;
    let total_sq = sample.iter().map(|v| (v - mean).powi(2)).sum::<f64>();
    if total_sq <= 0.0 {
        return None;
    }
    let mut sum = sample[..q].iter().sum::<f64>() - q as f64 * mean;
    let mut squares = sum * sum;
    for i in q..n {
        sum += sample[i] - sample[i - q];
        squares += sum * sum;
    }
    let normalization = (q * (n - q + 1)) as f64 * (1.0 - q as f64 / n as f64);
    let ratio = (squares / normalization) / (total_sq / (n - 1) as f64);
    if !robust {
        return Some(ratio);
    }
    let mut leverage = [0.0; 30];
    for (slot, value) in leverage.iter_mut().zip(sample) {
        *slot = (value - mean).powi(2) / total_sq;
    }
    let theta = (1..q)
        .map(|lag| {
            let paired = leverage[lag..n]
                .iter()
                .zip(&leverage[..n - lag])
                .map(|(a, b)| a * b)
                .sum::<f64>();
            4.0 * (1.0 - lag as f64 / q as f64).powi(2) * n as f64 * paired
        })
        .sum::<f64>();
    divide((n as f64).sqrt() * (ratio - 1.0), theta.sqrt())
}

fn variance_ratio(values: &Values, n: usize, q: usize) -> Series {
    with_windows(values, n, |sample| ratio_stats(sample, q, false))
}

fn robust_vr_z(values: &Values, n: usize, q: usize) -> Series {
    with_windows(values, n, |sample| ratio_stats(sample, q, true))
}

fn arch_lm(returns: &[f64]) -> Option<f64> {
    let n = returns.len();
    let mean = returns.iter().sum::<f64>() / n as f64;
    let mut squared = [0.0; 30];
    let squared = &mut squared[..n];
    for (slot, value) in squared.iter_mut().zip(returns) {
        *slot = (value - mean).powi(2);
    }
    // OLS R² is unchanged by scaling all squared residuals. Use a common
    // scale so very small minute returns do not cause spurious zero pivots.
    let scale = squared.iter().copied().fold(0.0, f64::max);
    if scale == 0.0 {
        return None;
    }
    squared.iter_mut().for_each(|value| *value /= scale);
    let columns = ARCH_LAGS + 1;
    let count = n - ARCH_LAGS;
    let mut xtx = vec![vec![0.0; columns]; columns];
    let mut xty = vec![0.0; columns];
    let mut y_sum = 0.0;
    let mut y_sq = 0.0;
    for window in squared.windows(columns) {
        let (lags, target) = window.split_at(ARCH_LAGS);
        let mut row = [1.0; ARCH_LAGS + 1];
        row[1..].copy_from_slice(lags);
        let y = target[0];
        y_sum += y;
        y_sq += y * y;
        for i in 0..columns {
            xty[i] += row[i] * y;
            for j in 0..columns {
                xtx[i][j] += row[i] * row[j];
            }
        }
    }
    let coefficients = solve_linear_system(xtx, xty.clone())?;
    let explained = coefficients
        .iter()
        .zip(xty)
        .map(|(a, b)| a * b)
        .sum::<f64>();
    let centered_mean = y_sum * y_sum / count as f64;
    let r2 = divide(explained - centered_mean, y_sq - centered_mean)?;
    Some(count as f64 * r2.clamp(0.0, 1.0))
}

fn evaluate(index: usize, variant: usize, inputs: &[Series]) -> Series {
    let n = WINDOWS[variant];
    if index >= 6 {
        return amplitude(index - 6, variant, inputs);
    }
    let close = &inputs[0];
    let returns = zip(close, &ts_delay(close, 1), log_return);
    let positive = |v: f64| v.max(0.0).tanh();
    if index == 2 {
        let window = variance_window(variant);
        let up = variance_ratio(&map(&returns, |r| r.max(0.0)), window, SHORT_VR);
        let down = variance_ratio(&map(&returns, |r| (-r).max(0.0)), window, SHORT_VR);
        return zip(&up, &down, |up, down| log_return(up, down).map(f64::tanh));
    }
    if index == 5 {
        let lm = with_windows(&returns, DUAL_VR_WINDOWS[variant], arch_lm);
        return (0..close.len())
            .map(|i| {
                let (c, h, l) = (close[i]?, inputs[2][i]?, inputs[3][i]?);
                if !(l > 0.0 && l <= c && c <= h) {
                    return None;
                }
                let clv = divide((c - l) - (h - c), h - l)?;
                Some(clv * -(-lm[i]? / 5.0).exp_m1())
            })
            .collect();
    }
    let direction = direction(close, &inputs[1], &returns, n);
    let score = match index {
        0 => map(
            &robust_vr_z(&returns, variance_window(variant), SHORT_VR),
            f64::tanh,
        ),
        1 => {
            let window = DUAL_VR_WINDOWS[variant];
            zip(
                &robust_vr_z(&returns, window, SHORT_VR),
                &robust_vr_z(&returns, window, LONG_VR[variant]),
                |s, l| Some(positive(s) * positive(-l) - positive(-s) * positive(l)),
            )
        }
        3 | 4 => {
            let weights = if index == 3 {
                let log_volume = map(&relative_volume(&inputs[2]), f64::ln_1p);
                zip(&log_volume, &previous_median(&log_volume, 5), divide)
            } else {
                let change = &inputs[2];
                zip(
                    change,
                    &previous_mean(&map(change, f64::abs), 5),
                    |change, scale| divide(change.max(0.0), scale).map(f64::tanh),
                )
            };
            let adjusted = zip(&returns, &weights, |r, w| Some(r * w));
            let window = WEIGHTED_VR_WINDOWS[variant];
            zip(
                &variance_ratio(&adjusted, window, SHORT_VR),
                &variance_ratio(&returns, window, SHORT_VR),
                |a, b| Some((a - b).tanh()),
            )
        }
        _ => unreachable!(),
    };
    zip(&direction, &score, |d, s| Some(d * s))
}

fn direction(close: &Values, open: &Values, returns: &Values, n: usize) -> Series {
    zip(
        &zip(close, open, log_return),
        &ts_delay(&sample_std(returns, n), 1),
        |body, scale| divide(body, scale).map(f64::tanh),
    )
}

struct Candle {
    upper: f64,
    lower: f64,
    body: f64,
}

fn amplitude(index: usize, variant: usize, inputs: &[Series]) -> Series {
    let (close, open, high, low) = (&inputs[0], &inputs[1], &inputs[2], &inputs[3]);
    let n = WINDOWS[variant];
    let candle = (0..close.len())
        .map(|i| {
            let (c, o, h, l) = (close[i]?, open[i]?, high[i]?, low[i]?);
            if l > o.min(c) || h < o.max(c) {
                return None;
            }
            Some(Candle {
                upper: log_return(h, o)?,
                lower: log_return(o, l)?,
                body: log_return(c, o)?,
            })
        })
        .collect::<Vec<_>>();
    let part = |f: fn(&Candle) -> f64| candle.iter().map(|c| c.as_ref().map(f)).collect::<Series>();
    let relative = || relative_volume(&inputs[4]);
    let imbalance = |up: &Values, down: &Values, width| {
        zip(
            &ts_sum(up, width, width),
            &ts_sum(down, width, width),
            |u, d| divide(u - d, u + d),
        )
    };
    match index {
        0 => return imbalance(&part(|c| c.upper.powi(2)), &part(|c| c.lower.powi(2)), n),
        1 => {
            let volume = relative();
            let weight = |wick: &Values| zip(wick, &volume, |w, v| Some(v.min(3.0) * w.powi(2)));
            let upper = weight(&part(|c| c.upper - c.body.max(0.0)));
            let lower = weight(&part(|c| c.lower + c.body.min(0.0)));
            return imbalance(&lower, &upper, VOLUME_ADJUSTED_WINDOWS[variant]);
        }
        3 => {
            let weight =
                |amplitude: &Values| zip(amplitude, &inputs[4], |a, oi| Some(oi.max(0.0) * a));
            return imbalance(&weight(&part(|c| c.upper)), &weight(&part(|c| c.lower)), n);
        }
        _ => {}
    }
    let range = part(|c| c.upper + c.lower);
    let returns = zip(close, &ts_delay(close, 1), log_return);
    let direction = direction(close, open, &returns, n);
    let score = match index {
        2 => {
            let adjusted = zip(&range, &relative(), |a, v| divide(a, v.sqrt()));
            zip(
                &adjusted,
                &previous_median(&adjusted, VOLUME_ADJUSTED_WINDOWS[variant]),
                |a, base| {
                    if a == 0.0 && base > 0.0 {
                        Some(0.0)
                    } else {
                        log_return(a, base).map(|change| change.max(0.0).tanh())
                    }
                },
            )
        }
        4 => candle
            .iter()
            .map(|c| {
                let c = c.as_ref()?;
                let body_sq = c.body.powi(2);
                let rs = c.upper * (c.upper - c.body) + c.lower * (c.lower + c.body);
                divide(body_sq - rs, body_sq + rs)
            })
            .collect(),
        5 => {
            let participation = zip(&inputs[5], &inputs[4], |oi, volume| {
                divide(oi.max(0.0), volume)
            });
            let compression = zip(&range, &previous_median(&range, n), |a, base| {
                divide(a, base).map(|x| (-x).exp())
            });
            let volume = map(&relative(), f64::ln);
            let sigmoid = |v: f64| 1.0 / (1.0 + (-v).exp());
            (0..close.len())
                .map(|i| {
                    Some(sigmoid(volume[i]?) * sigmoid(participation[i]? - 0.1) * compression[i]?)
                })
                .collect()
        }
        _ => unreachable!(),
    };
    zip(&direction, &score, |d, s| Some(d * s))
}

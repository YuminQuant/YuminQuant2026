use crate::core::{AssetClass, FactorContext, FactorSeries, FactorSpec, Frequency, Lookback};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::future::minute_input::{
    compute_future_minute, future_minute_request, FutureMinuteField,
    FutureMinuteField::ContractLevel,
};
use crate::factor::future::rolling::rolling_exponential_mean;
use crate::factor::Factor;
use crate::operators::{ts_delay, ts_max, ts_median, ts_min, ts_sum};

const SHORT: usize = 5;
const MOMENTUM_SCALE: usize = 5;
const EXPANSION_BASES: [usize; 3] = [10, 15, 20];
const CHANNEL_WINDOWS: [usize; 3] = [5, 10, 15];
const COMPRESSION_BASES: [usize; 3] = [5, 10, 10];
const TR_WEIGHTS: [usize; 3] = [5, 15, 25];
pub(super) const WINDOWS: [usize; 3] = [5, 15, 30];
pub(super) const VARIANT_SUFFIXES: [&str; 3] = ["short", "", "long"];
// Include the close anchoring the first lagged return or true range.
pub(super) const HISTORY_BARS: usize = 31;
pub(super) type Series = Vec<Option<f64>>;
pub(super) type Values = [Option<f64>];

const CLOSE: &[FutureMinuteField] = &[ContractLevel("close")];
const HIGH_LOW_CLOSE: &[FutureMinuteField] = &[
    ContractLevel("close"),
    ContractLevel("high"),
    ContractLevel("low"),
];
const OHLC: &[FutureMinuteField] = &[
    ContractLevel("close"),
    ContractLevel("high"),
    ContractLevel("low"),
    ContractLevel("open"),
];
const GAP_INPUT: &[FutureMinuteField] = &[ContractLevel("close"), ContractLevel("open")];

const FACTORS: [(&str, &str); 10] = [
    ("msv_f01_expansion_momentum", "Five-minute volatility-scaled momentum weighted by expansion relative to the preceding baseline."),
    ("msv_f02_expansion_signed_er", "Five-bar signed return path efficiency weighted by non-overlapping realized volatility expansion."),
    ("msv_f03_multihorizon_alignment", "Volatility expansion times short momentum, gated by agreement with standardized momentum over the preceding baseline."),
    ("msv_f04_semivariance_imbalance", "Five-bar upside versus downside semivariance imbalance weighted by volatility expansion."),
    ("msv_f05_semivariance_rotation", "Change in signed semivariance imbalance between five recent minutes and the preceding baseline, weighted by volatility expansion."),
    ("msv_f06_vol_of_vol_shock", "Five-bar standardized momentum weighted by expansion in the sample volatility of log five-bar realized volatility."),
    ("msv_f07_compression_breakout", "ATR-scaled minute channel breakout confirmed by prior compression and five-minute volatility release."),
    ("msv_f08_abnormal_range_pressure", "Five-bar signed candle bodies weighted by true-range expansion relative to the preceding median."),
    ("msv_f09_failed_breakout_reversal", "Abnormal true range amplifies a minute channel breach that closes back inside the channel."),
    ("msv_f10_gap_confirmation", "One-minute close-to-open gaps confirmed by the current minute's candle body. The shock baseline uses the median of nonzero prior gaps so ordinary zero-gap minutes do not erase it."),
];

struct VolatilityFactor {
    index: usize,
    variant: usize,
}

pub(super) fn create(index: usize) -> Box<dyn Factor> {
    create_variant(index, 1)
}

pub(super) fn create_variant(index: usize, variant: usize) -> Box<dyn Factor> {
    Box::new(VolatilityFactor { index, variant })
}

pub(super) fn factor_id(index: usize, variant: usize) -> String {
    let offset = match variant {
        0 => 142,
        1 => 120,
        2 => 164,
        _ => unreachable!("volatility factor variant"),
    };
    format!("yq_{:03}", offset + index)
}

fn fields(index: usize) -> &'static [FutureMinuteField] {
    match index {
        6 | 8 => HIGH_LOW_CLOSE,
        7 => OHLC,
        9 => GAP_INPUT,
        _ => CLOSE,
    }
}

impl Factor for VolatilityFactor {
    fn spec(&self) -> FactorSpec {
        let (alias, description) = FACTORS[self.index];
        let id = factor_id(self.index, self.variant);
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

fn window_description(index: usize, variant: usize) -> String {
    match index {
        0..=5 => format!(
            "The recent window is 5 minutes and the baseline is {} minutes.",
            EXPANSION_BASES[variant]
        ),
        6 => format!(
            "The channel is {} minutes, the baseline is {} minutes, and ATR uses {} minutes.",
            CHANNEL_WINDOWS[variant], COMPRESSION_BASES[variant], WINDOWS[variant]
        ),
        7 => format!("True range uses a {}-minute baseline.", TR_WEIGHTS[variant]),
        8 => format!(
            "The channel is {} minutes and the true-range baseline and ATR use {} minutes.",
            CHANNEL_WINDOWS[variant], WINDOWS[variant]
        ),
        9 => format!(
            "Reference statistics use {} prior minutes.",
            WINDOWS[variant]
        ),
        _ => unreachable!("volatility factor index"),
    }
}
pub(super) fn map(values: &Values, f: impl Fn(f64) -> f64) -> Series {
    values.iter().map(|value| finite(f((*value)?))).collect()
}

pub(super) fn zip(left: &Values, right: &Values, f: impl Fn(f64, f64) -> Option<f64>) -> Series {
    left.iter()
        .zip(right)
        .map(|(&a, &b)| finite(f(a?, b?)?))
        .collect()
}

fn finite(value: f64) -> Option<f64> {
    value.is_finite().then_some(value)
}

fn multiply(left: &Values, right: &Values) -> Series {
    zip(left, right, |a, b| Some(a * b))
}

pub(super) fn divide(n: f64, d: f64) -> Option<f64> {
    (d > 0.0).then(|| n / d)
}

pub(super) fn log_return(close: f64, previous: f64) -> Option<f64> {
    (close > 0.0 && previous > 0.0).then(|| close.ln() - previous.ln())
}

fn rv(returns: &Values, n: usize) -> Series {
    map(&ts_sum(&map(returns, |r| r * r), n, n), |sum| {
        (sum / n as f64).sqrt()
    })
}

fn log_expansion(current: f64, base: f64) -> Option<f64> {
    if base <= 0.0 || current < 0.0 {
        None
    } else if current == 0.0 {
        Some(0.0)
    } else {
        // Use kappa = 1 for all log-ratio weights.
        Some((current.ln() - base.ln()).max(0.0).tanh())
    }
}

pub(super) fn sample_std(values: &Values, n: usize) -> Series {
    map(&crate::operators::ts_std_dev(values, n, n), |std| {
        std * (n as f64 / (n - 1) as f64).sqrt()
    })
}

fn momentum(returns: &Values, n: usize) -> Series {
    let scale = ts_delay(&sample_std(returns, MOMENTUM_SCALE), 1);
    zip(&ts_sum(returns, n, n), &scale, |r, std| {
        divide(r, std * (n as f64).sqrt())
    })
}

fn imbalance(returns: &Values, n: usize) -> Series {
    let signed_square = map(returns, |r| r * r.abs());
    let squares = map(returns, |r| r * r);
    zip(
        &ts_sum(&signed_square, n, n),
        &ts_sum(&squares, n, n),
        divide,
    )
}

fn true_range(close: &Values, high: &Values, low: &Values) -> Series {
    (0..close.len())
        .map(|i| {
            let previous = close[i.checked_sub(1)?]?;
            let (h, l) = (high[i]?, low[i]?);
            (h >= l).then(|| (h - l).max((h - previous).abs()).max((l - previous).abs()))
        })
        .collect()
}

fn evaluate(index: usize, variant: usize, inputs: &[Series]) -> Series {
    let close = &inputs[0];
    if index == 9 {
        return gap_confirmation(close, &inputs[1], WINDOWS[variant]);
    }
    if index >= 6 {
        return range_factor(index, variant, inputs);
    }
    let returns = zip(close, &ts_delay(close, 1), log_return);
    let medium = EXPANSION_BASES[variant];
    if index == 5 {
        let log_rv = map(&rv(&returns, SHORT), f64::ln);
        let short = sample_std(&log_rv, SHORT);
        let medium = ts_delay(&sample_std(&log_rv, medium), SHORT);
        let shock = zip(&short, &medium, log_expansion);
        return multiply(&map(&momentum(&returns, SHORT), f64::tanh), &shock);
    }
    let expansion = zip(
        &rv(&returns, SHORT),
        &ts_delay(&rv(&returns, medium), SHORT),
        log_expansion,
    );
    let direction = match index {
        0 => map(&momentum(&returns, SHORT), f64::tanh),
        1 => zip(
            &ts_sum(&returns, SHORT, SHORT),
            &ts_sum(&map(&returns, f64::abs), SHORT, SHORT),
            divide,
        ),
        2 => zip(
            &momentum(&returns, SHORT),
            &ts_delay(&momentum(&returns, medium), SHORT),
            |short, medium| Some(short.tanh() * (short * medium).tanh().max(0.0)),
        ),
        3 => imbalance(&returns, SHORT),
        4 => zip(
            &imbalance(&returns, SHORT),
            &ts_delay(&imbalance(&returns, medium), SHORT),
            |short, medium| Some((short - medium) / 2.0),
        ),
        _ => unreachable!(),
    };
    multiply(&direction, &expansion)
}

fn range_factor(index: usize, variant: usize, inputs: &[Series]) -> Series {
    let (close, high, low) = (&inputs[0], &inputs[1], &inputs[2]);
    let tr = true_range(close, high, low);
    let channel = CHANNEL_WINDOWS[variant];
    let scale_window = WINDOWS[variant];
    let tr_window = if index == 7 {
        TR_WEIGHTS[variant]
    } else {
        scale_window
    };
    let unusual_range = || {
        zip(
            &tr,
            &ts_delay(&ts_median(&tr, tr_window, tr_window), 1),
            log_expansion,
        )
    };
    if index == 7 {
        let open = &inputs[3];
        let body = (0..close.len())
            .map(|i| {
                let (c, o, h, l) = (close[i]?, open[i]?, high[i]?, low[i]?);
                if h == l {
                    Some(0.0)
                } else {
                    divide(c - o, h - l)
                }
            })
            .collect::<Series>();
        let weight = unusual_range();
        return zip(
            &ts_sum(&multiply(&weight, &body), SHORT, SHORT),
            &ts_sum(&weight, SHORT, SHORT),
            |sum, weight| {
                if weight == 0.0 {
                    Some(0.0)
                } else {
                    Some(weight.tanh() * sum / weight)
                }
            },
        );
    }
    let upper = ts_delay(&ts_max(high, channel, channel), 1);
    let lower = ts_delay(&ts_min(low, channel, channel), 1);
    let scale = ts_delay(
        &rolling_exponential_mean(&tr, scale_window, 1.0 - 1.0 / scale_window as f64),
        1,
    );
    let breakout = |diff: f64, atr: f64| divide(diff.max(0.0), atr).map(f64::tanh);
    if index == 6 {
        let returns = zip(close, &ts_delay(close, 1), log_return);
        let baseline_window = COMPRESSION_BASES[variant];
        let compressed = ts_delay(&rv(&returns, channel), 1);
        let baseline = ts_delay(&rv(&returns, baseline_window), channel + 1);
        let compression = zip(&baseline, &compressed, log_expansion);
        let release = zip(&rv(&returns, SHORT), &compressed, log_expansion);
        return (0..close.len())
            .map(|i| {
                let (c, u, l, a) = (close[i]?, upper[i]?, lower[i]?, scale[i]?);
                finite(compression[i]? * release[i]? * (breakout(c - u, a)? - breakout(l - c, a)?))
            })
            .collect();
    }
    let weight = unusual_range();
    (0..close.len())
        .map(|i| {
            let (c, h, low, u, l, a) = (
                close[i]?, high[i]?, low[i]?, upper[i]?, lower[i]?, scale[i]?,
            );
            let inside = breakout(c - l, a)? * breakout(u - c, a)?;
            finite(weight[i]? * inside * (breakout(l - low, a)? - breakout(h - u, a)?))
        })
        .collect()
}

fn gap_confirmation(close: &Values, open: &Values, window: usize) -> Series {
    let previous_close = ts_delay(close, 1);
    let gaps = zip(open, &previous_close, log_return);
    let movement = zip(close, open, log_return);
    let returns = zip(close, &previous_close, log_return);
    let gap_scale = ts_delay(&sample_std(&gaps, window), 1);
    let movement_scale = ts_delay(&sample_std(&returns, window), 1);
    (0..close.len())
        .map(|i| {
            let (gap, movement) = (gaps[i]?, movement[i]?);
            let (gap_scale, movement_scale) = (gap_scale[i]?, movement_scale[i]?);
            if gap * movement <= 0.0 {
                return Some(0.0);
            }
            let typical_gap = nonzero_gap_median(&gaps[i - window..i])?;
            let shock = log_expansion(gap.abs(), typical_gap)?;
            let strength = divide(gap, gap_scale)?.tanh().abs();
            let confirmation = divide(movement, movement_scale)?.tanh().abs();
            Some(gap.signum() * shock * strength.min(confirmation))
        })
        .collect()
}

fn nonzero_gap_median(gaps: &Values) -> Option<f64> {
    let mut magnitudes = [0.0; 30];
    let mut len = 0;
    for gap in gaps {
        let magnitude = (*gap)?.abs();
        if magnitude > 0.0 {
            magnitudes[len] = magnitude;
            len += 1;
        }
    }
    if len == 0 {
        return None;
    }
    magnitudes[..len].sort_unstable_by(f64::total_cmp);
    Some((magnitudes[(len - 1) / 2] + magnitudes[len / 2]) / 2.0)
}

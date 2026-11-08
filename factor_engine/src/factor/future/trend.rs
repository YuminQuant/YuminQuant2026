use crate::core::{AssetClass, FactorContext, FactorSeries, FactorSpec, Frequency, Lookback};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::future::minute_input::{
    compute_future_minute, compute_future_minute_many, future_minute_request, FutureMinuteField,
    FutureMinuteField::{ContractDiff, Raw},
};
use crate::factor::future::rolling::{atr, rolling_ema, with_atr};
use crate::factor::Factor;
use crate::operators::{ts_delay, ts_max, ts_mean, ts_min, ts_pctchg, ts_std_dev, ts_sum};

#[derive(Clone, Copy)]
enum BandSignal {
    Sigmoid,
    Proximity,
    Breakout,
    UpperSlope,
    LowerSlope,
}

#[derive(Clone, Copy)]
enum Signal {
    KamaSpread(usize),
    SignedEfficiency(usize),
    Band(BandSignal, usize),
    PriorRange(usize),
    VolumeOiEfficiency(usize, usize),
    MacdMomentum,
}

use BandSignal::*;
use Signal::*;

const FIRST_ID: usize = 26;
const SIGNALS: [Signal; 26] = [
    KamaSpread(15),
    KamaSpread(30),
    SignedEfficiency(5),
    SignedEfficiency(15),
    SignedEfficiency(30),
    Band(Sigmoid, 5),
    Band(Sigmoid, 15),
    Band(Sigmoid, 30),
    Band(Proximity, 5),
    Band(Proximity, 15),
    Band(Proximity, 30),
    Band(Breakout, 5),
    Band(Breakout, 15),
    Band(Breakout, 30),
    Band(UpperSlope, 5),
    Band(UpperSlope, 15),
    Band(UpperSlope, 30),
    Band(LowerSlope, 5),
    Band(LowerSlope, 15),
    Band(LowerSlope, 30),
    PriorRange(5),
    PriorRange(15),
    PriorRange(30),
    VolumeOiEfficiency(5, 15),
    VolumeOiEfficiency(15, 30),
    MacdMomentum,
];
const OHLC: &[FutureMinuteField] = &[Raw("close"), Raw("high"), Raw("low")];
const CLOSE: &[FutureMinuteField] = &[Raw("close")];
const CLOSE_VOLUME_OI_CHANGE: &[FutureMinuteField] =
    &[Raw("close"), Raw("vol"), ContractDiff("oi")];
const MACD_FAST: usize = 15;
const MACD_SLOW: usize = 30;
const MACD_SIGNAL: usize = 5;

impl Signal {
    fn inputs(self) -> (&'static [FutureMinuteField], usize) {
        match self {
            KamaSpread(n) | Band(UpperSlope | LowerSlope, n) => (OHLC, 2 * n - 1),
            Band(_, n) | PriorRange(n) => (OHLC, n),
            SignedEfficiency(n) => (CLOSE, n),
            VolumeOiEfficiency(_, slow) => (CLOSE_VOLUME_OI_CHANGE, slow),
            MacdMomentum => (OHLC, MACD_SLOW + MACD_SIGNAL - 2),
        }
    }

    fn description(self) -> String {
        match self {
            KamaSpread(n) => format!(
                "{n}-bar finite-window KAMA minus finite-window EMA, divided by {n}-bar ATR."
            ),
            SignedEfficiency(n) => {
                format!("Sign of {n}-bar return times Kaufman efficiency over {n} price changes.")
            }
            Band(kind, n) => {
                let formula = match kind {
                    Sigmoid => "sigmoid((U-C)/ATR) - sigmoid((C-L)/ATR)",
                    Proximity => "exp(-abs(C-U)/ATR) - exp(-abs(C-L)/ATR)",
                    Breakout => "tanh(max(C-U,0)/ATR) - tanh(max(L-C,0)/ATR)",
                    UpperSlope => "exp(-abs(C-U)/ATR) * tanh((U-U_lag)/ATR)",
                    LowerSlope => "exp(-abs(C-L)/ATR) * tanh((L-L_lag)/ATR)",
                };
                format!("{formula}; {n}-bar Bollinger bands (1.5 population standard deviations), ATR and band lag.")
            }
            PriorRange(n) => format!(
                "Close position in the previous {n}-bar high-low range, excluding the current bar."
            ),
            VolumeOiEfficiency(fast, slow) => format!(
                "{slow}-bar Kaufman efficiency times {fast}-bar price direction, weighted by the sign of the {fast}/{slow}-bar volume ratio and direction-adjusted {slow}-bar open interest change."
            ),
            MacdMomentum => format!(
                "M * (1 + 0.5 * M * A), where M and A are tanh of MACD ({MACD_FAST}, {MACD_SLOW}, {MACD_SIGNAL}) DIF and histogram divided by {MACD_SLOW}-bar ATR."
            ),
        }
    }
}

struct TrendFactor(usize);

pub(super) fn create(index: usize) -> Box<dyn Factor> {
    Box::new(TrendFactor(index))
}

fn id(index: usize) -> String {
    format!("yq_{:03}", FIRST_ID + index)
}

fn spec(index: usize) -> FactorSpec {
    let signal = SIGNALS[index];
    FactorSpec {
        id: id(index),
        aliases: Vec::new(),
        name: id(index),
        asset_class: AssetClass::Future,
        frequency: Frequency::Minute1,
        version: "0.1.0".into(),
        tags: vec![
            "minute".into(),
            "price_volume".into(),
            "pv".into(),
            "yq".into(),
        ],
        description: signal.description(),
        dependencies: vec![future_minute_request(signal.inputs().0)],
        intraday_raw_dependencies: Vec::new(),
        lookback: Lookback { trading_days: 3 },
    }
}

impl Factor for TrendFactor {
    fn spec(&self) -> FactorSpec {
        spec(self.0)
    }

    fn compute_provider_key(&self) -> String {
        match SIGNALS[self.0] {
            Band(_, window) => format!("future-minute-yq-bands-{window}"),
            _ => self.spec().registry_key(),
        }
    }

    fn compute(&self, context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let signal = SIGNALS[self.0];
        let (fields, history) = signal.inputs();
        compute_future_minute(self.spec(), context, data, fields, history, |inputs| {
            let close = &inputs[0];
            match signal {
                KamaSpread(n) => kama_spread(close, &inputs[1], &inputs[2], n),
                SignedEfficiency(n) => signed_efficiency(close, n),
                Band(kind, n) => BandInputs::new(close, &inputs[1], &inputs[2], n).evaluate(kind),
                PriorRange(n) => prior_range(close, &inputs[1], &inputs[2], n),
                VolumeOiEfficiency(fast, slow) => {
                    volume_oi_efficiency(close, &inputs[1], &inputs[2], fast, slow)
                }
                MacdMomentum => macd_momentum(close, &inputs[1], &inputs[2]),
            }
        })
    }

    fn compute_many(
        &self,
        requested_ids: &[String],
        context: &FactorContext,
        data: &DataPool,
    ) -> Result<Vec<FactorSeries>> {
        let Band(_, window) = SIGNALS[self.0] else {
            return if requested_ids.contains(&id(self.0)) {
                Ok(vec![self.compute(context, data)?])
            } else {
                Ok(Vec::new())
            };
        };
        let outputs: Vec<_> = SIGNALS
            .iter()
            .enumerate()
            .filter_map(|(index, &signal)| match signal {
                Band(kind, n) if n == window && requested_ids.contains(&id(index)) => {
                    Some((index, kind))
                }
                _ => None,
            })
            .collect();
        let Some(history) = outputs.iter().map(|&(i, _)| SIGNALS[i].inputs().1).max() else {
            return Ok(Vec::new());
        };
        let specs: Vec<_> = outputs.iter().map(|&(i, _)| spec(i)).collect();
        compute_future_minute_many(&specs, context, data, OHLC, history, |inputs| {
            let bands = BandInputs::new(&inputs[0], &inputs[1], &inputs[2], window);
            outputs
                .iter()
                .map(|&(_, kind)| bands.evaluate(kind))
                .collect()
        })
    }
}

pub(super) fn efficiency_ratio(close: &[Option<f64>], n: usize) -> Vec<Option<f64>> {
    let mut output = vec![None; close.len()];
    if let Some(tail) = output.get_mut(n..) {
        for (slot, sample) in tail.iter_mut().zip(close.windows(n + 1)) {
            *slot = efficiency_window(sample);
        }
    }
    output
}

fn efficiency_window(sample: &[Option<f64>]) -> Option<f64> {
    let change = (sample[sample.len() - 1]? - sample[0]?).abs();
    let path = sample
        .windows(2)
        .try_fold(0.0, |sum, pair| Some(sum + (pair[1]? - pair[0]?).abs()))?;
    let value = if path == 0.0 { 0.0 } else { change / path };
    value.is_finite().then(|| value.min(1.0))
}

fn signed_efficiency(close: &[Option<f64>], n: usize) -> Vec<Option<f64>> {
    ts_pctchg(close, n)
        .into_iter()
        .zip(efficiency_ratio(close, n))
        .map(|(momentum, ratio)| Some(momentum?.signum() * ratio?))
        .collect()
}

fn volume_oi_efficiency(
    close: &[Option<f64>],
    volume: &[Option<f64>],
    oi_change: &[Option<f64>],
    fast: usize,
    slow: usize,
) -> Vec<Option<f64>> {
    let prior_close = ts_delay(close, fast);
    let fast_volume = ts_sum(volume, fast, fast);
    let slow_volume = ts_sum(volume, slow, slow);
    let oi_change = ts_sum(oi_change, slow, slow);
    let efficiency = efficiency_ratio(close, slow);

    (0..close.len())
        .map(|i| {
            let direction = sign(close[i]? - prior_close[i]?);
            let (fast_volume, slow_volume) = (fast_volume[i]?, slow_volume[i]?);
            let volume_sign = if slow_volume == 0.0 {
                0.0
            } else {
                sign(fast_volume / slow_volume)
            };
            let oi_sign = sign(direction * oi_change[i]?);
            Some(efficiency[i]? * direction * (1.0 + volume_sign + oi_sign))
        })
        .collect()
}

fn sign(value: f64) -> f64 {
    if value == 0.0 {
        0.0
    } else {
        value.signum()
    }
}

fn rolling_kama(close: &[Option<f64>], n: usize) -> Vec<Option<f64>> {
    let mut output = vec![None; close.len()];
    let er = efficiency_ratio(close, n);
    if let Some(tail) = output.get_mut(2 * n - 1..) {
        for (start, (slot, sample)) in tail.iter_mut().zip(close.windows(2 * n)).enumerate() {
            // Reset to the mean of the first n closes, then apply n adaptive updates.
            let seed = sample[..n]
                .iter()
                .try_fold(0.0, |sum, value| Some(sum + (*value)?));
            let fast = 2.0 / 3.0;
            let slow = 2.0 / 31.0;
            *slot = seed.and_then(|seed| {
                sample[n..]
                    .iter()
                    .zip(&er[start + n..start + 2 * n])
                    .try_fold(seed / n as f64, |previous, (&value, &er)| {
                        let smoothing = (slow + er? * (fast - slow)).powi(2);
                        Some(previous + smoothing * (value? - previous))
                    })
                    .filter(|value| value.is_finite())
            });
        }
    }
    output
}

fn kama_spread(
    close: &[Option<f64>],
    high: &[Option<f64>],
    low: &[Option<f64>],
    n: usize,
) -> Vec<Option<f64>> {
    rolling_kama(close, n)
        .into_iter()
        .zip(rolling_ema(close, n))
        .zip(atr(close, high, low, n))
        .map(|((kama, ema), range)| with_atr(range, |range| Some((kama? - ema?) / range)))
        .collect()
}

fn macd_momentum(
    close: &[Option<f64>],
    high: &[Option<f64>],
    low: &[Option<f64>],
) -> Vec<Option<f64>> {
    let dif: Vec<_> = rolling_ema(close, MACD_FAST)
        .into_iter()
        .zip(rolling_ema(close, MACD_SLOW))
        .map(|(fast, slow)| Some(fast? - slow?))
        .collect();
    let dea = rolling_ema(&dif, MACD_SIGNAL);

    dif.into_iter()
        .zip(dea)
        .zip(atr(close, high, low, MACD_SLOW))
        .map(|((dif, dea), range)| {
            with_atr(range, |range| {
                let dif = dif?;
                let momentum = (dif / range).tanh();
                let acceleration = ((dif - dea?) / range).tanh();
                Some(momentum * (1.0 + 0.5 * momentum * acceleration))
            })
        })
        .collect()
}

#[derive(Clone, Copy)]
struct Bands {
    upper: f64,
    lower: f64,
}

struct BandInputs<'a> {
    close: &'a [Option<f64>],
    bands: Vec<Option<Bands>>,
    atr: Vec<Option<f64>>,
    lag: usize,
}

impl<'a> BandInputs<'a> {
    fn new(close: &'a [Option<f64>], high: &[Option<f64>], low: &[Option<f64>], n: usize) -> Self {
        let bands = ts_mean(close, n, n)
            .into_iter()
            .zip(ts_std_dev(close, n, n))
            .map(|(mean, std)| {
                let (mean, width) = (mean?, 1.5 * std?);
                Some(Bands {
                    upper: mean + width,
                    lower: mean - width,
                })
            })
            .collect();
        Self {
            close,
            bands,
            atr: atr(close, high, low, n),
            lag: n,
        }
    }

    fn evaluate(&self, kind: BandSignal) -> Vec<Option<f64>> {
        (0..self.close.len())
            .map(|i| {
                with_atr(self.atr[i], |range| {
                    let Bands { upper, lower } = self.bands[i]?;
                    let close = self.close[i]?;
                    let upper_gap = (upper - close) / range;
                    let lower_gap = (close - lower) / range;
                    let previous = || i.checked_sub(self.lag).and_then(|i| self.bands[i]);
                    let sigmoid = |x: f64| 1.0 / (1.0 + (-x).exp());
                    Some(match kind {
                        Sigmoid => sigmoid(upper_gap) - sigmoid(lower_gap),
                        Proximity => (-upper_gap.abs()).exp() - (-lower_gap.abs()).exp(),
                        Breakout => (-upper_gap).max(0.0).tanh() - (-lower_gap).max(0.0).tanh(),
                        UpperSlope => {
                            (-upper_gap.abs()).exp() * ((upper - previous()?.upper) / range).tanh()
                        }
                        LowerSlope => {
                            (-lower_gap.abs()).exp() * ((lower - previous()?.lower) / range).tanh()
                        }
                    })
                })
            })
            .collect()
    }
}

fn prior_range(
    close: &[Option<f64>],
    high: &[Option<f64>],
    low: &[Option<f64>],
    n: usize,
) -> Vec<Option<f64>> {
    let high = ts_delay(&ts_max(high, n, n), 1);
    let low = ts_delay(&ts_min(low, n, n), 1);
    close
        .iter()
        .zip(high)
        .zip(low)
        .map(|((&close, high), low)| {
            let (close, low) = (close?, low?);
            let range = high? - low;
            (range != 0.0)
                .then(|| (close - low) / range)
                .filter(|value| value.is_finite())
        })
        .collect()
}

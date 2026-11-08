use crate::core::{AssetClass, FactorContext, FactorSeries, FactorSpec, Frequency, Lookback};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::future::minute_input::{
    compute_future_minute, future_minute_request, FutureMinuteField, FutureMinuteField::Raw,
};
use crate::factor::future::rolling::{atr, rolling_ema, rolling_exponential_mean, with_atr};
use crate::factor::future::trend::efficiency_ratio;
use crate::factor::Factor;
use crate::operators::{ts_diff, ts_mean, ts_std_dev};

const FIRST_ID: usize = 52;
const PERIOD: usize = 15;
const FAST: usize = 5;
const SLOW: usize = 30;
const LAG: usize = 5;
const WINDOW: usize = 30;
const EMA: usize = 5;
const TAU: f64 = 8.0;
const EPSILON: f64 = 1e-8;
const ATR: usize = 15;

#[derive(Clone, Copy)]
enum Signal {
    Level,
    Slope,
    Acceleration,
    Spread,
    Residual,
    Shock,
    ZScore,
    Persistence,
    Efficiency,
    Extreme,
    Boundary,
    Recovery,
    Agreement,
    Pullback,
    Divergence,
    PriceEfficiency,
}

use Signal::*;

const SIGNALS: [(Signal, &str); 16] = [
    (Level, "Centered 15-bar RSI scaled to [-1, 1]."),
    (Slope, "Five-bar change in 15-bar RSI, divided by 50."),
    (Acceleration, "Second difference of 15-bar RSI at five-bar intervals, divided by 50."),
    (Spread, "Five-bar RSI minus 30-bar RSI, divided by 50."),
    (Residual, "15-bar RSI minus its five-bar finite-window EMA, divided by 50."),
    (Shock, "Latest one-bar RSI change divided by its 30-bar rolling standard deviation plus epsilon."),
    (ZScore, "Deviation of RSI from its 30-bar rolling mean, scaled by its rolling standard deviation plus epsilon."),
    (Persistence, "Thirty-bar average of centered 15-bar RSI."),
    (Efficiency, "Signed 30-bar RSI change divided by the cumulative absolute change over the same window."),
    (Extreme, "Signed RSI excursion beyond 30 or 70, saturated with tanh at a scale of eight RSI points."),
    (Boundary, "Exponential proximity to RSI 70 minus proximity to RSI 30, at a scale of eight RSI points."),
    (Recovery, "Five-bar RSI recovery weighted by prior oversold or overbought proximity, at a scale of eight RSI points."),
    (Agreement, "Mean centered five- and 30-bar RSI discounted by their disagreement."),
    (Pullback, "Signed long-horizon RSI strength weighted by opposing short-horizon RSI strength."),
    (Divergence, "Opposing five-bar changes in close and RSI, scaled by 15-bar ATR and eight RSI points respectively."),
    (PriceEfficiency, "Centered 15-bar RSI multiplied by unsigned 30-bar Kaufman price efficiency."),
];

impl Signal {
    fn fields(self) -> &'static [FutureMinuteField] {
        match self {
            Divergence => &[Raw("close"), Raw("high"), Raw("low")],
            _ => &[Raw("close")],
        }
    }

    fn history(self) -> usize {
        match self {
            Level | Extreme | Boundary => PERIOD,
            Slope | Recovery => PERIOD + LAG,
            Divergence => (PERIOD + LAG).max(ATR),
            Acceleration => PERIOD + 2 * LAG,
            Spread | Agreement | Pullback => SLOW,
            Residual => PERIOD + EMA - 1,
            Shock | Efficiency => PERIOD + WINDOW,
            ZScore | Persistence => PERIOD + WINDOW - 1,
            PriceEfficiency => PERIOD.max(WINDOW),
        }
    }
}

struct RsiFactor(usize);

pub(super) fn create(index: usize) -> Box<dyn Factor> {
    Box::new(RsiFactor(index))
}

impl Factor for RsiFactor {
    fn spec(&self) -> FactorSpec {
        let (signal, description) = SIGNALS[self.0];
        let id = format!("yq_{:03}", FIRST_ID + self.0);
        FactorSpec {
            name: id.clone(),
            id,
            aliases: Vec::new(),
            asset_class: AssetClass::Future,
            frequency: Frequency::Minute1,
            version: "0.1.0".into(),
            tags: ["minute", "price_volume", "pv", "yq", "rsi"]
                .map(str::to_string)
                .to_vec(),
            description: format!("{description} RSI uses finite-window Wilder weights."),
            dependencies: vec![future_minute_request(signal.fields())],
            intraday_raw_dependencies: Vec::new(),
            lookback: Lookback { trading_days: 3 },
        }
    }

    fn compute(&self, context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let signal = SIGNALS[self.0].0;
        compute_future_minute(
            self.spec(),
            context,
            data,
            signal.fields(),
            signal.history(),
            |inputs| evaluate(signal, inputs),
        )
    }
}

fn rolling_rsi(close: &[Option<f64>], period: usize) -> Vec<Option<f64>> {
    let changes = ts_diff(close, 1);
    let gains = changes
        .iter()
        .map(|change| change.map(positive))
        .collect::<Vec<_>>();
    let losses = changes
        .iter()
        .map(|change| change.map(|value| positive(-value)))
        .collect::<Vec<_>>();
    let decay = 1.0 - 1.0 / period as f64;
    let up = rolling_exponential_mean(&gains, period, decay);
    let down = rolling_exponential_mean(&losses, period, decay);
    combine(&up, &down, |up, down| {
        if up + down == 0.0 {
            50.0
        } else {
            100.0 * (up / (up + down))
        }
    })
}

fn positive(value: f64) -> f64 {
    value.max(0.0)
}

fn centered(value: f64) -> f64 {
    (value - 50.0) / 50.0
}

fn sigmoid(value: f64) -> f64 {
    1.0 / (1.0 + (-value).exp())
}

fn combine(
    left: &[Option<f64>],
    right: &[Option<f64>],
    operation: impl Fn(f64, f64) -> f64,
) -> Vec<Option<f64>> {
    left.iter()
        .zip(right)
        .map(|(&a, &b)| Some(operation(a?, b?)).filter(|value| value.is_finite()))
        .collect()
}

fn map(values: &[Option<f64>], operation: impl Fn(f64) -> f64) -> Vec<Option<f64>> {
    values.iter().map(|value| value.map(&operation)).collect()
}

fn evaluate(signal: Signal, inputs: &[Vec<Option<f64>>]) -> Vec<Option<f64>> {
    let close = &inputs[0];
    if matches!(signal, Spread | Agreement | Pullback) {
        return combine(
            &rolling_rsi(close, FAST),
            &rolling_rsi(close, SLOW),
            |fast, slow| match signal {
                Spread => (fast - slow) / 50.0,
                Agreement => {
                    let (fast, slow) = (centered(fast), centered(slow));
                    (fast + slow) / 2.0 * (1.0 - (fast - slow).abs() / 2.0)
                }
                Pullback => {
                    let long = ((slow - 50.0) / 15.0).tanh();
                    let short = ((fast - 50.0) / 15.0).tanh();
                    long * positive(-long * short)
                }
                _ => unreachable!(),
            },
        );
    }
    let rsi = rolling_rsi(close, PERIOD);
    match signal {
        Level => map(&rsi, centered),
        Slope => map(&ts_diff(&rsi, LAG), |change| change / 50.0),
        Acceleration => map(&ts_diff(&ts_diff(&rsi, LAG), LAG), |change| change / 50.0),
        Residual => combine(&rsi, &rolling_ema(&rsi, EMA), |r, ema| (r - ema) / 50.0),
        Shock => {
            let change = ts_diff(&rsi, 1);
            combine(&change, &ts_std_dev(&change, WINDOW, WINDOW), |d, sd| {
                d / (sd + EPSILON)
            })
        }
        ZScore => {
            let deviation = combine(&rsi, &ts_mean(&rsi, WINDOW, WINDOW), |r, mean| r - mean);
            combine(&deviation, &ts_std_dev(&rsi, WINDOW, WINDOW), |d, sd| {
                d / (sd + EPSILON)
            })
        }
        Persistence => ts_mean(&map(&rsi, centered), WINDOW, WINDOW),
        Efficiency => combine(
            &ts_diff(&rsi, WINDOW),
            &efficiency_ratio(&rsi, WINDOW),
            |d, er| d.signum() * er,
        ),
        Extreme => map(&rsi, |r| {
            ((positive(r - 70.0) - positive(30.0 - r)) / TAU).tanh()
        }),
        Boundary => map(&rsi, |r| {
            (-(r - 70.0).abs() / TAU).exp() - (-(r - 30.0).abs() / TAU).exp()
        }),
        Recovery => rsi
            .iter()
            .enumerate()
            .map(|(i, &current)| {
                let previous = rsi[i.checked_sub(LAG)?]?;
                let recovery = ((current? - previous) / TAU).tanh();
                Some(
                    sigmoid((30.0 - previous) / TAU) * positive(recovery)
                        - sigmoid((previous - 70.0) / TAU) * positive(-recovery),
                )
            })
            .collect(),
        Divergence => {
            let price_change = ts_diff(close, LAG);
            let price = price_change
                .iter()
                .zip(atr(close, &inputs[1], &inputs[2], ATR))
                .map(|(&change, atr)| {
                    let change = change?;
                    with_atr(atr, |atr| Some((change / (LAG as f64 * atr)).tanh()))
                })
                .collect::<Vec<_>>();
            combine(&price, &ts_diff(&rsi, LAG), |p, change| {
                let q = (change / TAU).tanh();
                positive(-p) * positive(q) - positive(p) * positive(-q)
            })
        }
        PriceEfficiency => combine(&rsi, &efficiency_ratio(close, WINDOW), |r, er| {
            centered(r) * er
        }),
        Spread | Agreement | Pullback => unreachable!(),
    }
}

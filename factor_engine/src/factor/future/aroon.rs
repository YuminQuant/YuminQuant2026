use crate::core::{AssetClass, FactorContext, FactorSeries, FactorSpec, Frequency, Lookback};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::future::minute_input::{
    compute_future_minute, future_minute_request, FutureMinuteField, FutureMinuteField::Raw,
};
use crate::factor::future::rolling::{atr, rolling_ema, with_atr};
use crate::factor::future::trend::efficiency_ratio;
use crate::factor::Factor;
use crate::operators::{ts_delay, ts_diff, ts_max, ts_mean, ts_min};

const FIRST_ID: usize = 68;
const PERIOD: usize = 15;
const FAST: usize = 5;
const SLOW: usize = 30;
const EMA: usize = 5;
const LAG: usize = 5;
const WINDOW: usize = 30;
const ATR: usize = 15;

#[derive(Clone, Copy)]
enum Signal {
    Oscillator,
    Clarity,
    Slope,
    Persistence,
    Efficiency,
    Renewal,
    Breakout,
    Spread,
    Agreement,
    PriceEfficiency,
    Location,
    Pullback,
}

use Signal::*;

const SIGNALS: [(Signal, &str); 12] = [
    (Oscillator, "Normalized 15-period Aroon oscillator based on the recency of the latest high and low."),
    (Clarity, "Normalized Aroon oscillator discounted when both the high and the low are recent."),
    (Slope, "Five-bar slope of the five-bar finite-window EMA of the Aroon oscillator."),
    (Persistence, "Thirty-bar mean of the normalized Aroon oscillator."),
    (Efficiency, "Signed 30-bar net change in the Aroon oscillator divided by its cumulative absolute change."),
    (Renewal, "Thirty-bar mean of strict new-high events minus strict new-low events against the preceding 15 bars."),
    (Breakout, "Tanh of net high and low breakout magnitude against the preceding 15 bars, scaled by 15-bar ATR."),
    (Spread, "Half the difference between the normalized five- and 30-period Aroon oscillators."),
    (Agreement, "Thirty-period Aroon oscillator weighted by directional agreement with the five-period oscillator."),
    (PriceEfficiency, "Normalized Aroon oscillator multiplied by unsigned 15-bar Kaufman price efficiency."),
    (Location, "Normalized Aroon oscillator weighted by agreement with the close's location in its high-low range."),
    (Pullback, "Normalized Aroon oscillator weighted by the strength of an opposing close location in its high-low range."),
];

impl Signal {
    fn fields(self) -> &'static [FutureMinuteField] {
        match self {
            Breakout | PriceEfficiency | Location | Pullback => {
                &[Raw("high"), Raw("low"), Raw("close")]
            }
            _ => &[Raw("high"), Raw("low")],
        }
    }

    fn history(self) -> usize {
        match self {
            Slope => PERIOD + EMA - 1 + LAG,
            Persistence | Renewal => PERIOD + WINDOW - 1,
            Efficiency => PERIOD + WINDOW,
            Breakout => PERIOD.max(ATR),
            Spread | Agreement => SLOW,
            _ => PERIOD,
        }
    }
}

struct AroonFactor(usize);

pub(super) fn create(index: usize) -> Box<dyn Factor> {
    Box::new(AroonFactor(index))
}

impl Factor for AroonFactor {
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
            tags: ["minute", "price_volume", "pv", "yq", "aroon"]
                .map(str::to_string)
                .to_vec(),
            description: description.into(),
            dependencies: vec![future_minute_request(signal.fields())],
            lookback: Lookback { trading_days: 3 },
            intraday_raw_dependencies: Vec::new(),
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

struct Extremes {
    up: f64,
    down: f64,
    high: f64,
    low: f64,
}

impl Extremes {
    fn oscillator(&self) -> f64 {
        self.up - self.down
    }

    fn location(&self, close: f64) -> f64 {
        let range = self.high - self.low;
        if range == 0.0 {
            0.0
        } else {
            2.0 * (close - self.low) / range - 1.0
        }
    }
}

fn extremes(high: &[Option<f64>], low: &[Option<f64>], period: usize) -> Option<Extremes> {
    let (mut high_value, mut low_value) = (high[0]?, low[0]?);
    let (mut high_index, mut low_index) = (0, 0);
    // Equal extremes use the latest bar. Indices run from t-N (0) to t (N).
    for (index, (&h, &l)) in high.iter().zip(low).enumerate().skip(1) {
        let (h, l) = (h?, l?);
        if h >= high_value {
            (high_value, high_index) = (h, index);
        }
        if l <= low_value {
            (low_value, low_index) = (l, index);
        }
    }
    Some(Extremes {
        up: high_index as f64 / period as f64,
        down: low_index as f64 / period as f64,
        high: high_value,
        low: low_value,
    })
}

fn rolling_aroon(
    high: &[Option<f64>],
    low: &[Option<f64>],
    period: usize,
    operation: impl Fn(usize, Extremes) -> Option<f64>,
) -> Vec<Option<f64>> {
    let mut output = vec![None; high.len()];
    for (offset, (h, l)) in high
        .windows(period + 1)
        .zip(low.windows(period + 1))
        .enumerate()
    {
        let index = offset + period;
        output[index] = extremes(h, l, period).and_then(|state| operation(index, state));
    }
    output
}

fn oscillator(high: &[Option<f64>], low: &[Option<f64>], period: usize) -> Vec<Option<f64>> {
    rolling_aroon(high, low, period, |_, state| Some(state.oscillator()))
}

fn previous_breaks(
    high: &[Option<f64>],
    low: &[Option<f64>],
) -> (Vec<Option<f64>>, Vec<Option<f64>>) {
    let previous_high = ts_delay(&ts_max(high, PERIOD, PERIOD), 1);
    let previous_low = ts_delay(&ts_min(low, PERIOD, PERIOD), 1);
    let up = high
        .iter()
        .zip(previous_high)
        .map(|(&h, prior)| Some((h? - prior?).max(0.0)))
        .collect();
    let down = low
        .iter()
        .zip(previous_low)
        .map(|(&l, prior)| Some((prior? - l?).max(0.0)))
        .collect();
    (up, down)
}

fn combine(
    left: &[Option<f64>],
    right: &[Option<f64>],
    operation: impl Fn(f64, f64) -> f64,
) -> Vec<Option<f64>> {
    left.iter()
        .zip(right)
        .map(|(&a, &b)| Some(operation(a?, b?)))
        .collect()
}

fn evaluate(signal: Signal, inputs: &[Vec<Option<f64>>]) -> Vec<Option<f64>> {
    let (high, low) = (&inputs[0], &inputs[1]);
    match signal {
        Oscillator => oscillator(high, low, PERIOD),
        Clarity => rolling_aroon(high, low, PERIOD, |_, state| {
            Some(state.oscillator() * (1.0 - state.up.min(state.down)))
        }),
        Slope => ts_diff(&rolling_ema(&oscillator(high, low, PERIOD), EMA), LAG)
            .into_iter()
            .map(|value| value.map(|d| d / LAG as f64))
            .collect(),
        Persistence => ts_mean(&oscillator(high, low, PERIOD), WINDOW, WINDOW),
        Efficiency => {
            let osc = oscillator(high, low, PERIOD);
            combine(
                &ts_diff(&osc, WINDOW),
                &efficiency_ratio(&osc, WINDOW),
                |d, er| d.signum() * er,
            )
        }
        Renewal => {
            let (up, down) = previous_breaks(high, low);
            let events = combine(&up, &down, |u, d| {
                f64::from(u8::from(u > 0.0)) - f64::from(u8::from(d > 0.0))
            });
            ts_mean(&events, WINDOW, WINDOW)
        }
        Breakout => {
            let (up, down) = previous_breaks(high, low);
            let ranges = atr(&inputs[2], high, low, ATR);
            up.into_iter()
                .zip(down)
                .zip(ranges)
                .map(|((u, d), atr)| {
                    let change = u? - d?;
                    with_atr(atr, |atr| Some((change / atr).tanh()))
                })
                .collect()
        }
        Spread | Agreement => combine(
            &oscillator(high, low, FAST),
            &oscillator(high, low, SLOW),
            |fast, slow| match signal {
                Spread => (fast - slow) / 2.0,
                _ => slow * (1.0 + slow * fast) / 2.0,
            },
        ),
        PriceEfficiency => combine(
            &oscillator(high, low, PERIOD),
            &efficiency_ratio(&inputs[2], PERIOD),
            |o, er| o * er,
        ),
        Location | Pullback => rolling_aroon(high, low, PERIOD, |index, state| {
            let o = state.oscillator();
            let q = state.location(inputs[2][index]?);
            Some(match signal {
                Location => o * (1.0 + o * q) / 2.0,
                _ => o * (-o * q).max(0.0),
            })
        }),
    }
}

use crate::core::{
    AssetClass, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::future::minute_input::{
    compute_future_minute, future_minute_request, FutureMinuteField,
    FutureMinuteField::{ContractDiff, Raw},
};
use crate::factor::future::rolling::{rolling_ema, rolling_exponential_mean};
use crate::factor::Factor;
use crate::operators::{ts_mean, ts_pctchg, ts_sum, ts_zscore};

#[derive(Clone, Copy)]
enum Average {
    Ema,
    Trima,
    Dema,
}

#[derive(Clone, Copy)]
enum Signal {
    MaRatio(usize, usize),
    CloseMa(usize),
    Return(usize),
    AtrSpread(Average, usize, usize),
    EmaZscore(usize),
    VolumeEmaRatio(usize, usize),
    InterestEmaRatio(usize, usize),
    VolumeInterestMean(usize),
    VolumeNetInterest(usize),
}

use Average::{Dema, Ema, Trima};
use Signal::*;

const SIGNALS: [(Signal, &str); 25] = [
    (MaRatio(5, 15), "Symmetric relative spread between the 5- and 15-bar simple moving averages of close."),
    (CloseMa(5), "Symmetric relative spread between the current close and its 5-bar simple moving average."),
    (MaRatio(15, 30), "Symmetric relative spread between the 15- and 30-bar simple moving averages of close."),
    (Return(5), "Five-bar close-to-close return."),
    (Return(15), "Fifteen-bar close-to-close return."),
    (Return(30), "Thirty-bar close-to-close return."),
    (AtrSpread(Ema, 5, 15), "Spread between 5- and 15-bar finite-window close EMAs, divided by 15-bar finite-window Wilder-weighted ATR."),
    (AtrSpread(Ema, 15, 30), "Spread between 15- and 30-bar finite-window close EMAs, divided by 30-bar finite-window Wilder-weighted ATR."),
    (EmaZscore(5), "Thirty-bar rolling z-score of the 5-bar finite-window close EMA."),
    (EmaZscore(15), "Thirty-bar rolling z-score of the 15-bar finite-window close EMA."),
    (EmaZscore(30), "Thirty-bar rolling z-score of the 30-bar finite-window close EMA."),
    (AtrSpread(Trima, 5, 15), "Spread between 5- and 15-bar triangular moving averages of close, divided by 15-bar finite-window Wilder-weighted ATR."),
    (AtrSpread(Trima, 15, 30), "Spread between 15- and 30-bar triangular moving averages of close, divided by 30-bar finite-window Wilder-weighted ATR."),
    (AtrSpread(Dema, 5, 15), "Spread between 5- and 15-bar finite-window double exponential moving averages of close, divided by 15-bar finite-window Wilder-weighted ATR."),
    (AtrSpread(Dema, 15, 30), "Spread between 15- and 30-bar finite-window double exponential moving averages of close, divided by 30-bar finite-window Wilder-weighted ATR."),
    (VolumeEmaRatio(5, 15), "Symmetric relative spread between the 5- and 15-bar finite-window EMAs of per-minute trading volume."),
    (VolumeEmaRatio(15, 30), "Symmetric relative spread between the 15- and 30-bar finite-window EMAs of per-minute trading volume."),
    (InterestEmaRatio(5, 15), "Signed symmetric relative spread between 5- and 15-bar finite-window EMAs of the change in open interest. Changes use the same contract across trading days."),
    (InterestEmaRatio(15, 30), "Signed symmetric relative spread between 15- and 30-bar finite-window EMAs of the change in open interest. Changes use the same contract across trading days."),
    (VolumeInterestMean(5), "Five-bar mean of per-minute trading volume divided by open interest."),
    (VolumeInterestMean(15), "Fifteen-bar mean of per-minute trading volume divided by open interest."),
    (VolumeInterestMean(30), "Thirty-bar mean of per-minute trading volume divided by open interest."),
    (VolumeNetInterest(5), "Total trading volume over five bars divided by the signed net change in open interest over the same bars."),
    (VolumeNetInterest(15), "Total trading volume over fifteen bars divided by the signed net change in open interest over the same bars."),
    (VolumeNetInterest(30), "Total trading volume over thirty bars divided by the signed net change in open interest over the same bars."),
];

impl Signal {
    fn inputs(self) -> (&'static [FutureMinuteField], usize, usize) {
        match self {
            MaRatio(..) | CloseMa(_) | Return(_) => (&[Raw("close")], 1, 30),
            AtrSpread(..) => (&[Raw("close"), Raw("high"), Raw("low")], 3, 58),
            EmaZscore(_) => (&[Raw("close")], 3, 58),
            VolumeEmaRatio(..) => (&[Raw("vol")], 3, 30),
            InterestEmaRatio(..) => (&[ContractDiff("oi")], 3, 30),
            VolumeInterestMean(_) => (&[Raw("vol"), Raw("oi")], 3, 30),
            VolumeNetInterest(_) => (&[Raw("vol"), ContractDiff("oi")], 3, 30),
        }
    }
}

struct MovingAverageSignal(usize);

pub(super) fn create(index: usize) -> Box<dyn Factor> {
    Box::new(MovingAverageSignal(index))
}

fn id(index: usize) -> String {
    format!("yq_{:03}", index + 1)
}

fn spec(index: usize) -> FactorSpec {
    let (columns, days, _) = SIGNALS[index].0.inputs();
    FactorSpec {
        id: id(index),
        aliases: Vec::new(),
        name: id(index),
        asset_class: AssetClass::Future,
        frequency: Frequency::Minute1,
        version: "0.1.0".into(),
        tags: vec!["minute".into(), "price_volume".into(), "pv".into(), "yq".into()],
        description: SIGNALS[index].1.into(),
        dependencies: vec![future_minute_request(columns)],
        intraday_raw_dependencies: Vec::new(),
        lookback: Lookback { trading_days: days },
    }
}

impl Factor for MovingAverageSignal {
    fn spec(&self) -> FactorSpec {
        spec(self.0)
    }

    fn compute(&self, context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let signal = SIGNALS[self.0].0;
        let (columns, _, bars) = signal.inputs();
        compute_future_minute(self.spec(), context, data, columns, bars, |inputs| {
            compute_signal(signal, inputs)
        })
    }
}

fn compute_signal(signal: Signal, inputs: &[Vec<Option<f64>>]) -> Vec<Option<f64>> {
    let first = &inputs[0];
    match signal {
        MaRatio(fast, slow) => combine(
            &ts_mean(first, fast, fast),
            &ts_mean(first, slow, slow),
            symmetric,
        ),
        CloseMa(window) => combine(first, &ts_mean(first, window, window), symmetric),
        Return(window) => ts_pctchg(first, window),
        AtrSpread(kind, fast, slow) => {
            let spread = combine(
                &smooth(first, fast, kind),
                &smooth(first, slow, kind),
                |a, b| Some(a - b),
            );
            combine(&spread, &atr(first, &inputs[1], &inputs[2], slow), divide)
        }
        EmaZscore(window) => ts_zscore(&rolling_ema(first, window), 30, 30),
        VolumeEmaRatio(fast, slow) => combine(
            &rolling_ema(first, fast),
            &rolling_ema(first, slow),
            symmetric,
        ),
        InterestEmaRatio(fast, slow) => {
            combine(
                &rolling_ema(first, fast),
                &rolling_ema(first, slow),
                signed_symmetric,
            )
        }
        VolumeInterestMean(window) => ts_mean(&combine(first, &inputs[1], divide), window, window),
        VolumeNetInterest(window) => combine(
            &ts_sum(first, window, window),
            &ts_sum(&inputs[1], window, window),
            divide,
        ),
    }
}

fn combine(
    left: &[Option<f64>],
    right: &[Option<f64>],
    operation: impl Fn(f64, f64) -> Option<f64>,
) -> Vec<Option<f64>> {
    left.iter()
        .zip(right)
        .map(|(&left, &right)| operation(left?, right?).filter(|value| value.is_finite()))
        .collect()
}

fn divide(a: f64, b: f64) -> Option<f64> {
    (b != 0.0).then(|| a / b)
}

fn symmetric(a: f64, b: f64) -> Option<f64> {
    (a > 0.0 && b > 0.0).then(|| (a - b) / a.min(b))
}

fn signed_symmetric(a: f64, b: f64) -> Option<f64> {
    Some(if b == 0.0 {
        if a == 0.0 {
            0.0
        } else {
            1.0
        }
    } else if a == 0.0 {
        -b.signum()
    } else {
        (a - b) / a.abs().min(b.abs())
    })
}

fn smooth(values: &[Option<f64>], window: usize, kind: Average) -> Vec<Option<f64>> {
    match kind {
        Ema => rolling_ema(values, window),
        Trima => {
            let first = window.div_ceil(2);
            let second = window / 2 + 1;
            ts_mean(&ts_mean(values, first, first), second, second)
        }
        Dema => {
            let ema = rolling_ema(values, window);
            combine(&ema, &rolling_ema(&ema, window), |a, b| Some(2.0 * a - b))
        }
    }
}

fn atr(
    close: &[Option<f64>],
    high: &[Option<f64>],
    low: &[Option<f64>],
    window: usize,
) -> Vec<Option<f64>> {
    let ranges = (0..close.len())
        .map(|idx| {
            let (high, low) = (high[idx]?, low[idx]?);
            let previous = idx.checked_sub(1).and_then(|i| close[i]);
            let range = previous.map_or(high - low, |previous| {
                (high - low)
                    .max((high - previous).abs())
                    .max((low - previous).abs())
            });
            Some(range)
        })
        .collect::<Vec<_>>();
    rolling_exponential_mean(&ranges, window, 1.0 - 1.0 / window as f64)
}

use crate::core::{AssetClass, FactorContext, FactorSeries, FactorSpec, Frequency, Lookback};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::future::minute_input::{
    compute_future_minute_many, future_minute_request, FutureMinuteField,
    FutureMinuteField::{ContinuousMinute, ContractDiff, ContractLevel},
};
use crate::factor::future::multiscale_volatility::{Series, Values};
use crate::factor::Factor;
use std::collections::BTreeMap;

const FIRST_ID: usize = 688;
const THRESHOLD: f64 = 100.0;
const LAMBERT: f64 = 0.015;
const PRICE: &[FutureMinuteField] = &[
    ContractLevel("close"),
    ContractLevel("high"),
    ContractLevel("low"),
    ContinuousMinute,
];
const PARTICIPATION: &[FutureMinuteField] = &[
    PRICE[0],
    PRICE[1],
    PRICE[2],
    PRICE[3],
    ContractLevel("vol"),
    ContractDiff("oi"),
];

const FACTORS: [(&str, &str); 5] = [
    ("level", "Standard Lambert CCI of minute typical prices. A flat complete window yields zero."),
    ("trend_expansion", "Follow CCI outside the outer thresholds when it is still moving further from the center. The change in CCI velocity controls confirmation."),
    ("tail_reentry", "Fade the previous CCI extreme on reentry inside the outer thresholds, weighted by the excursion, tail persistence and price confirmation."),
    ("price_divergence", "Fade a new closing extreme when CCI diverges from the prior extreme, with a recency decay and a fitted oscillator slope."),
    ("participation_confirmation", "Follow signed CCI when current log volume exceeds its trailing median and net open interest increases. Median absolute deviation scales the volume surprise."),
];

fn periods(kind: usize, variant: usize) -> (usize, usize) {
    // Allocate history to the lagged CCI or extrema without exceeding 30 bars.
    match kind {
        1 | 2 => ([5, 15, 28][variant], 2),
        3 => ([5, 15, 25][variant], [2, 3, 5][variant]),
        _ => ([5, 15, 30][variant], 0),
    }
}

#[derive(Clone, Copy)]
struct CciFactor {
    kind: usize,
    variant: usize,
}

pub(super) fn create(id: usize) -> Box<dyn Factor> {
    let offset = id - FIRST_ID;
    Box::new(CciFactor {
        kind: offset / 3,
        variant: offset % 3,
    })
}

impl Factor for CciFactor {
    fn spec(&self) -> FactorSpec {
        let (alias, description) = FACTORS[self.kind];
        let (n, lag) = periods(self.kind, self.variant);
        let id = format!("yq_{:03}", FIRST_ID + self.kind * 3 + self.variant);
        FactorSpec {
            name: id.clone(),
            id,
            aliases: vec![format!("cci_{alias}_{n}m")],
            asset_class: AssetClass::Future,
            frequency: Frequency::Minute1,
            version: "0.1.0".into(),
            tags: ["minute", "pv", "yq", "cci", "timing"]
                .map(str::to_string)
                .to_vec(),
            description: format!(
                "{description} Uses a {n}-bar CCI and {} bars of total history.",
                n + lag
            ),
            dependencies: vec![future_minute_request(if self.kind == 4 {
                PARTICIPATION
            } else {
                PRICE
            })],
            intraday_raw_dependencies: Vec::new(),
            lookback: Lookback { trading_days: 0 },
        }
    }

    fn compute_provider_key(&self) -> String {
        format!("future-minute-cci-{}", self.kind == 4)
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
        let has_participation = self.kind == 4;
        let (specs, variants): (Vec<_>, Vec<_>) = ids
            .iter()
            .filter_map(|id| {
                let offset = id
                    .strip_prefix("yq_")?
                    .parse::<usize>()
                    .ok()?
                    .checked_sub(FIRST_ID)?;
                let factor = Self {
                    kind: offset / 3,
                    variant: offset % 3,
                };
                (factor.kind < FACTORS.len() && (factor.kind == 4) == has_participation)
                    .then(|| (factor.spec(), factor))
            })
            .unzip();
        if specs.is_empty() {
            return Ok(Vec::new());
        }
        let fields = if has_participation {
            PARTICIPATION
        } else {
            PRICE
        };
        compute_future_minute_many(&specs, context, data, fields, 0, |inputs| {
            evaluate(inputs, &variants)
        })
    }
}

#[derive(Clone, Copy)]
struct Cci {
    value: f64,
    deviation: f64,
}

fn cci_series(typical: &Values, links: &Values, n: usize) -> Vec<Option<Cci>> {
    let mut result = vec![None; typical.len()];
    let Some(output) = result.get_mut(n - 1..) else {
        return result;
    };
    for (start, (slot, prices)) in output.iter_mut().zip(typical.windows(n)).enumerate() {
        if links[start + 1..start + n].iter().any(|&x| x != Some(1.0)) {
            continue;
        }
        let Some(anchor) = prices[n - 1] else {
            continue;
        };
        let Some(total) = prices
            .iter()
            .try_fold(0.0, |sum, &x| Some(sum + (x? - anchor)))
        else {
            continue;
        };
        let average = total / n as f64;
        let deviation = prices
            .iter()
            .flatten()
            .map(|x| ((x - anchor) - average).abs())
            .sum::<f64>()
            / n as f64;
        let value = if deviation == 0.0 {
            0.0
        } else {
            -average / (LAMBERT * deviation)
        };
        if value.is_finite() && deviation.is_finite() {
            *slot = Some(Cci { value, deviation });
        }
    }
    result
}

fn evaluate(inputs: &[Series], factors: &[CciFactor]) -> Vec<Series> {
    let (close, high, low, links) = (&inputs[0], &inputs[1], &inputs[2], &inputs[3]);
    let typical = (0..close.len())
        .map(|i| {
            let (c, h, l) = (close[i]?, high[i]?, low[i]?);
            (l > 0.0 && l <= c && c <= h).then(|| (h + l + c) / 3.0)
        })
        .collect::<Series>();
    let mut profiles = BTreeMap::new();
    for f in factors {
        let (n, _) = periods(f.kind, f.variant);
        profiles
            .entry(n)
            .or_insert_with(|| cci_series(&typical, links, n));
    }
    factors
        .iter()
        .map(|f| {
            let (n, lag) = periods(f.kind, f.variant);
            let profile = &profiles[&n];
            (0..close.len())
                .map(|i| {
                    let now = profile[i]?;
                    match f.kind {
                        0 => Some(now.value),
                        1 | 2 => {
                            let previous = profile[i.checked_sub(1)?]?;
                            let older = profile[i.checked_sub(2)?]?.value;
                            Some(if f.kind == 1 {
                                expansion(now.value, previous.value, older)
                            } else {
                                reentry(now, previous, older, close[i]? - close[i - 1]?)
                            })
                        }
                        3 => divergence(profile, close, i, lag),
                        4 => participation(now.value, &inputs[4], &inputs[5], i, n),
                        _ => unreachable!("unknown CCI factor"),
                    }
                })
                .collect()
        })
        .collect()
}

fn positive_tanh(value: f64, scale: f64) -> f64 {
    if value <= 0.0 {
        0.0
    } else {
        (value / scale).tanh()
    }
}

fn smooth_confirmation(normalized: f64) -> f64 {
    0.5 * (1.0 + normalized.tanh())
}

fn expansion(now: f64, previous: f64, older: f64) -> f64 {
    let direction = now.signum();
    let speed = direction * (now - previous);
    let acceleration = speed - direction * (previous - older);
    direction
        * positive_tanh(now.abs() - THRESHOLD, THRESHOLD)
        * positive_tanh(speed, THRESHOLD)
        * smooth_confirmation(acceleration / THRESHOLD)
}

fn reentry(now: Cci, previous: Cci, older: f64, close_change: f64) -> f64 {
    if now.value.abs() >= THRESHOLD || previous.value.abs() <= THRESHOLD {
        return 0.0;
    }
    let direction = previous.value.signum();
    let tail_persistence = 0.5 + 0.5 * positive_tanh(direction * older - THRESHOLD, THRESHOLD);
    let price_scale = now.deviation.max(previous.deviation);
    let price_confirmation = smooth_confirmation(-direction * close_change / price_scale);
    -direction
        * positive_tanh(previous.value.abs() - THRESHOLD, THRESHOLD)
        * positive_tanh(THRESHOLD - direction * now.value, THRESHOLD)
        * tail_persistence
        * price_confirmation
}

fn divergence(profile: &[Option<Cci>], close: &Values, i: usize, lag: usize) -> Option<f64> {
    let start = i.checked_sub(lag)?;
    let mut high = start;
    let mut low = start;
    let midpoint = lag as f64 / 2.0;
    let baseline = profile[start]?.value;
    let mut covariance = 0.0;
    let mut time_variance = 0.0;
    for j in start..=i {
        let centered_time = (j - start) as f64 - midpoint;
        covariance += centered_time * (profile[j]?.value - baseline);
        time_variance += centered_time * centered_time;
        if j == i {
            break;
        }
        // Equal closing prices choose the most recent extreme.
        if close[j]? >= close[high]? {
            high = j;
        }
        if close[j]? <= close[low]? {
            low = j;
        }
    }
    let c = close[i]?;
    let now = profile[i]?;
    let upper = close[high]?;
    let lower = close[low]?;
    let scale = (upper - lower).max(now.deviation);
    let fitted_change = covariance / time_variance * lag as f64;
    let leg = |movement, oscillator_gap, pivot, slope_support| {
        let age = (i - pivot) as f64;
        positive_tanh(movement, scale)
            * positive_tanh(oscillator_gap, THRESHOLD)
            * (-age / lag as f64).exp2()
            * smooth_confirmation(slope_support / THRESHOLD)
    };
    let bearish = leg(
        c - upper,
        profile[high]?.value - now.value,
        high,
        -fitted_change,
    );
    let bullish = leg(
        lower - c,
        now.value - profile[low]?.value,
        low,
        fitted_change,
    );
    Some(bullish - bearish)
}

fn median(sample: &mut [f64]) -> f64 {
    sample.sort_unstable_by(f64::total_cmp);
    let middle = sample.len() / 2;
    if sample.len() % 2 == 0 {
        (sample[middle - 1] + sample[middle]) / 2.0
    } else {
        sample[middle]
    }
}

fn participation(cci: f64, volume: &Values, delta: &Values, i: usize, n: usize) -> Option<f64> {
    let previous = &volume[(i + 1).checked_sub(n)?..i];
    let mut history = [0.0; 29];
    let history = &mut history[..previous.len()];
    for (slot, &v) in history.iter_mut().zip(previous) {
        *slot = v.filter(|x| *x >= 0.0)?.ln_1p();
    }
    let center = median(history);
    for v in history.iter_mut() {
        *v = (*v - center).abs();
    }
    // A 25% change is the reference scale when the historical dispersion is tiny.
    let scale = median(history).max(0.25_f64.ln_1p());
    let current = volume[i].filter(|x| *x >= 0.0)?;
    let added = delta[i]?.max(0.0);
    if current == 0.0 {
        return Some(0.0);
    }
    let surprise = positive_tanh(current.ln_1p() - center, scale);
    Some((cci / THRESHOLD).tanh() * surprise * (added / current).tanh())
}

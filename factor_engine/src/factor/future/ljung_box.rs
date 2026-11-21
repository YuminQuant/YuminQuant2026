use crate::core::{AssetClass, FactorContext, FactorSeries, FactorSpec, Frequency, Lookback};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::future::minute_input::{
    compute_future_minute_many, future_minute_request, FutureMinuteField,
    FutureMinuteField::{ContinuousMinute, ContractDiff, ContractLevel},
};
use crate::factor::future::multiscale_volatility::{log_return, map, zip, Series, Values};
use crate::factor::future::variance_timing::{close_returns, enough, ReturnWindow};
use crate::factor::Factor;
use crate::operators::ts_delay;

const FIRST_ID: usize = 313;
const LAGS: usize = 5;
const PRICE_WINDOWS: [usize; 3] = [60, 120, 180];
const STATE_WINDOWS: [usize; 3] = [120, 180, 240];
const ACTIVITY_BASELINE: usize = 5;
const MIN_GROUP_PAIRS: usize = 12;
const BLOCK: usize = 5;
const MAX_BLOCKS: usize = STATE_WINDOWS[2] / BLOCK;

const CLOSE: &[FutureMinuteField] = &[ContractLevel("close"), ContinuousMinute];
const RANGE: &[FutureMinuteField] = &[
    CLOSE[0],
    ContinuousMinute,
    ContractLevel("high"),
    ContractLevel("low"),
];
const VOLUME: &[FutureMinuteField] = &[CLOSE[0], ContinuousMinute, ContractLevel("vol")];
const VOLUME_OI: &[FutureMinuteField] =
    &[VOLUME[0], ContinuousMinute, VOLUME[2], ContractDiff("oi")];
const RANGE_VOLUME: &[FutureMinuteField] = &[
    RANGE[0],
    ContinuousMinute,
    RANGE[2],
    RANGE[3],
    VOLUME[2],
    ContractLevel("open"),
];
const UNWIND: &[FutureMinuteField] = &[
    CLOSE[0],
    ContinuousMinute,
    ContractDiff("oi"),
    ContractLevel("open"),
];
const BODY: &[FutureMinuteField] = &[CLOSE[0], ContinuousMinute, ContractLevel("open")];
const TURNOVER: &[FutureMinuteField] = &[
    RANGE[0],
    ContinuousMinute,
    RANGE[2],
    RANGE[3],
    VOLUME[2],
    ContractLevel("oi"),
];
const AMOUNT: &[FutureMinuteField] = &[CLOSE[0], ContinuousMinute, ContractLevel("amount")];
const OHLC: &[FutureMinuteField] = &[RANGE[0], ContinuousMinute, RANGE[2], RANGE[3], BODY[2]];
const VOLUME_AMOUNT: &[FutureMinuteField] = &[VOLUME[0], ContinuousMinute, VOLUME[2], AMOUNT[2]];
const AMOUNT_OI: &[FutureMinuteField] =
    &[AMOUNT[0], ContinuousMinute, AMOUNT[2], ContractDiff("oi")];

const FACTORS: [(&str, &str); 27] = [
    ("lb_signed_return_dependence", "Current return times the balance of positive and negative Ljung-Box contributions over five lags."),
    ("lb_odd_even_reversal", "Fade the current return when negative odd-lag and positive even-lag autocorrelations jointly indicate alternating returns."),
    ("lb_sign_vs_magnitude", "Current return times the change in signed Ljung-Box strength when return amplitudes are replaced by their signs."),
    ("lb_ar_residual_projection", "Signed higher-lag residual projection, gated by joint residual dependence on a later subwindow after a prior AR(1) fit."),
    ("lb_range_cluster_shock_fade", "Fade a price shock beyond two historical standard deviations when log high-low ranges cluster and return dependence is weak."),
    ("lb_volume_conditioned_propagation", "Signed return weighted by the difference in lagged return dependence after high versus low relative volume."),
    ("lb_volume_oi_cross_lag", "Historical cross-lag response of returns to volume-weighted open-interest shocks, activated by the current signed shock."),
    ("lb_volume_cluster_wick_rejection", "Wick rejection in a compressed candle, weighted by persistent abnormal minute volume."),
    ("lb_unwind_cluster_reversal", "Fade the signed candle body during a negative open-interest shock when past net reductions cluster."),
    ("lb_realized_volatility_release", "Signed candle body during declining realized volatility, weighted by dependence across non-overlapping five-minute volatility blocks."),
    ("lb_turnover_close_acceptance", "Signed close location confirmed by persistent abnormal volume per unit of preceding open interest."),
    ("lb_volume_oi_correlation_regime", "Signed return multiplied by the latest five-minute volume and open-interest correlation and its historical block-dependence strength."),
    ("lb_amihud_cluster_reversal", "Fade the current return when unusually high Amihud illiquidity occurs in a persistent illiquidity regime."),
    ("lb_volume_leads_range", "Signed candle body weighted by historically predictable range expansion after relative volume shocks."),
    ("lb_signed_volume_proxy_dependence", "Current close-location volume proxy times the signed portmanteau strength of its own history."),
    ("lb_body_efficiency_dependence", "Signed candle body as a share of range, adjusted by persistence or reversal of candle-body efficiency."),
    ("lb_minute_opening_gap_dependence", "Standardized close-to-open price gaps within contiguous minutes, weighted by the signed dependence of earlier gaps."),
    ("lb_two_sided_wick_pressure", "Lower-wick versus upper-wick rejection weighted by each side's own serial dependence."),
    ("lb_price_shock_leads_volume", "Current price direction confirmed by predictability of future volume after large absolute price returns."),
    ("lb_price_oi_correlation_regime", "An open-interest shock mapped to price direction using recent price and open-interest correlation and its historical persistence."),
    ("lb_new_interest_per_lot", "Signed price response during sustained abnormal positive net open-interest change per traded lot."),
    ("lb_oi_state_flip", "Signed candle body during an open-interest sign reversal in a serially dependent positioning regime."),
    ("lb_trading_restart_fade", "Fade the first traded price shock after a zero-volume minute when zero-volume minutes cluster historically."),
    ("lb_close_trade_price_basis_change", "Change in the close-versus-average-execution-price basis weighted by signed serial dependence."),
    ("lb_amount_leads_inventory_activity", "Current price direction confirmed by lagged predictability of absolute open-interest shocks after traded-amount surprises."),
    ("lb_range_volume_correlation_rejection", "Wick rejection scaled by intrabar range-volume association and the persistence of that association across disjoint blocks."),
    ("lb_volume_acceleration", "Current price direction amplified or reversed by changes in minute volume in a serially dependent volume-acceleration regime."),
];

fn fields(kind: usize) -> &'static [FutureMinuteField] {
    match kind {
        4 => RANGE,
        5 | 18 | 22 | 26 => VOLUME,
        6 | 11 | 20 => VOLUME_OI,
        7 | 13 | 14 | 25 => RANGE_VOLUME,
        8 | 19 | 21 => UNWIND,
        9 | 16 => BODY,
        10 => TURNOVER,
        12 => AMOUNT,
        15 | 17 => OHLC,
        23 => VOLUME_AMOUNT,
        24 => AMOUNT_OI,
        _ => CLOSE,
    }
}

fn widths(kind: usize) -> &'static [usize; 3] {
    if kind >= 5 {
        &STATE_WINDOWS
    } else {
        &PRICE_WINDOWS
    }
}

#[derive(Clone, Copy)]
struct LjungBoxFactor {
    kind: usize,
    variant: usize,
}

pub(super) fn create(number: usize) -> Box<dyn Factor> {
    let offset = number - FIRST_ID;
    Box::new(LjungBoxFactor {
        kind: offset / PRICE_WINDOWS.len(),
        variant: offset % PRICE_WINDOWS.len(),
    })
}

impl Factor for LjungBoxFactor {
    fn spec(&self) -> FactorSpec {
        let (alias, description) = FACTORS[self.kind];
        let n = widths(self.kind)[self.variant];
        let id = format!(
            "yq_{:03}",
            FIRST_ID + self.kind * PRICE_WINDOWS.len() + self.variant
        );
        FactorSpec {
            name: id.clone(),
            id,
            aliases: vec![format!("{alias}_{n}m")],
            asset_class: AssetClass::Future,
            frequency: Frequency::Minute1,
            version: "0.1.0".into(),
            tags: ["minute", "pv", "yq", "ljung_box", "timing"]
                .map(str::to_string)
                .to_vec(),
            description: format!("{description} The lookback is {n} observed minute bars."),
            dependencies: vec![future_minute_request(fields(self.kind))],
            intraday_raw_dependencies: Vec::new(),
            lookback: Lookback { trading_days: 3 },
        }
    }

    fn compute_provider_key(&self) -> String {
        format!("future-minute-ljung-box-{}", self.kind)
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
        let (specs, sizes): (Vec<_>, Vec<_>) = widths(self.kind)
            .iter()
            .enumerate()
            .filter_map(|(variant, &n)| {
                let spec = Self { variant, ..*self }.spec();
                ids.contains(&spec.id).then_some((spec, n))
            })
            .unzip();
        let Some(&max_width) = sizes.last() else {
            return Ok(Vec::new());
        };
        // Reserve the five-bar activity baseline and the preceding level used for differencing.
        compute_future_minute_many(
            &specs,
            context,
            data,
            fields(self.kind),
            max_width + ACTIVITY_BASELINE + 1,
            |inputs| {
                let returns = close_returns(&inputs[0], &inputs[1]);
                let state = auxiliary(self.kind, inputs, &returns);
                let related = secondary(self.kind, inputs, &state);
                sizes
                    .iter()
                    .map(|&n| {
                        (0..returns.len())
                            .map(|t| {
                                evaluate(
                                    self.kind,
                                    n,
                                    t,
                                    inputs,
                                    &returns,
                                    &state,
                                    related.as_deref(),
                                )
                                .filter(|value| value.is_finite())
                            })
                            .collect()
                    })
                    .collect()
            },
        )
    }
}

fn auxiliary(kind: usize, inputs: &[Series], returns: &Values) -> Series {
    match kind {
        2 => map(returns, |r| if r == 0.0 { 0.0 } else { r.signum() }),
        5 => relative_positive(&inputs[2]),
        6 => {
            let volume = relative_positive(&inputs[2]);
            let shocks = oi_shocks(&inputs[3], &inputs[1]);
            volume
                .into_iter()
                .zip(shocks)
                .map(|(v, shock)| {
                    let v = v?;
                    Some(v / (1.0 + v) * shock?)
                })
                .collect()
        }
        7 => mask(&map(&relative_positive(&inputs[4]), f64::ln_1p), &inputs[1]),
        8 | 21 => oi_shocks(&inputs[2], &inputs[1]),
        9 => realized_volatility(returns),
        10 => map(&turnover(inputs), f64::ln_1p),
        11 => volume_oi_correlation(inputs),
        12 => map(
            &relative_positive(&zip(returns, &inputs[2], |r, amount| {
                (amount > 0.0).then(|| r.abs() / amount)
            })),
            f64::ln_1p,
        ),
        13 => map(&relative_positive(&inputs[4]), centered_ratio),
        14 => {
            let volume = relative_positive(&inputs[4]);
            (0..returns.len())
                .map(|i| {
                    (inputs[1][i] == Some(1.0)).then_some(())?;
                    let v = volume[i]?;
                    Some(close_location(inputs, i)? * v / (1.0 + v))
                })
                .collect()
        }
        15 => (0..returns.len())
            .map(|i| {
                let (close, high, low, open) = valid_candle(inputs, 4, i)?;
                Some((close - open) / (high - low))
            })
            .collect(),
        16 => (0..returns.len())
            .map(|i| {
                (inputs[1][i] == Some(1.0)).then_some(())?;
                log_return(inputs[2][i]?, inputs[0][i.checked_sub(1)?]?)
            })
            .collect(),
        17 => wick_series(inputs, 4, true),
        18 => map(returns, f64::abs),
        19 => block_correlation(returns, &oi_shocks(&inputs[2], &inputs[1])),
        20 => {
            let positive_oi = mask(&inputs[3], &inputs[1]);
            map(
                &relative_positive(&zip(&positive_oi, &inputs[2], |delta, vol| {
                    (vol > 0.0).then(|| delta.max(0.0) / vol)
                })),
                f64::ln_1p,
            )
        }
        22 => mask(
            &inputs[2]
                .iter()
                .map(|&vol| {
                    Some(if vol.filter(|x| *x >= 0.0)? == 0.0 {
                        1.0
                    } else {
                        0.0
                    })
                })
                .collect::<Series>(),
            &inputs[1],
        ),
        23 => {
            let average_trade = zip(&inputs[3], &inputs[2], log_return);
            let basis = zip(&map(&inputs[0], f64::ln), &average_trade, |close, vwap| {
                Some(close - vwap)
            });
            difference(&basis, &inputs[1])
        }
        24 => mask(&map(&relative_positive(&inputs[2]), f64::ln_1p), &inputs[1]),
        25 => block_correlation(&log_ranges(inputs), &log_volume(&inputs[4])),
        26 => difference(&log_volume(&inputs[2]), &inputs[1]),
        _ => Vec::new(),
    }
}

fn secondary(kind: usize, inputs: &[Series], state: &Values) -> Option<Series> {
    match kind {
        4 | 7 | 13 => Some(log_ranges(inputs)),
        8 => Some(map(state, |shock| (-shock).max(0.0))),
        17 => Some(wick_series(inputs, 4, false)),
        18 => Some(mask(
            &map(&relative_positive(&inputs[2]), f64::ln_1p),
            &inputs[1],
        )),
        19 => Some(oi_shocks(&inputs[2], &inputs[1])),
        24 => Some(map(&oi_shocks(&inputs[3], &inputs[1]), f64::abs)),
        _ => None,
    }
}

fn mask(values: &Values, links: &Values) -> Series {
    zip(values, links, |value, link| (link == 1.0).then_some(value))
}

fn log_ranges(inputs: &[Series]) -> Series {
    (0..inputs[0].len())
        .map(|i| {
            (inputs[1][i] == Some(1.0)).then_some(())?;
            let (close, high, low) = (inputs[0][i]?, inputs[2][i]?, inputs[3][i]?);
            (low <= close && close <= high).then_some(())?;
            log_return(high, low)
        })
        .collect()
}

fn close_location(inputs: &[Series], i: usize) -> Option<f64> {
    let (close, high, low) = (inputs[0][i]?, inputs[2][i]?, inputs[3][i]?);
    (0.0 < low && low <= close && close <= high && high > low)
        .then(|| ((close - low) - (high - close)) / (high - low))
}

fn valid_candle(inputs: &[Series], open_column: usize, i: usize) -> Option<(f64, f64, f64, f64)> {
    let (close, high, low, open) = (
        inputs[0][i]?,
        inputs[2][i]?,
        inputs[3][i]?,
        inputs[open_column][i]?,
    );
    (inputs[1][i] == Some(1.0)
        && 0.0 < low
        && low < high
        && low <= close.min(open)
        && close.max(open) <= high)
        .then_some((close, high, low, open))
}

fn wick_series(inputs: &[Series], open_column: usize, upper: bool) -> Series {
    (0..inputs[0].len())
        .map(|i| {
            let (close, high, low, open) = valid_candle(inputs, open_column, i)?;
            let wick = if upper {
                high - close.max(open)
            } else {
                close.min(open) - low
            };
            Some(wick / (high - low))
        })
        .collect()
}

fn log_volume(volume: &Values) -> Series {
    volume
        .iter()
        .map(|&vol| Some(vol.filter(|v| *v >= 0.0)?.ln_1p()))
        .collect()
}

fn difference(values: &Values, links: &Values) -> Series {
    mask(
        &zip(values, &ts_delay(values, 1), |current, previous| {
            Some(current - previous)
        }),
        links,
    )
}

fn centered_ratio(ratio: f64) -> f64 {
    (ratio - 1.0) / (ratio + 1.0)
}

fn relative_positive(values: &Values) -> Series {
    // Sum the five observations directly so zeros in the window stay exact
    // and the normalization is independent of the beginning of the batch.
    (0..values.len())
        .map(|i| {
            let start = i.checked_sub(ACTIVITY_BASELINE)?;
            let sum = values[start..i]
                .iter()
                .try_fold(0.0, |sum, &value| Some(sum + value.filter(|x| *x >= 0.0)?))?;
            (sum > 0.0).then_some(())?;
            Some(values[i].filter(|x| *x >= 0.0)? / (sum / ACTIVITY_BASELINE as f64))
        })
        .collect()
}

fn turnover(inputs: &[Series]) -> Series {
    let turns = (0..inputs[0].len())
        .map(|i| {
            (inputs[1][i] == Some(1.0)).then_some(())?;
            let (vol, oi) = (inputs[4][i]?, inputs[5][i.checked_sub(1)?]?);
            (vol >= 0.0 && oi > 0.0).then(|| vol / oi)
        })
        .collect::<Series>();
    relative_positive(&turns)
}

fn realized_volatility(returns: &Values) -> Series {
    trailing_blocks(returns.len(), |start, end| {
        let squares = returns[start..end]
            .iter()
            .try_fold(0.0, |sum, &r| Some(sum + r?.powi(2)))?;
        Some((squares / BLOCK as f64).sqrt())
    })
}

fn trailing_blocks(length: usize, f: impl Fn(usize, usize) -> Option<f64>) -> Series {
    let mut values = vec![None; length];
    for (i, value) in values.iter_mut().enumerate().skip(BLOCK - 1) {
        *value = f(i + 1 - BLOCK, i + 1);
    }
    values
}

fn volume_oi_correlation(inputs: &[Series]) -> Series {
    block_correlation(&log_volume(&inputs[2]), &oi_shocks(&inputs[3], &inputs[1]))
}

fn block_correlation(x: &Values, y: &Values) -> Series {
    trailing_blocks(x.len(), |start, end| {
        let mut pairs = Pairs::default();
        for i in start..end {
            pairs.push(x[i]?, y[i]?);
        }
        pairs.correlation()
    })
}

fn oi_shocks(changes: &Values, links: &Values) -> Series {
    let intraday_change = |i: usize| {
        (links[i] == Some(1.0)).then_some(())?;
        changes[i]
    };
    (0..changes.len())
        .map(|i| {
            let current = intraday_change(i)?;
            let start = i.checked_sub(ACTIVITY_BASELINE)?;
            let (n, sum) = (start..i)
                .filter_map(intraday_change)
                .fold((0, 0.0), |(n, sum), x| (n + 1, sum + x.abs()));
            if n < 3 {
                return None;
            }
            if sum == 0.0 {
                return (current == 0.0).then_some(0.0);
            }
            Some((current / (sum / n as f64)).tanh())
        })
        .collect()
}

#[derive(Default, Clone, Copy)]
struct Lag {
    correlation: f64,
    energy: f64,
}

impl Lag {
    fn new(correlation: f64, observations: usize, pairs: usize) -> Self {
        let n = observations as f64;
        Self {
            correlation,
            energy: n * (n + 2.0) * correlation.powi(2) / pairs as f64,
        }
    }
}

struct Profile([Lag; LAGS]);

impl Profile {
    fn autocorrelation(window: &ReturnWindow<'_>, width: usize) -> Option<Self> {
        let mut lags = [Lag::default(); LAGS];
        for (index, result) in lags.iter_mut().enumerate() {
            let lag = index + 1;
            let (pairs, sum) = window
                .lag_pairs(lag)
                .fold((0, 0.0), |(n, sum), (_, a, b)| (n + 1, sum + a * b));
            if !enough(pairs, width - lag) {
                return None;
            }
            *result = Lag::new(sum / window.centered_squares, window.count, pairs);
        }
        Some(Self(lags))
    }

    fn energy(&self) -> f64 {
        self.0.iter().map(|item| item.energy).sum()
    }

    fn signed(&self) -> f64 {
        let score = self
            .0
            .iter()
            .map(|item| item.correlation.signum() * item.energy)
            .sum::<f64>();
        let energy = self.energy();
        if energy > LAGS as f64 {
            strength(energy, LAGS) * score / energy
        } else {
            0.0
        }
    }

    fn alternation(&self) -> f64 {
        let mut odd = 0.0;
        let mut even = 0.0;
        for (index, item) in self.0.iter().enumerate() {
            if index % 2 == 0 && item.correlation < 0.0 {
                odd += item.energy;
            } else if index % 2 == 1 && item.correlation > 0.0 {
                even += item.energy;
            }
        }
        (strength(odd, 3) * strength(even, 2)).sqrt()
    }
}

// On a complete price window, each contribution uses the textbook n-k denominator.
// Gapped and conditional profiles use actual pair counts as a descriptive score.
fn strength(energy: f64, lags: usize) -> f64 {
    let base = lags as f64;
    (energy - base).max(0.0) / (energy + base)
}

#[derive(Default)]
struct Pairs {
    n: usize,
    mean_x: f64,
    mean_y: f64,
    ss_x: f64,
    ss_y: f64,
    cross: f64,
}

impl Pairs {
    fn push(&mut self, x: f64, y: f64) {
        self.n += 1;
        let n = self.n as f64;
        let dx = x - self.mean_x;
        let dy = y - self.mean_y;
        self.mean_x += dx / n;
        self.mean_y += dy / n;
        self.ss_x += dx * (x - self.mean_x);
        self.ss_y += dy * (y - self.mean_y);
        self.cross += dx * (y - self.mean_y);
    }

    fn correlation(&self) -> Option<f64> {
        let scale = self.ss_x.sqrt() * self.ss_y.sqrt();
        (scale.is_finite() && scale > 0.0).then(|| (self.cross / scale).clamp(-1.0, 1.0))
    }
}

fn residual_projection(returns: &Values, start: usize, t: usize) -> Option<f64> {
    let mid = (start + t) / 2;
    let mut train = Pairs::default();
    for pair in returns[start..mid].windows(2) {
        if let (Some(x), Some(y)) = (pair[0], pair[1]) {
            train.push(x, y);
        }
    }
    if !enough(train.n, mid - start - 1) || train.ss_x <= 0.0 {
        return None;
    }
    let slope = train.cross / train.ss_x;
    let residual =
        |i: usize| Some(returns[i]? - train.mean_y - slope * (returns[i - 1]? - train.mean_x));
    let mut valid = [None; PRICE_WINDOWS[2] / 2];
    for (slot, i) in valid.iter_mut().zip(mid..t) {
        *slot = residual(i);
    }
    let width = t - mid;
    let sample = ReturnWindow::new(&valid[..width])?;
    let profile = Profile::autocorrelation(&sample, width)?;
    // Apply the residual model only if the full forecast path is continuous.
    returns[t - LAGS..=t]
        .iter()
        .all(Option::is_some)
        .then_some(())?;
    let mut forecast = 0.0;
    for lag in 2..=LAGS {
        let e = residual(t + 1 - lag)? - sample.mean;
        forecast += (LAGS + 1 - lag) as f64 * profile.0[lag - 1].correlation * e;
    }
    Some((forecast / (10.0 * sample.std())).tanh() * strength(profile.energy(), LAGS))
}

fn volume_profile_spread(history: &ReturnWindow<'_>, volume: &Values) -> Option<f64> {
    let (mut active, mut quiet) = ([Lag::default(); LAGS], [Lag::default(); LAGS]);
    for lag in 1..=LAGS {
        let mut high = Pairs::default();
        let mut low = Pairs::default();
        for (i, x, y) in history.lag_pairs(lag) {
            match volume[i] {
                Some(v) if v >= 1.5 => high.push(x, y),
                Some(v) if v <= 0.8 => low.push(x, y),
                _ => {}
            }
        }
        // Use a common count in both scores to compare the two correlation structures.
        let pairs = high.n.min(low.n);
        if pairs < MIN_GROUP_PAIRS {
            return None;
        }
        active[lag - 1] = Lag::new(high.correlation()?, pairs + lag, pairs);
        quiet[lag - 1] = Lag::new(low.correlation()?, pairs + lag, pairs);
    }
    Some((Profile(active).signed() - Profile(quiet).signed()) / 2.0)
}

fn cross_lag_profile(history: &ReturnWindow<'_>, width: usize, shocks: &Values) -> Option<Profile> {
    let mut lags = [Lag::default(); LAGS];
    for lag in 1..=LAGS {
        let mut pairs = Pairs::default();
        for (i, _, future) in history.lag_pairs(lag) {
            if let Some(shock) = shocks[i] {
                pairs.push(shock, future);
            }
        }
        if !enough(pairs.n, width - lag) {
            return None;
        }
        lags[lag - 1] = Lag::new(pairs.correlation()?, pairs.n + lag, pairs.n);
    }
    Some(Profile(lags))
}

fn lagged_response(profile: &Profile) -> f64 {
    profile
        .0
        .iter()
        .enumerate()
        .map(|(i, lag)| (LAGS - i) as f64 * lag.correlation)
        .sum::<f64>()
        / 15.0
}

fn wick_rejection(
    inputs: &[Series],
    ranges: &Values,
    volume: &Values,
    start: usize,
    t: usize,
) -> Option<f64> {
    let state = ReturnWindow::new(&volume[start..t])?;
    let intensity = strength(Profile::autocorrelation(&state, t - start)?.energy(), LAGS);
    let range = ranges[t]?;
    let mut sum = 0.0;
    let mut observations = 0;
    for &value in ranges[start..t].iter().flatten() {
        sum += value;
        observations += 1;
    }
    if !enough(observations, t - start) || sum <= 0.0 {
        return None;
    }
    let (close, high, low, open) = (inputs[0][t]?, inputs[2][t]?, inputs[3][t]?, inputs[5][t]?);
    if !(0.0 < low && low < high && low <= open && open <= high) {
        return None;
    }
    let lower = close.min(open) - low;
    let upper = high - close.max(open);
    let compression = (1.0 - range / (sum / observations as f64)).max(0.0);
    let abnormal_volume = ((volume[t]? - std::f64::consts::LN_2) / state.std())
        .tanh()
        .max(0.0);
    Some((lower - upper) / (high - low) * compression * abnormal_volume * intensity)
}

fn block_profile(blocks: &Values, start: usize, t: usize) -> Option<Profile> {
    // Consecutive block endpoints are five bars apart. An invalid block remains
    // in place so a lag cannot bridge a break or a missing observation.
    let mut samples = [None; MAX_BLOCKS];
    let endpoints = (start + BLOCK - 1..t).step_by(BLOCK);
    let count = endpoints.len();
    for (slot, index) in samples.iter_mut().zip(endpoints) {
        *slot = blocks[index];
    }
    Profile::autocorrelation(&ReturnWindow::new(&samples[..count])?, count)
}

fn state_profile(state: &Values, start: usize, t: usize) -> Option<Profile> {
    Profile::autocorrelation(&ReturnWindow::new(&state[start..t])?, t - start)
}

fn abnormal(log_ratio: f64) -> f64 {
    (log_ratio - std::f64::consts::LN_2).tanh().max(0.0)
}

fn forecast_activity(
    target: &Values,
    source: &Values,
    start: usize,
    t: usize,
    surprise: f64,
) -> Option<f64> {
    let response = cross_lag_profile(
        &ReturnWindow::new(&target[start..t])?,
        t - start,
        &source[start..t],
    )?;
    Some(
        (surprise * lagged_response(&response)).tanh().max(0.0) * strength(response.energy(), LAGS),
    )
}

fn body(inputs: &[Series], open_column: usize, t: usize, scale: f64) -> Option<f64> {
    Some((log_return(inputs[0][t]?, inputs[open_column][t]?)? / scale).tanh())
}

fn evaluate(
    kind: usize,
    n: usize,
    t: usize,
    inputs: &[Series],
    returns: &Values,
    state: &Values,
    related: Option<&Values>,
) -> Option<f64> {
    let start = t.checked_sub(n)?;
    match kind {
        3 => return residual_projection(returns, start, t),
        7 => return wick_rejection(inputs, related?, state, start, t),
        10 => {
            return Some(
                close_location(inputs, t)?
                    * abnormal(state[t]?)
                    * state_profile(state, start, t)?.signed().max(0.0),
            )
        }
        14 | 15 => return Some(state[t]? * state_profile(state, start, t)?.signed()),
        16 => {
            let past = ReturnWindow::new(&state[start..t])?;
            let profile = Profile::autocorrelation(&past, n)?;
            return Some(((state[t]? - past.mean) / past.std()).tanh() * profile.signed());
        }
        17 => {
            let lower = related?;
            let low = strength(state_profile(lower, start, t)?.energy(), LAGS);
            let high = strength(state_profile(state, start, t)?.energy(), LAGS);
            return Some(lower[t]? * low - state[t]? * high);
        }
        25 => {
            let profile = block_profile(state, start, t)?;
            let (c, h, l, o) = valid_candle(inputs, 5, t)?;
            let wick = ((c.min(o) - l) - (h - c.max(o))) / (h - l);
            return Some(wick * state[t]? * strength(profile.energy(), LAGS));
        }
        _ => {}
    }

    let price = ReturnWindow::new(&returns[start..t])?;
    let shock = returns[t]? / price.std();
    let direction = shock.tanh();
    match kind {
        0 => Some(direction * Profile::autocorrelation(&price, n)?.signed()),
        1 => Some(-direction * Profile::autocorrelation(&price, n)?.alternation()),
        2 => {
            let signs = state_profile(state, start, t)?.signed();
            Some(direction * (signs - Profile::autocorrelation(&price, n)?.signed()) / 2.0)
        }
        4 => {
            let range = related?;
            range[t]?;
            let amplitude = strength(state_profile(range, start, t)?.energy(), LAGS);
            let price_dependence = strength(Profile::autocorrelation(&price, n)?.energy(), LAGS);
            let extreme = shock.signum() * (shock.abs() - 2.0).max(0.0).tanh();
            Some(-extreme * amplitude * (1.0 - price_dependence))
        }
        5 => Some(
            direction
                * centered_ratio(state[t]?)
                * volume_profile_spread(&price, &state[start..t])?,
        ),
        6 => {
            let profile = cross_lag_profile(&price, n, &state[start..t])?;
            Some((state[t]? * lagged_response(&profile)).tanh() * strength(profile.energy(), LAGS))
        }
        8 => {
            let reduction = related?;
            let regime = state_profile(reduction, start, t)?.signed().max(0.0);
            Some(-body(inputs, 3, t, price.std())? * reduction[t]? * regime)
        }
        9 => {
            let previous = state[t - BLOCK]?;
            let current = state[t]?;
            if previous <= 0.0 {
                return None;
            }
            let relief = (1.0 - current / previous).max(0.0);
            let regime = strength(block_profile(state, start, t)?.energy(), LAGS);
            Some(body(inputs, 2, t, price.std())? * relief * regime)
        }
        11 => {
            let regime = strength(block_profile(state, start, t)?.energy(), LAGS);
            Some(direction * state[t]? * regime)
        }
        12 => {
            let regime = state_profile(state, start, t)?.signed().max(0.0);
            Some(-direction * abnormal(state[t]?) * regime)
        }
        13 => {
            let expected = forecast_activity(related?, state, start, t, state[t]?)?;
            Some(body(inputs, 5, t, price.std())? * expected)
        }
        18 => {
            let past = ReturnWindow::new(&state[start..t])?;
            let normalized = (state[t]? - past.mean) / past.std();
            Some(direction * forecast_activity(related?, state, start, t, normalized)?)
        }
        19 => {
            let regime = strength(block_profile(state, start, t)?.energy(), LAGS);
            Some(related?[t]? * state[t]? * regime)
        }
        20 => Some(
            direction * abnormal(state[t]?) * state_profile(state, start, t)?.signed().max(0.0),
        ),
        21 => {
            let flip = (-state[t]? * state[t - 1]?).max(0.0);
            let regime = strength(state_profile(state, start, t)?.energy(), LAGS);
            Some(body(inputs, 3, t, price.std())? * flip * regime)
        }
        22 => {
            let regime = state_profile(state, start, t)?.signed().max(0.0);
            Some(if state[t - 1]? == 1.0 && state[t]? == 0.0 {
                -direction * regime
            } else {
                0.0
            })
        }
        23 | 26 => {
            let past = ReturnWindow::new(&state[start..t])?;
            let profile = Profile::autocorrelation(&past, n)?;
            let normalized = ((state[t]? - past.mean) / past.std()).tanh();
            if kind == 23 {
                Some(normalized * profile.signed())
            } else {
                Some(direction * normalized * strength(profile.energy(), LAGS))
            }
        }
        24 => {
            let past = ReturnWindow::new(&state[start..t])?;
            let normalized = (state[t]? - past.mean) / past.std();
            Some(direction * forecast_activity(related?, state, start, t, normalized)?)
        }
        _ => unreachable!("unknown Ljung-Box factor"),
    }
}

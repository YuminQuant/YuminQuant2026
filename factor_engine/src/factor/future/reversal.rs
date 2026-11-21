use crate::core::{AssetClass, FactorContext, FactorSeries, FactorSpec, Frequency, Lookback};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::future::minute_input::{
    compute_future_minute_many, future_minute_request, FutureMinuteField,
    FutureMinuteField::{ContinuousMinute, ContractDiff, ContractLevel},
};
use crate::factor::future::multiscale_volatility::{log_return, Series, Values};
use crate::factor::future::variance_timing::close_returns;
use crate::factor::Factor;

const FIRST_ID: usize = 565;
const WINDOWS: [(usize, usize); 3] = [(5, 2), (15, 5), (30, 10)];
const C: FutureMinuteField = ContractLevel("close");
const O: FutureMinuteField = ContractLevel("open");
const H: FutureMinuteField = ContractLevel("high");
const L: FutureMinuteField = ContractLevel("low");
const V: FutureMinuteField = ContractLevel("vol");
const I: FutureMinuteField = ContractLevel("oi");
const D: FutureMinuteField = ContractDiff("oi");
const A: FutureMinuteField = ContractLevel("amount");
const LINK: FutureMinuteField = ContinuousMinute;

const FACTORS: [(&str, &str); 36] = [
    ("nested_mean_flip", "Opposite signs of the nested full-window and recent mean minute returns. Signal direction follows the recent mean."),
    ("two_leg_reversal", "Reversal between disjoint price legs, sized by the smaller absolute displacement relative to realized return volatility."),
    ("breadth_flip", "Follow a price reversal when the majority of minutes in each leg agree with its own direction."),
    ("adaptive_pivot", "Choose the internal close that maximizes the matched price excursion on either side of the turning point."),
    ("hinge_fit", "A continuous two-slope price fit with opposing slopes. Signal strength reflects the gain in fit over a single trend line."),
    ("half_reclaim", "First crossing of the halfway recovery level of a fixed price excursion during the right leg."),
    ("overshoot_fatigue", "Fade a right leg that more than reclaims the left leg when its latest increment loses speed."),
    ("negv_residual", "Trade the remaining recovery after two opposite volume-weighted same-sign return runs. The shock must still be partially unrecovered."),
    ("flash_recovery", "Recovery from the largest single return in the older leg. Its share of total older-leg movement sizes the impulse."),
    ("duration_asymmetry", "A rapid price correction after a longer same-sign run, weighted by the speed advantage of the correction."),
    ("double_test", "A four-part alternating price path with a higher second low or lower second high and a reclaim of the intervening close."),
    ("failed_recovery", "Follow the original direction when a third price leg erases at least half of the intervening rebound."),
    ("pivot_wick", "Two-sided price excursion around the fixed pivot confirmed by an opposing-side wick on the pivot candle."),
    ("close_acceptance", "Price reversal confirmed by the shift in candle close location from the older to the newer leg."),
    ("range_reentry", "A price reversal that crosses back into the first candle's range after the pivot left that range."),
    ("range_contraction", "A two-leg price reversal confirmed by lower average intrabar range in the recovery leg."),
    ("semivariance_transfer", "Follow the reversal when opposing realized return power dominates on entry and aligned power dominates on recovery."),
    ("amihud_relief", "Price reversal with a decline in absolute return per unit of traded amount from the shock to the recovery leg."),
    ("volume_decay", "Follow an incomplete reversal after participation decreases from the shock to the recovery."),
    ("volume_timing", "Price reversal whose peak volume occurs close to the fixed turning point and is concentrated relative to window volume."),
    ("close_location_flow", "Price reversal confirmed by the sign change in volume-weighted close-location pressure between legs."),
    ("pivot_absorption", "Recovery after high pivot volume coincides with a small pivot candle body relative to its range."),
    ("turnover_peak", "Reversal around a pivot with high turnover per preceding outstanding lot relative to both surrounding legs."),
    ("deleveraging_rebuild", "Price recovery during a switch from net open interest contraction to expansion."),
    ("continuing_deleveraging", "Price reversal when both legs reduce open interest but net reduction slows during the recovery."),
    ("position_direction_transfer", "A reversal with net position growth concentrated on price moves aligned with the recovery leg."),
    ("position_churn_relief", "Recovery after absolute net open interest change per traded lot declines."),
    ("turnover_position_confirmation", "Recovery confirmed by rising turnover and a positive fraction of traded lots reflected in net position expansion."),
    ("vwap_path", "Two opposing legs in the amount-per-lot execution price proxy, using actual-contract minute changes."),
    ("vwap_recovery_failure", "Fade a close-to-close reversal when the execution price proxy moves against its recovery direction."),
    ("vwap_basis_reclaim", "Follow a price reversal when the close to VWAP log basis strengthens in the recovery direction from the pivot."),
    ("notional_diffusion", "Trade the recovery when notional is less concentrated across its minutes than across the original shock."),
    ("gap_to_body", "Follow a reversal when the older move is borne mainly by close-to-open gaps and recovery by open-to-close bodies."),
    ("distributed_recovery", "Follow a reversal sustained by multiple recovery minutes after capping their amplitudes at the shock-leg RMS."),
    ("recovery_after_stale", "Price recovery following more zero-volume observations in the shock leg and higher recovery-leg participation."),
    ("range_impact_relief", "Price reversal with lower aggregate high-low range per traded lot in the recovery leg."),
];

#[derive(Clone, Copy, PartialEq, Eq)]
enum Inputs {
    Price,
    Candle,
    Volume,
    CandleVolume,
    Turnover,
    Position,
    PositionVolume,
    Amount,
    Vwap,
}

impl Inputs {
    fn for_kind(kind: usize) -> Self {
        match kind {
            12..=15 | 32 => Self::Candle,
            7 | 18 | 19 | 34 => Self::Volume,
            20 | 21 | 35 => Self::CandleVolume,
            22 => Self::Turnover,
            23..=25 => Self::Position,
            26 | 27 => Self::PositionVolume,
            17 | 31 => Self::Amount,
            28..=30 => Self::Vwap,
            _ => Self::Price,
        }
    }

    fn fields(self) -> &'static [FutureMinuteField] {
        match self {
            Self::Price => &[C, LINK],
            Self::Candle => &[C, LINK, O, H, L],
            Self::Volume => &[C, LINK, V],
            Self::CandleVolume => &[C, LINK, O, H, L, V],
            Self::Turnover => &[C, LINK, V, I],
            Self::Position => &[C, LINK, D],
            Self::PositionVolume => &[C, LINK, D, V, I],
            Self::Amount => &[C, LINK, A],
            Self::Vwap => &[C, LINK, V, A],
        }
    }
}

#[derive(Clone, Copy)]
struct Reversal {
    kind: usize,
    variant: usize,
}

pub(super) fn create(number: usize) -> Box<dyn Factor> {
    let offset = number - FIRST_ID;
    Box::new(Reversal {
        kind: offset / WINDOWS.len(),
        variant: offset % WINDOWS.len(),
    })
}

impl Factor for Reversal {
    fn spec(&self) -> FactorSpec {
        let (alias, description) = FACTORS[self.kind];
        let (width, short) = WINDOWS[self.variant];
        let id = format!(
            "yq_{:03}",
            FIRST_ID + self.kind * WINDOWS.len() + self.variant
        );
        FactorSpec {
            name: id.clone(), id, aliases: vec![format!("rvcont_{alias}_{width}m")],
            asset_class: AssetClass::Future, frequency: Frequency::Minute1, version: "0.1.0".into(),
            tags: ["minute", "pv", "yq", "reversal_continuation"].map(str::to_string).to_vec(),
            description: format!("{description} The full window is {width} minutes and the fixed recent leg, where used, is {short} minutes."),
            dependencies: vec![future_minute_request(Inputs::for_kind(self.kind).fields())],
            intraday_raw_dependencies: Vec::new(), lookback: Lookback { trading_days: 0 },
        }
    }

    fn compute_provider_key(&self) -> String {
        format!(
            "future-reversal-continuation-{}",
            Inputs::for_kind(self.kind) as u8
        )
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
        let group = Inputs::for_kind(self.kind);
        let (specs, factors): (Vec<_>, Vec<_>) = ids
            .iter()
            .filter_map(|id| {
                id.strip_prefix("yq_")?
                    .parse::<usize>()
                    .ok()?
                    .checked_sub(FIRST_ID)
            })
            .filter(|&offset| offset < FACTORS.len() * WINDOWS.len())
            .map(|offset| Self {
                kind: offset / WINDOWS.len(),
                variant: offset % WINDOWS.len(),
            })
            .filter(|factor| Inputs::for_kind(factor.kind) == group)
            .map(|factor| (factor.spec(), factor))
            .unzip();
        if specs.is_empty() {
            return Ok(Vec::new());
        }
        compute_future_minute_many(&specs, context, data, group.fields(), 0, |inputs| {
            Market::new(group, inputs).calculate(&factors)
        })
    }
}

#[derive(Clone, Copy)]
struct Candle {
    high: f64,
    low: f64,
    open: f64,
    close: f64,
}

impl Candle {
    fn range(self) -> f64 {
        self.high - self.low
    }
    fn body(self) -> f64 {
        self.close - self.open
    }
    fn location(self) -> f64 {
        fraction(2.0 * (self.close - self.low) - self.range(), self.range())
    }
    fn wick(self) -> f64 {
        fraction(
            self.close.min(self.open) - self.low - (self.high - self.close.max(self.open)),
            self.range(),
        )
    }
}

struct Market<'a> {
    group: Inputs,
    inputs: &'a [Series],
    returns: Series,
    candles: Vec<Option<Candle>>,
    traded: Series,
}

impl<'a> Market<'a> {
    fn new(group: Inputs, inputs: &'a [Series]) -> Self {
        let returns = close_returns(&inputs[0], &inputs[1]);
        let candles = if matches!(group, Inputs::Candle | Inputs::CandleVolume) {
            (0..returns.len())
                .map(|i| {
                    let c = inputs[0][i]?;
                    let o = inputs[2][i]?;
                    let h = inputs[3][i]?;
                    let l = inputs[4][i]?;
                    (l > 0.0 && l <= o && l <= c && o <= h && c <= h).then(|| Candle {
                        high: h.ln(),
                        low: l.ln(),
                        open: o.ln(),
                        close: c.ln(),
                    })
                })
                .collect()
        } else {
            Vec::new()
        };
        let traded = if group == Inputs::Vwap {
            (0..returns.len())
                .map(|i| log_return(inputs[3][i]?, inputs[2][i]?))
                .collect()
        } else {
            Vec::new()
        };
        Self {
            group,
            inputs,
            returns,
            candles,
            traded,
        }
    }

    fn column(&self, field: FutureMinuteField) -> &Values {
        let idx = self
            .group
            .fields()
            .iter()
            .position(|f| match (f, field) {
                (ContractLevel(a), ContractLevel(b)) | (ContractDiff(a), ContractDiff(b)) => {
                    *a == b
                }
                _ => false,
            })
            .expect("declared reversal input");
        &self.inputs[idx]
    }

    fn volumes(&self, i: usize) -> Option<f64> {
        let v = self.column(V)[i]?;
        (v >= 0.0).then_some(v)
    }

    fn calculate(&self, factors: &[Reversal]) -> Vec<Series> {
        let len = self.returns.len();
        let mut result = vec![vec![None; len]; factors.len()];
        for (variant, (n, k)) in WINDOWS.into_iter().enumerate() {
            let selected = factors
                .iter()
                .enumerate()
                .filter(|(_, f)| f.variant == variant)
                .collect::<Vec<_>>();
            if selected.is_empty() {
                continue;
            }
            for t in n..len {
                let Some(w) = Window::new(&self.returns[t + 1 - n..=t], k) else {
                    continue;
                };
                let start = t + 1 - n;
                for &(output, factor) in &selected {
                    result[output][t] = self.eval(factor.kind, start, &w).filter(|x| x.is_finite());
                }
            }
        }
        result
    }

    fn eval(&self, kind: usize, start: usize, w: &Window) -> Option<f64> {
        let n = w.n;
        let end = start + n;
        let pivot = start + w.split;
        let sum = |f: &dyn Fn(usize) -> Option<f64>, from, to| {
            (from..to).try_fold(0.0, |sum, i| Some(sum + f(i)?))
        };
        let averages = |f: &dyn Fn(usize) -> Option<f64>| {
            Some((
                sum(f, start, pivot)? / w.split as f64,
                sum(f, pivot, end)? / w.short() as f64,
            ))
        };

        // These shapes either choose their own turning point or compare whole paths.
        match kind {
            0 => {
                let full = w.path[n] / n as f64;
                let right = w.right / w.short() as f64;
                return Some(against(full, right, w.rms));
            }
            2 => {
                let breadth = |values: &[f64]| {
                    values.iter().map(|x| sign(*x)).sum::<f64>() / values.len() as f64
                };
                return Some(
                    w.direction
                        * (-w.direction * breadth(w.before())).max(0.0)
                        * (w.direction * breadth(w.after())).max(0.0),
                );
            }
            3 => return Some(adaptive(w)),
            4 => return Some(hinge(w)),
            7 | 9 => {
                let mut series = w.returns;
                if kind == 7 {
                    for (j, r) in series[..n].iter_mut().enumerate() {
                        *r *= self.volumes(start + j)?;
                    }
                }
                return Some(terminal_runs(&series[..n], kind == 7));
            }
            10 => return Some(double_test(w)),
            11 => return Some(failed_recovery(w)),
            28 => {
                let mut changes = [None; 30];
                for (j, i) in (start..end).enumerate() {
                    changes[j] = Some(self.traded[i]? - self.traded[i - 1]?);
                }
                return Some(Window::new(&changes[..n], w.short())?.base());
            }
            _ => {}
        }

        let base = w.base();
        if base == 0.0 {
            return Some(0.0);
        }
        let d = w.direction;
        let shock = w.left.abs();
        let recovery = w.right.abs();
        match kind {
            1 => Some(base),
            5 => {
                let previous = d * (w.path[n - 1] - w.left);
                let level = shock / 2.0;
                Some(if previous < level && recovery >= level {
                    base
                } else {
                    0.0
                })
            }
            6 => {
                let pace = recovery / w.short() as f64;
                let latest = d * w.returns[n - 1];
                Some(
                    -base
                        * positive(balance(recovery, shock))
                        * positive((pace - latest) / (pace + latest.abs())),
                )
            }
            8 => {
                let (j, jump) = w
                    .before()
                    .iter()
                    .copied()
                    .enumerate()
                    .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))?;
                let post = w.path[n] - w.path[j + 1];
                let concentration = jump.abs() / w.before().iter().map(|v| v.abs()).sum::<f64>();
                Some(if sign(jump) == -d && sign(post) == d {
                    against(jump, post, w.scale()) * concentration
                } else {
                    0.0
                })
            }
            12 => {
                let wick = self.candles[pivot - 1]?.wick();
                Some(base * positive(d * wick))
            }
            13 => {
                let (left, right) = averages(&|i| Some(self.candles[i]?.location()))?;
                Some(base * positive(d * (right - left) / 2.0))
            }
            14 => {
                let anchor = self.candles[start - 1]?;
                let turn = self.candles[pivot - 1]?.close;
                let latest = self.candles[end - 1]?.close;
                let previous = self.candles[end - 2]?.close;
                let boundary = if d > 0.0 { anchor.low } else { anchor.high };
                Some(
                    if d * (turn - boundary) < 0.0
                        && d * (previous - boundary) < 0.0
                        && d * (latest - boundary) >= 0.0
                    {
                        base
                    } else {
                        0.0
                    },
                )
            }
            15 => {
                let (left, right) = averages(&|i| Some(self.candles[i]?.range()))?;
                Some(base * positive(balance(left, right)))
            }
            16 => {
                let quality = |r: &[f64], direction| {
                    let net = r.iter().map(|x| sign(*x) * x * x).sum::<f64>();
                    let power = r.iter().map(|x| x * x).sum::<f64>();
                    positive(fraction(direction * net, power))
                };
                Some(base * quality(w.before(), -d) * quality(w.after(), d))
            }
            17 => {
                let amount = self.column(A);
                let impact = |from, to, returns: &[f64]| {
                    let v = sum(&|i| nonnegative(amount[i]?), from, to)?;
                    (v > 0.0).then(|| returns.iter().map(|r| r.abs()).sum::<f64>() / v)
                };
                let left = impact(start, pivot, w.before())?;
                let right = impact(pivot, end, w.after())?;
                Some(base * positive(balance(left, right)))
            }
            18 => {
                let (left, right) = averages(&|i| self.volumes(i))?;
                Some(base * positive(balance(left, right)) * positive(balance(shock, recovery)))
            }
            19 => {
                let mut max = 0.0;
                let mut max_index = start;
                let mut total = 0.0;
                for i in start..end {
                    let volume = self.volumes(i)?;
                    total += volume;
                    if volume > max {
                        max = volume;
                        max_index = i;
                    }
                }
                let distance = max_index.abs_diff(pivot - 1);
                let locality = 1.0 - (distance as f64 / w.short() as f64).min(1.0);
                Some(
                    base * locality
                        * positive((fraction(max, total) * n as f64 - 1.0) / (n - 1) as f64),
                )
            }
            20 => {
                let flow = |from, to| {
                    let net = sum(
                        &|i| Some(self.candles[i]?.location() * self.volumes(i)?),
                        from,
                        to,
                    )?;
                    let volume = sum(&|i| self.volumes(i), from, to)?;
                    (volume > 0.0).then(|| net / volume)
                };
                let left = flow(start, pivot)?;
                let right = flow(pivot, end)?;
                Some(base * positive(-d * left).min(positive(d * right)))
            }
            21 => {
                let volume = self.volumes(pivot - 1)?;
                let others = sum(&|i| self.volumes(i), start, pivot - 1)? / (w.split - 1) as f64;
                let candle = self.candles[pivot - 1]?;
                let absorption = 1.0 - fraction(candle.body().abs(), candle.range());
                Some(base * positive(balance(volume, others)) * absorption)
            }
            22 => {
                let oi = self.column(I);
                let turnover = |i: usize| {
                    let prior = oi[i - 1]?;
                    (prior > 0.0).then_some(())?;
                    Some(self.volumes(i)? / prior)
                };
                let peak = turnover(pivot - 1)?;
                let older = sum(&turnover, start, pivot - 1)? / (w.split - 1) as f64;
                let newer = sum(&turnover, pivot, end)? / w.short() as f64;
                Some(base * positive(balance(peak, older.max(newer))))
            }
            23 | 24 => {
                let flow = self.column(D);
                let (left, right) = averages(&|i| flow[i])?;
                if left >= 0.0 {
                    return Some(0.0);
                }
                let strength = if kind == 23 {
                    if right > 0.0 {
                        2.0 * right.min(-left) / (right - left)
                    } else {
                        0.0
                    }
                } else if right < 0.0 {
                    positive(balance(-left, -right))
                } else {
                    0.0
                };
                Some(base * strength)
            }
            25 => {
                let change = self.column(D);
                let weighted = sum(
                    &|i| Some(d * sign(self.returns[i]?) * positive(change[i]?)),
                    pivot,
                    end,
                )?;
                let total = sum(&|i| Some(positive(change[i]?)), pivot, end)?;
                Some(base * positive(fraction(weighted, total)))
            }
            26 => {
                let delta = self.column(D);
                let absolute_share = |from, to| {
                    let v = sum(&|i| self.volumes(i), from, to)?;
                    (v > 0.0).then_some(())?;
                    Some(sum(&|i| Some(delta[i]?.abs()), from, to)? / v)
                };
                Some(
                    base * positive(balance(
                        absolute_share(start, pivot)?,
                        absolute_share(pivot, end)?,
                    )),
                )
            }
            27 => {
                let oi = self.column(I);
                let delta = self.column(D);
                let (left, right) = averages(&|i| {
                    let outstanding = oi[i - 1]?;
                    (outstanding > 0.0).then_some(())?;
                    Some(self.volumes(i)? / outstanding)
                })?;
                let traded = sum(&|i| self.volumes(i), pivot, end)?;
                let added = sum(&|i| delta[i], pivot, end)?.max(0.0);
                (traded > 0.0).then_some(())?;
                Some(base * positive(balance(right, left)) * fraction(added, added + traded))
            }
            29 => {
                let traded = self.traded[end - 1]? - self.traded[pivot - 1]?;
                Some(-base * strength(positive(-d * traded), w.scale()))
            }
            30 => {
                let basis = |i: usize| Some(log_return(self.inputs[0][i]?, 1.0)? - self.traded[i]?);
                let movement = d * (basis(end - 1)? - basis(pivot - 1)?);
                Some(base * strength(positive(movement), w.scale()))
            }
            31 => {
                let amount = self.column(A);
                let hhi = |from, to| {
                    let total = sum(&|i| nonnegative(amount[i]?), from, to)?;
                    let sq = sum(&|i| Some(nonnegative(amount[i]?)?.powi(2)), from, to)?;
                    (total > 0.0).then_some(())?;
                    let length = (to - from) as f64;
                    Some(((length * sq / (total * total) - 1.0) / (length - 1.0)).clamp(0.0, 1.0))
                };
                Some(base * positive(hhi(start, pivot)? - hhi(pivot, end)?))
            }
            32 => {
                let gap = |i: usize| Some(self.candles[i]?.open - self.candles[i - 1]?.close);
                let old_gap = positive(-d * sum(&gap, start, pivot)?);
                let new_body = positive(d * sum(&|i| Some(self.candles[i]?.body()), pivot, end)?);
                let old_body =
                    positive(-d * sum(&|i| Some(self.candles[i]?.body()), start, pivot)?);
                let new_gap = positive(d * sum(&gap, pivot, end)?);
                Some(
                    base * positive(balance(old_gap, old_body))
                        * positive(balance(new_body, new_gap)),
                )
            }
            33 => {
                let rms = (w.before().iter().map(|r| r * r).sum::<f64>() / w.split as f64).sqrt();
                let cap = rms;
                let clipped =
                    positive(d * w.after().iter().map(|r| r.clamp(-cap, cap)).sum::<f64>());
                let breadth = w.after().iter().filter(|&&x| sign(x) == d).count();
                Some(
                    base * fraction(clipped.min(recovery), recovery)
                        * breadth.saturating_sub(1) as f64
                        / (w.short() - 1) as f64,
                )
            }
            34 => {
                let (left, right) =
                    averages(&|i| Some(if self.volumes(i)? == 0.0 { 1.0 } else { 0.0 }))?;
                let (older, newer) = averages(&|i| self.volumes(i))?;
                Some(base * positive(left - right) * positive(balance(newer, older)))
            }
            35 => {
                let friction = |from, to| {
                    let traded = sum(&|i| self.volumes(i), from, to)?;
                    (traded > 0.0).then_some(())?;
                    Some(sum(&|i| Some(self.candles[i]?.range()), from, to)? / traded)
                };
                Some(base * positive(balance(friction(start, pivot)?, friction(pivot, end)?)))
            }
            _ => unreachable!("unknown reversal factor"),
        }
    }
}

struct Window {
    n: usize,
    split: usize,
    path: [f64; 31],
    returns: [f64; 30],
    rms: f64,
    left: f64,
    right: f64,
    direction: f64,
}

impl Window {
    fn new(values: &Values, short: usize) -> Option<Self> {
        let n = values.len();
        let split = n - short;
        let mut path = [0.0; 31];
        let mut returns = [0.0; 30];
        let mut energy = 0.0;
        for (j, &value) in values.iter().enumerate() {
            let r = value?;
            returns[j] = r;
            path[j + 1] = path[j] + r;
            energy += r * r;
        }
        let left = path[split];
        let right = path[n] - left;
        Some(Self {
            n,
            split,
            path,
            returns,
            rms: (energy / n as f64).sqrt(),
            left,
            right,
            direction: if left * right < 0.0 { sign(right) } else { 0.0 },
        })
    }

    fn short(&self) -> usize {
        self.n - self.split
    }
    fn before(&self) -> &[f64] {
        &self.returns[..self.split]
    }
    fn after(&self) -> &[f64] {
        &self.returns[self.split..self.n]
    }
    fn scale(&self) -> f64 {
        self.rms * (self.n as f64).sqrt()
    }
    fn base(&self) -> f64 {
        against(self.left, self.right, self.scale())
    }
}

fn sign(x: f64) -> f64 {
    if x == 0.0 {
        0.0
    } else {
        x.signum()
    }
}
fn positive(x: f64) -> f64 {
    x.max(0.0)
}
fn nonnegative(x: f64) -> Option<f64> {
    (x >= 0.0).then_some(x)
}
fn fraction(x: f64, base: f64) -> f64 {
    if base > 0.0 {
        x / base
    } else {
        0.0
    }
}
fn balance(a: f64, b: f64) -> f64 {
    fraction(a - b, a + b)
}
fn strength(x: f64, scale: f64) -> f64 {
    fraction(x, x + scale)
}
fn against(before: f64, after: f64, scale: f64) -> f64 {
    if before * after < 0.0 {
        sign(after) * strength(before.abs().min(after.abs()), scale)
    } else {
        0.0
    }
}

fn adaptive(w: &Window) -> f64 {
    let end = w.path[w.n];
    let side = |s: f64| {
        let matched = w.path[1..w.n]
            .iter()
            .map(|&pivot| positive((-s * pivot).min(s * (end - pivot))))
            .fold(0.0, f64::max);
        strength(matched, w.scale())
    };
    side(1.0) - side(-1.0)
}

fn hinge(w: &Window) -> f64 {
    let y = &w.path[..=w.n];
    let count = y.len() as f64;
    let x_mean = w.n as f64 / 2.0;
    let h_mean = (w.short() * (w.short() + 1)) as f64 / (2.0 * count);
    let y_mean = y.iter().sum::<f64>() / count;
    let (mut xx, mut hh, mut xh, mut xy, mut hy, mut yy) = (0., 0., 0., 0., 0., 0.);
    for (j, &value) in y.iter().enumerate() {
        let x = j as f64 - x_mean;
        let h = j.saturating_sub(w.split) as f64 - h_mean;
        let y = value - y_mean;
        xx += x * x;
        hh += h * h;
        xh += x * h;
        xy += x * y;
        hy += h * y;
        yy += y * y;
    }
    let determinant = xx * hh - xh * xh;
    if determinant <= 0.0 || yy <= 0.0 {
        return 0.0;
    }
    let a = (hh * xy - xh * hy) / determinant;
    let extra = (xx * hy - xh * xy) / determinant;
    let b = a + extra;
    if a * b >= 0.0 {
        return 0.0;
    }
    let gain = ((a * xy + extra * hy - xy * xy / xx) / yy)
        .clamp(0.0, 1.0)
        .sqrt();
    sign(b) * gain
}

fn terminal_runs(r: &[f64], residual: bool) -> f64 {
    let d = sign(r[r.len() - 1]);
    if d == 0.0 {
        return 0.0;
    }
    let end = r.iter().rev().take_while(|&&x| sign(x) == d).count();
    let split = r.len() - end;
    let before = r[..split]
        .iter()
        .rev()
        .take_while(|&&x| sign(x) == -d)
        .count();
    if before == 0 {
        return 0.0;
    }
    let a = -d * r[split - before..split].iter().sum::<f64>();
    let b = d * r[split..].iter().sum::<f64>();
    let left = a / before as f64;
    let right = b / end as f64;
    let result = if residual {
        positive(balance(a, b)) * fraction(b, a + b)
    } else {
        positive(balance(before as f64, end as f64)) * positive(balance(right, left))
    };
    d * result
}

fn double_test(w: &Window) -> f64 {
    let [p, q, r, last] = [
        w.path[w.n / 4],
        w.path[w.n / 2],
        w.path[3 * w.n / 4],
        w.path[w.n],
    ];
    let d = sign(last - r);
    if d * p >= 0.0
        || d * (q - p) <= 0.0
        || d * (r - q) >= 0.0
        || d * (r - p) <= 0.0
        || d * (last - q) <= 0.0
    {
        return 0.0;
    }
    let reclaim = d * (last - q);
    let structure = d * (r - p);
    d * strength(reclaim, w.scale()) * strength(structure, w.scale())
}

fn failed_recovery(w: &Window) -> f64 {
    let mid = w.path[w.n / 3];
    let turn = w.path[2 * w.n / 3];
    let original = sign(mid);
    let bounce = -original * (turn - mid);
    let failure = original * (w.path[w.n] - turn);
    if bounce <= 0.0 || bounce >= mid.abs() || failure < bounce / 2.0 {
        return 0.0;
    }
    original * strength(failure, w.scale()) * strength(bounce, w.scale())
}

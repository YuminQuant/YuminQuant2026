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

const FIRST_ID: usize = 475;
const WINDOWS: [usize; 3] = [30, 60, 120];
const RECENT: usize = 5;
const EXTRA_HISTORY: usize = 20;
const C: FutureMinuteField = ContractLevel("close");
const O: FutureMinuteField = ContractLevel("open");
const H: FutureMinuteField = ContractLevel("high");
const L: FutureMinuteField = ContractLevel("low");
const V: FutureMinuteField = ContractLevel("vol");
const I: FutureMinuteField = ContractLevel("oi");
const D: FutureMinuteField = ContractDiff("oi");
const A: FutureMinuteField = ContractLevel("amount");
const LINK: FutureMinuteField = ContinuousMinute;
const PRICE: &[FutureMinuteField] = &[C, LINK];
const OHLC: &[FutureMinuteField] = &[C, LINK, O, H, L];
const VOLUME: &[FutureMinuteField] = &[C, LINK, O, V];
const CANDLE_VOLUME: &[FutureMinuteField] = &[C, LINK, O, H, L, V];
const TURNOVER: &[FutureMinuteField] = &[C, LINK, V, I];
const POSITION: &[FutureMinuteField] = &[C, LINK, D, I, O];
const POSITION_VOLUME: &[FutureMinuteField] = &[C, LINK, D, I, V];
const NOTIONAL: &[FutureMinuteField] = &[C, LINK, A, O];
const TRADED: &[FutureMinuteField] = &[C, LINK, A, V];

const FACTORS: [(&str, &str); 30] = [
    ("return_sign", "Five-bar return direction follows clustered nonnegative/negative runs and reverses when those states alternate unusually often."),
    ("body_vs_close", "Candle direction weighted by the clustering advantage of open-to-close signs over close-to-close signs."),
    ("large_return", "A large signed return is confirmed or faded by the historical clustering or alternation of large versus ordinary absolute returns."),
    ("clustering_transition", "Five-bar price direction weighted by the change in return-sign clustering between two disjoint historical halves."),
    ("terminal_streak", "Signed terminal price streak weighted by the conditional probability of observing its repeats under random ordering."),
    ("range_half_occupancy", "Close displacement from the midpoint of the preceding five-bar range follows clustered occupancy of its upper or lower half and reverses under alternation."),
    ("gap_fill", "Follow unfilled minute opening gaps or fade filled gaps when gap-filling events show historical clustering."),
    ("range_expansion_switch", "Candle direction on a switch from below-normal to above-normal minute range, weighted by historical range-state clustering."),
    ("wick_polarity", "Lower-versus-upper wick imbalance is confirmed or reversed by the runs of wick dominance."),
    ("close_acceptance", "Directional body confirmed when closes are accepted near the same side of the candle range in clustered episodes."),
    ("extreme_renewal", "Upper and lower five-bar extreme renewals each have their own event runs and weight the signed breakout magnitude."),
    ("volatility_acceleration", "Five-bar price direction on rising realized volatility, conditioned on runs of expansion and contraction across five-minute blocks."),
    ("candle_overlap", "Fade the candle direction when its overlap with the preceding candle is large and large overlap clusters."),
    ("volume_change", "Candle direction on a rise in minute volume, confirmed or faded by the runs of volume increases and decreases."),
    ("volume_absorption", "Fade recent drift during clustered high-volume small-body events."),
    ("turnover", "Recent price direction conditioned on runs of high and low turnover per preceding outstanding lot."),
    ("position_sign", "Recent price direction is confirmed on net position expansion and faded on contraction, weighted by directional position runs."),
    ("inventory_intensity", "Direction of price and net position change weighted by clustered high absolute net open interest change per traded lot."),
    ("price_position_concordance", "Recent price direction weighted by the persistence of matching minute return and net position signs."),
    ("liquidation_reversal", "Candle direction on a reversal of the preceding price move during net position contraction and clustered liquidation reversals."),
    ("amihud_recovery", "Candle direction on improving return impact per unit of turnover in a clustered Amihud impact regime."),
    ("vwap_basis", "Current close-versus-VWAP basis displacement confirmed or faded by runs of positive and negative displacements."),
    ("traded_price_disagreement", "Follow the direction of the execution-price proxy during clustered disagreement with close-to-close price direction."),
    ("notional_concentration", "Fade recent price direction when concentration of traded amount within a five-minute block occurs in clustered episodes."),
    ("range_per_lot", "Wick rejection on unusually high intrabar range per traded lot and historical clustering of high-impact states."),
    ("pair_alignment", "Disjoint two-return pairs form aligned or opposed states. Pair direction follows unusual alignment clustering and reverses under alternation."),
    ("restart_after_stall", "Fade the first price impulse after a zero-volume minute when no-trade minutes cluster."),
    ("joint_volume_range_expansion", "Candle direction on simultaneous above-baseline range and volume, weighted by clustering of their joint expansions."),
    ("failed_breakout", "Upward and downward failed breakouts have separate event runs. Fade the broken boundary when failures cluster."),
    ("directional_impact", "Fade an above-baseline Amihud shock weighted by the clustering of unusually high price impact on its own side."),
];

fn fields(kind: usize) -> &'static [FutureMinuteField] {
    match kind {
        1 | 5..=10 | 12 | 28 => OHLC,
        13 | 26 => VOLUME,
        14 | 24 | 27 => CANDLE_VOLUME,
        15 => TURNOVER,
        16 | 18 | 19 => POSITION,
        17 => POSITION_VOLUME,
        20 | 23 | 29 => NOTIONAL,
        21 | 22 => TRADED,
        _ => PRICE,
    }
}

#[derive(Clone, Copy)]
struct RunsFactor {
    kind: usize,
    variant: usize,
}

pub(super) fn create(id: usize) -> Box<dyn Factor> {
    let offset = id - FIRST_ID;
    Box::new(RunsFactor {
        kind: offset / WINDOWS.len(),
        variant: offset % WINDOWS.len(),
    })
}

impl Factor for RunsFactor {
    fn spec(&self) -> FactorSpec {
        let (alias, description) = FACTORS[self.kind];
        let width = WINDOWS[self.variant];
        let id = format!(
            "yq_{:03}",
            FIRST_ID + self.kind * WINDOWS.len() + self.variant
        );
        FactorSpec {
            name: id.clone(),
            id,
            aliases: vec![format!("runs_{alias}_{width}m")],
            asset_class: AssetClass::Future,
            frequency: Frequency::Minute1,
            version: "0.1.0".into(),
            tags: ["minute", "pv", "yq", "runs", "timing"]
                .map(str::to_string)
                .to_vec(),
            description: format!(
                "{description} The historical window spans {width} one-minute bars."
            ),
            dependencies: vec![future_minute_request(fields(self.kind))],
            intraday_raw_dependencies: Vec::new(),
            lookback: Lookback { trading_days: 3 },
        }
    }

    fn compute_provider_key(&self) -> String {
        format!("future-minute-runs-{}", self.kind)
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
        let (specs, widths): (Vec<_>, Vec<_>) = WINDOWS
            .iter()
            .enumerate()
            .filter_map(|(variant, &width)| {
                let spec = Self { variant, ..*self }.spec();
                ids.contains(&spec.id).then_some((spec, width))
            })
            .unzip();
        let Some(&max_width) = widths.last() else {
            return Ok(Vec::new());
        };
        compute_future_minute_many(
            &specs,
            context,
            data,
            fields(self.kind),
            max_width + EXTRA_HISTORY,
            |inputs| {
                Market {
                    kind: self.kind,
                    inputs,
                    returns: close_returns(&inputs[0], &inputs[1]),
                }
                .compute(&widths)
            },
        )
    }
}

type Bits = Vec<Option<bool>>;

#[derive(Clone, Copy, Default)]
struct Counts {
    states: [usize; 2],
    edges: usize,
    switches: usize,
    wedges: usize,
}

struct RunWindow {
    span: usize,
    prefix: Vec<Counts>,
}

impl RunWindow {
    fn new(bits: &[Option<bool>], span: usize) -> Self {
        let mut prefix = Vec::<Counts>::with_capacity(bits.len());
        for (i, bit) in bits.iter().enumerate() {
            // Separate prefix counts for each phase of block-sampled states.
            let mut counts = i.checked_sub(span).map(|j| prefix[j]).unwrap_or_default();
            if let Some(current) = bit {
                counts.states[usize::from(*current)] += 1;
                if let Some(prior) = i.checked_sub(span).and_then(|j| bits[j]) {
                    counts.edges += 1;
                    counts.switches += usize::from(prior != *current);
                    counts.wedges +=
                        usize::from(i.checked_sub(2 * span).and_then(|j| bits[j]).is_some());
                }
            }
            prefix.push(counts);
        }
        Self { span, prefix }
    }

    fn counts(&self, width: usize, last: usize) -> Option<Counts> {
        let before = (last + self.span).checked_sub(width)?;
        let start = before.checked_sub(self.span);
        let earlier = start.map(|i| self.prefix[i]).unwrap_or_default();
        let current = self.prefix[last];
        let n = current.states.iter().sum::<usize>() - earlier.states.iter().sum::<usize>();
        if n < 4 || 5 * n < 3 * (width / self.span) {
            return None;
        }
        let excluded_edge = self.prefix[before];
        let excluded_wedge = self.prefix[before + self.span];
        Some(Counts {
            states: [
                current.states[0] - earlier.states[0],
                current.states[1] - earlier.states[1],
            ],
            edges: current.edges - excluded_edge.edges,
            switches: current.switches - excluded_edge.switches,
            wedges: current.wedges - excluded_wedge.wedges,
        })
    }

    fn score(&self, width: usize, last: usize) -> Option<f64> {
        let Counts {
            states: [a, b],
            edges,
            switches,
            wedges,
        } = self.counts(width, last)?;
        if edges < 3 {
            return None;
        }
        if a == 0 || b == 0 {
            return Some(0.0);
        }
        let n = (a + b) as f64;
        let (a, b, m, w) = (a as f64, b as f64, edges as f64, wedges as f64);
        // Conditional permutation moments on the observed adjacency graph.
        // Two edges either share a vertex or have four distinct endpoints.
        let p = 2.0 * a * b / (n * (n - 1.0));
        let joint = 4.0 * a * (a - 1.0) * b * (b - 1.0) / (n * (n - 1.0) * (n - 2.0) * (n - 3.0));
        let disjoint = m * (m - 1.0) / 2.0 - w;
        let var = m * p * (1.0 - p) + 2.0 * (w * (p / 2.0 - p * p) + disjoint * (joint - p * p));
        if var <= 0.0 {
            return Some(0.0);
        }
        Some(((m * p - switches as f64) / (2.0 * var.sqrt())).tanh())
    }
}

struct Market<'a> {
    kind: usize,
    inputs: &'a [Series],
    returns: Series,
}

impl Market<'_> {
    fn linked(&self, i: usize) -> bool {
        self.inputs[1][i] == Some(1.0)
    }

    fn build(&self, value: impl Fn(usize) -> Option<f64>) -> Series {
        (0..self.returns.len())
            .map(|i| {
                if self.linked(i) {
                    value(i).filter(|x| x.is_finite())
                } else {
                    None
                }
            })
            .collect()
    }

    fn bits(&self, value: impl Fn(usize) -> Option<bool>) -> Bits {
        (0..self.returns.len())
            .map(|i| if self.linked(i) { value(i) } else { None })
            .collect()
    }

    fn column(&self, name: &str) -> &Values {
        let idx = fields(self.kind).iter().position(|field| match field {
            ContractLevel(col) => *col == name,
            ContractDiff("oi") => name == "delta_oi",
            _ => false,
        });
        &self.inputs[idx.expect("declared runs factor input")]
    }

    fn body(&self, i: usize) -> Option<f64> {
        log_return(self.inputs[0][i]?, self.column("open")[i]?)
    }

    fn candle(&self, i: usize) -> Option<Candle> {
        let c = self.inputs[0][i]?;
        let o = self.column("open")[i]?;
        let h = self.column("high")[i]?;
        let l = self.column("low")[i]?;
        (l > 0.0 && l <= c && l <= o && c <= h && o <= h).then_some(Candle { c, o, h, l })
    }

    fn drift(&self, i: usize) -> Option<f64> {
        let mean = prior_stat(&self.returns, i + 1, |x| x)?;
        let scale = rms(&self.returns, i + 1)?;
        if scale == 0.0 {
            Some(0.0)
        } else {
            ratio(mean, scale)
        }
    }

    fn volume(&self) -> Series {
        let v = self.column("vol");
        self.build(|i| nonnegative(v[i]?))
    }

    fn ranges(&self) -> Series {
        self.build(|i| {
            let bar = self.candle(i)?;
            log_return(bar.h, bar.l)
        })
    }

    fn price_impact(&self) -> Series {
        let value = self.column("amount");
        self.build(|i| ratio(self.returns[i]?.abs(), value[i]?))
    }

    fn price_bits(&self) -> Bits {
        self.bits(|i| Some(self.returns[i]? >= 0.0))
    }

    fn changes(&self, values: &Values) -> Series {
        self.build(|i| Some(values[i]? - values[i.checked_sub(1)?]?))
    }

    fn series(
        &self,
        widths: &[usize],
        signal: impl Fn(usize, usize, usize) -> Option<f64>,
    ) -> Vec<Series> {
        widths
            .iter()
            .map(|&width| self.build(|i| signal(i, width, i.checked_sub(RECENT)?)))
            .collect()
    }

    fn weighted(
        &self,
        widths: &[usize],
        bits: &Bits,
        span: usize,
        cluster_only: bool,
        direction: impl Fn(usize) -> Option<f64>,
    ) -> Vec<Series> {
        let stat = RunWindow::new(bits, span);
        self.series(widths, |i, n, last| {
            let weight = stat.score(n, last)?;
            Some(
                direction(i)?
                    * if cluster_only {
                        weight.max(0.0)
                    } else {
                        weight
                    },
            )
        })
    }

    fn sides(
        &self,
        widths: &[usize],
        up: &Bits,
        down: &Bits,
        legs: impl Fn(usize) -> Option<(f64, f64)>,
    ) -> Vec<Series> {
        let a = RunWindow::new(up, 1);
        let b = RunWindow::new(down, 1);
        self.series(widths, |i, n, last| {
            let (positive, negative) = legs(i)?;
            let leg = |x, score: &RunWindow| {
                if x == 0.0 {
                    Some(0.0)
                } else {
                    Some(x * score.score(n, last)?.max(0.0))
                }
            };
            Some(leg(positive, &a)? - leg(negative, &b)?)
        })
    }

    fn above_recent(&self, state: &Values) -> Bits {
        self.bits(|i| Some(relative(state, i)? > 0.0))
    }

    fn compute(&self, widths: &[usize]) -> Vec<Series> {
        let r = &self.returns;
        match self.kind {
            0 => self.weighted(widths, &self.price_bits(), 1, false, |i| self.drift(i)),
            1 => {
                let body = self.bits(|i| Some(self.body(i)? >= 0.0));
                let a = RunWindow::new(&body, 1);
                let b = RunWindow::new(&self.price_bits(), 1);
                self.series(widths, |i, n, last| {
                    Some(self.candle(i)?.body()? * (a.score(n, last)? - b.score(n, last)?) / 2.0)
                })
            }
            2 => {
                let excess = self.build(|i| balance(r[i]?.abs(), rms(r, i)?));
                let state = self.bits(|i| Some(excess[i]? > 0.0));
                self.weighted(widths, &state, 1, false, |i| {
                    Some(sign(r[i]?) * excess[i]?.max(0.0))
                })
            }
            3 => {
                let stat = RunWindow::new(&self.price_bits(), 1);
                self.series(widths, |i, n, last| {
                    let h = n / 2;
                    Some(
                        self.drift(i)?
                            * (stat.score(h, last)? - stat.score(h, last.checked_sub(h)?)?)
                            / 2.0,
                    )
                })
            }
            4 => {
                let state = self.bits(|i| {
                    let r = r[i]?;
                    (r != 0.0).then_some(r > 0.0)
                });
                let history = RunWindow::new(&state, 1);
                self.series(widths, |i, n, last| {
                    let now = state[i]?;
                    let streak = trailing(&r[..], i + 1)?
                        .iter()
                        .rev()
                        .take_while(|r| r.is_some_and(|x| x != 0.0 && (x > 0.0) == now))
                        .count();
                    if streak < 2 {
                        return Some(0.0);
                    }
                    let [down, up] = history.counts(n, last)?.states;
                    let side = [down, up][usize::from(now)];
                    if side == 0 || down + up == side {
                        return None;
                    }
                    let tail_chance = (1..streak).fold(1.0, |p, k| {
                        p * side.saturating_sub(k) as f64 / (down + up - k) as f64
                    });
                    Some(
                        sign(r[i]?) * (1.0 - tail_chance) * (streak - 1) as f64
                            / (RECENT - 1) as f64,
                    )
                })
            }
            5 => {
                let (high, low) = self.bounds();
                let position = self.build(|i| {
                    let c = self.inputs[0][i]?.ln();
                    let (h, l) = (high[i]?, low[i]?);
                    let width = h - l;
                    let displacement = 2.0 * (c - l) - width;
                    signed_scale(displacement, width)
                });
                let state = self.bits(|i| Some(position[i]? > 0.0));
                self.weighted(widths, &state, 1, false, |i| position[i])
            }
            6 => {
                let gap = self.build(|i| {
                    let p = self.inputs[0][i.checked_sub(1)?]?;
                    let bar = self.candle(i)?;
                    let gap = log_return(bar.o, p)?;
                    let range = log_return(bar.h.max(p), bar.l.min(p))?;
                    if range == 0.0 {
                        Some(0.0)
                    } else {
                        ratio(gap, range)
                    }
                });
                let filled = self.bits(|i| {
                    let gap = gap[i]?;
                    let previous = self.inputs[0][i.checked_sub(1)?]?;
                    let bar = self.candle(i)?;
                    Some((gap > 0.0 && bar.l <= previous) || (gap < 0.0 && bar.h >= previous))
                });
                self.weighted(widths, &filled, 1, true, |i| {
                    Some(if filled[i]? { -gap[i]? } else { gap[i]? })
                })
            }
            7 => {
                let ranges = self.ranges();
                let state = self.above_recent(&ranges);
                self.weighted(widths, &state, 1, true, |i| {
                    let switched = state[i]? && !state[i.checked_sub(1)?]?;
                    Some(if switched {
                        sign(self.body(i)?) * relative(&ranges, i)?.max(0.0)
                    } else {
                        0.0
                    })
                })
            }
            8 => {
                let wick = self.build(|i| self.candle(i)?.wick());
                let state = self.bits(|i| Some(wick[i]? > 0.0));
                self.weighted(widths, &state, 1, false, |i| wick[i])
            }
            9 => {
                let location = self.build(|i| self.candle(i)?.location());
                let state = self.bits(|i| Some(location[i]? >= 0.0));
                self.weighted(widths, &state, 1, true, |i| {
                    let body = self.candle(i)?.body()?;
                    let position = location[i]?;
                    Some(if body * position > 0.0 {
                        body * position.abs()
                    } else {
                        0.0
                    })
                })
            }
            10 | 28 => {
                let (high, low) = self.bounds();
                let strength = self.build(|i| Some(high[i]? - low[i]?));
                let failed = self.kind == 28;
                let up = self.build(|i| {
                    let bar = self.candle(i)?;
                    let ceiling = high[i]?;
                    let beyond = (bar.h.ln() - ceiling).max(0.0);
                    if failed && bar.c.ln() >= ceiling {
                        Some(0.0)
                    } else {
                        Some(beyond)
                    }
                });
                let down = self.build(|i| {
                    let bar = self.candle(i)?;
                    let floor = low[i]?;
                    let beyond = (floor - bar.l.ln()).max(0.0);
                    if failed && bar.c.ln() <= floor {
                        Some(0.0)
                    } else {
                        Some(beyond)
                    }
                });
                let a = self.bits(|i| Some(up[i]? > 0.0));
                let b = self.bits(|i| Some(down[i]? > 0.0));
                self.sides(widths, &a, &b, |i| {
                    let s = if failed { -1.0 } else { 1.0 };
                    Some((
                        s * signed_scale(up[i]?, strength[i]?)?,
                        s * signed_scale(down[i]?, strength[i]?)?,
                    ))
                })
            }
            11 => {
                let rv = self.build(|i| rms(r, i + 1));
                let acceleration = self.build(|i| balance(rv[i]?, rv[i.checked_sub(RECENT)?]?));
                let state = self.bits(|i| Some(acceleration[i]? > 0.0));
                self.weighted(widths, &state, RECENT, false, |i| {
                    Some(self.drift(i)? * acceleration[i]?.max(0.0))
                })
            }
            12 => {
                let overlap = self.build(|i| {
                    let now = self.candle(i)?;
                    let prev = self.candle(i.checked_sub(1)?)?;
                    let intersection = (now.h.min(prev.h) - now.l.max(prev.l)).max(0.0);
                    let union = now.h.max(prev.h) - now.l.min(prev.l);
                    ratio(intersection, union)
                });
                let state = self.bits(|i| Some(overlap[i]? > 0.5));
                self.weighted(widths, &state, 1, true, |i| {
                    Some(-self.candle(i)?.body()? * (2.0 * overlap[i]? - 1.0).max(0.0))
                })
            }
            13 => {
                let volume = self.volume();
                let change = self.build(|i| balance(volume[i]?, volume[i.checked_sub(1)?]?));
                let state = self.bits(|i| Some(change[i]? > 0.0));
                self.weighted(widths, &state, 1, false, |i| {
                    Some(sign(self.body(i)?) * change[i]?.max(0.0))
                })
            }
            14 => {
                let vol = self.volume();
                let event = self.bits(|i| {
                    Some(relative(&vol, i)? > 0.0 && self.candle(i)?.body()?.abs() < 0.25)
                });
                self.weighted(widths, &event, 1, true, |i| {
                    Some(if event[i]? {
                        -self.drift(i)? * relative(&vol, i)?
                    } else {
                        0.0
                    })
                })
            }
            15 => {
                let vol = self.volume();
                let oi = self.column("oi");
                let turnover = self.build(|i| ratio(vol[i]?, oi[i.checked_sub(1)?]?));
                let state = self.above_recent(&turnover);
                self.weighted(widths, &state, 1, false, |i| {
                    Some(self.drift(i)? * relative(&turnover, i)?)
                })
            }
            16 => {
                let change = self.column("delta_oi");
                let prior = self.column("oi");
                let flow = self.build(|i| ratio(change[i]?, prior[i.checked_sub(1)?]?));
                let state = self.bits(|i| Some(flow[i]? > 0.0));
                self.weighted(widths, &state, 1, true, |i| {
                    Some(self.drift(i)? * signed_scale(flow[i]?, prior_stat(&flow, i, f64::abs)?)?)
                })
            }
            17 => {
                let vol = self.volume();
                let change = self.column("delta_oi");
                let intensity = self.build(|i| ratio(change[i]?.abs(), vol[i]?));
                let state = self.above_recent(&intensity);
                self.weighted(widths, &state, 1, true, |i| {
                    Some(sign(change[i]?) * self.drift(i)? * relative(&intensity, i)?.max(0.0))
                })
            }
            18 => {
                let change = self.column("delta_oi");
                let agreement = self.build(|i| Some(sign(r[i]?) * sign(change[i]?)));
                let state = self.bits(|i| Some(agreement[i]? > 0.0));
                self.weighted(widths, &state, 1, true, |i| {
                    Some(self.drift(i)? * agreement[i]?)
                })
            }
            19 => {
                let change = self.column("delta_oi");
                let reversal = self.bits(|i| {
                    Some(
                        change[i]? < 0.0 && sign(self.body(i)?) * sign(r[i.checked_sub(1)?]?) < 0.0,
                    )
                });
                self.weighted(widths, &reversal, 1, true, |i| {
                    Some(if reversal[i]? {
                        sign(self.body(i)?)
                    } else {
                        0.0
                    })
                })
            }
            20 => {
                let impact = self.price_impact();
                let state = self.above_recent(&impact);
                self.weighted(widths, &state, 1, true, |i| {
                    Some(sign(self.body(i)?) * (-relative(&impact, i)?).max(0.0))
                })
            }
            21 | 22 => {
                let vol = self.volume();
                let amount = self.column("amount");
                // In price returns and centered log basis, the fixed lot multiplier cancels.
                let traded = self.build(|i| log_return(amount[i]?, vol[i]?));
                if self.kind == 21 {
                    let basis = self.build(|i| Some(self.inputs[0][i]?.ln() - traded[i]?));
                    let residual = self.build(|i| Some(basis[i]? - prior_stat(&basis, i, |x| x)?));
                    let state = self.bits(|i| Some(residual[i]? > 0.0));
                    let change = self.changes(&basis);
                    self.weighted(widths, &state, 1, false, |i| {
                        signed_scale(residual[i]?, prior_stat(&change, i, f64::abs)?)
                    })
                } else {
                    let traded_r = self.changes(&traded);
                    let state = self.bits(|i| Some(sign(traded_r[i]?) * sign(r[i]?) < 0.0));
                    self.weighted(widths, &state, 1, true, |i| {
                        Some(if state[i]? { sign(traded_r[i]?) } else { 0.0 })
                    })
                }
            }
            23 => {
                let amount = self.build(|i| nonnegative(self.column("amount")[i]?));
                let concentration = self.build(|i| {
                    let mean = prior_stat(&amount, i + 1, |x| x)?;
                    let power = prior_stat(&amount, i + 1, |x| x * x)?;
                    ratio(power, RECENT as f64 * mean * mean)
                });
                let state = self.bits(|i| Some(concentration[i]? > 0.4));
                self.weighted(widths, &state, RECENT, true, |i| {
                    Some(-self.drift(i)? * ((concentration[i]? - 0.4) / 0.6).max(0.0))
                })
            }
            24 => {
                let ranges = self.ranges();
                let volume = self.volume();
                let impact = self.build(|i| ratio(ranges[i]?, volume[i]?));
                let state = self.above_recent(&impact);
                self.weighted(widths, &state, 1, true, |i| {
                    Some(self.candle(i)?.wick()? * relative(&impact, i)?.max(0.0))
                })
            }
            25 => {
                let aligned = self.bits(|i| Some(sign(r[i]?) * sign(r[i.checked_sub(1)?]?) > 0.0));
                self.weighted(widths, &aligned, 2, false, |i| {
                    let current = r[i]?;
                    let previous = r[i.checked_sub(1)?]?;
                    Some(if aligned[i]? {
                        sign(current + previous)
                    } else {
                        0.0
                    })
                })
            }
            26 => {
                let v = self.volume();
                let halted = self.bits(|i| Some(v[i]? == 0.0));
                self.weighted(widths, &halted, 1, true, |i| {
                    Some(if !halted[i]? && halted[i.checked_sub(1)?]? {
                        -sign(self.body(i)?)
                    } else {
                        0.0
                    })
                })
            }
            27 => {
                let range = self.ranges();
                let vol = self.volume();
                let state =
                    self.bits(|i| Some(relative(&range, i)? > 0.0 && relative(&vol, i)? > 0.0));
                self.weighted(widths, &state, 1, true, |i| {
                    Some(if state[i]? {
                        self.candle(i)?.body()? * (relative(&range, i)? * relative(&vol, i)?).sqrt()
                    } else {
                        0.0
                    })
                })
            }
            29 => {
                let impact = self.price_impact();
                let shock = self.build(|i| Some(relative(&impact, i)?.max(0.0)));
                let up = self.bits(|i| Some(shock[i]? > 0.0 && r[i]? > 0.0));
                let down = self.bits(|i| Some(shock[i]? > 0.0 && r[i]? < 0.0));
                self.sides(widths, &down, &up, |i| {
                    Some((
                        if down[i]? { shock[i]? } else { 0.0 },
                        if up[i]? { shock[i]? } else { 0.0 },
                    ))
                })
            }
            _ => unreachable!("unknown runs factor"),
        }
    }

    fn bounds(&self) -> (Series, Series) {
        let high = self.build(|i| Some(self.candle(i)?.h.ln()));
        let low = self.build(|i| Some(self.candle(i)?.l.ln()));
        (
            self.build(|i| extremum(&high, i, f64::max)),
            self.build(|i| extremum(&low, i, f64::min)),
        )
    }
}

struct Candle {
    c: f64,
    o: f64,
    h: f64,
    l: f64,
}

impl Candle {
    fn body(&self) -> Option<f64> {
        ratio(self.c - self.o, self.h - self.l)
    }

    fn location(&self) -> Option<f64> {
        ratio(2.0 * (self.c - self.l) - (self.h - self.l), self.h - self.l)
    }

    fn wick(&self) -> Option<f64> {
        ratio(
            self.c.min(self.o) - self.l - (self.h - self.c.max(self.o)),
            self.h - self.l,
        )
    }
}

fn nonnegative(x: f64) -> Option<f64> {
    (x >= 0.0).then_some(x)
}

fn sign(x: f64) -> f64 {
    if x == 0.0 {
        0.0
    } else {
        x.signum()
    }
}

fn ratio(a: f64, b: f64) -> Option<f64> {
    (b > 0.0).then(|| a / b)
}

fn balance(a: f64, b: f64) -> Option<f64> {
    if a == 0.0 && b == 0.0 {
        Some(0.0)
    } else {
        ratio(a - b, a + b)
    }
}

fn signed_scale(x: f64, scale: f64) -> Option<f64> {
    if x == 0.0 {
        Some(0.0)
    } else {
        ratio(x, x.abs() + scale)
    }
}

fn trailing(values: &Values, end: usize) -> Option<&Values> {
    Some(&values[end.checked_sub(RECENT)?..end])
}

fn extremum(values: &Values, end: usize, f: impl Fn(f64, f64) -> f64) -> Option<f64> {
    let (first, rest) = trailing(values, end)?.split_first()?;
    rest.iter().try_fold((*first)?, |x, &y| Some(f(x, y?)))
}

fn prior_stat(values: &Values, end: usize, f: impl Fn(f64) -> f64) -> Option<f64> {
    let sum = trailing(values, end)?
        .iter()
        .try_fold(0.0, |sum, &x| Some(sum + f(x?)))?;
    Some(sum / RECENT as f64)
}

fn rms(values: &Values, end: usize) -> Option<f64> {
    Some(prior_stat(values, end, |r| r * r)?.sqrt())
}

fn relative(values: &Values, i: usize) -> Option<f64> {
    balance(values[i]?, prior_stat(values, i, |x| x)?)
}

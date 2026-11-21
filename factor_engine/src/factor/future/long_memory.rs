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

const FIRST_ID: usize = 394;
const WINDOWS: [usize; 3] = [180, 240, 360];
const RECENT: usize = 5;
const HISTORY_DAYS: usize = 5;
// One extra bar for differencing, a five-bar feature and the five-bar holdout.
const WARMUP_MARGIN: usize = 2 * RECENT + 1;

const CLOSE: FutureMinuteField = ContractLevel("close");
const OPEN: FutureMinuteField = ContractLevel("open");
const HIGH: FutureMinuteField = ContractLevel("high");
const LOW: FutureMinuteField = ContractLevel("low");
const VOL: FutureMinuteField = ContractLevel("vol");
const OI: FutureMinuteField = ContractLevel("oi");
const OI_CHANGE: FutureMinuteField = ContractDiff("oi");
const AMOUNT: FutureMinuteField = ContractLevel("amount");

const PRICE: &[FutureMinuteField] = &[CLOSE, ContinuousMinute];
const CANDLE: &[FutureMinuteField] = &[CLOSE, ContinuousMinute, OPEN, HIGH, LOW];
const BODY: &[FutureMinuteField] = &[CLOSE, ContinuousMinute, OPEN];
const VOLUME: &[FutureMinuteField] = &[CLOSE, ContinuousMinute, OPEN, VOL];
const CANDLE_VOLUME: &[FutureMinuteField] = &[CLOSE, ContinuousMinute, OPEN, HIGH, LOW, VOL];
const TURNOVER: &[FutureMinuteField] = &[CLOSE, ContinuousMinute, VOL, OI];
const POSITION: &[FutureMinuteField] = &[CLOSE, ContinuousMinute, OI, OI_CHANGE];
const PARTICIPATION: &[FutureMinuteField] = &[CLOSE, ContinuousMinute, VOL, OI, OI_CHANGE];
const NOTIONAL: &[FutureMinuteField] = &[CLOSE, ContinuousMinute, OPEN, AMOUNT];
const EXECUTION_PRICE: &[FutureMinuteField] = &[CLOSE, ContinuousMinute, VOL, AMOUNT];

const FACTORS: [(&str, &str); 27] = [
    ("return_persistence", "Recent signed price displacement weighted by the bias-adjusted rescaled-range slope of minute returns."),
    ("return_sign_excess", "Directional breadth weighted by the excess persistence of return signs over return amplitudes."),
    ("semivariance_asymmetry", "Upside versus downside realized power, each weighted by the persistence of its own positive or negative semivariance series."),
    ("short_dependence_adjusted", "Current return relative to recent RMS, weighted by rescaled-range persistence using a two-lag Bartlett variance adjustment."),
    ("scaling_curvature", "Recent price drift weighted by the change in return scaling from small to large subwindows."),
    ("persistence_transition", "Recent price drift weighted by the change in the return scaling slope between two disjoint historical halves."),
    ("range_expansion_fade", "Fade the candle direction on unusual intrabar range expansion in a persistent high-low range regime."),
    ("volatility_release", "Candle direction confirmed by a decline between non-overlapping realized volatility observations when volatility is persistent."),
    ("true_range_gap_memory", "Signed opening gap as a fraction of log true range, extended or faded according to the scaling of normalized historical gaps."),
    ("body_efficiency", "Signed candle body as a share of range, weighted by its own scaling persistence or antipersistence."),
    ("wick_asymmetry", "Lower wick support versus upper wick rejection, each weighted by the persistence of its own normalized wick series."),
    ("close_location_innovation", "Current close location relative to its recent reference, weighted by the persistence of close location within the candle."),
    ("volume_innovation", "Candle direction on a positive participation surprise, extended or faded according to the scaling of minute log-volume innovations."),
    ("volume_absorption", "Fade recent price displacement when persistent heavy volume produces a small candle body relative to its range."),
    ("turnover_dry_up", "Fade recent price displacement when lots traded per prior outstanding lot fall in a persistent turnover regime."),
    ("volume_position_correlation", "Price direction weighted by the current volume and net-position correlation and its persistence across non-overlapping blocks."),
    ("position_expansion_contraction", "Follow price on net position expansion and fade it on contraction, weighted by each side's own persistence."),
    ("price_position_concordance", "Recent price direction conditioned on price and net-position concordance and the scaling slope of that concordance."),
    ("position_churn", "Fade recent price displacement when persistent intermediation coincides with low absolute net open interest change per traded lot."),
    ("amihud_liquidity_recovery", "Candle direction confirmed by a decline in return impact per unit of traded amount when that impact persists."),
    ("vwap_acceptance", "Recent displacement of the close-to-minute-VWAP log basis weighted by its persistence. The constant contract multiplier cancels."),
    ("vwap_return_leadership", "Recent VWAP displacement weighted by its scaling advantage over close-to-close returns."),
    ("notional_concentration_fade", "Fade recent drift when traded amount is concentrated in one minute and concentration persists across disjoint blocks."),
    ("range_per_lot_fragility", "Fade the candle direction when high-low movement per lot is elevated in a persistent range-impact regime."),
    ("two_bar_alignment", "Two-bar return direction conditioned on within-pair alignment and its persistence across disjoint pairs."),
    ("tail_robustness", "Recent price drift weighted by the scaling difference between locally clipped and raw returns."),
    ("extreme_extension_balance", "Upper versus lower five-bar extreme extensions, weighted by the persistence of breakouts on each side."),
];

fn fields(kind: usize) -> &'static [FutureMinuteField] {
    match kind {
        6 | 8..=11 | 26 => CANDLE,
        7 => BODY,
        12 => VOLUME,
        13 | 23 => CANDLE_VOLUME,
        14 => TURNOVER,
        15 | 18 => PARTICIPATION,
        16 | 17 => POSITION,
        19 | 22 => NOTIONAL,
        20 | 21 => EXECUTION_PRICE,
        _ => PRICE,
    }
}

#[derive(Clone, Copy)]
struct LongMemoryFactor {
    kind: usize,
    variant: usize,
}

pub(super) fn create(id: usize) -> Box<dyn Factor> {
    let offset = id - FIRST_ID;
    Box::new(LongMemoryFactor {
        kind: offset / WINDOWS.len(),
        variant: offset % WINDOWS.len(),
    })
}

impl Factor for LongMemoryFactor {
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
            aliases: vec![format!("hurst_{alias}_{width}m")],
            asset_class: AssetClass::Future,
            frequency: Frequency::Minute1,
            version: "0.1.0".into(),
            tags: ["minute", "pv", "yq", "hurst", "timing"]
                .map(str::to_string)
                .to_vec(),
            description: format!("{description} Scaling uses {width} one-minute bars."),
            dependencies: vec![future_minute_request(fields(self.kind))],
            intraday_raw_dependencies: Vec::new(),
            lookback: Lookback {
                trading_days: HISTORY_DAYS,
            },
        }
    }

    fn compute_provider_key(&self) -> String {
        format!("future-minute-long-memory-{}", self.kind)
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
        let columns = fields(self.kind);
        compute_future_minute_many(
            &specs,
            context,
            data,
            columns,
            max_width + WARMUP_MARGIN,
            |inputs| {
                let market = Market {
                    kind: self.kind,
                    inputs,
                    returns: close_returns(&inputs[0], &inputs[1]),
                };
                market.compute(&widths)
            },
        )
    }
}

#[derive(Clone, Copy)]
struct Scale {
    samples: usize,
    iid_range: f64,
}

const MINUTE_SCALES: [Scale; 3] = [
    Scale {
        samples: 5,
        iid_range: 1.927_362_733_094_497_3,
    },
    Scale {
        samples: 15,
        iid_range: 3.881_221_076_490_524_6,
    },
    Scale {
        samples: 30,
        iid_range: 5.844_259_003_414_316,
    },
];
const PAIR_SCALES: [Scale; 3] = [
    MINUTE_SCALES[0],
    Scale {
        samples: 10,
        iid_range: 3.023_331_086_565_937,
    },
    MINUTE_SCALES[1],
];
const FIVE_BAR_SCALES: [Scale; 3] = [
    Scale {
        samples: 3,
        iid_range: 1.350_474_474_235_659_1,
    },
    Scale {
        samples: 6,
        iid_range: 2.176_725_718_709_772,
    },
    Scale {
        samples: 12,
        iid_range: 3.386_375_345_292_063_5,
    },
];

fn rescaled_range(
    values: &Values,
    end: usize,
    scale: usize,
    stride: usize,
    hac: bool,
) -> Option<f64> {
    let start = (end + stride).checked_sub(scale * stride)?;
    let mut sample = [0.0; 30];
    let sample = &mut sample[..scale];
    let reference = values[start]?;
    let mut sum = 0.0;
    for (slot, &value) in sample
        .iter_mut()
        .zip(values[start..=end].iter().step_by(stride))
    {
        *slot = value? - reference;
        sum += *slot;
    }
    let mean = sum / scale as f64;
    let (mut path, mut low, mut high, mut energy) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0);
    for value in sample.iter_mut() {
        *value -= mean;
        path += *value;
        low = low.min(path);
        high = high.max(path);
        energy += value.powi(2);
    }
    if hac {
        // Bartlett weights are 2/3 at lag one and 1/3 at lag two.
        for lag in 1..=2 {
            let covariance = sample
                .iter()
                .zip(&sample[lag..])
                .map(|(a, b)| a * b)
                .sum::<f64>();
            energy += 2.0 * (1.0 - lag as f64 / 3.0) * covariance;
        }
    }
    (energy.is_finite() && energy > 0.0 && high > low)
        .then(|| (high - low) / (energy / scale as f64).sqrt())
}

struct Memory {
    blocks: [Series; 3],
    log_sizes: [f64; 3],
    scales: [Scale; 3],
    stride: usize,
}

impl Memory {
    fn new(source: &Values, stride: usize, hac: bool) -> Self {
        let scales = match stride {
            2 => PAIR_SCALES,
            RECENT => FIVE_BAR_SCALES,
            _ => MINUTE_SCALES,
        };
        let blocks = scales.map(|scale| {
            (0..source.len())
                .map(|end| rescaled_range(source, end, scale.samples, stride, hac))
                .collect()
        });
        Self {
            blocks,
            log_sizes: scales.map(|s| (s.samples as f64).ln()),
            scales,
            stride,
        }
    }

    fn scaling(&self, width: usize, end: usize) -> Option<[f64; 3]> {
        end.checked_sub(width)?;
        let mut normalized = [0.0; 3];
        for (i, (values, scale)) in self.blocks.iter().zip(self.scales).enumerate() {
            let span = scale.samples * self.stride;
            let blocks = width / span;
            let mut sum = 0.0;
            let mut valid = 0;
            for offset in (0..blocks).map(|j| j * span) {
                if let Some(value) = values[end - 1 - offset] {
                    sum += value;
                    valid += 1;
                }
            }
            if valid < 2 || 2 * valid < blocks {
                return None;
            }
            // Exact expected R/S for independent Gaussian samples with
            // population S. Removing it centers each scale's finite-sample bias.
            normalized[i] = (sum / valid as f64 / scale.iid_range).ln();
        }
        Some(normalized)
    }

    fn slope(&self, width: usize, end: usize) -> Option<f64> {
        let y = self.scaling(width, end)?;
        let x = self.log_sizes;
        let mean = x.iter().sum::<f64>() / x.len() as f64;
        let weighted = x.iter().zip(y).map(|(x, y)| (x - mean) * y).sum::<f64>();
        let variance = x.iter().map(|x| (x - mean).powi(2)).sum::<f64>();
        Some(weighted / variance)
    }

    fn score(&self, width: usize, end: usize) -> Option<f64> {
        Some((2.0 * self.slope(width, end)?).tanh())
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

    fn column(&self, name: &str) -> &Values {
        let index = fields(self.kind).iter().position(|field| match field {
            ContractLevel(column) => *column == name,
            ContractDiff("oi") => name == "delta_oi",
            _ => false,
        });
        &self.inputs[index.expect("factor input declared")]
    }

    fn body(&self, i: usize) -> Option<f64> {
        log_return(self.inputs[0][i]?, self.column("open")[i]?)
    }

    fn ohlc(&self, i: usize) -> Option<Candle> {
        let (c, o, h, l) = (
            self.inputs[0][i]?,
            self.column("open")[i]?,
            self.column("high")[i]?,
            self.column("low")[i]?,
        );
        (l > 0.0 && o >= l && c >= l && o <= h && c <= h).then_some(Candle { c, o, h, l })
    }

    fn flow(&self, i: usize) -> Option<f64> {
        let previous_oi = self.column("oi")[i.checked_sub(1)?]?;
        (previous_oi > 0.0).then_some(())?;
        Some(self.column("delta_oi")[i]? / previous_oi)
    }

    fn impulse(&self, i: usize) -> Option<f64> {
        impulse(&self.returns, i)
    }

    fn series(
        &self,
        widths: &[usize],
        signal: impl Fn(usize, usize, usize) -> Option<f64>,
    ) -> Vec<Series> {
        widths
            .iter()
            .map(|&width| {
                self.build(|i| {
                    let end = (i + 1).checked_sub(RECENT)?;
                    signal(i, width, end)
                })
            })
            .collect()
    }

    fn compute(&self, widths: &[usize]) -> Vec<Series> {
        let ret = &self.returns;
        match self.kind {
            0 => {
                let m = Memory::new(ret, 1, false);
                self.series(widths, |i, n, end| {
                    Some(self.impulse(i)? * m.score(n, end)?)
                })
            }
            1 => {
                let signs = self.build(|i| Some(sign(ret[i]?)));
                let a = Memory::new(&signs, 1, false);
                let b = Memory::new(ret, 1, false);
                self.series(widths, |i, n, end| {
                    Some(prior_mean(&signs, i + 1)? * (a.slope(n, end)? - b.slope(n, end)?).tanh())
                })
            }
            2 => {
                let up = self.build(|i| Some(ret[i]?.max(0.0).powi(2)));
                let down = self.build(|i| Some((-ret[i]?).max(0.0).powi(2)));
                let a = Memory::new(&up, 1, false);
                let b = Memory::new(&down, 1, false);
                self.series(widths, |i, n, end| {
                    let u = prior_mean(&up, i + 1)?;
                    let d = prior_mean(&down, i + 1)?;
                    if u + d == 0.0 {
                        return Some(0.0);
                    }
                    Some(
                        (u * positive(a.score(n, end)?) - d * positive(b.score(n, end)?)) / (u + d),
                    )
                })
            }
            3 => {
                let m = Memory::new(ret, 1, true);
                self.series(widths, |i, n, end| {
                    Some(return_surprise(ret, i)? * m.score(n, end)?)
                })
            }
            4 => {
                let m = Memory::new(ret, 1, false);
                self.series(widths, |i, n, end| {
                    let [a, b, c] = m.scaling(n, end)?;
                    let [x, y, z] = m.log_sizes;
                    Some(self.impulse(i)? * ((c - b) / (z - y) - (b - a) / (y - x)).tanh())
                })
            }
            5 => {
                let m = Memory::new(ret, 1, false);
                self.series(widths, |i, n, end| {
                    let half = n / 2;
                    let old_end = end.checked_sub(half)?;
                    let change = m.slope(half, end)? - m.slope(half, old_end)?;
                    Some(self.impulse(i)? * change.tanh())
                })
            }
            6 => {
                let range = self.build(|i| self.ohlc(i)?.log_range());
                self.weighted(widths, &range, 1, |i| {
                    Some(-sign(self.body(i)?) * positive(relative(&range, i)?))
                })
            }
            7 => {
                let rv = self.build(|i| Some(prior_power(ret, i + 1)?.sqrt()));
                self.weighted(widths, &rv, RECENT, |i| {
                    Some(
                        sign(self.body(i)?)
                            * positive(balance(rv[i.checked_sub(RECENT)?]?, rv[i]?)?),
                    )
                })
            }
            8 => {
                let gap = self.build(|i| {
                    let p = self.inputs[0][i.checked_sub(1)?]?;
                    let bar = self.ohlc(i)?;
                    let tr = log_return(bar.h.max(p), bar.l.min(p))?;
                    let opening_gap = log_return(bar.o, p)?;
                    if tr == 0.0 {
                        Some(0.0)
                    } else {
                        ratio(opening_gap, tr)
                    }
                });
                let m = Memory::new(&gap, 1, false);
                self.series(widths, |i, n, end| Some(gap[i]? * m.score(n, end)?))
            }
            9 => {
                let body = self.build(|i| self.ohlc(i)?.body_fraction());
                let m = Memory::new(&body, 1, false);
                self.series(widths, |i, n, end| Some(body[i]? * m.score(n, end)?))
            }
            10 => {
                let lower = self.build(|i| self.ohlc(i)?.wick(false));
                let upper = self.build(|i| self.ohlc(i)?.wick(true));
                self.two_sided(widths, &lower, &upper, |i| Some((lower[i]?, upper[i]?)))
            }
            11 => {
                let clv = self.build(|i| self.ohlc(i)?.location());
                self.weighted(widths, &clv, 1, |i| {
                    Some((clv[i]? - prior_mean(&clv, i)?) / 2.0)
                })
            }
            12 => {
                let volume = self.column("vol");
                let innovation = self.build(|i| {
                    let current = nonnegative(volume[i]?)?;
                    let prior = nonnegative(volume[i.checked_sub(1)?]?)?;
                    Some(current.ln_1p() - prior.ln_1p())
                });
                let m = Memory::new(&innovation, 1, false);
                self.series(widths, |i, n, end| {
                    Some(sign(self.body(i)?) * positive(innovation[i]?.tanh()) * m.score(n, end)?)
                })
            }
            13 => {
                let volume = self.build(|i| nonnegative(self.column("vol")[i]?));
                let log_volume = self.build(|i| Some(volume[i]?.ln_1p()));
                self.weighted(widths, &log_volume, 1, |i| {
                    let absorption = 1.0 - self.ohlc(i)?.body_fraction()?.abs();
                    Some(-self.impulse(i)? * positive(relative(&volume, i)?) * absorption)
                })
            }
            14 => {
                let turns = self.build(|i| {
                    ratio(
                        nonnegative(self.column("vol")[i]?)?,
                        self.column("oi")[i.checked_sub(1)?]?,
                    )
                });
                self.weighted(widths, &turns, 1, |i| {
                    Some(-self.impulse(i)? * positive(-relative(&turns, i)?))
                })
            }
            15 => {
                let volume = self.build(|i| Some(nonnegative(self.column("vol")[i]?)?.ln_1p()));
                let flow = self.build(|i| self.flow(i));
                let corr = self.build(|i| block_correlation(&volume, &flow, i));
                self.weighted(widths, &corr, RECENT, |i| Some(self.impulse(i)? * corr[i]?))
            }
            16 => {
                let up = self.build(|i| Some(positive(self.flow(i)?)));
                let down = self.build(|i| Some(positive(-self.flow(i)?)));
                self.series(widths, {
                    let a = Memory::new(&up, 1, false);
                    let b = Memory::new(&down, 1, false);
                    move |i, n, end| {
                        let u = prior_mean(&up, i + 1)?;
                        let d = prior_mean(&down, i + 1)?;
                        if u + d == 0.0 {
                            return Some(0.0);
                        }
                        Some(
                            self.impulse(i)?
                                * (u * positive(a.score(n, end)?) - d * positive(b.score(n, end)?))
                                / (u + d),
                        )
                    }
                })
            }
            17 => {
                let agreement = self.build(|i| Some(sign(ret[i]?) * self.flow(i)?));
                let m = Memory::new(&agreement, 1, false);
                self.series(widths, |i, n, end| {
                    let current = agreement[i]?;
                    Some(
                        self.impulse(i)?
                            * balance_signed(current, prior_abs_mean(&agreement, i)?)?
                            * m.score(n, end)?,
                    )
                })
            }
            18 => {
                let net_per_lot = self
                    .build(|i| ratio(self.column("delta_oi")[i]?.abs(), self.column("vol")[i]?));
                self.weighted(widths, &net_per_lot, 1, |i| {
                    Some(-self.impulse(i)? * positive(-relative(&net_per_lot, i)?))
                })
            }
            19 => {
                let illiquidity = self.build(|i| ratio(ret[i]?.abs(), self.column("amount")[i]?));
                self.weighted(widths, &illiquidity, 1, |i| {
                    Some(sign(self.body(i)?) * positive(-relative(&illiquidity, i)?))
                })
            }
            20 | 21 => {
                let vwap = self.build(|i| ratio(self.column("amount")[i]?, self.column("vol")[i]?));
                if self.kind == 20 {
                    let basis = self.build(|i| log_return(self.inputs[0][i]?, vwap[i]?));
                    let m = Memory::new(&basis, 1, false);
                    // A contract multiplier enters the log basis as a constant.
                    // It cancels in both block centering and the displacement.
                    let change = self.build(|i| Some(basis[i]? - basis[i.checked_sub(1)?]?));
                    self.series(widths, |i, n, end| {
                        Some(
                            balance_signed(
                                basis[i]? - prior_mean(&basis, i)?,
                                prior_abs_mean(&change, i)?,
                            )? * m.score(n, end)?,
                        )
                    })
                } else {
                    let traded = self.build(|i| log_return(vwap[i]?, vwap[i.checked_sub(1)?]?));
                    let a = Memory::new(&traded, 1, false);
                    let b = Memory::new(ret, 1, false);
                    self.series(widths, |i, n, end| {
                        Some(impulse(&traded, i)? * (a.slope(n, end)? - b.slope(n, end)?).tanh())
                    })
                }
            }
            22 => {
                let amount = self.build(|i| nonnegative(self.column("amount")[i]?));
                let concentration = self.build(|i| {
                    let short = trailing(&amount, i + 1)?;
                    let (sum, energy) =
                        short.iter().try_fold((0.0, 0.0), |(sum, energy), &x| {
                            let x = x?;
                            Some((sum + x, energy + x * x))
                        })?;
                    ratio(energy, sum * sum)
                });
                self.weighted(widths, &concentration, RECENT, |i| {
                    Some(-self.impulse(i)? * ((concentration[i]? - 0.2) / 0.8).clamp(0.0, 1.0))
                })
            }
            23 => {
                let impact =
                    self.build(|i| ratio(self.ohlc(i)?.log_range()?, self.column("vol")[i]?));
                self.weighted(widths, &impact, 1, |i| {
                    Some(-sign(self.body(i)?) * positive(relative(&impact, i)?))
                })
            }
            24 => {
                let agreement = self.build(|i| {
                    let (a, b) = (ret[i.checked_sub(1)?]?, ret[i]?);
                    let energy = a * a + b * b;
                    if energy == 0.0 {
                        Some(0.0)
                    } else {
                        Some(2.0 * a * b / energy)
                    }
                });
                let m = Memory::new(&agreement, 2, false);
                self.series(widths, |i, n, end| {
                    let direction = sign(ret[i]? + ret[i.checked_sub(1)?]?);
                    Some(direction * agreement[i]? * m.score(n, end)?)
                })
            }
            25 => {
                let clipped = self.build(|i| {
                    let limit = 2.0 * prior_power(ret, i)?.sqrt();
                    Some(ret[i]?.clamp(-limit, limit))
                });
                let a = Memory::new(&clipped, 1, false);
                let b = Memory::new(ret, 1, false);
                self.series(widths, |i, n, end| {
                    Some(self.impulse(i)? * (a.slope(n, end)? - b.slope(n, end)?).tanh())
                })
            }
            26 => {
                let high = self.build(|i| Some(self.ohlc(i)?.h.ln()));
                let low = self.build(|i| Some(self.ohlc(i)?.l.ln()));
                let up = self.build(|i| Some(positive(high[i]? - extremum(&high, i, f64::max)?)));
                let down = self.build(|i| Some(positive(extremum(&low, i, f64::min)? - low[i]?)));
                let range = self.build(|i| Some(high[i]? - low[i]?));
                self.two_sided(widths, &up, &down, |i| {
                    let scale = prior_mean(&range, i)?;
                    Some((
                        balance_signed(up[i]?, scale)?,
                        balance_signed(down[i]?, scale)?,
                    ))
                })
            }
            _ => unreachable!("unknown long-memory factor"),
        }
    }

    fn weighted(
        &self,
        widths: &[usize],
        state: &Values,
        stride: usize,
        direction: impl Fn(usize) -> Option<f64>,
    ) -> Vec<Series> {
        let m = Memory::new(state, stride, false);
        self.series(widths, |i, n, end| {
            Some(direction(i)? * positive(m.score(n, end)?))
        })
    }

    fn two_sided(
        &self,
        widths: &[usize],
        lower: &Values,
        upper: &Values,
        direction: impl Fn(usize) -> Option<(f64, f64)>,
    ) -> Vec<Series> {
        let a = Memory::new(lower, 1, false);
        let b = Memory::new(upper, 1, false);
        self.series(widths, |i, n, end| {
            let (left, right) = direction(i)?;
            let leg = |value, memory: &Memory| {
                if value == 0.0 {
                    Some(0.0)
                } else {
                    Some(value * positive(memory.score(n, end)?))
                }
            };
            Some(leg(left, &a)? - leg(right, &b)?)
        })
    }
}

struct Candle {
    c: f64,
    o: f64,
    h: f64,
    l: f64,
}

impl Candle {
    fn log_range(&self) -> Option<f64> {
        log_return(self.h, self.l)
    }

    fn body_fraction(&self) -> Option<f64> {
        ratio(self.c - self.o, self.h - self.l)
    }

    fn location(&self) -> Option<f64> {
        ratio(2.0 * (self.c - self.l) - (self.h - self.l), self.h - self.l)
    }

    fn wick(&self, upper: bool) -> Option<f64> {
        let span = if upper {
            self.h - self.c.max(self.o)
        } else {
            self.c.min(self.o) - self.l
        };
        ratio(span, self.h - self.l)
    }
}

fn sign(value: f64) -> f64 {
    if value == 0.0 {
        0.0
    } else {
        value.signum()
    }
}

fn positive(value: f64) -> f64 {
    value.max(0.0)
}

fn nonnegative(value: f64) -> Option<f64> {
    (value >= 0.0).then_some(value)
}

fn ratio(value: f64, base: f64) -> Option<f64> {
    (base > 0.0).then(|| value / base)
}

fn balance(a: f64, b: f64) -> Option<f64> {
    let sum = a + b;
    if sum == 0.0 {
        Some(0.0)
    } else {
        ratio(a - b, sum)
    }
}

fn balance_signed(value: f64, scale: f64) -> Option<f64> {
    if value == 0.0 {
        Some(0.0)
    } else {
        ratio(value, value.abs() + scale)
    }
}

fn relative(values: &Values, i: usize) -> Option<f64> {
    balance(values[i]?, prior_mean(values, i)?)
}

fn trailing(values: &Values, end: usize) -> Option<&Values> {
    Some(&values[end.checked_sub(RECENT)?..end])
}

fn prior_stat(values: &Values, end: usize, transform: impl Fn(f64) -> f64) -> Option<f64> {
    let sum = trailing(values, end)?
        .iter()
        .try_fold(0.0, |sum, &x| Some(sum + transform(x?)))?;
    Some(sum / RECENT as f64)
}

fn prior_mean(values: &Values, end: usize) -> Option<f64> {
    prior_stat(values, end, |x| x)
}

fn prior_abs_mean(values: &Values, end: usize) -> Option<f64> {
    prior_stat(values, end, f64::abs)
}

fn prior_power(values: &Values, end: usize) -> Option<f64> {
    prior_stat(values, end, |x| x * x)
}

fn return_surprise(returns: &Values, i: usize) -> Option<f64> {
    let r = returns[i]?;
    let rms = prior_power(returns, i)?.sqrt();
    if rms == 0.0 {
        return (r == 0.0).then_some(0.0);
    }
    Some((r / rms).tanh())
}

fn impulse(returns: &Values, i: usize) -> Option<f64> {
    let end = i + 1;
    let mean = prior_mean(returns, end)?;
    let scale = prior_power(returns, end)?.sqrt();
    if scale == 0.0 {
        Some(0.0)
    } else {
        ratio(mean, scale)
    }
}

fn extremum(values: &Values, end: usize, f: impl Fn(f64, f64) -> f64) -> Option<f64> {
    let (first, rest) = trailing(values, end)?.split_first()?;
    rest.iter().try_fold((*first)?, |x, &y| Some(f(x, y?)))
}

fn block_correlation(x: &Values, y: &Values, i: usize) -> Option<f64> {
    let (a, b) = (trailing(x, i + 1)?, trailing(y, i + 1)?);
    let (x_mean, y_mean) = (prior_mean(x, i + 1)?, prior_mean(y, i + 1)?);
    if a.iter().all(|value| *value == a[0]) || b.iter().all(|value| *value == b[0]) {
        return None;
    }
    let mut xx = 0.0;
    let mut yy = 0.0;
    let mut xy = 0.0;
    for (&a, &b) in a.iter().zip(b) {
        let (a, b) = (a? - x_mean, b? - y_mean);
        xx += a * a;
        yy += b * b;
        xy += a * b;
    }
    Some(ratio(xy, xx.sqrt() * yy.sqrt())?.clamp(-1.0, 1.0))
}

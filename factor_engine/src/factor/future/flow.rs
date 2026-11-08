use std::collections::{BTreeSet, HashMap, VecDeque};

use crate::core::{
    AssetClass, DatasetId, FactorContext, FactorRowKey, FactorSeries, FactorSpec, FactorValue,
    Frequency, Lookback,
};
use crate::data::DataPool;
use crate::error::{err, Result};
use crate::factor::future::minute_input::{
    compute_future_minute, future_minute_request, product, FutureMinuteField,
    FutureMinuteField::{ContractDiff, ContractLevel, Raw, VwapNotional},
    FutureWarmupSource,
};
use crate::factor::future::rolling::{atr, with_atr};
use crate::factor::Factor;
use crate::operators::{ts_corr, ts_delay, ts_mean, ts_median, ts_rank, ts_std_dev, ts_sum};

const FIRST_ID: usize = 80;
const N: usize = 15;
const K: usize = 5;
const W: usize = 30;
const EPS: f64 = 1e-8;
const SIGMA_FLOOR: f64 = 1e-6;
const OI_FLOOR: f64 = 1.0;
const MIN_SEASONAL_DAYS: usize = 5;
type Values = [Option<f64>];
type Series = Vec<Option<f64>>;

const CLOSE_VOL: &[FutureMinuteField] = &[Raw("close"), Raw("vol")];
const CLOSE_OI_CHANGE: &[FutureMinuteField] = &[Raw("close"), ContractDiff("oi")];
const CLOSE_VOL_OI_CHANGE: &[FutureMinuteField] = &[Raw("close"), Raw("vol"), ContractDiff("oi")];
const OHLCV: &[FutureMinuteField] = &[
    Raw("close"),
    Raw("vol"),
    Raw("high"),
    Raw("low"),
    Raw("open"),
];

const DESCRIPTIONS: [&str; 40] = [
    "Continuous 15-bar price trend weighted by bounded relative traded volume.",
    "Continuous price trend weighted by the ratio of relative volume in two consecutive five-bar windows.",
    "Fifteen-bar volume-weighted average of continuous one-bar price direction.",
    "Continuous price direction weighted by excess relative volume over 15 bars.",
    "Continuous price direction weighted by squared traded volume over 15 bars.",
    "Difference between volume-weighted price direction in the latest five bars and the preceding five bars.",
    "Continuous price trend weighted by the 15-bar mean of bounded relative volume.",
    "Signed volume flow discounted by the normalized concentration of volume across 15 bars.",
    "Normalized Herfindahl concentration of traded volume across 15 bars.",
    "Current log-one-plus volume standardized against the same clock time in the prior 30 trading days. At least five valid observations are required.",
    "Continuous price trend weighted by positive net open interest change, scaled by its lagged rolling median magnitude.",
    "Continuous price trend weighted by negative net open interest change, scaled by its lagged rolling median magnitude.",
    "Continuous price trend times the tanh of signed net open interest change over its lagged rolling median magnitude.",
    "Price direction weighted by positive open interest changes and normalized by total absolute open interest change.",
    "Price direction weighted by negative open interest changes and normalized by total absolute open interest change.",
    "Continuous price trend weighted by unsigned 15-bar open interest path efficiency.",
    "Lagged correlation between open interest change and next-bar log return, multiplied by the standardized current open interest change.",
    "Signed 15-bar net open interest change divided by its cumulative absolute change.",
    "One minus unsigned 15-bar open interest path efficiency. A window with no open interest movement returns zero.",
    "Centered 30-bar percentile rank of open interest within the same actual contract, using average ranks for ties.",
    "Standardized 15-bar log return divided by one plus relative traded volume.",
    "Relative-volume-weighted log return divided by lagged return volatility.",
    "Signed log-return efficiency weighted by positive deviations from normal volume.",
    "Signed log-return efficiency weighted by negative deviations from normal volume.",
    "Fifteen-bar correlation between log returns and log relative volume.",
    "Normalized difference in mean relative volume between up and down bars. Both directions are required.",
    "Fifteen-bar volume-weighted close location within each bar's high-low range.",
    "Fifteen-bar volume-weighted open-to-close body divided by each bar's high-low range.",
    "Close minus 15-bar VWAP, scaled by 15-bar ATR. Uses multiplier-normalized traded amount where available and volume-weighted typical price otherwise.",
    "Volume-weighted signed log-return efficiency minus unweighted signed log-return efficiency.",
    "Continuous price trend times summed positive open interest changes per traded lot.",
    "Continuous price trend times summed negative open interest changes per traded lot.",
    "Continuous price trend times net open interest change per traded lot.",
    "Price-direction-weighted positive open interest changes per traded lot.",
    "Price-direction-weighted negative open interest changes per traded lot.",
    "Price-direction-weighted positive open interest changes on excess-volume bars per excess-volume-weighted traded lot.",
    "Price-direction-weighted negative open interest changes on excess-volume bars per excess-volume-weighted traded lot.",
    "Total absolute open interest change per traded lot over 15 bars.",
    "Fifteen-bar correlation between log-one-plus volume and log-one-plus absolute open interest change.",
    "Twice the lesser of accumulated positive and negative open interest changes per traded lot.",
];

struct FlowFactor(usize);

pub(super) fn create(index: usize) -> Box<dyn Factor> {
    Box::new(FlowFactor(index))
}

fn fields(index: usize) -> &'static [FutureMinuteField] {
    match index {
        8 | 9 => &[Raw("vol")],
        10..=16 => CLOSE_OI_CHANGE,
        17 | 18 => &[ContractDiff("oi")],
        19 => &[ContractLevel("oi")],
        26 => &[Raw("close"), Raw("vol"), Raw("high"), Raw("low")],
        27 => OHLCV,
        28 => &[
            Raw("close"),
            Raw("vol"),
            Raw("high"),
            Raw("low"),
            VwapNotional,
        ],
        30..=36 => CLOSE_VOL_OI_CHANGE,
        37..=39 => &[Raw("vol"), ContractDiff("oi")],
        _ => CLOSE_VOL,
    }
}

impl Factor for FlowFactor {
    fn spec(&self) -> FactorSpec {
        let id = format!("yq_{:03}", FIRST_ID + self.0);
        let family = [
            "volume",
            "open_interest",
            "price_volume",
            "volume_open_interest",
        ][self.0 / 10];
        FactorSpec {
            name: id.clone(),
            id,
            aliases: Vec::new(),
            asset_class: AssetClass::Future,
            frequency: Frequency::Minute1,
            version: "0.1.0".into(),
            tags: ["minute", "pv", "yq", family].map(str::to_string).to_vec(),
            description: DESCRIPTIONS[self.0].into(),
            dependencies: vec![future_minute_request(fields(self.0))],
            intraday_raw_dependencies: Vec::new(),
            lookback: Lookback {
                trading_days: if self.0 == 9 { W } else { 3 },
            },
        }
    }

    fn compute(&self, context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let spec = self.spec();
        if self.0 == 9 {
            return volume_surprise(spec, context, data);
        }
        compute_future_minute(
            spec,
            context,
            data,
            fields(self.0),
            N + W,
            |inputs| match self.0 {
                0..=8 => volume(self.0, inputs),
                10..=19 => open_interest(self.0 - 10, inputs),
                20..=29 => price_volume(self.0 - 20, inputs),
                _ => volume_interest(self.0 - 30, inputs),
            },
        )
    }
}

fn map(values: &Values, op: impl Fn(f64) -> f64) -> Series {
    values
        .iter()
        .map(|value| Some(op((*value)?)).filter(|v| v.is_finite()))
        .collect()
}

fn zip(a: &Values, b: &Values, op: impl Fn(f64, f64) -> Option<f64>) -> Series {
    a.iter()
        .zip(b)
        .map(|(&a, &b)| op(a?, b?).filter(|v| v.is_finite()))
        .collect()
}

fn mul(a: &Values, b: &Values) -> Series {
    zip(a, b, |a, b| Some(a * b))
}

fn ratio(n: f64, d: f64) -> Option<f64> {
    if d != 0.0 {
        Some(n / d)
    } else {
        (n == 0.0).then_some(0.0)
    }
}

fn sum(values: &Values, window: usize) -> Series {
    ts_sum(values, window, window)
}

fn lag(values: &Values, offset: usize) -> Series {
    ts_delay(values, offset)
}

fn log_returns(close: &Values) -> Series {
    zip(close, &lag(close, 1), |a, b| {
        (a > 0.0 && b > 0.0).then(|| a.ln() - b.ln())
    })
}

fn sigma(returns: &Values) -> Series {
    map(&lag(&ts_std_dev(returns, W, W), 1), |s| s.max(SIGMA_FLOOR))
}

fn direction(returns: &Values) -> Series {
    zip(returns, &sigma(returns), |r, s| Some((r / s).tanh()))
}

fn standardized_trend(returns: &Values) -> Series {
    zip(&sum(returns, N), &sigma(returns), |r, s| {
        Some(r / ((N as f64).sqrt() * s))
    })
}

fn trend(returns: &Values) -> Series {
    map(&standardized_trend(returns), f64::tanh)
}

fn phi(value: f64) -> f64 {
    value / (1.0 + value)
}

fn baseline(vol: &Values) -> Series {
    lag(&ts_mean(vol, W, W), 1)
}

fn relative_volume(vol: &Values) -> Series {
    zip(vol, &baseline(vol), |v, normal| {
        (normal > 0.0).then(|| v / normal)
    })
}

fn rvol(vol: &Values) -> Series {
    zip(&sum(vol, N), &sum(&baseline(vol), N), |v, normal| {
        (normal > 0.0).then(|| v / normal)
    })
}

fn nonnegative(values: &Values) -> Series {
    values
        .iter()
        .map(|v| v.filter(|x| x.is_finite() && *x >= 0.0))
        .collect()
}

fn excess(relative: &Values, high: bool) -> Series {
    map(relative, |v| {
        if high {
            (v - 1.0).max(0.0)
        } else {
            (1.0 - v).max(0.0)
        }
    })
}

fn weighted_mean(values: &Values, weights: &Values, window: usize, epsilon: f64) -> Series {
    zip(
        &sum(&mul(values, weights), window),
        &sum(weights, window),
        |n, d| ratio(n, d + epsilon),
    )
}

fn signed_efficiency(values: &Values, epsilon: f64) -> Series {
    zip(&sum(values, N), &sum(&map(values, f64::abs), N), |n, d| {
        ratio(n, d + epsilon)
    })
}

fn concentration(vol: &Values) -> Series {
    zip(
        &sum(&map(vol, |v| v * v), N),
        &sum(vol, N),
        |squares, total| {
            Some(if total == 0.0 {
                0.0
            } else {
                ((N as f64 * (squares / (total * total)) - 1.0) / (N - 1) as f64).clamp(0.0, 1.0)
            })
        },
    )
}

fn volume(index: usize, inputs: &[Series]) -> Series {
    if index == 8 {
        return concentration(&nonnegative(&inputs[0]));
    }
    let returns = log_returns(&inputs[0]);
    let vol = nonnegative(&inputs[1]);
    match index {
        0 => mul(&trend(&returns), &map(&rvol(&vol), phi)),
        1 => {
            let activity = sum(&relative_volume(&vol), K);
            let acceleration = zip(&activity, &lag(&activity, K), |a, b| {
                Some(phi(a / (b + EPS)))
            });
            mul(&trend(&returns), &acceleration)
        }
        2 => weighted_mean(&direction(&returns), &vol, N, 0.0),
        3 => weighted_mean(
            &direction(&returns),
            &excess(&relative_volume(&vol), true),
            N,
            EPS,
        ),
        4 => weighted_mean(&direction(&returns), &map(&vol, |v| v * v), N, 0.0),
        5 => {
            let flow = weighted_mean(&direction(&returns), &vol, K, 0.0);
            zip(&flow, &lag(&flow, K), |a, b| Some(a - b))
        }
        6 => mul(
            &trend(&returns),
            &ts_mean(&map(&relative_volume(&vol), phi), N, N),
        ),
        7 => mul(
            &weighted_mean(&direction(&returns), &vol, N, 0.0),
            &map(&concentration(&vol), |h| 1.0 - h),
        ),
        _ => unreachable!(),
    }
}

fn oi_leg(change: &Values, positive: bool) -> Series {
    map(
        change,
        |d| if positive { d.max(0.0) } else { (-d).max(0.0) },
    )
}

fn robust_scale(values: &Values) -> Series {
    map(&lag(&ts_median(&map(values, f64::abs), W, W), 1), |v| {
        v.max(OI_FLOOR)
    })
}

fn open_interest(index: usize, inputs: &[Series]) -> Series {
    if index == 9 {
        return map(&ts_rank(&nonnegative(&inputs[0]), W, W), |rank| {
            2.0 * (rank - 1.0) / (W - 1) as f64 - 1.0
        });
    }
    if index == 7 {
        return signed_efficiency(&inputs[0], 0.0);
    }
    if index == 8 {
        let change = &inputs[0];
        return zip(
            &sum(change, N),
            &sum(&map(change, f64::abs), N),
            |net, path| {
                Some(if path == 0.0 {
                    0.0
                } else {
                    1.0 - net.abs() / path
                })
            },
        );
    }
    let returns = log_returns(&inputs[0]);
    let change = &inputs[1];
    match index {
        0..=2 => {
            // Summing actual-contract differences avoids treating a roll as an OI jump.
            let net = sum(change, N);
            let normalized = zip(&net, &robust_scale(&net), |d, scale| Some(d / scale));
            let score = map(&normalized, |x| match index {
                0 => phi(x.max(0.0)),
                1 => phi((-x).max(0.0)),
                _ => x.tanh(),
            });
            mul(&trend(&returns), &score)
        }
        3 | 4 => {
            let flow = sum(&mul(&direction(&returns), &oi_leg(change, index == 3)), N);
            zip(&flow, &sum(&map(change, f64::abs), N), |n, d| {
                Some(n / (d + EPS))
            })
        }
        5 => mul(
            &trend(&returns),
            &map(&signed_efficiency(change, 0.0), f64::abs),
        ),
        6 => {
            // At t, the most recent pair is (change[t-2], return[t-1]).
            let correlation = lag(&ts_corr(&lag(change, 1), &returns, W - 1, W - 1), 1);
            let shock = zip(change, &robust_scale(change), |d, scale| {
                Some((d / scale).tanh())
            });
            mul(&correlation, &shock)
        }
        _ => unreachable!(),
    }
}

fn bar_position(close: &Values, high: &Values, low: &Values, open: Option<&Values>) -> Series {
    (0..close.len())
        .map(|i| {
            let (c, h, l) = (close[i]?, high[i]?, low[i]?);
            let numerator = match open {
                Some(open) => c - open[i]?,
                None => (c - l) - (h - c),
            };
            if h == l {
                Some(0.0)
            } else {
                ratio(numerator, h - l)
            }
        })
        .collect()
}

fn volume_asymmetry(returns: &Values, relative: &Values) -> Series {
    let mut result = vec![None; returns.len()];
    for (index, (returns, relative)) in returns.windows(N).zip(relative.windows(N)).enumerate() {
        result[index + N - 1] = (|| {
            let (mut up, mut down, mut up_count, mut down_count) = (0.0, 0.0, 0, 0);
            for (&r, &v) in returns.iter().zip(relative) {
                let (r, v) = (r?, v?);
                if r > 0.0 {
                    up += v;
                    up_count += 1;
                } else if r < 0.0 {
                    down += v;
                    down_count += 1;
                }
            }
            if up_count == 0 || down_count == 0 {
                return None;
            }
            let (up, down) = (up / up_count as f64, down / down_count as f64);
            ratio(up - down, up + down)
        })();
    }
    result
}

fn price_volume(index: usize, inputs: &[Series]) -> Series {
    let close = &inputs[0];
    let vol = nonnegative(&inputs[1]);
    match index {
        6 | 7 => {
            let open = inputs.get(4).map(Vec::as_slice);
            weighted_mean(
                &bar_position(close, &inputs[2], &inputs[3], open),
                &vol,
                N,
                0.0,
            )
        }
        8 => {
            let (high, low, notional) = (&inputs[2], &inputs[3], &inputs[4]);
            let price_volume = (0..close.len())
                .map(|i| {
                    let volume = vol[i]?;
                    if volume == 0.0 {
                        return Some(0.0);
                    }
                    notional[i].or_else(|| Some((close[i]? + high[i]? + low[i]?) / 3.0 * volume))
                })
                .collect::<Series>();
            let vwap = zip(&sum(&price_volume, N), &sum(&vol, N), |value, volume| {
                (volume > 0.0).then(|| value / volume)
            });
            let deviation = zip(close, &vwap, |c, w| Some(c - w));
            deviation
                .into_iter()
                .zip(atr(close, high, low, N))
                .map(|(d, a)| {
                    let d = d?;
                    with_atr(a, |a| Some(d / a))
                })
                .collect()
        }
        _ => {
            let returns = log_returns(close);
            match index {
                0 => zip(&standardized_trend(&returns), &rvol(&vol), |r, v| {
                    Some(r / (1.0 + v))
                }),
                1 => zip(
                    &weighted_mean(&returns, &relative_volume(&vol), N, 0.0),
                    &sigma(&returns),
                    |r, s| Some(r / s),
                ),
                2 | 3 => signed_efficiency(
                    &mul(&returns, &excess(&relative_volume(&vol), index == 2)),
                    EPS,
                ),
                4 => ts_corr(&returns, &map(&relative_volume(&vol), f64::ln), N, N),
                5 => volume_asymmetry(&returns, &relative_volume(&vol)),
                9 => zip(
                    &signed_efficiency(&mul(&returns, &vol), 0.0),
                    &signed_efficiency(&returns, 0.0),
                    |a, b| Some(a - b),
                ),
                _ => unreachable!(),
            }
        }
    }
}

fn volume_interest(index: usize, inputs: &[Series]) -> Series {
    let has_price = index < 7;
    let offset = usize::from(has_price);
    let vol = nonnegative(&inputs[offset]);
    let change = &inputs[offset + 1];
    let per_volume = |activity: &Values| zip(activity, &sum(&vol, N), |n, d| Some(n / (d + EPS)));
    match index {
        7 => per_volume(&sum(&map(change, f64::abs), N)),
        8 => ts_corr(
            &map(&vol, f64::ln_1p),
            &map(change, |d| d.abs().ln_1p()),
            N,
            N,
        ),
        9 => per_volume(&zip(
            &sum(&oi_leg(change, true), N),
            &sum(&oi_leg(change, false), N),
            |u, d| Some(2.0 * u.min(d)),
        )),
        _ => {
            let returns = log_returns(&inputs[0]);
            match index {
                0..=2 => {
                    let activity = match index {
                        0 => oi_leg(change, true),
                        1 => oi_leg(change, false),
                        _ => change.to_vec(),
                    };
                    mul(&trend(&returns), &per_volume(&sum(&activity, N)))
                }
                3 | 4 => per_volume(&sum(
                    &mul(&direction(&returns), &oi_leg(change, index == 3)),
                    N,
                )),
                5 | 6 => {
                    let activity = mul(&direction(&returns), &oi_leg(change, index == 5));
                    let excess = excess(&relative_volume(&vol), true);
                    zip(
                        &sum(&mul(&activity, &excess), N),
                        &sum(&mul(&vol, &excess), N),
                        |n, d| Some(n / (d + EPS)),
                    )
                }
                _ => unreachable!(),
            }
        }
    }
}

type Slots<'a> = HashMap<(&'a str, &'a str), f64>;

fn volume_surprise(
    spec: FactorSpec,
    context: &FactorContext,
    data: &DataPool,
) -> Result<FactorSeries> {
    let input = data
        .future_minute_input()
        .ok_or_else(|| err("missing futures minute input configuration"))?;
    let same_contract = input.options().warmup_source == FutureWarmupSource::SameContract;
    let targets = context
        .target_dates
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let mut prior = VecDeque::<(i32, Slots<'_>)>::new();
    let mut output = Vec::new();
    for &date in &context.load_dates {
        let mut slots = Slots::new();
        if let Some(table) = data
            .minute(DatasetId::FutureMinute1m, date)
            .filter(|table| table.len > 0)
        {
            let day = data
                .future_minute_day_index(date)
                .ok_or_else(|| err("missing futures minute day index"))?;
            let times = table.required_utf8("trade_time")?;
            let vol = nonnegative(&table.required_f64_cast("vol")?);
            let active = input
                .selection(date)
                .map(|m| m.values().map(String::as_str).collect::<BTreeSet<_>>());
            for (code, indices) in &day.groups {
                let code = code.as_str();
                let emit = targets.contains(&date)
                    && active.as_ref().is_none_or(|active| active.contains(code));
                let product = if active.is_some() && emit && !same_contract {
                    product(code)
                } else {
                    None
                };
                let history = if emit {
                    prior
                        .iter()
                        .filter_map(|(date, slots)| {
                            input
                                .history_contract(*date, code, product.as_deref(), same_contract)
                                .map(|code| (code, slots))
                        })
                        .collect::<Vec<_>>()
                } else {
                    Vec::new()
                };
                for &row in indices {
                    let time = times[row].as_deref().expect("indexed minute timestamp");
                    let clock = time.get(11..19);
                    let current = vol[row].map(f64::ln_1p);
                    if let (Some(clock), Some(value)) = (clock, current) {
                        if slots.insert((code, clock), value).is_some() {
                            return Err(err(format!(
                                "duplicate minute clock {clock} for {code} on {date}"
                            )));
                        }
                    }
                    if emit {
                        let value = current.zip(clock).and_then(|(current, clock)| {
                            let mut count = 0;
                            let (mut mean, mut m2) = (0.0, 0.0);
                            for &(code, slots) in &history {
                                if let Some(value) = slots.get(&(code, clock)) {
                                    count += 1;
                                    let delta = value - mean;
                                    mean += delta / count as f64;
                                    m2 += delta * (value - mean);
                                }
                            }
                            (count >= MIN_SEASONAL_DAYS).then(|| {
                                (current - mean) / ((m2 / count as f64).max(0.0).sqrt() + EPS)
                            })
                        });
                        output.push(FactorValue {
                            key: FactorRowKey::Minute {
                                trade_date: date,
                                trade_time: time.into(),
                                ts_code: code.into(),
                            },
                            value,
                        });
                    }
                }
            }
        }
        prior.push_back((date, slots));
        if prior.len() > W {
            prior.pop_front();
        }
    }
    Ok(FactorSeries {
        spec,
        values: output,
    })
}

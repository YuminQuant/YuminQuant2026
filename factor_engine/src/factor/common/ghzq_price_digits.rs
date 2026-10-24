use std::collections::BTreeMap;

use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::{err, Result};
use crate::factor::common::stock_daily_ops::{is_bj_stock, neutralize_size_sector};
use crate::factor::common::{DailyPanel, PanelColumn};
use crate::factor::Factor;
use crate::operators::{cs_pctrank, ts_mean};

pub const IDS: [&str; 8] = [
    "multi_scale_grid_consensus",
    "multi_scale_grid_consensus_atr",
    "touch_rate20",
    "days_since_last_touch",
    "multi_barrier_touch_count",
    "barrier_acceptance20",
    "barrier_side_pressure20",
    "barrier_volatility_compression",
];
const ALIASES: [&str; 8] = [
    "MultiScaleGridConsensus",
    "MultiScaleGridConsensusATR",
    "TouchRate20",
    "DaysSinceLastTouch",
    "MultiBarrierTouchCount",
    "BarrierAcceptance20",
    "BarrierSidePressure20",
    "BarrierVolatilityCompression",
];

pub struct PriceDigits(pub usize);

impl Factor for PriceDigits {
    fn spec(&self) -> FactorSpec {
        spec(self.0)
    }

    fn compute_provider_key(&self) -> String {
        "stock|daily|ghzq_price_digits".into()
    }

    fn compute(&self, context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        self.compute_many(&[IDS[self.0].into()], context, data)?
            .pop()
            .ok_or_else(|| err("GHZQ price digits provider returned no output"))
    }

    fn compute_many(
        &self,
        requested: &[String],
        _: &FactorContext,
        data: &DataPool,
    ) -> Result<Vec<FactorSeries>> {
        let wanted = std::array::from_fn(|i| requested.iter().any(|id| id == IDS[i]));
        let panel = data.daily_panel(DatasetId::StockDailyPv)?;
        let turnover = if wanted[6] {
            Some(panel.column_from_panel(
                data.daily_panel(DatasetId::StockDailyBasic)?,
                "turnover_rate_f",
            )?)
        } else {
            None
        };
        raw_columns(panel, turnover.as_ref(), &wanted)?
            .into_iter()
            .map(|(i, column)| {
                Ok(neutralize_size_sector(&column, panel, data)?.to_factor_series(spec(i)))
            })
            .collect()
    }
}

fn spec(i: usize) -> FactorSpec {
    let atr = matches!(i, 1 | 6 | 7);
    let range = atr || matches!(i, 2 | 3 | 4);
    let mut columns = vec!["close"];
    if range {
        columns.extend(["high", "low"]);
    }
    if atr {
        columns.push("pre_close");
    }
    let mut dependencies = vec![
        DataRequest::new(DatasetId::StockDailyPv, &columns).with_panel_only(),
        DataRequest::new(DatasetId::StockBarraDaily, &["SIZE"]).with_panel_only(),
        DataRequest::new(DatasetId::StockSwClassification, &["l1_code"]),
    ];
    if i == 6 {
        dependencies.push(
            DataRequest::new(DatasetId::StockDailyBasic, &["turnover_rate_f"]).with_panel_only(),
        );
    }
    let formula = match i {
        0 => "mean of three CS percentile ranks of price-normalized negative grid distances",
        1 => "mean of three CS percentile ranks of ATR20-normalized negative grid distances",
        2 => "20-date mean of low<=nearest historical-day adaptive barrier<=high over valid observations",
        3 => "trading dates since the latest touch in the inclusive 252-date window; no touch or unknown current touch is null",
        4 => "floor(high/grid)-ceil(low/grid)+1, nonnegative; grid fixed by current close",
        5 => "20-date mean sign(historical close - current nearest 1 yuan barrier); equality=0",
        6 => "APPROXIMATION (report table 6 does not specify formula): sum(sign(close_u-B)*w_u)/sum(w_u), w_u=turnover_rate_f_u*exp(-abs(close_u-B)/ATR20_u), B=current nearest 1 yuan barrier; skip invalid samples, zero total weight is null",
        _ => "CS percentile rank of -adaptive distance/ATR20 times (1-ATR5/ATR20)",
    };
    FactorSpec {
        id: IDS[i].into(), aliases: vec![ALIASES[i].into()], name: ALIASES[i].into(),
        asset_class: AssetClass::Stock, frequency: Frequency::Daily, version: "0.1.0".into(),
        tags: ["GHZQ", "price_volume", "price_digits", "daily", "neutralize", "barra", "size", "sector"].into_iter().map(str::to_string).collect(),
        description: format!("{}: {}. Unadjusted close/OHLC in rounded integer cents; nearest-grid ties round upward. Adaptive grid=1/5/10/50 yuan for close<100/<500/<1000/otherwise. TR=max(H-L,abs(H-pre_close),abs(L-pre_close)); ATR uses simple mean with min_periods=1. Inclusive trading-date windows, skip missing observations, all missing remains null. CS ranks ascending with average ties on non-BJ present stocks. Final SW L1+Barra SIZE residual; no winsorization, zscore, filling or sign flip.", ALIASES[i], formula),
        dependencies, intraday_raw_dependencies: vec![],
        lookback: Lookback { trading_days: match i { 3 => 251, 6 => 38, 1 | 2 | 5 | 7 => 19, _ => 0 } },
    }
}

fn cents(value: Option<f64>) -> Option<i64> {
    value
        .filter(|v| v.is_finite() && *v > 0.0 && *v < 1e12)
        .map(|v| (v * 100.0).round() as i64)
        .filter(|v| *v > 0)
}

fn grid(p: i64) -> i64 {
    match p {
        0..=9_999 => 100,
        10_000..=49_999 => 500,
        50_000..=99_999 => 1_000,
        _ => 5_000,
    }
}

fn barrier(p: i64, g: i64) -> i64 {
    ((p + g / 2) / g) * g
}
fn distance(p: i64, g: i64) -> f64 {
    (p - barrier(p, g)).abs() as f64
}

fn raw_columns(
    panel: &DailyPanel,
    turnover: Option<&PanelColumn>,
    wanted: &[bool; 8],
) -> Result<BTreeMap<usize, PanelColumn>> {
    let n = panel.instruments().len();
    let days = panel.dates().len();
    let close = panel.column("close")?;
    let needs_atr = wanted[1] || wanted[6] || wanted[7];
    let needs_range = needs_atr || wanted[2] || wanted[3] || wanted[4];
    let high = if needs_range {
        Some(panel.column("high")?)
    } else {
        None
    };
    let low = if needs_range {
        Some(panel.column("low")?)
    } else {
        None
    };
    let pre = if needs_atr {
        Some(panel.column("pre_close")?)
    } else {
        None
    };
    let mut atr_panel = (wanted[1] || wanted[7]).then(|| vec![None; panel.shape_len()]);
    let mut outputs: BTreeMap<usize, Vec<Option<f64>>> = (0..IDS.len())
        .filter(|i| wanted[*i])
        .map(|i| (i, vec![None; panel.shape_len()]))
        .collect();
    for (stock, code) in panel.instruments().iter().enumerate() {
        if is_bj_stock(code) {
            continue;
        }
        let prices: Vec<_> = (0..days)
            .map(|day| {
                let offset = day * n + stock;
                if panel.is_present_offset(offset) {
                    cents(close.values()[offset])
                } else {
                    None
                }
            })
            .collect();
        let mut touch = vec![None; days];
        let mut counts = vec![None; days];
        let mut tr = vec![None; days];
        if needs_range {
            for day in 0..days {
                let offset = day * n + stock;
                let (Some(p), Some(h), Some(l)) = (
                    prices[day],
                    cents(high.as_ref().unwrap().values()[offset]),
                    cents(low.as_ref().unwrap().values()[offset]),
                ) else {
                    continue;
                };
                if l > h || p < l || p > h {
                    continue;
                }
                let g = grid(p);
                let b = barrier(p, g);
                touch[day] = Some(if l <= b && b <= h { 1.0 } else { 0.0 });
                counts[day] = Some((h / g - (l + g - 1) / g + 1).max(0) as f64);
                if let Some(prev) = pre.as_ref().and_then(|c| cents(c.values()[offset])) {
                    tr[day] = Some((h - l).max((h - prev).abs()).max((l - prev).abs()) as f64);
                }
            }
        }
        // ATR and distances are both in cents; their ratio is unit invariant.
        let atr20 = if needs_atr {
            ts_mean(&tr, 20, 1)
        } else {
            vec![]
        };
        let atr5 = if wanted[7] {
            ts_mean(&tr, 5, 1)
        } else {
            vec![]
        };
        let touch_mean = if wanted[2] {
            ts_mean(&touch, 20, 1)
        } else {
            vec![]
        };
        let mut last_touch = None;
        for day in 0..days {
            if touch[day] == Some(1.0) {
                last_touch = Some(day);
            }
            let offset = day * n + stock;
            let Some(p) = prices[day] else {
                continue;
            };
            if let Some(buffer) = atr_panel.as_mut() {
                buffer[offset] = atr20[day].filter(|a| *a > 0.0);
            }
            if !panel.is_target_date(panel.dates()[day]) {
                continue;
            }
            if let Some(out) = outputs.get_mut(&2) {
                out[offset] = touch_mean[day];
            }
            if let Some(out) = outputs.get_mut(&3) {
                out[offset] = touch[day].and_then(|_| {
                    last_touch
                        .filter(|last| day - last < 252)
                        .map(|last| (day - last) as f64)
                });
            }
            if let Some(out) = outputs.get_mut(&4) {
                out[offset] = counts[day];
            }
            if wanted[5] || wanted[6] {
                let b = barrier(p, 100);
                let mut sum = 0.0;
                let mut valid = 0usize;
                let mut weighted = 0.0;
                let mut weight_sum = 0.0;
                for u in day.saturating_sub(19)..=day {
                    let Some(c) = prices[u] else {
                        continue;
                    };
                    let sign = (c - b).signum() as f64;
                    sum += sign;
                    valid += 1;
                    if wanted[6] {
                        let rate = turnover
                            .and_then(|t| t.values()[u * n + stock])
                            .filter(|v| v.is_finite() && *v >= 0.0);
                        if let (Some(rate), Some(a)) = (rate, atr20[u].filter(|a| *a > 0.0)) {
                            let w = rate * (-((c - b).abs() as f64) / a).exp();
                            weighted += sign * w;
                            weight_sum += w;
                        }
                    }
                }
                if let Some(out) = outputs.get_mut(&5) {
                    out[offset] = (valid > 0).then(|| sum / valid as f64);
                }
                if let Some(out) = outputs.get_mut(&6) {
                    out[offset] = (weight_sum > 0.0).then(|| weighted / weight_sum);
                }
            }
            if let Some(out) = outputs.get_mut(&7) {
                out[offset] = atr5[day]
                    .zip(atr20[day].filter(|a| *a > 0.0))
                    .map(|(a, b)| 1.0 - a / b);
            }
        }
    }
    // Only three temporary cross-sections, not three additional wide panels.
    if wanted[0] || wanted[1] || wanted[7] {
        for (day, date) in panel.dates().iter().enumerate() {
            if !panel.is_target_date(*date) {
                continue;
            }
            for atr_scaled in [false, true] {
                if (!atr_scaled && !wanted[0]) || (atr_scaled && !wanted[1] && !wanted[7]) {
                    continue;
                }
                let mut combined = vec![Some(0.0); n];
                for scale in 0..3 {
                    if atr_scaled && !wanted[1] && scale > 0 {
                        break;
                    }
                    let cross: Vec<_> = (0..n)
                        .map(|stock| {
                            let offset = day * n + stock;
                            if is_bj_stock(&panel.instruments()[stock])
                                || !panel.is_present_offset(offset)
                            {
                                return None;
                            }
                            let p = cents(close.values()[offset])?;
                            let denominator = if atr_scaled {
                                atr_panel.as_ref()?[offset]?
                            } else {
                                p as f64
                            };
                            Some(-distance(p, [grid(p), 50, 10][scale]) / denominator)
                        })
                        .collect();
                    let ranks = cs_pctrank(&cross, true);
                    if atr_scaled && scale == 0 {
                        if let Some(out) = outputs.get_mut(&7) {
                            for stock in 0..n {
                                let o = day * n + stock;
                                out[o] = out[o].zip(ranks[stock]).map(|(a, b)| a * b);
                            }
                        }
                    }
                    for stock in 0..n {
                        combined[stock] =
                            combined[stock].zip(ranks[stock]).map(|(a, b)| a + b / 3.0);
                    }
                }
                if let Some(out) = outputs.get_mut(&(if atr_scaled { 1 } else { 0 })) {
                    out[day * n..(day + 1) * n].copy_from_slice(&combined);
                }
            }
        }
    }
    outputs
        .into_iter()
        .map(|(i, values)| Ok((i, panel.column_from_values(values)?)))
        .collect()
}

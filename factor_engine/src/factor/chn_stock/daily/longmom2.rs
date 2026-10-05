use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::stock_daily_ops::is_bj_stock;
use crate::factor::common::vector::clean;
use crate::factor::common::{DailyPanel, PanelColumn};
use crate::factor::Factor;
use crate::operators::cs_regression_residual;

const WINDOW: usize = 160;
const REVERSE_WINDOW: usize = 20;
const LOW_QUANTILE: f64 = 0.70;

pub struct StockDailyLongMom2;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyLongMom2)
}

impl Factor for StockDailyLongMom2 {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "longmom2".into(),
            aliases: vec!["LongMom2".into(), "LONGMOM2".into()],
            name: "Long Momentum 2.0".into(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: "0.1.0".into(),
            tags: ["KYZQ", "price_volume", "momentum", "amplitude", "excess_return",
                "reversal_neutralize", "daily"].into_iter().map(str::to_string).collect(),
            description: "KYZQ LongMom2: first take 160 trading dates inclusive of today, then discard missing/nontraded and closing-limit-up/down observations without extending the window. Amplitude=(high-low)/pre_close. Sum daily pct_chg/100 minus the equal-weight return of present, traded, valid non-BJ stocks over amplitudes <= the linearly interpolated 70th percentile (boundary ties retained). Closing-limit stocks participate in the market mean. Finally take cross-sectional OLS residual against the compounded last 20 daily returns with intercept. Both rolling windows use min_periods=1, including partial early windows; skip missing observations, with all-missing windows remaining null. Excludes BJ; no industry/SIZE neutralization or final standardization.".into(),
            dependencies: vec![
                DataRequest::new(DatasetId::StockDailyPv,
                    &["close", "high", "low", "pre_close", "pct_chg", "vol"]),
                DataRequest::new(DatasetId::StockDailyLimit, &["up_limit", "down_limit"]),
            ],
            intraday_raw_dependencies: vec![],
            lookback: Lookback { trading_days: WINDOW - 1 },
        }
    }

    fn compute(&self, _: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let panel = data.daily_panel(DatasetId::StockDailyPv)?;
        let close = panel.column("close")?;
        let high = panel.column("high")?;
        let low = panel.column("low")?;
        let pre_close = panel.column("pre_close")?;
        let pct_chg = panel.column("pct_chg")?;
        let volume = panel.column("vol")?;
        let limits = data.daily(DatasetId::StockDailyLimit)?;
        let up_limit = panel.column_from_table(limits, "up_limit")?;
        let down_limit = panel.column_from_table(limits, "down_limit")?;
        let market = market_returns(panel, &close, &pre_close, &pct_chg, &volume);
        let n = panel.instruments().len();
        let mut raw = vec![None; panel.shape_len()];
        let mut reverse = vec![None; panel.shape_len()];
        for (stock, code) in panel.instruments().iter().enumerate() {
            if is_bj_stock(code) {
                continue;
            }
            // Keep per-stock history locally rather than materializing amplitude/excess panels.
            let mut returns = vec![None; panel.dates().len()];
            let mut amplitudes = vec![None; panel.dates().len()];
            let mut excess = vec![None; panel.dates().len()];
            for day in 0..panel.dates().len() {
                let offset = day * n + stock;
                if !panel.is_present_offset(offset) {
                    continue;
                }
                returns[day] = daily_return(
                    close.values()[offset],
                    pre_close.values()[offset],
                    pct_chg.values()[offset],
                );
                amplitudes[day] = eligible_amplitude(
                    close.values()[offset],
                    high.values()[offset],
                    low.values()[offset],
                    pre_close.values()[offset],
                    volume.values()[offset],
                    up_limit.values()[offset],
                    down_limit.values()[offset],
                );
                excess[day] = returns[day]
                    .zip(market[day])
                    .and_then(|(r, m)| clean(Some(r - m)));
            }
            let sums = low_amplitude_series(&amplitudes, &excess);
            let ret20 = compounded_return_series(&returns);
            for (day, date) in panel.dates().iter().enumerate() {
                if panel.is_target_date(*date) {
                    raw[day * n + stock] = sums[day];
                    reverse[day * n + stock] = ret20[day];
                }
            }
        }
        let raw = panel.column_from_values(raw)?;
        let reverse = panel.column_from_values(reverse)?;
        Ok(raw
            .cs_binary(&reverse, cs_regression_residual)?
            .to_factor_series(self.spec()))
    }
}

fn positive(value: Option<f64>) -> Option<f64> {
    clean(value).filter(|v| *v > 0.0)
}

fn daily_return(close: Option<f64>, pre_close: Option<f64>, pct_chg: Option<f64>) -> Option<f64> {
    positive(close)?;
    positive(pre_close)?;
    clean(pct_chg).filter(|v| *v >= -100.0).map(|v| v / 100.0)
}

fn eligible_amplitude(
    close: Option<f64>,
    high: Option<f64>,
    low: Option<f64>,
    pre_close: Option<f64>,
    volume: Option<f64>,
    up_limit: Option<f64>,
    down_limit: Option<f64>,
) -> Option<f64> {
    let close = positive(close)?;
    let high = positive(high)?;
    let low = positive(low)?;
    let pre_close = positive(pre_close)?;
    positive(volume)?;
    let up = positive(up_limit)?;
    let down = positive(down_limit)?;
    // Daily prices are quoted in cents; rounding handles float32 parquet representation.
    let cents = |value: f64| (value * 100.0).round();
    if high < low || up <= down || cents(close) >= cents(up) || cents(close) <= cents(down) {
        return None;
    }
    clean(Some((high - low) / pre_close))
}

fn market_returns(
    panel: &DailyPanel,
    close: &PanelColumn,
    pre_close: &PanelColumn,
    pct_chg: &PanelColumn,
    volume: &PanelColumn,
) -> Vec<Option<f64>> {
    let n = panel.instruments().len();
    (0..panel.dates().len())
        .map(|day| {
            let mut sum = 0.0;
            let mut count = 0usize;
            for (stock, code) in panel.instruments().iter().enumerate() {
                let offset = day * n + stock;
                if is_bj_stock(code)
                    || !panel.is_present_offset(offset)
                    || positive(volume.values()[offset]).is_none()
                {
                    continue;
                }
                if let Some(r) = daily_return(
                    close.values()[offset],
                    pre_close.values()[offset],
                    pct_chg.values()[offset],
                ) {
                    sum += r;
                    count += 1;
                }
            }
            if count == 0 {
                None
            } else {
                clean(Some(sum / count as f64))
            }
        })
        .collect()
}

fn low_amplitude_series(amplitudes: &[Option<f64>], excess: &[Option<f64>]) -> Vec<Option<f64>> {
    let mut output = vec![None; amplitudes.len()];
    let mut samples = Vec::with_capacity(WINDOW);
    for end in 0..amplitudes.len() {
        samples.clear();
        for q in (end + 1).saturating_sub(WINDOW)..=end {
            if let (Some(a), Some(r)) = (clean(amplitudes[q]), clean(excess[q])) {
                samples.push((a, r));
            }
        }
        output[end] = low_amplitude_sum(&mut samples);
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{ColumnData, Table};
    use std::collections::{BTreeMap, HashMap};

    fn near(actual: f64, expected: f64) {
        assert!((actual - expected).abs() < 1e-10, "{actual} != {expected}");
    }

    #[test]
    fn longmom2_cleans_suspensions_and_closing_limits_but_keeps_intraday_touches() {
        let amp = |close, high, low, volume, up, down| {
            eligible_amplitude(close, high, low, Some(10.0), volume, up, down)
        };
        near(
            amp(
                Some(10.0),
                Some(11.0),
                Some(9.0),
                Some(1.0),
                Some(11.0),
                Some(9.0),
            )
            .unwrap(),
            0.2,
        );
        assert_eq!(
            amp(
                Some(11.0),
                Some(11.0),
                Some(9.5),
                Some(1.0),
                Some(11.0),
                Some(9.0)
            ),
            None
        );
        assert_eq!(
            amp(
                Some(9.0),
                Some(10.0),
                Some(9.0),
                Some(1.0),
                Some(11.0),
                Some(9.0)
            ),
            None
        );
        // Float32 representations of the same cent must classify identically.
        assert_eq!(
            amp(
                Some(10.98999977),
                Some(11.0),
                Some(10.0),
                Some(1.0),
                Some(10.99),
                Some(9.0)
            ),
            None
        );
        assert_eq!(
            amp(
                Some(10.0),
                Some(11.0),
                Some(9.0),
                Some(0.0),
                Some(11.0),
                Some(9.0)
            ),
            None
        );
        assert_eq!(
            amp(
                Some(10.0),
                Some(11.0),
                Some(9.0),
                None,
                Some(11.0),
                Some(9.0)
            ),
            None
        );
        assert_eq!(
            amp(
                Some(10.0),
                Some(11.0),
                Some(9.0),
                Some(1.0),
                None,
                Some(9.0)
            ),
            None
        );
        assert_eq!(
            eligible_amplitude(
                Some(10.0),
                Some(11.0),
                Some(9.0),
                Some(0.0),
                Some(1.0),
                Some(11.0),
                Some(9.0)
            ),
            None
        );
        assert_eq!(
            amp(
                Some(10.0),
                Some(9.0),
                Some(10.0),
                Some(1.0),
                Some(11.0),
                Some(9.0)
            ),
            None
        );
        near(
            daily_return(Some(10.2), Some(10.0), Some(2.0)).unwrap(),
            0.02,
        );
        assert_eq!(daily_return(Some(10.0), Some(10.0), Some(f64::NAN)), None);
    }

    #[test]
    fn longmom2_uses_fixed_calendar_window_and_quantile_boundary_ties() {
        let a: Vec<_> = (0..161).map(|i| Some(i as f64)).collect();
        let r: Vec<_> = (0..161).map(|i| Some(i as f64)).collect();
        let result = low_amplitude_series(&a, &r);
        near(result[0].unwrap(), 0.0);
        near(result[1].unwrap(), 0.0);
        near(result[159].unwrap(), (0..112).sum::<usize>() as f64);
        near(result[160].unwrap(), (1..113).sum::<usize>() as f64);
        // Removing the first 60 dates leaves 100 within the original window, not 160.
        let mut filtered = a.clone();
        filtered[..60].fill(None);
        near(
            low_amplitude_series(&filtered, &r)[159].unwrap(),
            (60..130).sum::<usize>() as f64,
        );
        assert!(low_amplitude_series(&[None; 160], &r[..160])[159].is_none());
        let mut tied = vec![(1.0, 1.0); 8];
        tied.extend([(2.0, 100.0); 2]);
        near(low_amplitude_sum(&mut tied).unwrap(), 8.0);
        // No additional minimum eligible-day requirement was specified.
        near(low_amplitude_sum(&mut [(2.0, 3.0)]).unwrap(), 3.0);
    }

    #[test]
    fn longmom2_partial_quantile_selection_matches_sorted_reference() {
        for n in 1..=WINDOW {
            for seed in 0..5 {
                let mut samples: Vec<_> = (0..n)
                    .map(|q| (((q * 17 + seed * 11) % 37) as f64, q as f64 - 70.0))
                    .collect();
                let mut sorted = samples.clone();
                sorted.sort_by(|a, b| a.0.total_cmp(&b.0));
                let p = (n - 1) as f64 * 0.7;
                let lo = p.floor() as usize;
                let hi = p.ceil() as usize;
                let cutoff = sorted[lo].0 + (p - lo as f64) * (sorted[hi].0 - sorted[lo].0);
                let expected = sorted
                    .iter()
                    .filter(|(a, _)| *a <= cutoff)
                    .map(|(_, r)| r)
                    .sum();
                near(low_amplitude_sum(&mut samples).unwrap(), expected);
            }
        }
    }

    #[test]
    fn longmom2_reverse_is_twenty_returns_compounded_without_long_window_filters() {
        let mut r = vec![Some(0.01); 21];
        let result = compounded_return_series(&r);
        near(result[0].unwrap(), 0.01);
        near(result[18].unwrap(), 1.01_f64.powi(19) - 1.0);
        near(result[19].unwrap(), 1.01_f64.powi(20) - 1.0);
        near(result[20].unwrap(), 1.01_f64.powi(20) - 1.0);
        r[0] = None;
        let result = compounded_return_series(&r);
        assert!(result[0].is_none());
        near(result[19].unwrap(), 1.01_f64.powi(19) - 1.0);
        assert!(compounded_return_series(&[None; 25])
            .iter()
            .all(Option::is_none));
        assert_eq!(
            low_amplitude_series(&[None, Some(1.0)], &[Some(0.2), Some(0.3)]),
            vec![None, Some(0.3)]
        );
        near(result[20].unwrap(), 1.01_f64.powi(20) - 1.0);
    }

    #[test]
    fn longmom2_market_excludes_bj_absent_nontraded_and_invalid_but_includes_limits() {
        let panel = DailyPanel::from_index(
            vec![20250102],
            vec![
                "000001.SZ",
                "000002.SZ",
                "430001.BJ",
                "000003.SZ",
                "000004.SZ",
                "000005.SZ",
                "000006.SZ",
            ]
            .into_iter()
            .map(str::to_string)
            .collect(),
            &[20250102],
            vec![true, true, true, false, true, true, true],
        )
        .unwrap();
        let close = panel
            .column_from_values(vec![
                Some(11.0),
                Some(10.0),
                Some(10.0),
                Some(10.0),
                Some(10.0),
                Some(10.0),
                Some(10.0),
            ])
            .unwrap();
        let pre = panel.column_from_values(vec![Some(10.0); 7]).unwrap();
        let pct = panel
            .column_from_values(vec![
                Some(10.0),
                Some(2.0),
                Some(50.0),
                Some(80.0),
                Some(60.0),
                Some(90.0),
                Some(f64::NAN),
            ])
            .unwrap();
        let vol = panel
            .column_from_values(vec![
                Some(1.0),
                Some(1.0),
                Some(1.0),
                Some(1.0),
                Some(0.0),
                None,
                Some(1.0),
            ])
            .unwrap();
        near(
            market_returns(&panel, &close, &pre, &pct, &vol)[0].unwrap(),
            0.06,
        );
    }

    fn pool(first: usize, last: usize, targets: &[i32]) -> (FactorContext, DataPool) {
        let dates: Vec<_> = (1..=12)
            .flat_map(|m| (1..=28).map(move |d| 20250000 + m * 100 + d))
            .take(176)
            .collect();
        let selected = &dates[first..=last];
        let context = FactorContext {
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            start_date: targets[0],
            end_date: *targets.last().unwrap(),
            load_start_date: selected[0],
            load_dates: selected.to_vec(),
            target_dates: targets.to_vec(),
        };
        let rows: Vec<_> = (first..=last)
            .flat_map(|d| (0..7).map(move |s| (d, s)))
            .filter(|(d, s)| !(*d == 24 && *s == 3))
            .collect();
        let code = |s| {
            if s == 6 {
                "430001.BJ".to_string()
            } else {
                format!("{:06}.SZ", s + 1)
            }
        };
        let pct = |d, s| {
            if d == 18 && s == 1 {
                10.0
            } else {
                ((d * 7 + s * 3) % 19) as f64 * 0.15 - 1.35
            }
        };
        let value = |field: &str, d: usize, s: usize| match field {
            "pct_chg" => pct(d, s),
            "close" => 10.0 * (1.0 + pct(d, s) / 100.0),
            "pre_close" => 10.0,
            "high" => {
                (10.0 * (1.0 + pct(d, s) / 100.0)).max(10.0) + ((d * 5 + s * 3) % 11) as f64 * 0.03
            }
            "low" => (10.0 * (1.0 + pct(d, s) / 100.0)).min(10.0) - 0.03,
            "vol" => {
                if d == 13 && s == 2 {
                    0.0
                } else {
                    100.0
                }
            }
            _ => unreachable!(),
        };
        let mut pv = BTreeMap::from([
            (
                "ts_code".into(),
                ColumnData::Utf8(rows.iter().map(|(_, s)| Some(code(*s))).collect()),
            ),
            (
                "trade_date".into(),
                ColumnData::I32(rows.iter().map(|(d, _)| Some(dates[*d])).collect()),
            ),
        ]);
        for field in ["close", "high", "low", "pre_close", "pct_chg", "vol"] {
            pv.insert(
                field.into(),
                ColumnData::F64(
                    rows.iter()
                        .map(|(d, s)| Some(value(field, *d, *s)))
                        .collect(),
                ),
            );
        }
        let limit_rows: Vec<_> = rows
            .iter()
            .rev()
            .copied()
            .filter(|(d, s)| !(*d == 70 && *s == 4))
            .collect();
        let limit = Table::new(BTreeMap::from([
            (
                "ts_code".into(),
                ColumnData::Utf8(limit_rows.iter().map(|(_, s)| Some(code(*s))).collect()),
            ),
            (
                "trade_date".into(),
                ColumnData::I32(limit_rows.iter().map(|(d, _)| Some(dates[*d])).collect()),
            ),
            (
                "up_limit".into(),
                ColumnData::F64(
                    limit_rows
                        .iter()
                        .map(|(_, s)| Some(11.0 + *s as f64 * 0.1))
                        .collect(),
                ),
            ),
            (
                "down_limit".into(),
                ColumnData::F64(
                    limit_rows
                        .iter()
                        .map(|(_, s)| Some(9.0 - *s as f64 * 0.1))
                        .collect(),
                ),
            ),
        ]))
        .unwrap();
        let data = DataPool::from_daily_tables_for_test(
            HashMap::from([
                (DatasetId::StockDailyPv, Table::new(pv).unwrap()),
                (DatasetId::StockDailyLimit, limit),
            ]),
            &context,
        )
        .unwrap();
        (context, data)
    }

    #[test]
    fn longmom2_pv_alignment_batch_history_and_cross_sectional_reversal_neutrality() {
        let dates: Vec<_> = (1..=12)
            .flat_map(|m| (1..=28).map(move |d| 20250000 + m * 100 + d))
            .take(176)
            .collect();
        let targets = &dates[159..];
        let (context, data) = pool(0, 175, targets);
        let full = StockDailyLongMom2.compute(&context, &data).unwrap();
        assert!(full.values.iter().any(|v| v.value.is_some()));
        assert!(full
            .values
            .iter()
            .all(|v| targets.contains(&v.key.trade_date())));
        for target in targets {
            let idx = dates.iter().position(|d| d == target).unwrap();
            let (c, d) = pool(idx + 1 - WINDOW, idx, &[*target]);
            let single = StockDailyLongMom2.compute(&c, &d).unwrap();
            let expected: Vec<_> = full
                .values
                .iter()
                .filter(|v| v.key.trade_date() == *target)
                .collect();
            assert_eq!(single.values.len(), expected.len());
            for (actual, expected) in single.values.iter().zip(expected) {
                assert_eq!(actual.key, expected.key);
                match (actual.value, expected.value) {
                    (Some(a), Some(b)) => near(a, b),
                    (None, None) => (),
                    _ => panic!("batch mismatch"),
                }
            }
            let mut sum = 0.0;
            let mut exposure = 0.0;
            let mut valid = 0;
            for row in &single.values {
                let crate::core::FactorRowKey::Daily { ts_code, .. } = &row.key else {
                    panic!("daily key");
                };
                if ts_code.ends_with(".BJ") {
                    assert!(row.value.is_none());
                    continue;
                }
                if let Some(residual) = row.value {
                    let stock = ts_code[..6].parse::<usize>().unwrap() - 1;
                    let gross: f64 = (idx + 1 - REVERSE_WINDOW..=idx)
                        .map(|day| {
                            1.0 + (((day * 7 + stock * 3) % 19) as f64 * 0.15 - 1.35) / 100.0
                        })
                        .product();
                    sum += residual;
                    exposure += residual * (gross - 1.0);
                    valid += 1;
                }
            }
            assert!(valid >= 5);
            near(sum, 0.0);
            near(exposure, 0.0);
        }
    }
}

fn low_amplitude_sum(samples: &mut [(f64, f64)]) -> Option<f64> {
    if samples.is_empty() {
        return None;
    }
    let position = (samples.len() - 1) as f64 * LOW_QUANTILE;
    let lower = position.floor() as usize;
    let fraction = position - lower as f64;
    let cutoff = {
        let (_, mid, upper) = samples.select_nth_unstable_by(lower, |a, b| a.0.total_cmp(&b.0));
        let upper_value = if fraction > 0.0 {
            upper.iter().map(|v| v.0).reduce(f64::min)?
        } else {
            mid.0
        };
        mid.0 + fraction * (upper_value - mid.0)
    };
    clean(Some(
        samples
            .iter()
            .filter(|(a, _)| *a <= cutoff)
            .map(|(_, r)| r)
            .sum(),
    ))
}

fn compounded_return_series(returns: &[Option<f64>]) -> Vec<Option<f64>> {
    let mut output = vec![None; returns.len()];
    for end in 0..returns.len() {
        let mut gross = 1.0;
        let mut count = 0usize;
        for value in &returns[(end + 1).saturating_sub(REVERSE_WINDOW)..=end] {
            let Some(value) = clean(*value) else {
                continue;
            };
            gross *= 1.0 + value;
            count += 1;
        }
        if count >= 1 {
            output[end] = clean(Some(gross - 1.0));
        }
    }
    output
}

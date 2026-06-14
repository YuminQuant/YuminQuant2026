use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::stock_daily_ops::{is_bj_stock, mask_bj};
use crate::factor::common::{DailyPanel, PanelColumn};
use crate::factor::Factor;

const VERSION: &str = "0.1.0";
const RW: usize = 5;
const HW: usize = 20;
const HOLDING_TIME: usize = 5;
const THRESHOLD: f64 = 0.40;
const MIN_SIMILAR_SAMPLES: usize = 2;
const LOOKBACK: usize = HW + RW - 1;

pub struct StockDailySimilarLowVolatility;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailySimilarLowVolatility)
}

impl Factor for StockDailySimilarLowVolatility {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "similar_low_volatility".to_string(),
            aliases: vec!["SimilarLowVolatility".to_string(), "SLV".to_string()],
            name: "Similar Low Volatility".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: [
                "XNZQ",
                "price_volume",
                "price_pattern",
                "correlation",
                "excess_return",
                "volatility",
                "daily",
            ]
            .iter()
            .map(|value| value.to_string())
            .collect(),
            description: "XNZQ similar low-volatility factor: for each non-BJ stock, match the latest 5-day raw close pattern against 5-day slices in the prior 20-day history by absolute Pearson correlation >= 0.40, then take the population standard deviation of sign-adjusted 5-day cumulative excess returns after the matched slices. Excess return is stock raw-close return minus equal-weight non-BJ A-share return.".to_string(),
            dependencies: vec![DataRequest::new(DatasetId::StockDailyPv, &["close"])],
            intraday_raw_dependencies: Vec::new(),
            lookback: Lookback {
                trading_days: LOOKBACK,
            },
        }
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let panel = data.daily_panel(DatasetId::StockDailyPv)?;
        let close = panel.column("close")?;
        let stock_return = close.ts(price_return_series)?;
        let eligible = eligible_instruments(panel);
        let market_return = equal_weight_market_return(panel, stock_return.values(), &eligible)?;
        let excess_return = stock_return.zip_binary(&market_return, subtract)?;
        let raw = close.ts_binary(&excess_return, similar_low_volatility_series)?;
        let factor = mask_bj(&raw, panel)?;
        Ok(factor.to_factor_series(self.spec()))
    }
}

fn eligible_instruments(panel: &DailyPanel) -> Vec<bool> {
    panel
        .instruments()
        .iter()
        .map(|ts_code| !is_bj_stock(ts_code))
        .collect()
}

fn price_return_series(close: &[Option<f64>]) -> Vec<Option<f64>> {
    let mut output = vec![None; close.len()];
    for idx in 1..close.len() {
        output[idx] = simple_return(close[idx], close[idx - 1]);
    }
    output
}

fn equal_weight_market_return(
    panel: &DailyPanel,
    returns: &[Option<f64>],
    eligible: &[bool],
) -> Result<PanelColumn> {
    let instrument_count = panel.instruments().len();
    let mut values = vec![None; panel.shape_len()];
    for date_idx in 0..panel.dates().len() {
        let offset = date_idx * instrument_count;
        let mut sum = 0.0;
        let mut count = 0usize;
        for instrument_idx in 0..instrument_count {
            if !eligible.get(instrument_idx).copied().unwrap_or(false) {
                continue;
            }
            let Some(value) = finite(returns[offset + instrument_idx]) else {
                continue;
            };
            sum += value;
            count += 1;
        }
        if count == 0 {
            continue;
        }
        let mean = sum / count as f64;
        if !mean.is_finite() {
            continue;
        }
        for instrument_idx in 0..instrument_count {
            values[offset + instrument_idx] = Some(mean);
        }
    }
    panel.column_from_values(values)
}

fn similar_low_volatility_series(
    close: &[Option<f64>],
    excess_returns: &[Option<f64>],
) -> Vec<Option<f64>> {
    let mut output = vec![None; close.len()];
    for idx in 0..close.len() {
        if idx + 1 < LOOKBACK + 1 {
            continue;
        }
        let current_start = idx + 1 - RW;
        let history_start = current_start - HW;
        let Some(current) = collect_window(close, current_start, RW) else {
            continue;
        };

        let mut samples = Vec::with_capacity(HW - RW + 1);
        for candidate_start in history_start..=history_start + HW - RW {
            let Some(history) = collect_window(close, candidate_start, RW) else {
                continue;
            };
            let Some(correlation) = pearson_corr(&current, &history) else {
                continue;
            };
            if correlation.abs() < THRESHOLD {
                continue;
            }
            let future_start = candidate_start + RW;
            let Some(cumulative_return) =
                cumulative_excess_return(excess_returns, future_start, HOLDING_TIME)
            else {
                continue;
            };
            let adjusted = if correlation < 0.0 {
                -cumulative_return
            } else {
                cumulative_return
            };
            if adjusted.is_finite() {
                samples.push(adjusted);
            }
        }

        if samples.len() >= MIN_SIMILAR_SAMPLES {
            output[idx] = population_std(&samples);
        }
    }
    output
}

fn collect_window(values: &[Option<f64>], start: usize, len: usize) -> Option<Vec<f64>> {
    if start.checked_add(len)? > values.len() {
        return None;
    }
    values[start..start + len]
        .iter()
        .map(|value| finite(*value))
        .collect()
}

fn cumulative_excess_return(
    excess_returns: &[Option<f64>],
    start: usize,
    len: usize,
) -> Option<f64> {
    if start.checked_add(len)? > excess_returns.len() {
        return None;
    }
    let mut accumulator = 1.0;
    for idx in start..start + len {
        let value = finite(excess_returns[idx])?;
        accumulator *= 1.0 + value;
        if !accumulator.is_finite() {
            return None;
        }
    }
    Some(accumulator - 1.0).filter(|value| value.is_finite())
}

fn pearson_corr(left: &[f64], right: &[f64]) -> Option<f64> {
    if left.len() != right.len() || left.is_empty() {
        return None;
    }
    let left_mean = mean(left)?;
    let right_mean = mean(right)?;
    let mut numerator = 0.0;
    let mut left_ss = 0.0;
    let mut right_ss = 0.0;
    for (left_value, right_value) in left.iter().zip(right) {
        let left_diff = *left_value - left_mean;
        let right_diff = *right_value - right_mean;
        numerator += left_diff * right_diff;
        left_ss += left_diff * left_diff;
        right_ss += right_diff * right_diff;
    }
    let denominator = (left_ss * right_ss).sqrt();
    if denominator <= f64::EPSILON {
        return None;
    }
    let value = numerator / denominator;
    value.is_finite().then_some(value.clamp(-1.0, 1.0))
}

fn population_std(values: &[f64]) -> Option<f64> {
    let mean = mean(values)?;
    let variance = values
        .iter()
        .map(|value| {
            let diff = *value - mean;
            diff * diff
        })
        .sum::<f64>()
        / values.len() as f64;
    let std = variance.max(0.0).sqrt();
    std.is_finite().then_some(std)
}

fn mean(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let value = values.iter().sum::<f64>() / values.len() as f64;
    value.is_finite().then_some(value)
}

fn simple_return(close: Option<f64>, previous_close: Option<f64>) -> Option<f64> {
    let close = finite(close)?;
    let previous_close = finite(previous_close)?;
    if close <= f64::EPSILON || previous_close <= f64::EPSILON {
        return None;
    }
    let value = close / previous_close - 1.0;
    value.is_finite().then_some(value)
}

fn subtract(left: Option<f64>, right: Option<f64>) -> Option<f64> {
    match (finite(left), finite(right)) {
        (Some(left), Some(right)) => Some(left - right).filter(|value| value.is_finite()),
        _ => None,
    }
}

fn finite(value: Option<f64>) -> Option<f64> {
    value.filter(|value| value.is_finite())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1e-12,
            "actual={actual}, expected={expected}"
        );
    }

    #[test]
    fn similar_low_volatility_spec_has_xnzq_tag_and_close_dependency() {
        let spec = StockDailySimilarLowVolatility.spec();
        assert_eq!(spec.id, "similar_low_volatility");
        assert!(spec.tags.contains(&"XNZQ".to_string()));
        assert_eq!(spec.dependencies.len(), 1);
        assert_eq!(spec.lookback.trading_days, 24);
    }

    #[test]
    fn similar_low_volatility_constants_match_report_windows() {
        assert_eq!(RW, 5);
        assert_eq!(HW, 20);
        assert_eq!(HW - RW + 1, 16);
        assert_eq!(HOLDING_TIME, 5);
        assert_eq!(LOOKBACK, 24);
    }

    #[test]
    fn equal_weight_market_return_excludes_bj_stocks() {
        let panel = DailyPanel::from_index(
            vec![20200102],
            vec![
                "000001.SZ".to_string(),
                "600000.SH".to_string(),
                "430001.BJ".to_string(),
            ],
            &[20200102],
            vec![true; 3],
        )
        .expect("panel");
        let eligible = eligible_instruments(&panel);
        assert_eq!(eligible, vec![true, true, false]);

        let returns = vec![Some(0.10), Some(0.30), Some(0.90)];
        let market =
            equal_weight_market_return(&panel, &returns, &eligible).expect("market return");

        for value in market.values() {
            assert_close(value.unwrap(), 0.20);
        }
    }

    #[test]
    fn similar_low_volatility_uses_abs_corr_and_sign_corrects_negative_matches() {
        let mut close = vec![None; LOOKBACK + 1];
        for (idx, value) in [1.0, 2.0, 3.0, 4.0, 5.0].iter().enumerate() {
            close[idx] = Some(*value);
            close[20 + idx] = Some(*value);
        }
        for (idx, value) in [5.0, 4.0, 3.0, 2.0, 1.0].iter().enumerate() {
            close[10 + idx] = Some(*value);
        }

        let mut excess_returns = vec![Some(0.0); LOOKBACK + 1];
        for idx in 5..10 {
            excess_returns[idx] = Some(0.01);
        }
        for idx in 15..20 {
            excess_returns[idx] = Some(0.02);
        }

        let output = similar_low_volatility_series(&close, &excess_returns);
        let positive = (1.01_f64).powi(HOLDING_TIME as i32) - 1.0;
        let negative_source = (1.02_f64).powi(HOLDING_TIME as i32) - 1.0;
        let expected = (positive + negative_source) / 2.0;

        assert_close(output[LOOKBACK].unwrap(), expected);
    }

    #[test]
    fn similar_low_volatility_requires_two_similar_samples() {
        let mut close = vec![None; LOOKBACK + 1];
        for (idx, value) in [1.0, 2.0, 3.0, 4.0, 5.0].iter().enumerate() {
            close[idx] = Some(*value);
            close[20 + idx] = Some(*value);
        }
        let excess_returns = vec![Some(0.01); LOOKBACK + 1];

        let output = similar_low_volatility_series(&close, &excess_returns);

        assert_eq!(output[LOOKBACK], None);
    }

    #[test]
    fn pearson_corr_rejects_flat_windows() {
        assert_eq!(
            pearson_corr(&[1.0, 1.0, 1.0, 1.0, 1.0], &[1.0, 2.0, 3.0, 4.0, 5.0]),
            None
        );
        assert_close(
            pearson_corr(&[1.0, 2.0, 3.0, 4.0, 5.0], &[5.0, 4.0, 3.0, 2.0, 1.0]).unwrap(),
            -1.0,
        );
    }
}

use crate::core::DatasetId;
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::panel::{DailyPanel, PanelColumn};
use crate::factor::common::vector::clean;

pub const CHIP_WINDOW: usize = 250;
pub const RET_WINDOW: usize = 20;
pub const MIN_PERIODS: usize = 1;

pub fn daily_vwap(amount_thousand_yuan: Option<f64>, volume_lots: Option<f64>) -> Option<f64> {
    match (finite(amount_thousand_yuan), finite(volume_lots)) {
        (Some(amount), Some(volume)) if volume.abs() > f64::EPSILON => Some(amount * 10.0 / volume),
        _ => None,
    }
}

pub fn percent_to_decimal(value: Option<f64>) -> Option<f64> {
    finite(value).map(|value| value / 100.0)
}

pub fn adjusted_close(close: &PanelColumn, adj_factor: &PanelColumn) -> Result<PanelColumn> {
    close.zip_binary(adj_factor, multiply)
}

pub fn ret20_from_adjusted_close(adj_close: &PanelColumn) -> Result<PanelColumn> {
    adj_close.ts(|values| rolling_price_return(values, RET_WINDOW, MIN_PERIODS))
}

pub fn std20_from_adjusted_close(adj_close: &PanelColumn) -> Result<PanelColumn> {
    adj_close.ts(|values| {
        let returns = daily_returns(values);
        crate::operators::ts_std_dev(&returns, RET_WINDOW, MIN_PERIODS)
    })
}

pub fn holding_ret(
    panel: &DailyPanel,
    price_avg: &PanelColumn,
    amount: &PanelColumn,
    turnover: &PanelColumn,
) -> Result<PanelColumn> {
    let date_count = panel.dates().len();
    let instrument_count = panel.instruments().len();
    let mut output = vec![None; panel.shape_len()];

    for instrument_idx in 0..instrument_count {
        for date_idx in 0..date_count {
            let offset = date_idx * instrument_count + instrument_idx;
            let Some(current_price) = finite(price_avg.values()[offset]) else {
                continue;
            };

            let Some(cost) = chip_cost_for_date(
                date_idx,
                instrument_idx,
                instrument_count,
                price_avg.values(),
                amount.values(),
                turnover.values(),
            ) else {
                continue;
            };
            if cost.abs() <= f64::EPSILON {
                continue;
            }
            output[offset] = Some(current_price / cost - 1.0);
        }
    }

    panel.column_from_values(output)
}

pub fn holding_ret_from_data(data: &DataPool) -> Result<(DailyPanel, PanelColumn)> {
    let panel = data.daily_panel(DatasetId::StockDailyPv)?;
    let amount = panel.column("amount")?;
    let volume = panel.column("vol")?;
    let price_avg = amount.zip_binary(&volume, daily_vwap)?;
    let turnover = panel
        .column_from_table(data.daily(DatasetId::StockDailyBasic)?, "turnover_rate_f")?
        .map_values(percent_to_decimal);
    let holding = holding_ret(&panel, &price_avg, &amount, &turnover)?;
    Ok((panel.clone(), holding))
}

pub fn ret20_from_data(panel: &DailyPanel, data: &DataPool) -> Result<PanelColumn> {
    let close = panel.column("close")?;
    let adj_factor =
        panel.column_from_table(data.daily(DatasetId::StockAdjFactor)?, "adj_factor")?;
    let adj_close = adjusted_close(&close, &adj_factor)?;
    ret20_from_adjusted_close(&adj_close)
}

pub fn cross_section_mean_constant(values: &PanelColumn) -> Result<PanelColumn> {
    values.cs(|cross_section| {
        let valid = cross_section
            .iter()
            .filter_map(|value| finite(*value))
            .collect::<Vec<_>>();
        if valid.is_empty() {
            return vec![None; cross_section.len()];
        }
        let mean = valid.iter().sum::<f64>() / valid.len() as f64;
        vec![Some(mean); cross_section.len()]
    })
}

pub fn sign_adjust(value: Option<f64>, signal: Option<f64>) -> Option<f64> {
    match (finite(value), finite(signal)) {
        (Some(value), Some(signal)) if signal > 0.0 => Some(value),
        (Some(value), Some(signal)) if signal < 0.0 => Some(-value),
        (Some(_), Some(_)) => Some(0.0),
        _ => None,
    }
}

pub fn sign_adjust_minus_2pct(value: Option<f64>, market: Option<f64>) -> Option<f64> {
    finite(market)
        .map(|market| market + 0.02)
        .and_then(|signal| sign_adjust(value, Some(signal)))
}

pub fn enhanced(ret20: Option<f64>, adjusted_holding_ret: Option<f64>) -> Option<f64> {
    match (finite(ret20), finite(adjusted_holding_ret)) {
        (Some(ret20), Some(adjusted_holding_ret)) => {
            Some(ret20 * (1.0 - ret20) + (1.0 - ret20) * adjusted_holding_ret)
        }
        _ => None,
    }
}

pub fn finite(value: Option<f64>) -> Option<f64> {
    clean(value).filter(|value| value.is_finite())
}

fn chip_cost_for_date(
    date_idx: usize,
    instrument_idx: usize,
    instrument_count: usize,
    price_avg: &[Option<f64>],
    amount: &[Option<f64>],
    turnover: &[Option<f64>],
) -> Option<f64> {
    if date_idx == 0 {
        return None;
    }

    let start = date_idx.saturating_sub(CHIP_WINDOW);
    let mut survival = 1.0;
    let mut numerator = 0.0;
    let mut denominator = 0.0;

    for chip_date_idx in (start..date_idx).rev() {
        let survival_offset = (chip_date_idx + 1) * instrument_count + instrument_idx;
        let Some(turnover_value) = finite(turnover[survival_offset]) else {
            break;
        };
        survival *= 1.0 - turnover_value;
        if !survival.is_finite() {
            break;
        }

        let chip_offset = chip_date_idx * instrument_count + instrument_idx;
        let (Some(price), Some(amount_value)) =
            (finite(price_avg[chip_offset]), finite(amount[chip_offset]))
        else {
            continue;
        };
        let retained_amount = amount_value * survival;
        numerator += price * retained_amount;
        denominator += retained_amount;
    }

    if denominator.abs() > f64::EPSILON {
        Some(numerator / denominator)
    } else {
        None
    }
}

fn rolling_price_return(
    values: &[Option<f64>],
    window: usize,
    min_periods: usize,
) -> Vec<Option<f64>> {
    let mut output = vec![None; values.len()];
    for idx in 0..values.len() {
        let Some(current) = finite(values[idx]) else {
            continue;
        };
        let start = idx.saturating_sub(window);
        let mut prior_count = 0usize;
        let mut base = None;
        for candidate in start..idx {
            if let Some(value) = finite(values[candidate]) {
                prior_count += 1;
                if base.is_none() {
                    base = Some(value);
                }
            }
        }
        let Some(base) = base else {
            continue;
        };
        if prior_count >= min_periods && base.abs() > f64::EPSILON {
            output[idx] = Some(current / base - 1.0);
        }
    }
    output
}

fn daily_returns(values: &[Option<f64>]) -> Vec<Option<f64>> {
    let mut output = vec![None; values.len()];
    for idx in 1..values.len() {
        let (Some(current), Some(previous)) = (finite(values[idx]), finite(values[idx - 1])) else {
            continue;
        };
        if previous.abs() > f64::EPSILON {
            output[idx] = Some(current / previous - 1.0);
        }
    }
    output
}

fn multiply(left: Option<f64>, right: Option<f64>) -> Option<f64> {
    match (finite(left), finite(right)) {
        (Some(left), Some(right)) => Some(left * right),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: Option<f64>, expected: f64) {
        let actual = actual.expect("expected value");
        assert!(
            (actual - expected).abs() < 1e-12,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn daily_vwap_uses_daily_amount_and_volume_units() {
        assert_close(daily_vwap(Some(1000.0), Some(100.0)), 100.0);
        assert_eq!(daily_vwap(Some(1.0), Some(0.0)), None);
    }

    #[test]
    fn rolling_price_return_uses_available_prior_price_with_min_periods_one() {
        let values = vec![Some(100.0), Some(110.0), Some(121.0)];
        let returns = rolling_price_return(&values, 20, 1);

        assert_eq!(returns[0], None);
        assert_close(returns[1], 0.1);
        assert_close(returns[2], 0.21);
    }

    #[test]
    fn chip_cost_excludes_current_day_and_applies_current_turnover_survival() {
        let price = vec![Some(10.0), Some(20.0), Some(30.0)];
        let amount = vec![Some(100.0), Some(100.0), Some(100.0)];
        let turnover = vec![Some(0.1), Some(0.2), Some(0.5)];

        let cost = chip_cost_for_date(2, 0, 1, &price, &amount, &turnover);
        let expected =
            (20.0 * 100.0 * 0.5 + 10.0 * 100.0 * 0.8 * 0.5) / (100.0 * 0.5 + 100.0 * 0.8 * 0.5);
        assert_close(cost, expected);
    }

    #[test]
    fn sign_adjust_minus_two_percent_uses_shifted_market_threshold() {
        assert_close(sign_adjust_minus_2pct(Some(3.0), Some(-0.019)), 3.0);
        assert_close(sign_adjust_minus_2pct(Some(3.0), Some(-0.021)), -3.0);
    }
}

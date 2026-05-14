use crate::core::DatasetId;
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::vector::clean;
use crate::factor::common::PanelColumn;
use crate::factor::common::{ClassificationLevel, ClassificationMap, DailyPanel};
use crate::operators::{cs_regression_residual, ts_mean};

pub const PLUS_TURNOVER_WINDOW: usize = 20;
pub const PLUS_TURNOVER_MIN_PERIODS: usize = 1;
const ROLLING_MEAN_DESIZE_WINDOW: usize = 20;
const ROLLING_MEAN_DESIZE_MIN_PERIODS: usize = 1;
const RET20_WINDOW: usize = 20;

pub fn rolling_mean_desize(values: PanelColumn, size: &PanelColumn) -> Result<PanelColumn> {
    values
        .ts(|series| {
            ts_mean(
                series,
                ROLLING_MEAN_DESIZE_WINDOW,
                ROLLING_MEAN_DESIZE_MIN_PERIODS,
            )
        })?
        .cs_neutralize_regression(&[size], None)
}

pub fn is_bj_stock(ts_code: &str) -> bool {
    ts_code.to_ascii_uppercase().ends_with(".BJ")
}

pub fn mask_bj(values: &PanelColumn, panel: &DailyPanel) -> Result<PanelColumn> {
    let instrument_count = panel.instruments().len();
    let instruments = panel.instruments();
    let masked = values
        .values()
        .iter()
        .enumerate()
        .map(|(idx, value)| {
            let instrument_idx = idx % instrument_count;
            if is_bj_stock(&instruments[instrument_idx]) {
                None
            } else {
                *value
            }
        })
        .collect();
    panel.column_from_values(masked)
}

pub fn neutralize_size_sector(
    values: &PanelColumn,
    panel: &DailyPanel,
    data: &DataPool,
) -> Result<PanelColumn> {
    let size = panel.column_from_table(data.daily(DatasetId::StockBarraDaily)?, "SIZE")?;
    let sector_map = ClassificationMap::from_table(
        data.daily(DatasetId::StockSwClassification)?,
        ClassificationLevel::Sector,
    )?;
    values.cs_neutralize_regression_by_group(&[&size], None, |trade_date, ts_codes| {
        sector_map.groups_for(trade_date, ts_codes)
    })
}

pub fn adjusted_20d_return(data: &DataPool, panel: &DailyPanel) -> Result<PanelColumn> {
    let close = panel.column_from_table(data.daily(DatasetId::StockDailyPv)?, "close")?;
    let adj_factor =
        panel.column_from_table(data.daily(DatasetId::StockAdjFactor)?, "adj_factor")?;
    let adj_close = close.zip_binary(&adj_factor, multiply_pair_value)?;
    adj_close.ts(|series| exact_price_return(series, RET20_WINDOW))
}

pub fn neutralize_ret20_size_sector(
    values: &PanelColumn,
    panel: &DailyPanel,
    data: &DataPool,
) -> Result<PanelColumn> {
    let masked = mask_bj(values, panel)?;
    let ret20 = adjusted_20d_return(data, panel)?;
    let size = panel.column_from_table(data.daily(DatasetId::StockBarraDaily)?, "SIZE")?;
    let sector_map = ClassificationMap::from_table(
        data.daily(DatasetId::StockSwClassification)?,
        ClassificationLevel::Sector,
    )?;
    let neutralized = masked.cs_neutralize_regression_by_group(
        &[&ret20, &size],
        None,
        |trade_date, ts_codes| sector_map.groups_for(trade_date, ts_codes),
    )?;
    mask_bj(&neutralized, panel)
}

pub fn plus_factor(
    close: &PanelColumn,
    high: &PanelColumn,
    low: &PanelColumn,
    pre_close: &PanelColumn,
) -> Result<PanelColumn> {
    close
        .zip_ternary(high, low, |close, high, low| {
            match (clean(close), clean(high), clean(low)) {
                (Some(close), Some(high), Some(low)) => Some(2.0 * close - high - low),
                _ => None,
            }
        })?
        .zip_binary(pre_close, safe_div)
}

pub fn turn_deplus(turnover: &PanelColumn, plus: &PanelColumn) -> Result<PanelColumn> {
    turnover.cs_binary(plus, cs_regression_residual)
}

pub fn plus_deturn(plus: &PanelColumn, turnover: &PanelColumn) -> Result<PanelColumn> {
    plus.cs_binary(turnover, cs_regression_residual)
}

pub fn nonnegative_shift(values: &[Option<f64>]) -> Vec<Option<f64>> {
    let Some(min_value) = values
        .iter()
        .filter_map(|value| clean(*value))
        .reduce(f64::min)
    else {
        return vec![None; values.len()];
    };
    values
        .iter()
        .map(|value| clean(*value).map(|value| value - min_value))
        .collect()
}

pub fn multiply_pair(left: &PanelColumn, right: &PanelColumn) -> Result<PanelColumn> {
    left.zip_binary(right, |left, right| match (clean(left), clean(right)) {
        (Some(left), Some(right)) => Some(left * right),
        _ => None,
    })
}

fn multiply_pair_value(left: Option<f64>, right: Option<f64>) -> Option<f64> {
    match (clean(left), clean(right)) {
        (Some(left), Some(right)) => Some(left * right),
        _ => None,
    }
}

fn exact_price_return(values: &[Option<f64>], window: usize) -> Vec<Option<f64>> {
    let mut output = vec![None; values.len()];
    for idx in window..values.len() {
        let (Some(current), Some(previous)) = (clean(values[idx]), clean(values[idx - window]))
        else {
            continue;
        };
        if previous.abs() > f64::EPSILON {
            let value = current / previous - 1.0;
            if value.is_finite() {
                output[idx] = Some(value);
            }
        }
    }
    output
}

fn safe_div(numerator: Option<f64>, denominator: Option<f64>) -> Option<f64> {
    match (clean(numerator), clean(denominator)) {
        (Some(numerator), Some(denominator)) if denominator.abs() > f64::EPSILON => {
            Some(numerator / denominator)
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: Option<f64>, expected: Option<f64>) {
        match (actual, expected) {
            (Some(actual), Some(expected)) => assert!(
                (actual - expected).abs() < 1e-10,
                "expected {expected}, got {actual}"
            ),
            (None, None) => {}
            _ => panic!("expected {:?}, got {:?}", expected, actual),
        }
    }

    #[test]
    fn stock_daily_ops_safe_div_rejects_zero_denominator() {
        assert_eq!(safe_div(Some(1.0), Some(0.0)), None);
        assert_close(safe_div(Some(6.0), Some(3.0)), Some(2.0));
    }

    #[test]
    fn stock_daily_ops_identifies_bj_suffix_case_insensitively() {
        assert!(is_bj_stock("920087.BJ"));
        assert!(is_bj_stock("920087.bj"));
        assert!(!is_bj_stock("000001.SZ"));
    }

    #[test]
    fn stock_daily_ops_exact_price_return_requires_exact_window_price() {
        let mut values = vec![None; 21];
        values[0] = Some(100.0);
        values[19] = Some(999.0);
        values[20] = Some(110.0);

        let returns = exact_price_return(&values, 20);

        assert_eq!(returns[19], None);
        assert_close(returns[20], Some(0.10));
    }

    #[test]
    fn stock_daily_ops_nonnegative_shift_preserves_order_and_sets_min_to_zero() {
        let shifted = nonnegative_shift(&[Some(-1.5), Some(0.5), None, Some(2.0)]);

        assert_close(shifted[0], Some(0.0));
        assert_close(shifted[1], Some(2.0));
        assert_eq!(shifted[2], None);
        assert_close(shifted[3], Some(3.5));
    }
}

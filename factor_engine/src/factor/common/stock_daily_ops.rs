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

/// Equivalent to regression-prediction imputation followed by the same neutralization.
pub fn neutralize_and_fill_missing(
    raw: &[Option<f64>],
    size: &[Option<f64>],
    groups: &[Option<String>],
) -> Vec<Option<f64>> {
    if size.len() != raw.len() || groups.len() != raw.len() {
        return vec![None; raw.len()];
    }
    let mut residual = crate::operators::cross_sectional::cs_neutralize_regression(
        raw,
        &[size],
        Some(groups),
        None,
    );
    let supported: std::collections::HashSet<&str> = residual
        .iter()
        .enumerate()
        .filter(|(_, v)| v.is_some_and(f64::is_finite))
        .filter_map(|(i, _)| groups[i].as_deref())
        .collect();
    for i in 0..raw.len() {
        if raw[i].filter(|v| v.is_finite()).is_none()
            && size[i].is_some_and(f64::is_finite)
            && groups[i].as_deref().is_some_and(|g| supported.contains(g))
        {
            residual[i] = Some(0.0);
        }
    }
    residual
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
    neutralize_size_sector_with_inputs(values, panel, &size, &sector_map)
}

pub fn neutralize_size_only(
    values: &PanelColumn,
    panel: &DailyPanel,
    data: &DataPool,
) -> Result<PanelColumn> {
    let size = panel.column_from_table(data.daily(DatasetId::StockBarraDaily)?, "SIZE")?;
    values.cs_neutralize_regression(&[&size], None)
}

pub fn neutralize_size_sector_with_inputs(
    values: &PanelColumn,
    _panel: &DailyPanel,
    size: &PanelColumn,
    sector_map: &ClassificationMap,
) -> Result<PanelColumn> {
    values.cs_neutralize_regression_by_group(&[&size], None, |trade_date, ts_codes| {
        sector_map.groups_for(trade_date, ts_codes)
    })
}

pub fn adjusted_20d_return(data: &DataPool, panel: &DailyPanel) -> Result<PanelColumn> {
    let close = panel.column_from_table(data.daily(DatasetId::StockDailyPv)?, "close")?;
    let adj_factor =
        panel.column_from_table(data.daily(DatasetId::StockAdjFactor)?, "adj_factor")?;
    let adj_close = close.zip_binary(&adj_factor, multiply_pair_value)?;
    adj_close.ts(|series| inclusive_price_return(series, RET20_WINDOW))
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

fn inclusive_price_return(values: &[Option<f64>], observations: usize) -> Vec<Option<f64>> {
    let mut output = vec![None; values.len()];
    if observations < 2 {
        return output;
    }
    let offset = observations - 1;
    for idx in offset..values.len() {
        let (Some(current), Some(previous)) = (clean(values[idx]), clean(values[idx - offset]))
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

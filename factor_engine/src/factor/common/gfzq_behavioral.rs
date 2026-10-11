use crate::core::DatasetId;
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::chip::{daily_vwap, percent_to_decimal};
use crate::factor::common::stock_daily_ops::{is_bj_stock, neutralize_size_sector};
use crate::factor::common::vector::clean;
use crate::factor::common::{DailyPanel, PanelColumn};
use crate::operators::cs_zscore;

pub const CGO_WINDOW: usize = 100;
pub const CGO_MIN_PERIODS: usize = 50;
pub const LOSS_WINDOW: usize = 100;
pub const LOSS_MIN_PERIODS: usize = 50;
pub const ST_WINDOW: usize = 20;
pub const ST_MIN_PERIODS: usize = 1;
pub const ST_THETA: f64 = 0.9;
pub const ST_DELTA: f64 = 0.9;
pub const SALIENCE_WINDOW: usize = 20;
pub const SALIENCE_MIN_PERIODS: usize = 1;
pub const SALIENCE_THETA: f64 = 0.9;
pub const SALIENCE_DELTA: f64 = 0.9;
pub const STV_RETURN_THRESHOLD: f64 = 0.07;

const EPS: f64 = 1e-12;

pub fn cgo_from_data(data: &DataPool) -> Result<(DailyPanel, PanelColumn)> {
    let panel = data.daily_panel(DatasetId::StockDailyPv)?;
    let close = panel.column("close")?;
    let amount = panel.column("amount")?;
    let vol = panel.column("vol")?;
    let vwap = amount.zip_binary(&vol, daily_vwap)?;
    let turnover = panel
        .column_from_table(data.daily(DatasetId::StockDailyBasic)?, "turnover_rate_f")?
        .map_values(percent_to_decimal);
    let cgo = cgo_panel(&panel, &close, &vwap, &turnover)?;
    Ok((panel.clone(), cgo))
}

pub fn st_salience_from_data(data: &DataPool) -> Result<(DailyPanel, PanelColumn)> {
    let panel = data.daily_panel(DatasetId::StockDailyPv)?;
    let close = panel.column("close")?;
    let adj_factor =
        panel.column_from_table(data.daily(DatasetId::StockAdjFactor)?, "adj_factor")?;
    let adj_close = close.zip_binary(&adj_factor, multiply)?;
    let returns = daily_returns(&panel, &adj_close)?;
    let market_returns = market_equal_returns_ex_bj(&panel, &returns);
    let st = st_salience_panel(&panel, &returns, &market_returns)?;
    Ok((panel.clone(), st))
}

pub fn loss_from_data(data: &DataPool) -> Result<(DailyPanel, PanelColumn)> {
    let panel = data.daily_panel(DatasetId::StockDailyPv)?;
    let amount = panel.column("amount")?;
    let vol = panel.column("vol")?;
    let vwap = amount.zip_binary(&vol, daily_vwap)?;
    let turnover = panel
        .column_from_table(data.daily(DatasetId::StockDailyBasic)?, "turnover_rate_f")?
        .map_values(percent_to_decimal);
    let loss = loss_panel(&panel, &vwap, &turnover)?;
    Ok((panel.clone(), loss))
}

pub fn salience_return_from_data(data: &DataPool) -> Result<(DailyPanel, PanelColumn)> {
    let panel = data.daily_panel(DatasetId::StockDailyPv)?;
    let close = panel.column("close")?;
    let adj_factor =
        panel.column_from_table(data.daily(DatasetId::StockAdjFactor)?, "adj_factor")?;
    let adj_close = close.zip_binary(&adj_factor, multiply)?;
    let returns = daily_returns(&panel, &adj_close)?;
    let market_returns = market_equal_returns_ex_bj(&panel, &returns);
    let raw = salience_return_panel(&panel, &returns, &market_returns)?;
    let factor = postprocess_salience_factor(data, &panel, &raw)?;
    Ok((panel.clone(), factor))
}

pub fn stt_from_data(data: &DataPool) -> Result<(DailyPanel, PanelColumn)> {
    let panel = data.daily_panel(DatasetId::StockDailyPv)?;
    let close = panel.column("close")?;
    let adj_factor =
        panel.column_from_table(data.daily(DatasetId::StockAdjFactor)?, "adj_factor")?;
    let adj_close = close.zip_binary(&adj_factor, multiply)?;
    let returns = daily_returns(&panel, &adj_close)?;
    let turnover = panel
        .column_from_table(data.daily(DatasetId::StockDailyBasic)?, "turnover_rate_f")?
        .map_values(percent_to_decimal);
    let market_turnover = market_equal_values_ex_bj(&panel, &turnover);
    let raw = stt_panel(&panel, &returns, &turnover, &market_turnover)?;
    let factor = postprocess_salience_factor(data, &panel, &raw)?;
    Ok((panel.clone(), factor))
}

pub fn stt2_from_data(data: &DataPool) -> Result<(DailyPanel, PanelColumn)> {
    let panel = data.daily_panel(DatasetId::StockDailyBasic)?;
    let turnover = panel
        .column("turnover_rate_f")?
        .map_values(percent_to_decimal);
    let market_turnover = market_equal_values_ex_bj(&panel, &turnover);
    let raw = stt2_panel(&panel, &turnover, &market_turnover)?;
    let factor = postprocess_salience_factor(data, &panel, &raw)?;
    Ok((panel.clone(), factor))
}

pub fn stv_from_data(data: &DataPool) -> Result<(DailyPanel, PanelColumn)> {
    let panel = data.daily_panel(DatasetId::StockDailyPv)?;
    let close = panel.column("close")?;
    let adj_factor =
        panel.column_from_table(data.daily(DatasetId::StockAdjFactor)?, "adj_factor")?;
    let adj_close = close.zip_binary(&adj_factor, multiply)?;
    let returns = daily_returns(&panel, &adj_close)?;
    let turnover = panel
        .column_from_table(data.daily(DatasetId::StockDailyBasic)?, "turnover_rate_f")?
        .map_values(percent_to_decimal);
    let raw = stv_panel(&panel, &returns, &turnover)?;
    let factor = postprocess_salience_factor(data, &panel, &raw)?;
    Ok((panel.clone(), factor))
}

pub fn cgo_panel(
    panel: &DailyPanel,
    close: &PanelColumn,
    vwap: &PanelColumn,
    turnover: &PanelColumn,
) -> Result<PanelColumn> {
    let date_count = panel.dates().len();
    let instrument_count = panel.instruments().len();
    let mut output = vec![None; panel.shape_len()];

    for instrument_idx in 0..instrument_count {
        for date_idx in 1..date_count {
            let offset = date_idx * instrument_count + instrument_idx;
            let prev_offset = (date_idx - 1) * instrument_count + instrument_idx;
            let Some(prev_close) = finite(close.values()[prev_offset]) else {
                continue;
            };
            let Some(reference_price) =
                cgo_reference_price(date_idx, instrument_idx, instrument_count, vwap, turnover)
            else {
                continue;
            };
            if reference_price.abs() <= EPS {
                continue;
            }
            output[offset] = finite_value((prev_close - reference_price) / reference_price);
        }
    }

    panel.column_from_values(output)
}

pub fn loss_panel(
    panel: &DailyPanel,
    vwap: &PanelColumn,
    turnover: &PanelColumn,
) -> Result<PanelColumn> {
    let date_count = panel.dates().len();
    let instrument_count = panel.instruments().len();
    let mut output = vec![None; panel.shape_len()];

    for instrument_idx in 0..instrument_count {
        for date_idx in 0..date_count {
            let offset = date_idx * instrument_count + instrument_idx;
            let Some(current_price) = finite(vwap.values()[offset]).filter(|value| *value > EPS)
            else {
                continue;
            };
            output[offset] = loss_for_date(
                date_idx,
                instrument_idx,
                instrument_count,
                current_price,
                vwap,
                turnover,
            );
        }
    }

    panel.column_from_values(output)
}

fn cgo_reference_price(
    date_idx: usize,
    instrument_idx: usize,
    instrument_count: usize,
    vwap: &PanelColumn,
    turnover: &PanelColumn,
) -> Option<f64> {
    let start = date_idx.saturating_sub(CGO_WINDOW);
    let mut survival = 1.0;
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    let mut valid_count = 0usize;

    for hist_date_idx in (start..date_idx).rev() {
        let offset = hist_date_idx * instrument_count + instrument_idx;
        let turnover_value = finite(turnover.values()[offset]);

        if let (Some(price), Some(turnover_value)) = (finite(vwap.values()[offset]), turnover_value)
        {
            valid_count += 1;
            let weight = turnover_value * survival;
            if weight.is_finite() && weight > 0.0 {
                numerator += price * weight;
                denominator += weight;
            }
        }

        if let Some(turnover_value) = turnover_value {
            survival *= 1.0 - turnover_value;
            if !survival.is_finite() {
                break;
            }
        }
    }

    if valid_count < CGO_MIN_PERIODS || denominator.abs() <= EPS {
        None
    } else {
        finite_value(numerator / denominator)
    }
}

fn loss_for_date(
    date_idx: usize,
    instrument_idx: usize,
    instrument_count: usize,
    current_price: f64,
    vwap: &PanelColumn,
    turnover: &PanelColumn,
) -> Option<f64> {
    let start = date_idx.saturating_sub(LOSS_WINDOW);
    let mut survival = 1.0;
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    let mut valid_count = 0usize;

    for hist_date_idx in (start..date_idx).rev() {
        let offset = hist_date_idx * instrument_count + instrument_idx;
        let turnover_value = finite(turnover.values()[offset]);

        if let (Some(history_price), Some(turnover_value)) = (
            finite(vwap.values()[offset]).filter(|value| *value > EPS),
            turnover_value,
        ) {
            valid_count += 1;
            let weight = turnover_value * survival;
            if weight.is_finite() && weight > 0.0 {
                let loss = if current_price < history_price {
                    (history_price - current_price) / history_price
                } else {
                    0.0
                };
                numerator += loss * weight;
                denominator += weight;
            }
        }

        if let Some(turnover_value) = turnover_value {
            survival *= 1.0 - turnover_value;
            if !survival.is_finite() {
                break;
            }
        }
    }

    if valid_count < LOSS_MIN_PERIODS || denominator.abs() <= EPS {
        None
    } else {
        finite_value(numerator / denominator)
    }
}

pub fn st_salience_panel(
    panel: &DailyPanel,
    returns: &PanelColumn,
    market_returns: &[Option<f64>],
) -> Result<PanelColumn> {
    let date_count = panel.dates().len();
    let instrument_count = panel.instruments().len();
    let mut output = vec![None; panel.shape_len()];

    for instrument_idx in 0..instrument_count {
        for date_idx in 0..date_count {
            let offset = date_idx * instrument_count + instrument_idx;
            output[offset] = st_salience_for_date(
                date_idx,
                instrument_idx,
                instrument_count,
                returns.values(),
                market_returns,
            );
        }
    }

    panel.column_from_values(output)
}

pub fn salience_return_panel(
    panel: &DailyPanel,
    returns: &PanelColumn,
    market_returns: &[Option<f64>],
) -> Result<PanelColumn> {
    let date_count = panel.dates().len();
    let instrument_count = panel.instruments().len();
    let mut output = vec![None; panel.shape_len()];

    for instrument_idx in 0..instrument_count {
        for date_idx in 0..date_count {
            let offset = date_idx * instrument_count + instrument_idx;
            output[offset] = salience_return_for_date(
                date_idx,
                instrument_idx,
                instrument_count,
                returns.values(),
                market_returns,
            );
        }
    }

    panel.column_from_values(output)
}

pub fn stt_panel(
    panel: &DailyPanel,
    returns: &PanelColumn,
    turnover: &PanelColumn,
    market_turnover: &[Option<f64>],
) -> Result<PanelColumn> {
    let date_count = panel.dates().len();
    let instrument_count = panel.instruments().len();
    let mut output = vec![None; panel.shape_len()];

    for instrument_idx in 0..instrument_count {
        for date_idx in 0..date_count {
            let offset = date_idx * instrument_count + instrument_idx;
            output[offset] = turnover_salience_for_date(
                date_idx,
                instrument_idx,
                instrument_count,
                returns.values(),
                turnover.values(),
                market_turnover,
            );
        }
    }

    panel.column_from_values(output)
}

pub fn stt2_panel(
    panel: &DailyPanel,
    turnover: &PanelColumn,
    market_turnover: &[Option<f64>],
) -> Result<PanelColumn> {
    let date_count = panel.dates().len();
    let instrument_count = panel.instruments().len();
    let mut output = vec![None; panel.shape_len()];

    for instrument_idx in 0..instrument_count {
        for date_idx in 0..date_count {
            let offset = date_idx * instrument_count + instrument_idx;
            output[offset] = turnover_salience_for_date(
                date_idx,
                instrument_idx,
                instrument_count,
                turnover.values(),
                turnover.values(),
                market_turnover,
            );
        }
    }

    panel.column_from_values(output)
}

pub fn stv_panel(
    panel: &DailyPanel,
    returns: &PanelColumn,
    turnover: &PanelColumn,
) -> Result<PanelColumn> {
    let date_count = panel.dates().len();
    let instrument_count = panel.instruments().len();
    let mut output = vec![None; panel.shape_len()];

    for instrument_idx in 0..instrument_count {
        for date_idx in 0..date_count {
            let offset = date_idx * instrument_count + instrument_idx;
            output[offset] = stv_for_date(
                date_idx,
                instrument_idx,
                instrument_count,
                returns.values(),
                turnover.values(),
            );
        }
    }

    panel.column_from_values(output)
}

fn st_salience_for_date(
    date_idx: usize,
    instrument_idx: usize,
    instrument_count: usize,
    returns: &[Option<f64>],
    market_returns: &[Option<f64>],
) -> Option<f64> {
    let start = (date_idx + 1).saturating_sub(ST_WINDOW);
    let mut rows = Vec::with_capacity(ST_WINDOW);
    for hist_date_idx in start..=date_idx {
        let offset = hist_date_idx * instrument_count + instrument_idx;
        let (Some(stock_ret), Some(market_ret)) = (
            finite(returns[offset]),
            finite(market_returns[hist_date_idx]),
        ) else {
            continue;
        };
        let salience = salience_value(stock_ret, market_ret)?;
        rows.push((stock_ret, salience));
    }
    if rows.len() < ST_MIN_PERIODS {
        return None;
    }

    let ranks = descending_pct_ranks(rows.iter().map(|row| row.1).collect::<Vec<_>>().as_slice());
    let delta_powers = ranks
        .iter()
        .map(|rank| ST_DELTA.powf(*rank))
        .collect::<Vec<_>>();
    let mean_delta = delta_powers.iter().sum::<f64>() / delta_powers.len() as f64;
    if mean_delta.abs() <= EPS || !mean_delta.is_finite() {
        return None;
    }

    let mean_ret = rows.iter().map(|row| row.0).sum::<f64>() / rows.len() as f64;
    let weighted_mean = rows
        .iter()
        .zip(delta_powers.iter())
        .map(|(row, delta_power)| row.0 * delta_power / mean_delta)
        .sum::<f64>()
        / rows.len() as f64;
    finite_value(weighted_mean - mean_ret)
}

fn salience_value(stock_ret: f64, market_ret: f64) -> Option<f64> {
    let denominator = stock_ret.abs() + market_ret.abs() + ST_THETA;
    if denominator.abs() <= EPS {
        return None;
    }
    finite_value(((stock_ret - market_ret).abs() / denominator) * (stock_ret - market_ret).exp())
}

fn salience_return_for_date(
    date_idx: usize,
    instrument_idx: usize,
    instrument_count: usize,
    returns: &[Option<f64>],
    market_returns: &[Option<f64>],
) -> Option<f64> {
    let start = (date_idx + 1).saturating_sub(SALIENCE_WINDOW);
    let mut rows = Vec::with_capacity(SALIENCE_WINDOW);
    for hist_date_idx in start..=date_idx {
        let offset = hist_date_idx * instrument_count + instrument_idx;
        let (Some(stock_ret), Some(market_ret)) = (
            finite(returns[offset]),
            finite(market_returns[hist_date_idx]),
        ) else {
            continue;
        };
        let salience = salience_value_without_exp(stock_ret, market_ret)?;
        rows.push((stock_ret, salience));
    }
    salience_weighted_difference(&rows)
}

fn turnover_salience_for_date(
    date_idx: usize,
    instrument_idx: usize,
    instrument_count: usize,
    payoff_values: &[Option<f64>],
    turnover: &[Option<f64>],
    market_turnover: &[Option<f64>],
) -> Option<f64> {
    let start = (date_idx + 1).saturating_sub(SALIENCE_WINDOW);
    let mut rows = Vec::with_capacity(SALIENCE_WINDOW);
    for hist_date_idx in start..=date_idx {
        let offset = hist_date_idx * instrument_count + instrument_idx;
        let (Some(payoff), Some(stock_turnover), Some(market_turnover)) = (
            finite(payoff_values[offset]),
            finite(turnover[offset]),
            finite(market_turnover[hist_date_idx]),
        ) else {
            continue;
        };
        let salience = salience_value_without_exp(stock_turnover, market_turnover)?;
        rows.push((payoff, salience));
    }
    salience_weighted_difference(&rows)
}

fn stv_for_date(
    date_idx: usize,
    instrument_idx: usize,
    instrument_count: usize,
    returns: &[Option<f64>],
    turnover: &[Option<f64>],
) -> Option<f64> {
    let start = (date_idx + 1).saturating_sub(SALIENCE_WINDOW);
    let mut rows = Vec::with_capacity(SALIENCE_WINDOW);
    for hist_date_idx in start..=date_idx {
        let offset = hist_date_idx * instrument_count + instrument_idx;
        let (Some(stock_ret), Some(turnover)) = (finite(returns[offset]), finite(turnover[offset]))
        else {
            continue;
        };
        let salience = if stock_ret.abs() >= STV_RETURN_THRESHOLD {
            stock_ret.abs() * 1000.0
        } else {
            turnover
        };
        rows.push((stock_ret, salience));
    }
    salience_weighted_difference(&rows)
}

fn salience_value_without_exp(value: f64, market_value: f64) -> Option<f64> {
    let denominator = value.abs() + market_value.abs() + SALIENCE_THETA;
    if denominator.abs() <= EPS {
        return None;
    }
    finite_value((value - market_value).abs() / denominator)
}

fn salience_weighted_difference(rows: &[(f64, f64)]) -> Option<f64> {
    if rows.len() < SALIENCE_MIN_PERIODS {
        return None;
    }
    let ranks =
        descending_one_based_ranks(rows.iter().map(|row| row.1).collect::<Vec<_>>().as_slice());
    let delta_powers = ranks
        .iter()
        .map(|rank| SALIENCE_DELTA.powf(*rank))
        .collect::<Vec<_>>();
    let denominator = delta_powers.iter().sum::<f64>();
    if denominator.abs() <= EPS || !denominator.is_finite() {
        return None;
    }

    let mean_payoff = rows.iter().map(|row| row.0).sum::<f64>() / rows.len() as f64;
    let weighted_payoff = rows
        .iter()
        .zip(delta_powers.iter())
        .map(|(row, delta_power)| row.0 * delta_power / denominator)
        .sum::<f64>();
    finite_value(weighted_payoff - mean_payoff)
}

fn descending_one_based_ranks(values: &[f64]) -> Vec<f64> {
    let mut pairs = values
        .iter()
        .enumerate()
        .map(|(idx, value)| (idx, *value))
        .collect::<Vec<_>>();
    pairs.sort_by(|left, right| {
        right
            .1
            .partial_cmp(&left.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.0.cmp(&right.0))
    });

    let mut output = vec![0.0; values.len()];
    let mut start = 0usize;
    while start < pairs.len() {
        let mut end = start + 1;
        while end < pairs.len() && (pairs[end].1 - pairs[start].1).abs() <= EPS {
            end += 1;
        }
        let avg_rank = (start + 1 + end) as f64 / 2.0;
        for idx in start..end {
            output[pairs[idx].0] = avg_rank;
        }
        start = end;
    }
    output
}

fn descending_pct_ranks(values: &[f64]) -> Vec<f64> {
    if values.len() <= 1 {
        return vec![0.0; values.len()];
    }
    let mut pairs = values
        .iter()
        .enumerate()
        .map(|(idx, value)| (idx, *value))
        .collect::<Vec<_>>();
    pairs.sort_by(|left, right| {
        right
            .1
            .partial_cmp(&left.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.0.cmp(&right.0))
    });

    let mut output = vec![0.0; values.len()];
    let mut start = 0usize;
    while start < pairs.len() {
        let mut end = start + 1;
        while end < pairs.len() && (pairs[end].1 - pairs[start].1).abs() <= EPS {
            end += 1;
        }
        let avg_rank = (start + end - 1) as f64 / 2.0;
        let pct_rank = avg_rank / (values.len() - 1) as f64;
        for idx in start..end {
            output[pairs[idx].0] = pct_rank;
        }
        start = end;
    }
    output
}

fn daily_returns(panel: &DailyPanel, adj_close: &PanelColumn) -> Result<PanelColumn> {
    let date_count = panel.dates().len();
    let instrument_count = panel.instruments().len();
    let mut output = vec![None; panel.shape_len()];
    for instrument_idx in 0..instrument_count {
        for date_idx in 1..date_count {
            let offset = date_idx * instrument_count + instrument_idx;
            let prev_offset = (date_idx - 1) * instrument_count + instrument_idx;
            let (Some(current), Some(previous)) = (
                finite(adj_close.values()[offset]),
                finite(adj_close.values()[prev_offset]),
            ) else {
                continue;
            };
            if previous.abs() > EPS {
                output[offset] = finite_value(current / previous - 1.0);
            }
        }
    }
    panel.column_from_values(output)
}

fn market_equal_returns_ex_bj(panel: &DailyPanel, returns: &PanelColumn) -> Vec<Option<f64>> {
    market_equal_values_ex_bj(panel, returns)
}

fn market_equal_values_ex_bj(panel: &DailyPanel, values: &PanelColumn) -> Vec<Option<f64>> {
    let instrument_count = panel.instruments().len();
    let instruments = panel.instruments();
    let mut output = Vec::with_capacity(panel.dates().len());
    for date_idx in 0..panel.dates().len() {
        let mut sum = 0.0;
        let mut count = 0usize;
        for (instrument_idx, ts_code) in instruments.iter().enumerate() {
            if is_bj_stock(ts_code) {
                continue;
            }
            let offset = date_idx * instrument_count + instrument_idx;
            if let Some(value) = finite(values.values()[offset]) {
                sum += value;
                count += 1;
            }
        }
        output.push((count > 0).then_some(sum / count as f64));
    }
    output
}

fn postprocess_salience_factor(
    data: &DataPool,
    panel: &DailyPanel,
    raw: &PanelColumn,
) -> Result<PanelColumn> {
    let winsorized = raw.cs(three_sigma_winsorize)?;
    let standardized = winsorized.cs(cs_zscore)?;
    neutralize_size_sector(&standardized, panel, data)
}

fn three_sigma_winsorize(values: &[Option<f64>]) -> Vec<Option<f64>> {
    let finite_values = values
        .iter()
        .filter_map(|value| finite(*value))
        .collect::<Vec<_>>();
    if finite_values.is_empty() {
        return vec![None; values.len()];
    }
    let mean = finite_values.iter().sum::<f64>() / finite_values.len() as f64;
    let variance = finite_values
        .iter()
        .map(|value| {
            let diff = value - mean;
            diff * diff
        })
        .sum::<f64>()
        / finite_values.len() as f64;
    let std = variance.sqrt();
    if !std.is_finite() || std <= EPS {
        return values.iter().map(|value| finite(*value)).collect();
    }
    let lower = mean - 3.0 * std;
    let upper = mean + 3.0 * std;
    values
        .iter()
        .map(|value| finite(*value).map(|value| value.clamp(lower, upper)))
        .collect()
}

fn finite(value: Option<f64>) -> Option<f64> {
    clean(value).filter(|value| value.is_finite())
}

fn finite_value(value: f64) -> Option<f64> {
    value.is_finite().then_some(value)
}

fn multiply(left: Option<f64>, right: Option<f64>) -> Option<f64> {
    match (finite(left), finite(right)) {
        (Some(left), Some(right)) => finite_value(left * right),
        _ => None,
    }
}

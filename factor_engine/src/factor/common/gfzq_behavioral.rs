use crate::core::DatasetId;
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::chip::{daily_vwap, percent_to_decimal};
use crate::factor::common::stock_daily_ops::is_bj_stock;
use crate::factor::common::vector::clean;
use crate::factor::common::{DailyPanel, PanelColumn};

pub const CGO_WINDOW: usize = 100;
pub const CGO_MIN_PERIODS: usize = 50;
pub const ST_WINDOW: usize = 20;
pub const ST_THETA: f64 = 0.9;
pub const ST_DELTA: f64 = 0.9;

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

fn st_salience_for_date(
    date_idx: usize,
    instrument_idx: usize,
    instrument_count: usize,
    returns: &[Option<f64>],
    market_returns: &[Option<f64>],
) -> Option<f64> {
    if date_idx + 1 < ST_WINDOW {
        return None;
    }
    let start = date_idx + 1 - ST_WINDOW;
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
    if rows.len() < ST_WINDOW {
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
            if let Some(value) = finite(returns.values()[offset]) {
                sum += value;
                count += 1;
            }
        }
        output.push((count > 0).then_some(sum / count as f64));
    }
    output
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
    fn gfzq_cgo_reference_uses_surviving_turnover_weights() {
        let panel = test_panel((0..51).collect(), vec!["000001.SZ".to_string()]);
        let mut vwap_values = vec![Some(999.0); 51];
        vwap_values[48] = Some(10.0);
        vwap_values[49] = Some(20.0);
        let vwap = panel.column_from_values(vwap_values).unwrap();
        let mut turnover_values = vec![Some(0.0); 51];
        turnover_values[48] = Some(0.1);
        turnover_values[49] = Some(0.2);
        let turnover = panel.column_from_values(turnover_values).unwrap();

        let cost = cgo_reference_price(50, 0, 1, &vwap, &turnover);
        let expected = (20.0 * 0.2 + 10.0 * 0.1 * 0.8) / (0.2 + 0.1 * 0.8);
        assert_close(cost, expected);
    }

    #[test]
    fn gfzq_cgo_requires_minimum_effective_days() {
        let panel = test_panel((0..50).collect(), vec!["000001.SZ".to_string()]);
        let vwap = panel.column_from_values(vec![Some(10.0); 50]).unwrap();
        let turnover = panel.column_from_values(vec![Some(0.1); 50]).unwrap();

        assert_eq!(cgo_reference_price(49, 0, 1, &vwap, &turnover), None);

        let panel = test_panel((0..51).collect(), vec!["000001.SZ".to_string()]);
        let vwap = panel.column_from_values(vec![Some(10.0); 51]).unwrap();
        let turnover = panel.column_from_values(vec![Some(0.1); 51]).unwrap();
        assert_close(cgo_reference_price(50, 0, 1, &vwap, &turnover), 10.0);
    }

    #[test]
    fn gfzq_st_ranks_salience_descending_and_computes_covariance() {
        let stock_returns = (0..ST_WINDOW)
            .map(|idx| Some(0.001 * (idx as f64 + 1.0)))
            .collect::<Vec<_>>();
        let mut returns = Vec::new();
        returns.extend(stock_returns.iter().copied());
        let market_returns = vec![Some(0.001); ST_WINDOW];

        let actual =
            st_salience_for_date(ST_WINDOW - 1, 0, 1, &returns, &market_returns).expect("st value");
        let mut rows = Vec::new();
        for idx in 0..ST_WINDOW {
            let r = stock_returns[idx].unwrap();
            rows.push((r, salience_value(r, 0.001).unwrap()));
        }
        let ranks =
            descending_pct_ranks(rows.iter().map(|row| row.1).collect::<Vec<_>>().as_slice());
        let delta_powers = ranks
            .iter()
            .map(|rank| ST_DELTA.powf(*rank))
            .collect::<Vec<_>>();
        let mean_delta = delta_powers.iter().sum::<f64>() / ST_WINDOW as f64;
        let mean_ret = rows.iter().map(|row| row.0).sum::<f64>() / ST_WINDOW as f64;
        let weighted = rows
            .iter()
            .zip(delta_powers.iter())
            .map(|(row, delta_power)| row.0 * delta_power / mean_delta)
            .sum::<f64>()
            / ST_WINDOW as f64;
        assert!((actual - (weighted - mean_ret)).abs() < 1e-12);
    }

    #[test]
    fn gfzq_st_market_return_excludes_bj_stocks() {
        let panel = test_panel(
            vec![1],
            vec!["000001.SZ".to_string(), "430001.BJ".to_string()],
        );
        let returns = panel
            .column_from_values(vec![Some(0.02), Some(0.40)])
            .unwrap();
        let market = market_equal_returns_ex_bj(&panel, &returns);
        assert_eq!(market, vec![Some(0.02)]);
    }

    fn test_panel(dates: Vec<i32>, instruments: Vec<String>) -> DailyPanel {
        let present = vec![true; dates.len() * instruments.len()];
        DailyPanel::from_index(dates.clone(), instruments, &dates, present).unwrap()
    }
}

use std::collections::HashMap;

use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::vector::clean;
use crate::factor::common::{DailyPanel, PanelColumn};
use crate::factor::Factor;
use crate::operators::ts_delay;

const MARKET_INDEX: &str = "000985.CSI";
const VERSION: &str = "0.1.0";
const REGRESSION_WINDOW: usize = 20;
const LAG_STEP: usize = 20;
const LAG_COUNT: usize = 6;
const LOOKBACK_DAYS: usize = REGRESSION_WINDOW - 1 + LAG_STEP * LAG_COUNT;

pub struct StockDailyIdVolDecorr;

#[derive(Clone, Copy, Debug)]
struct PortfolioEntry {
    sort_key: f64,
    weight: f64,
    stock_return: f64,
}

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyIdVolDecorr)
}

impl Factor for StockDailyIdVolDecorr {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "id_vol_decorr".to_string(),
            aliases: vec!["ID_Vol_deCorr".to_string()],
            name: "ID Vol deCorr".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: [
                "price_volume",
                "return",
                "volatility",
                "idiosyncratic_volatility",
                "regression",
                "daily",
                "DWZQ",
            ]
            .iter()
            .map(|value| value.to_string())
            .collect(),
            description: "ID Vol deCorr factor from 20-day Fama-French three-factor residual volatility, decorrelated against six 20-trading-day lagged ID Vol cross sections.".to_string(),
            dependencies: vec![
                DataRequest::new(DatasetId::StockDailyPv, &["close", "pre_close"]),
                DataRequest::new(DatasetId::StockDailyBasic, &["circ_mv", "pb"]),
                DataRequest::index_daily(MARKET_INDEX, &["close", "pre_close"]),
            ],
            intraday_raw_dependencies: Vec::new(),
            lookback: Lookback {
                trading_days: LOOKBACK_DAYS,
            },
        }
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let panel = data.daily_panel(DatasetId::StockDailyPv)?;
        let basic_table = data.daily(DatasetId::StockDailyBasic)?;

        let close = panel.column("close")?;
        let pre_close = panel.column("pre_close")?;
        let stock_return = close.zip_binary(&pre_close, ret)?;
        let circ_mv = panel.column_from_table(basic_table, "circ_mv")?;
        let pb = panel.column_from_table(basic_table, "pb")?;

        let index_panel = data.index_daily_panel(MARKET_INDEX)?;
        let index_return = index_panel
            .column("close")?
            .zip_binary(&index_panel.column("pre_close")?, ret)?;
        let market_return = expand_index_column(&panel, index_panel, &index_return)?;

        let smb = weighted_third_spread_column(&panel, &stock_return, &circ_mv, &circ_mv)?;
        let hml = weighted_third_spread_column(&panel, &stock_return, &circ_mv, &pb)?;
        let id_vol = id_vol_column(&panel, &stock_return, &market_return, &smb, &hml)?;

        let lag20 = id_vol.ts(|values| ts_delay(values, LAG_STEP))?;
        let lag40 = id_vol.ts(|values| ts_delay(values, LAG_STEP * 2))?;
        let lag60 = id_vol.ts(|values| ts_delay(values, LAG_STEP * 3))?;
        let lag80 = id_vol.ts(|values| ts_delay(values, LAG_STEP * 4))?;
        let lag100 = id_vol.ts(|values| ts_delay(values, LAG_STEP * 5))?;
        let lag120 = id_vol.ts(|values| ts_delay(values, LAG_STEP * 6))?;
        let factor = id_vol
            .cs_neutralize_regression(&[&lag20, &lag40, &lag60, &lag80, &lag100, &lag120], None)?;

        Ok(factor.to_factor_series(self.spec()))
    }
}

fn expand_index_column(
    stock_panel: &DailyPanel,
    index_panel: &DailyPanel,
    index_column: &PanelColumn,
) -> Result<PanelColumn> {
    let index_instrument_count = index_panel.instruments().len();
    if index_instrument_count == 0 {
        return stock_panel.column_from_values(vec![None; stock_panel.shape_len()]);
    }

    let mut by_date = HashMap::new();
    for (date_idx, trade_date) in index_panel.dates().iter().enumerate() {
        by_date.insert(
            *trade_date,
            index_column.values()[date_idx * index_instrument_count],
        );
    }

    let mut values = Vec::with_capacity(stock_panel.shape_len());
    for trade_date in stock_panel.dates() {
        let value = by_date.get(trade_date).copied().unwrap_or(None);
        for _ in stock_panel.instruments() {
            values.push(value);
        }
    }
    stock_panel.column_from_values(values)
}

fn weighted_third_spread_column(
    panel: &DailyPanel,
    stock_return: &PanelColumn,
    weight: &PanelColumn,
    sort_key: &PanelColumn,
) -> Result<PanelColumn> {
    let instrument_count = panel.instruments().len();
    let mut values = Vec::with_capacity(panel.shape_len());
    for date_idx in 0..panel.dates().len() {
        let offset = date_idx * instrument_count;
        let spread = weighted_third_spread(
            &stock_return.values()[offset..offset + instrument_count],
            &weight.values()[offset..offset + instrument_count],
            &sort_key.values()[offset..offset + instrument_count],
        );
        for _ in 0..instrument_count {
            values.push(spread);
        }
    }
    panel.column_from_values(values)
}

fn id_vol_column(
    panel: &DailyPanel,
    stock_return: &PanelColumn,
    market_return: &PanelColumn,
    smb: &PanelColumn,
    hml: &PanelColumn,
) -> Result<PanelColumn> {
    let date_count = panel.dates().len();
    let instrument_count = panel.instruments().len();
    let mut output = vec![None; panel.shape_len()];
    for instrument_idx in 0..instrument_count {
        let mut stock_series = Vec::with_capacity(date_count);
        let mut market_series = Vec::with_capacity(date_count);
        let mut smb_series = Vec::with_capacity(date_count);
        let mut hml_series = Vec::with_capacity(date_count);
        for date_idx in 0..date_count {
            let offset = date_idx * instrument_count + instrument_idx;
            stock_series.push(stock_return.values()[offset]);
            market_series.push(market_return.values()[offset]);
            smb_series.push(smb.values()[offset]);
            hml_series.push(hml.values()[offset]);
        }

        let computed = id_vol_series(&stock_series, &market_series, &smb_series, &hml_series);
        for (date_idx, value) in computed.into_iter().enumerate() {
            output[date_idx * instrument_count + instrument_idx] = value;
        }
    }
    panel.column_from_values(output)
}

fn weighted_third_spread(
    stock_return: &[Option<f64>],
    weight: &[Option<f64>],
    sort_key: &[Option<f64>],
) -> Option<f64> {
    let mut rows = Vec::new();
    for idx in 0..stock_return.len() {
        let (Some(stock_return), Some(weight), Some(sort_key)) = (
            finite(stock_return[idx]),
            finite(weight[idx]),
            finite(sort_key[idx]),
        ) else {
            continue;
        };
        if weight <= 0.0 {
            continue;
        }
        rows.push(PortfolioEntry {
            sort_key,
            weight,
            stock_return,
        });
    }

    rows.sort_by(|left, right| left.sort_key.total_cmp(&right.sort_key));
    let group_size = rows.len() / 3;
    if group_size == 0 {
        return None;
    }

    let low = weighted_return(&rows[..group_size])?;
    let high = weighted_return(&rows[rows.len() - group_size..])?;
    Some(low - high)
}

fn weighted_return(rows: &[PortfolioEntry]) -> Option<f64> {
    let mut weighted_sum = 0.0;
    let mut weight_sum = 0.0;
    for row in rows {
        weighted_sum += row.weight * row.stock_return;
        weight_sum += row.weight;
    }
    if weight_sum <= f64::EPSILON {
        return None;
    }
    Some(weighted_sum / weight_sum)
}

fn id_vol_series(
    stock_return: &[Option<f64>],
    market_return: &[Option<f64>],
    smb: &[Option<f64>],
    hml: &[Option<f64>],
) -> Vec<Option<f64>> {
    let mut output = vec![None; stock_return.len()];
    if stock_return.len() < REGRESSION_WINDOW {
        return output;
    }

    for end in REGRESSION_WINDOW - 1..stock_return.len() {
        let start = end + 1 - REGRESSION_WINDOW;
        let mut y = Vec::with_capacity(REGRESSION_WINDOW);
        let mut x = Vec::with_capacity(REGRESSION_WINDOW);
        let mut valid = true;
        for idx in start..=end {
            let (Some(y_value), Some(mkt_value), Some(smb_value), Some(hml_value)) = (
                finite(stock_return[idx]),
                finite(market_return[idx]),
                finite(smb[idx]),
                finite(hml[idx]),
            ) else {
                valid = false;
                break;
            };
            y.push(y_value);
            x.push([1.0, mkt_value, smb_value, hml_value]);
        }
        if !valid {
            continue;
        }
        output[end] = regression_residual_std(&y, &x);
    }
    output
}

fn regression_residual_std(y: &[f64], x: &[[f64; 4]]) -> Option<f64> {
    if y.len() != x.len() || y.len() < 4 {
        return None;
    }

    let mut xtx = vec![vec![0.0; 4]; 4];
    let mut xty = vec![0.0; 4];
    for (row, y_value) in x.iter().zip(y) {
        for i in 0..4 {
            xty[i] += row[i] * y_value;
            for j in 0..4 {
                xtx[i][j] += row[i] * row[j];
            }
        }
    }
    let beta = solve_linear_system(xtx, xty)?;

    let mut residual_sum_squares = 0.0;
    for (row, y_value) in x.iter().zip(y) {
        let fitted = row
            .iter()
            .zip(&beta)
            .map(|(x_value, beta)| x_value * beta)
            .sum::<f64>();
        residual_sum_squares += (y_value - fitted).powi(2);
    }
    Some((residual_sum_squares / y.len() as f64).sqrt())
}

fn solve_linear_system(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    for col in 0..n {
        let pivot =
            (col..n).max_by(|left, right| a[*left][col].abs().total_cmp(&a[*right][col].abs()))?;
        if a[pivot][col].abs() <= 1e-12 {
            return None;
        }
        if pivot != col {
            a.swap(pivot, col);
            b.swap(pivot, col);
        }

        let pivot_value = a[col][col];
        for j in col..n {
            a[col][j] /= pivot_value;
        }
        b[col] /= pivot_value;

        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = a[row][col];
            if factor.abs() <= f64::EPSILON {
                continue;
            }
            for j in col..n {
                a[row][j] -= factor * a[col][j];
            }
            b[row] -= factor * b[col];
        }
    }
    Some(b)
}

fn ret(numerator: Option<f64>, denominator: Option<f64>) -> Option<f64> {
    match (finite(numerator), finite(denominator)) {
        (Some(numerator), Some(denominator)) if denominator.abs() > f64::EPSILON => {
            Some(numerator / denominator - 1.0)
        }
        _ => None,
    }
}

fn finite(value: Option<f64>) -> Option<f64> {
    clean(value).filter(|value| value.is_finite())
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
    fn weighted_third_spread_uses_low_minus_high_weighted_returns() {
        let stock_return = vec![
            Some(0.10),
            Some(0.20),
            Some(0.30),
            Some(0.40),
            Some(0.50),
            Some(0.60),
        ];
        let weight = vec![
            Some(1.0),
            Some(3.0),
            Some(1.0),
            Some(1.0),
            Some(2.0),
            Some(2.0),
        ];
        let sort_key = vec![
            Some(1.0),
            Some(2.0),
            Some(3.0),
            Some(4.0),
            Some(5.0),
            Some(6.0),
        ];

        let low = (0.10 * 1.0 + 0.20 * 3.0) / 4.0;
        let high = (0.50 * 2.0 + 0.60 * 2.0) / 4.0;
        assert_close(
            weighted_third_spread(&stock_return, &weight, &sort_key),
            Some(low - high),
        );
    }

    #[test]
    fn regression_residual_std_is_zero_for_exact_three_factor_fit() {
        let mut y = Vec::new();
        let mut x = Vec::new();
        for idx in 0..REGRESSION_WINDOW {
            let mkt = idx as f64 / 100.0;
            let smb = ((idx * idx) as f64) / 10_000.0;
            let hml = if idx % 2 == 0 { 0.02 } else { -0.01 };
            y.push(0.01 + 0.5 * mkt - 0.2 * smb + 0.3 * hml);
            x.push([1.0, mkt, smb, hml]);
        }

        let actual = regression_residual_std(&y, &x).expect("std");
        assert!(
            actual.abs() < 1e-10,
            "expected zero residual std, got {actual}"
        );
    }

    #[test]
    fn id_vol_series_requires_complete_twenty_day_window() {
        let mut stock_return = Vec::new();
        let mut market_return = Vec::new();
        let mut smb = Vec::new();
        let mut hml = Vec::new();
        for idx in 0..REGRESSION_WINDOW {
            let mkt = idx as f64 / 100.0;
            let smb_value = ((idx * idx) as f64) / 10_000.0;
            let hml_value = if idx % 2 == 0 { 0.02 } else { -0.01 };
            stock_return.push(Some(0.01 + 0.5 * mkt - 0.2 * smb_value + 0.3 * hml_value));
            market_return.push(Some(mkt));
            smb.push(Some(smb_value));
            hml.push(Some(hml_value));
        }

        let output = id_vol_series(&stock_return, &market_return, &smb, &hml);

        assert!(output[..REGRESSION_WINDOW - 1].iter().all(Option::is_none));
        assert!(output[REGRESSION_WINDOW - 1].is_some());

        market_return[REGRESSION_WINDOW - 1] = None;
        let output = id_vol_series(&stock_return, &market_return, &smb, &hml);
        assert_eq!(output[REGRESSION_WINDOW - 1], None);
    }

    #[test]
    fn solve_linear_system_rejects_singular_matrix() {
        let a = vec![vec![1.0, 2.0], vec![2.0, 4.0]];
        let b = vec![1.0, 2.0];

        assert_eq!(solve_linear_system(a, b), None);
    }
}

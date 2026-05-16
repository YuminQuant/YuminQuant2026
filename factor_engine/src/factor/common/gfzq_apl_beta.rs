use crate::core::DatasetId;
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::stock_daily_ops::{is_bj_stock, neutralize_size_sector};
use crate::factor::common::vector::clean;
use crate::factor::common::{DailyPanel, PanelColumn};
use crate::operators::cs_zscore;

pub const APL_REGRESSION_WINDOW: usize = 20;
pub const APL_MIN_OBS: usize = 5;
pub const MARKET_INDEX: &str = "000985.CSI";

const EPS: f64 = 1e-12;
const REGRESSOR_COUNT: usize = 5;

#[derive(Clone, Copy, Debug)]
struct PortfolioEntry {
    sort_key: f64,
    weight: f64,
    stock_return: f64,
}

pub fn apl_beta_from_data(data: &DataPool) -> Result<(DailyPanel, PanelColumn)> {
    let panel = data.daily_panel(DatasetId::StockDailyPv)?;
    let close = panel.column("close")?;
    let pre_close = panel.column("pre_close")?;
    let stock_return = close.zip_binary(&pre_close, ret)?;

    let basic = data.daily(DatasetId::StockDailyBasic)?;
    let circ_mv = panel.column_from_table(basic, "circ_mv")?;
    let pb = panel.column_from_table(basic, "pb")?;

    let limit = data.daily(DatasetId::StockDailyLimit)?;
    let up_limit = panel.column_from_table(limit, "up_limit")?;
    let down_limit = panel.column_from_table(limit, "down_limit")?;
    let delta_lh_by_date = limit_hit_ratio_ex_bj(panel, &close, &up_limit, &down_limit);
    let delta_lh = expand_daily_values(panel, &delta_lh_by_date)?;

    let index_panel = data.index_daily_panel(MARKET_INDEX)?;
    let index_return = index_panel
        .column("close")?
        .zip_binary(&index_panel.column("pre_close")?, ret)?;
    let market_return = expand_index_column(panel, index_panel, &index_return)?;

    let smb = weighted_third_spread_column(panel, &stock_return, &circ_mv, &circ_mv)?;
    let hml = weighted_third_spread_column(panel, &stock_return, &circ_mv, &pb)?;
    let raw = apl_beta_column(panel, &stock_return, &delta_lh, &market_return, &smb, &hml)?;
    let standardized = raw.cs(cs_zscore)?;
    let factor = neutralize_size_sector(&standardized, panel, data)?;
    Ok((panel.clone(), factor))
}

pub fn limit_hit_ratio_ex_bj(
    panel: &DailyPanel,
    close: &PanelColumn,
    up_limit: &PanelColumn,
    down_limit: &PanelColumn,
) -> Vec<Option<f64>> {
    let instrument_count = panel.instruments().len();
    let instruments = panel.instruments();
    let mut output = Vec::with_capacity(panel.dates().len());
    for date_idx in 0..panel.dates().len() {
        let mut total = 0usize;
        let mut hitters = 0usize;
        for instrument_idx in 0..instrument_count {
            if is_bj_stock(&instruments[instrument_idx]) {
                continue;
            }
            let offset = date_idx * instrument_count + instrument_idx;
            let (Some(close), Some(up_limit), Some(down_limit)) = (
                finite(close.values()[offset]),
                finite(up_limit.values()[offset]),
                finite(down_limit.values()[offset]),
            ) else {
                continue;
            };
            total += 1;
            let close = round_price(close);
            if close >= round_price(up_limit) || close <= round_price(down_limit) {
                hitters += 1;
            }
        }
        output.push((total > 0).then_some(hitters as f64 / total as f64));
    }
    output
}

pub fn weighted_third_spread_column(
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

pub fn apl_beta_column(
    panel: &DailyPanel,
    stock_return: &PanelColumn,
    delta_lh: &PanelColumn,
    market_return: &PanelColumn,
    smb: &PanelColumn,
    hml: &PanelColumn,
) -> Result<PanelColumn> {
    let date_count = panel.dates().len();
    let instrument_count = panel.instruments().len();
    let mut output = vec![None; panel.shape_len()];

    for instrument_idx in 0..instrument_count {
        for end in 0..date_count {
            let start = (end + 1).saturating_sub(APL_REGRESSION_WINDOW);
            let mut y = Vec::with_capacity(APL_REGRESSION_WINDOW);
            let mut x = Vec::with_capacity(APL_REGRESSION_WINDOW);
            for date_idx in start..=end {
                let offset = date_idx * instrument_count + instrument_idx;
                let (
                    Some(y_value),
                    Some(apl_value),
                    Some(mkt_value),
                    Some(smb_value),
                    Some(hml_value),
                ) = (
                    finite(stock_return.values()[offset]),
                    finite(delta_lh.values()[offset]),
                    finite(market_return.values()[offset]),
                    finite(smb.values()[offset]),
                    finite(hml.values()[offset]),
                )
                else {
                    continue;
                };
                y.push(y_value);
                x.push([1.0, apl_value, mkt_value, smb_value, hml_value]);
            }
            if y.len() < APL_MIN_OBS {
                continue;
            }
            let Some(beta) = ols_beta(&y, &x) else {
                continue;
            };
            output[end * instrument_count + instrument_idx] = finite_value(beta[1].abs());
        }
    }

    panel.column_from_values(output)
}

fn expand_daily_values(panel: &DailyPanel, daily_values: &[Option<f64>]) -> Result<PanelColumn> {
    let mut values = Vec::with_capacity(panel.shape_len());
    for date_idx in 0..panel.dates().len() {
        let value = daily_values.get(date_idx).copied().unwrap_or(None);
        for _ in panel.instruments() {
            values.push(value);
        }
    }
    panel.column_from_values(values)
}

fn expand_index_column(
    stock_panel: &DailyPanel,
    index_panel: &DailyPanel,
    index_column: &PanelColumn,
) -> Result<PanelColumn> {
    use std::collections::HashMap;

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
    if weight_sum <= EPS {
        return None;
    }
    Some(weighted_sum / weight_sum)
}

fn ols_beta(y: &[f64], x: &[[f64; REGRESSOR_COUNT]]) -> Option<[f64; REGRESSOR_COUNT]> {
    if y.len() != x.len() || y.len() < REGRESSOR_COUNT {
        return None;
    }

    let mut xtx = vec![vec![0.0; REGRESSOR_COUNT]; REGRESSOR_COUNT];
    let mut xty = vec![0.0; REGRESSOR_COUNT];
    for (row, y_value) in x.iter().zip(y.iter()) {
        for i in 0..REGRESSOR_COUNT {
            xty[i] += row[i] * y_value;
            for j in 0..REGRESSOR_COUNT {
                xtx[i][j] += row[i] * row[j];
            }
        }
    }

    let beta = solve_linear_system(xtx, xty)?;
    let mut output = [0.0; REGRESSOR_COUNT];
    for (idx, value) in beta.into_iter().enumerate() {
        output[idx] = value;
    }
    Some(output)
}

fn solve_linear_system(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    for pivot in 0..n {
        let mut max_row = pivot;
        let mut max_value = a[pivot][pivot].abs();
        for (row_idx, row) in a.iter().enumerate().skip(pivot + 1) {
            let value = row[pivot].abs();
            if value > max_value {
                max_value = value;
                max_row = row_idx;
            }
        }
        if max_value <= EPS {
            return None;
        }
        if max_row != pivot {
            a.swap(max_row, pivot);
            b.swap(max_row, pivot);
        }

        let pivot_value = a[pivot][pivot];
        for col in pivot..n {
            a[pivot][col] /= pivot_value;
        }
        b[pivot] /= pivot_value;

        for row_idx in 0..n {
            if row_idx == pivot {
                continue;
            }
            let factor = a[row_idx][pivot];
            if factor.abs() <= EPS {
                continue;
            }
            for col in pivot..n {
                a[row_idx][col] -= factor * a[pivot][col];
            }
            b[row_idx] -= factor * b[pivot];
        }
    }
    Some(b)
}

fn round_price(value: f64) -> f64 {
    (value * 100.0).round() / 100.0
}

fn ret(close: Option<f64>, pre_close: Option<f64>) -> Option<f64> {
    match (finite(close), finite(pre_close)) {
        (Some(close), Some(pre_close)) if pre_close.abs() > EPS => {
            finite_value(close / pre_close - 1.0)
        }
        _ => None,
    }
}

fn finite(value: Option<f64>) -> Option<f64> {
    clean(value).filter(|value| value.is_finite())
}

fn finite_value(value: f64) -> Option<f64> {
    value.is_finite().then_some(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: Option<f64>, expected: f64) {
        let actual = actual.expect("expected value");
        assert!(
            (actual - expected).abs() < 1e-10,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn gfzq_apl_limit_ratio_rounds_prices_and_excludes_bj() {
        let panel = test_panel(
            vec![20260102],
            vec![
                "000001.SZ".to_string(),
                "430001.BJ".to_string(),
                "600000.SH".to_string(),
            ],
        );
        let close = panel
            .column_from_values(vec![Some(10.004), Some(20.0), Some(8.996)])
            .unwrap();
        let up_limit = panel
            .column_from_values(vec![Some(10.00), Some(20.0), Some(11.0)])
            .unwrap();
        let down_limit = panel
            .column_from_values(vec![Some(8.00), Some(18.0), Some(9.00)])
            .unwrap();

        let ratio = limit_hit_ratio_ex_bj(&panel, &close, &up_limit, &down_limit);

        assert_eq!(ratio, vec![Some(1.0)]);
    }

    #[test]
    fn gfzq_apl_limit_ratio_none_when_no_valid_non_bj_stock() {
        let panel = test_panel(vec![20260102], vec!["430001.BJ".to_string()]);
        let close = panel.column_from_values(vec![Some(20.0)]).unwrap();
        let up_limit = panel.column_from_values(vec![Some(20.0)]).unwrap();
        let down_limit = panel.column_from_values(vec![Some(18.0)]).unwrap();

        let ratio = limit_hit_ratio_ex_bj(&panel, &close, &up_limit, &down_limit);

        assert_eq!(ratio, vec![None]);
    }

    #[test]
    fn gfzq_apl_weighted_third_spread_uses_low_minus_high() {
        let stock_return = vec![
            Some(0.01),
            Some(0.02),
            Some(0.03),
            Some(0.04),
            Some(0.05),
            Some(0.06),
        ];
        let weight = vec![Some(1.0); 6];
        let sort_key = vec![
            Some(1.0),
            Some(2.0),
            Some(3.0),
            Some(4.0),
            Some(5.0),
            Some(6.0),
        ];

        assert_close(
            weighted_third_spread(&stock_return, &weight, &sort_key),
            0.015 - 0.055,
        );
    }

    #[test]
    fn gfzq_apl_ols_extracts_absolute_apl_beta() {
        let panel = test_panel((0..6).collect(), vec!["000001.SZ".to_string()]);
        let mut y = Vec::new();
        let mut apl = Vec::new();
        let mut mkt = Vec::new();
        let mut smb = Vec::new();
        let mut hml = Vec::new();
        for idx in 0..6 {
            let apl_value = idx as f64 * 0.1;
            let mkt_value = (idx % 2) as f64;
            let smb_value = (idx % 3) as f64;
            let hml_value = (idx % 5) as f64;
            y.push(Some(
                1.0 + 2.0 * apl_value + 0.5 * mkt_value - 0.3 * smb_value + 0.2 * hml_value,
            ));
            apl.push(Some(apl_value));
            mkt.push(Some(mkt_value));
            smb.push(Some(smb_value));
            hml.push(Some(hml_value));
        }
        let y = panel.column_from_values(y).unwrap();
        let apl = panel.column_from_values(apl).unwrap();
        let mkt = panel.column_from_values(mkt).unwrap();
        let smb = panel.column_from_values(smb).unwrap();
        let hml = panel.column_from_values(hml).unwrap();

        let beta = apl_beta_column(&panel, &y, &apl, &mkt, &smb, &hml).unwrap();

        assert_close(beta.values()[5], 2.0);
    }

    #[test]
    fn gfzq_apl_ols_rejects_singular_design() {
        let panel = test_panel((0..5).collect(), vec!["000001.SZ".to_string()]);
        let y = panel.column_from_values(vec![Some(1.0); 5]).unwrap();
        let x = panel.column_from_values(vec![Some(1.0); 5]).unwrap();

        let beta = apl_beta_column(&panel, &y, &x, &x, &x, &x).unwrap();

        assert_eq!(beta.values()[4], None);
    }

    fn test_panel(dates: Vec<i32>, instruments: Vec<String>) -> DailyPanel {
        let present = vec![true; dates.len() * instruments.len()];
        DailyPanel::from_index(dates.clone(), instruments, &dates, present).unwrap()
    }
}

use std::collections::{BTreeMap, BTreeSet};

use rayon::prelude::*;

use crate::core::{
    AssetClass, FactorContext, FactorRowKey, FactorSeries, FactorSpec, FactorValue, Frequency,
    IntradayDailyRawRequest, IntradayDailyRawSeries, IntradayDailyRawSpec, Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::{clean_intraday_value, stock_minute_raw_spec, DailyPanel, PanelColumn};
use crate::factor::Factor;
use crate::operators::{cs_zscore, ts_mean};

pub const DAWN_FOG_RAW_ID: &str = "daily_dawn_fog_tstd";
pub const F_ALL_RAW_ID: &str = "daily_regression_f_all";
pub const T_INTERCEPT_RAW_ID: &str = "daily_regression_t_intercept";

const RAW_VERSION: &str = "0.1.0";
const VERSION: &str = "0.1.0";
const WINDOW: usize = 20;
const PARAMETER_COUNT: usize = 7;
const SLOPE_COUNT: usize = 6;
const REGRESSION_OFFSET: usize = 6;
const CORR_BLOCK_SIZE: usize = 64;

pub struct StockDailyFlowerHiddenForest;

#[derive(Clone, Copy, Debug)]
struct RegressionMetrics {
    dawn_fog_tstd: Option<f64>,
    f_all: Option<f64>,
    t_intercept: Option<f64>,
}

#[derive(Clone, Debug)]
struct RegressionResult {
    f_all: f64,
    t_values: [f64; PARAMETER_COUNT],
}

#[derive(Clone, Debug)]
struct RegressionRow {
    y: f64,
    x: [f64; PARAMETER_COUNT],
}

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyFlowerHiddenForest)
}

fn raw_spec(raw_id: &str) -> IntradayDailyRawSpec {
    stock_minute_raw_spec(raw_id, RAW_VERSION, &["close", "vol"], 1)
}

impl Factor for StockDailyFlowerHiddenForest {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "flower_hidden_forest".to_string(),
            aliases: Vec::new(),
            name: "Flower Hidden in Forest".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: [
                "price_volume",
                "return",
                "volume",
                "regression",
                "intraday",
                "minute_agg",
                "correlation",
                "composite",
                "daily",
                "FZZQ",
            ]
            .iter()
            .map(|value| value.to_string())
            .collect(),
            description: "Intraday minute return on lagged volume-increase regression composite from dawn fog, shadow, and evening components.".to_string(),
            dependencies: Vec::new(),
            intraday_raw_dependencies: vec![
                IntradayDailyRawRequest::new(DAWN_FOG_RAW_ID, WINDOW - 1),
                IntradayDailyRawRequest::new(F_ALL_RAW_ID, WINDOW - 1),
                IntradayDailyRawRequest::new(T_INTERCEPT_RAW_ID, WINDOW - 1),
            ],
            lookback: Lookback {
                trading_days: WINDOW - 1,
            },
        }
    }

    fn intraday_raw_specs(&self) -> Vec<IntradayDailyRawSpec> {
        vec![
            raw_spec(DAWN_FOG_RAW_ID),
            raw_spec(F_ALL_RAW_ID),
            raw_spec(T_INTERCEPT_RAW_ID),
        ]
    }

    fn minute_compute(
        &self,
        raw_id: &str,
        context: &FactorContext,
        data: &DataPool,
    ) -> Result<Option<IntradayDailyRawSeries>> {
        let raw_ids = vec![raw_id.to_string()];
        Ok(self
            .minute_compute_many(&raw_ids, context, data)?
            .into_iter()
            .next())
    }

    fn minute_compute_many(
        &self,
        raw_ids: &[String],
        context: &FactorContext,
        data: &DataPool,
    ) -> Result<Vec<IntradayDailyRawSeries>> {
        let requested = raw_ids.iter().map(String::as_str).collect::<BTreeSet<_>>();
        let wants_dawn = requested.contains(DAWN_FOG_RAW_ID);
        let wants_f_all = requested.contains(F_ALL_RAW_ID);
        let wants_intercept = requested.contains(T_INTERCEPT_RAW_ID);
        if !wants_dawn && !wants_f_all && !wants_intercept {
            return Ok(Vec::new());
        }

        let mut dawn_values = Vec::new();
        let mut f_all_values = Vec::new();
        let mut intercept_values = Vec::new();
        for trade_date in &context.target_dates {
            let Some(table) = data.minute(raw_spec(DAWN_FOG_RAW_ID).source_dataset, *trade_date)
            else {
                continue;
            };
            let ts_codes = table.required_utf8("ts_code")?;
            let trade_times = table.required_utf8("trade_time")?;
            let close = table.required_f64_cast("close")?;
            let volume = table.required_f64_cast("vol")?;
            let mut grouped = BTreeMap::<String, Vec<usize>>::new();
            for idx in 0..table.len {
                let Some(ts_code) = ts_codes[idx].clone() else {
                    continue;
                };
                if trade_times[idx].is_none() {
                    continue;
                }
                grouped.entry(ts_code).or_default().push(idx);
            }

            let rows = grouped
                .into_iter()
                .collect::<Vec<_>>()
                .into_par_iter()
                .map(|(ts_code, mut indices)| {
                    indices.sort_by(|left, right| trade_times[*left].cmp(&trade_times[*right]));
                    let metrics = regression_metrics_from_rows(&indices, &close, &volume);
                    (ts_code, metrics)
                })
                .collect::<Vec<_>>();

            for (ts_code, metrics) in rows {
                if wants_dawn {
                    dawn_values.push(FactorValue {
                        key: FactorRowKey::Daily {
                            trade_date: *trade_date,
                            ts_code: ts_code.clone(),
                        },
                        value: metrics.dawn_fog_tstd,
                    });
                }
                if wants_f_all {
                    f_all_values.push(FactorValue {
                        key: FactorRowKey::Daily {
                            trade_date: *trade_date,
                            ts_code: ts_code.clone(),
                        },
                        value: metrics.f_all,
                    });
                }
                if wants_intercept {
                    intercept_values.push(FactorValue {
                        key: FactorRowKey::Daily {
                            trade_date: *trade_date,
                            ts_code,
                        },
                        value: metrics.t_intercept,
                    });
                }
            }
        }

        let mut output = Vec::new();
        if wants_dawn {
            output.push(IntradayDailyRawSeries {
                spec: raw_spec(DAWN_FOG_RAW_ID),
                values: dawn_values,
            });
        }
        if wants_f_all {
            output.push(IntradayDailyRawSeries {
                spec: raw_spec(F_ALL_RAW_ID),
                values: f_all_values,
            });
        }
        if wants_intercept {
            output.push(IntradayDailyRawSeries {
                spec: raw_spec(T_INTERCEPT_RAW_ID),
                values: intercept_values,
            });
        }
        Ok(output)
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let panel = data.intraday_daily_raw_panel(DAWN_FOG_RAW_ID)?;
        let dawn_raw = panel.column(DAWN_FOG_RAW_ID)?;
        let f_all = panel.column(F_ALL_RAW_ID)?;
        let t_intercept = panel.column(T_INTERCEPT_RAW_ID)?;

        let dawn_fog = dawn_raw
            .ts(|values| ts_mean(values, WINDOW, 1))?
            .cs(cs_zscore)?;
        let shadow_daily = shadow_daily(&f_all, &t_intercept)?;
        let shadow = shadow_daily
            .ts(|values| ts_mean(values, WINDOW, 1))?
            .cs(cs_zscore)?;
        let evening = evening_component(panel, &t_intercept, WINDOW)?.cs(cs_zscore)?;
        let factor = combine_components(&dawn_fog, &shadow, &evening)?;

        Ok(factor.to_factor_series(self.spec()))
    }
}

fn regression_metrics_from_rows(
    indices: &[usize],
    close: &[Option<f64>],
    volume: &[Option<f64>],
) -> RegressionMetrics {
    let close_series = indices
        .iter()
        .map(|idx| clean_intraday_value(close[*idx]))
        .collect::<Vec<_>>();
    let volume_series = indices
        .iter()
        .map(|idx| clean_intraday_value(volume[*idx]))
        .collect::<Vec<_>>();
    let rows = regression_rows_from_series(&close_series, &volume_series);
    let Some(result) = ols_regression(&rows) else {
        return RegressionMetrics {
            dawn_fog_tstd: None,
            f_all: None,
            t_intercept: None,
        };
    };

    RegressionMetrics {
        dawn_fog_tstd: dawn_fog_tstd(&result.t_values),
        f_all: Some(result.f_all),
        t_intercept: clean(Some(result.t_values[PARAMETER_COUNT - 1])),
    }
}

fn regression_rows_from_series(
    close: &[Option<f64>],
    volume: &[Option<f64>],
) -> Vec<RegressionRow> {
    if close.len() < REGRESSION_OFFSET + 2 || volume.len() < REGRESSION_OFFSET + 2 {
        return Vec::new();
    }

    let returns = close
        .windows(2)
        .map(|window| ret(window[1], window[0]))
        .collect::<Vec<_>>();
    let volume_diff = volume
        .windows(2)
        .map(|window| diff(window[1], window[0]))
        .collect::<Vec<_>>();
    let mut rows = Vec::new();
    for ret_idx in REGRESSION_OFFSET..returns.len() {
        let Some(y) = clean(returns[ret_idx]) else {
            continue;
        };
        let mut x = [0.0; PARAMETER_COUNT];
        let mut valid = true;
        for lag in 0..SLOPE_COUNT {
            let Some(value) = clean(volume_diff[ret_idx - lag]) else {
                valid = false;
                break;
            };
            x[lag] = value;
        }
        if !valid {
            continue;
        }
        x[PARAMETER_COUNT - 1] = 1.0;
        rows.push(RegressionRow { y, x });
    }
    rows
}

fn ols_regression(rows: &[RegressionRow]) -> Option<RegressionResult> {
    let residual_df = rows.len().checked_sub(PARAMETER_COUNT + 1)?;
    if residual_df == 0 {
        return None;
    }

    let scales = regression_scales(rows);
    let mut xtx = [[0.0; PARAMETER_COUNT]; PARAMETER_COUNT];
    let mut xty = [0.0; PARAMETER_COUNT];
    let mut y_sum = 0.0;
    for row in rows {
        y_sum += row.y;
        let x = scaled_design(&row.x, &scales);
        for i in 0..PARAMETER_COUNT {
            xty[i] += x[i] * row.y;
            for j in 0..PARAMETER_COUNT {
                xtx[i][j] += x[i] * x[j];
            }
        }
    }
    let xtx_inv = pseudo_inverse_symmetric(xtx)?;
    let beta = mat_vec_mul(&xtx_inv, &xty);
    let y_mean = y_sum / rows.len() as f64;

    let mut rss = 0.0;
    let mut tss = 0.0;
    for row in rows {
        let x = scaled_design(&row.x, &scales);
        let y_pred = dot(&x, &beta);
        rss += (row.y - y_pred).powi(2);
        tss += (row.y - y_mean).powi(2);
    }
    let rss_mean = rss / residual_df as f64;
    if rss_mean <= f64::EPSILON || !rss_mean.is_finite() {
        return None;
    }
    let ess = (tss - rss).max(0.0);
    let f_all = (ess / PARAMETER_COUNT as f64) / rss_mean;
    if !f_all.is_finite() {
        return None;
    }

    let mut t_values = [f64::NAN; PARAMETER_COUNT];
    for idx in 0..PARAMETER_COUNT {
        let variance = xtx_inv[idx][idx] * rss_mean;
        if variance <= f64::EPSILON || !variance.is_finite() {
            return None;
        }
        t_values[idx] = beta[idx] / variance.sqrt();
        if !t_values[idx].is_finite() {
            return None;
        }
    }
    Some(RegressionResult { f_all, t_values })
}

fn regression_scales(rows: &[RegressionRow]) -> [f64; PARAMETER_COUNT] {
    let mut scales = [1.0_f64; PARAMETER_COUNT];
    for row in rows {
        for (idx, scale) in scales.iter_mut().enumerate().take(SLOPE_COUNT) {
            *scale = (*scale).max(row.x[idx].abs());
        }
    }
    for scale in scales.iter_mut().take(SLOPE_COUNT) {
        if *scale <= f64::EPSILON || !scale.is_finite() {
            *scale = 1.0;
        }
    }
    scales
}

fn scaled_design(
    x: &[f64; PARAMETER_COUNT],
    scales: &[f64; PARAMETER_COUNT],
) -> [f64; PARAMETER_COUNT] {
    let mut output = [0.0; PARAMETER_COUNT];
    for idx in 0..PARAMETER_COUNT {
        output[idx] = x[idx] / scales[idx];
    }
    output
}

fn pseudo_inverse_symmetric(
    mut matrix: [[f64; PARAMETER_COUNT]; PARAMETER_COUNT],
) -> Option<[[f64; PARAMETER_COUNT]; PARAMETER_COUNT]> {
    let mut eigenvectors = [[0.0; PARAMETER_COUNT]; PARAMETER_COUNT];
    for idx in 0..PARAMETER_COUNT {
        eigenvectors[idx][idx] = 1.0;
    }

    for _ in 0..100 {
        let mut pivot_row = 0usize;
        let mut pivot_col = 1usize;
        let mut max_offdiag = 0.0;
        for row in 0..PARAMETER_COUNT {
            for col in (row + 1)..PARAMETER_COUNT {
                let value = matrix[row][col].abs();
                if value > max_offdiag {
                    max_offdiag = value;
                    pivot_row = row;
                    pivot_col = col;
                }
            }
        }
        if max_offdiag <= 1e-10 {
            break;
        }
        let p = pivot_row;
        let q = pivot_col;
        let app = matrix[p][p];
        let aqq = matrix[q][q];
        let apq = matrix[p][q];
        if !app.is_finite() || !aqq.is_finite() || !apq.is_finite() {
            return None;
        }
        let angle = 0.5 * (2.0 * apq).atan2(aqq - app);
        let c = angle.cos();
        let s = angle.sin();

        for idx in 0..PARAMETER_COUNT {
            if idx == p || idx == q {
                continue;
            }
            let aip = matrix[idx][p];
            let aiq = matrix[idx][q];
            matrix[idx][p] = c * aip - s * aiq;
            matrix[p][idx] = matrix[idx][p];
            matrix[idx][q] = s * aip + c * aiq;
            matrix[q][idx] = matrix[idx][q];
        }

        matrix[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
        matrix[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
        matrix[p][q] = 0.0;
        matrix[q][p] = 0.0;

        for idx in 0..PARAMETER_COUNT {
            let vip = eigenvectors[idx][p];
            let viq = eigenvectors[idx][q];
            eigenvectors[idx][p] = c * vip - s * viq;
            eigenvectors[idx][q] = s * vip + c * viq;
        }
    }

    let max_eigenvalue = (0..PARAMETER_COUNT)
        .map(|idx| matrix[idx][idx].abs())
        .fold(0.0, f64::max);
    if max_eigenvalue <= f64::EPSILON || !max_eigenvalue.is_finite() {
        return None;
    }
    let tolerance = max_eigenvalue * 1e-12;
    let mut inverse = [[0.0; PARAMETER_COUNT]; PARAMETER_COUNT];
    let mut rank = 0usize;
    for eig_idx in 0..PARAMETER_COUNT {
        let eigenvalue = matrix[eig_idx][eig_idx];
        if eigenvalue.abs() <= tolerance {
            continue;
        }
        rank += 1;
        let inv_eigenvalue = 1.0 / eigenvalue;
        for row in 0..PARAMETER_COUNT {
            for col in 0..PARAMETER_COUNT {
                inverse[row][col] +=
                    eigenvectors[row][eig_idx] * inv_eigenvalue * eigenvectors[col][eig_idx];
            }
        }
    }
    (rank > 0).then_some(inverse)
}

fn mat_vec_mul(
    matrix: &[[f64; PARAMETER_COUNT]; PARAMETER_COUNT],
    vector: &[f64; PARAMETER_COUNT],
) -> [f64; PARAMETER_COUNT] {
    let mut output = [0.0; PARAMETER_COUNT];
    for row in 0..PARAMETER_COUNT {
        output[row] = matrix[row]
            .iter()
            .zip(vector)
            .map(|(left, right)| left * right)
            .sum();
    }
    output
}

fn dot(left: &[f64; PARAMETER_COUNT], right: &[f64; PARAMETER_COUNT]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}

fn dawn_fog_tstd(t_values: &[f64; PARAMETER_COUNT]) -> Option<f64> {
    mean_std(t_values[1..6].iter().copied()).map(|(_, std)| std)
}

fn shadow_daily(f_all: &PanelColumn, t_intercept: &PanelColumn) -> Result<PanelColumn> {
    let sign = f_all.cs(|cross| {
        let Some(mean) = mean(cross.iter().filter_map(|value| clean(*value))) else {
            return vec![None; cross.len()];
        };
        cross
            .iter()
            .map(|value| clean(*value).map(|value| if value >= mean { 1.0 } else { -1.0 }))
            .collect()
    })?;
    t_intercept.zip_binary(&sign, |intercept, sign| {
        match (clean(intercept), clean(sign)) {
            (Some(intercept), Some(sign)) => Some(intercept.abs() * sign),
            _ => None,
        }
    })
}

fn evening_component(
    panel: &DailyPanel,
    t_intercept: &PanelColumn,
    window: usize,
) -> Result<PanelColumn> {
    let date_count = panel.dates().len();
    let code_count = panel.instruments().len();
    let mut output = vec![None; panel.shape_len()];
    for end_date_idx in 0..date_count {
        if end_date_idx + 1 < window {
            continue;
        }
        let start_date_idx = end_date_idx + 1 - window;
        let mut matrix = vec![None; window * code_count];
        for row_idx in 0..window {
            let date_idx = start_date_idx + row_idx;
            for code_idx in 0..code_count {
                matrix[row_idx * code_count + code_idx] =
                    t_intercept.values()[date_idx * code_count + code_idx];
            }
        }
        let correlations = mean_abs_column_corr_complete(&matrix, window, code_count);
        for code_idx in 0..code_count {
            output[end_date_idx * code_count + code_idx] = correlations[code_idx];
        }
    }
    panel.column_from_values(output)
}

fn mean_abs_column_corr_complete(
    matrix: &[Option<f64>],
    row_count: usize,
    column_count: usize,
) -> Vec<Option<f64>> {
    if row_count < 2 || column_count < 2 {
        return vec![None; column_count];
    }
    let mut original_columns = Vec::new();
    let mut normalized = Vec::new();
    for column_idx in 0..column_count {
        let mut values = Vec::with_capacity(row_count);
        let mut complete = true;
        for row_idx in 0..row_count {
            match clean(matrix[row_idx * column_count + column_idx]) {
                Some(value) => values.push(value),
                None => {
                    complete = false;
                    break;
                }
            }
        }
        if !complete {
            continue;
        }
        let Some((mean, std)) = mean_std(values.iter().copied()) else {
            continue;
        };
        if std <= f64::EPSILON {
            continue;
        }
        original_columns.push(column_idx);
        normalized.push(
            values
                .into_iter()
                .map(|value| (value - mean) / std)
                .collect::<Vec<_>>(),
        );
    }

    let valid_count = normalized.len();
    if valid_count < 2 {
        return vec![None; column_count];
    }
    let chunk_starts = (0..valid_count)
        .step_by(CORR_BLOCK_SIZE)
        .collect::<Vec<_>>();
    let partials = chunk_starts
        .into_par_iter()
        .map(|start| {
            let end = (start + CORR_BLOCK_SIZE).min(valid_count);
            let mut sums = vec![0.0; valid_count];
            let mut counts = vec![0usize; valid_count];
            for left_idx in start..end {
                for right_idx in (left_idx + 1)..valid_count {
                    let dot = normalized[left_idx]
                        .iter()
                        .zip(&normalized[right_idx])
                        .map(|(left, right)| left * right)
                        .sum::<f64>();
                    let corr = dot / row_count as f64;
                    if corr.is_nan() {
                        continue;
                    }
                    let abs_corr = corr.abs();
                    sums[left_idx] += abs_corr;
                    sums[right_idx] += abs_corr;
                    counts[left_idx] += 1;
                    counts[right_idx] += 1;
                }
            }
            (sums, counts)
        })
        .collect::<Vec<_>>();

    let mut sums = vec![0.0; valid_count];
    let mut counts = vec![0usize; valid_count];
    for (partial_sums, partial_counts) in partials {
        for idx in 0..valid_count {
            sums[idx] += partial_sums[idx];
            counts[idx] += partial_counts[idx];
        }
    }

    let mut output = vec![None; column_count];
    for (valid_idx, original_idx) in original_columns.into_iter().enumerate() {
        if counts[valid_idx] > 0 {
            output[original_idx] = Some(sums[valid_idx] / counts[valid_idx] as f64);
        }
    }
    output
}

fn combine_components(
    dawn_fog: &PanelColumn,
    shadow: &PanelColumn,
    evening: &PanelColumn,
) -> Result<PanelColumn> {
    dawn_fog.zip_ternary(shadow, evening, |dawn, shadow, evening| {
        match (clean(dawn), clean(shadow), clean(evening)) {
            (Some(dawn), Some(shadow), Some(evening)) => Some(dawn + shadow - evening),
            _ => None,
        }
    })
}

fn ret(numerator: Option<f64>, denominator: Option<f64>) -> Option<f64> {
    match (clean(numerator), clean(denominator)) {
        (Some(numerator), Some(denominator)) if denominator.abs() > f64::EPSILON => {
            Some(numerator / denominator - 1.0)
        }
        _ => None,
    }
}

fn diff(current: Option<f64>, previous: Option<f64>) -> Option<f64> {
    match (clean(current), clean(previous)) {
        (Some(current), Some(previous)) => Some(current - previous),
        _ => None,
    }
}

fn clean(value: Option<f64>) -> Option<f64> {
    value.filter(|value| value.is_finite())
}

fn mean(values: impl IntoIterator<Item = f64>) -> Option<f64> {
    let values = values
        .into_iter()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64)
}

fn mean_std(values: impl IntoIterator<Item = f64>) -> Option<(f64, f64)> {
    let values = values
        .into_iter()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    if values.is_empty() {
        return None;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / values.len() as f64;
    Some((mean, variance.sqrt()))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::{
        combine_components, dawn_fog_tstd, evening_component, mean_abs_column_corr_complete,
        ols_regression, regression_rows_from_series, shadow_daily, RegressionRow,
    };
    use crate::core::{AssetClass, FactorContext, Frequency};
    use crate::data::{ColumnData, Table};
    use crate::factor::common::DailyPanel;

    fn assert_close(actual: Option<f64>, expected: Option<f64>) {
        match (actual, expected) {
            (Some(actual), Some(expected)) => assert!(
                (actual - expected).abs() < 1e-8,
                "actual={actual}, expected={expected}"
            ),
            (None, None) => {}
            _ => panic!("actual={actual:?}, expected={expected:?}"),
        }
    }

    #[test]
    fn regression_rows_align_returns_and_lagged_volume_diff_like_python() {
        let close = (1..=10).map(|value| Some(value as f64)).collect::<Vec<_>>();
        let volume = (1..=10)
            .map(|value| Some((value * value) as f64))
            .collect::<Vec<_>>();
        let rows = regression_rows_from_series(&close, &volume);

        assert_eq!(rows.len(), 3);
        assert_close(Some(rows[0].y), Some(8.0 / 7.0 - 1.0));
        assert_eq!(rows[0].x, [15.0, 13.0, 11.0, 9.0, 7.0, 5.0, 1.0]);
    }

    #[test]
    fn ols_returns_stats_for_full_rank_sample() {
        let mut rows = Vec::new();
        for idx in 0..20 {
            let x = [
                idx as f64,
                (idx % 3) as f64,
                (idx % 5) as f64,
                (idx * idx) as f64,
                (idx as f64).sin(),
                (idx as f64).cos(),
                1.0,
            ];
            let y = 0.7 * x[0] - 0.2 * x[1] + 0.05 * x[3] + 1.0 + 0.01 * (idx % 4) as f64;
            rows.push(RegressionRow { y, x });
        }

        let result = ols_regression(&rows).expect("regression");

        assert!(result.f_all.is_finite() && result.f_all > 0.0);
        assert!(result.t_values.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn ols_requires_positive_residual_degrees_of_freedom() {
        let rows = (0..8)
            .map(|idx| RegressionRow {
                y: idx as f64,
                x: [idx as f64, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0],
            })
            .collect::<Vec<_>>();

        assert!(ols_regression(&rows).is_none());
    }

    #[test]
    fn dawn_fog_uses_only_lag_one_to_lag_five_t_values() {
        let t_values = [100.0, 1.0, 2.0, 3.0, 4.0, 5.0, 200.0];

        assert_close(dawn_fog_tstd(&t_values), Some(2.0_f64.sqrt()));
    }

    #[test]
    fn shadow_daily_flips_below_cross_section_mean_f_all() {
        let panel = sample_panel(vec![Some(1.0), Some(2.0), Some(3.0)], "f");
        let f = panel.column("f").expect("f");
        let intercept = panel
            .column_from_values(vec![Some(-4.0), Some(5.0), Some(-6.0)])
            .expect("intercept");
        let shadow = shadow_daily(&f, &intercept).expect("shadow");

        assert_eq!(shadow.values(), &[Some(-4.0), Some(5.0), Some(6.0)]);
    }

    #[test]
    fn mean_abs_column_corr_excludes_self_correlation() {
        let matrix = vec![
            Some(1.0),
            Some(1.0),
            Some(3.0),
            Some(2.0),
            Some(2.0),
            Some(2.0),
            Some(3.0),
            Some(3.0),
            Some(1.0),
        ];
        let output = mean_abs_column_corr_complete(&matrix, 3, 3);

        assert_close(output[0], Some(1.0));
        assert_close(output[1], Some(1.0));
        assert_close(output[2], Some(1.0));
    }

    #[test]
    fn evening_component_uses_complete_twenty_day_window() {
        let mut dates = Vec::new();
        let mut codes = Vec::new();
        let mut values = Vec::new();
        for idx in 0..20 {
            let date = 20260101 + idx;
            dates.extend([Some(date), Some(date), Some(date)]);
            codes.extend([
                Some("a".to_string()),
                Some("b".to_string()),
                Some("c".to_string()),
            ]);
            values.extend([
                Some(idx as f64),
                Some((idx * 2) as f64),
                Some((20 - idx) as f64),
            ]);
        }
        let table = Table::new(BTreeMap::from([
            ("trade_date".to_string(), ColumnData::I32(dates)),
            ("ts_code".to_string(), ColumnData::Utf8(codes)),
            ("t".to_string(), ColumnData::F64(values)),
        ]))
        .expect("table");
        let context = FactorContext {
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            start_date: 20260120,
            end_date: 20260120,
            load_start_date: 20260101,
            load_dates: (0..20).map(|idx| 20260101 + idx).collect(),
            target_dates: vec![20260120],
        };
        let panel = DailyPanel::from_table(&table, &context).expect("panel");
        let t = panel.column("t").expect("t");
        let evening = evening_component(&panel, &t, 20).expect("evening");

        assert!(evening.values().iter().take(57).all(Option::is_none));
        assert_close(evening.values()[57], Some(1.0));
        assert_close(evening.values()[58], Some(1.0));
        assert_close(evening.values()[59], Some(1.0));
    }

    #[test]
    fn combine_components_uses_python_direction() {
        let panel = sample_panel(vec![Some(1.0)], "dawn");
        let dawn = panel.column("dawn").expect("dawn");
        let shadow = panel.column_from_values(vec![Some(2.0)]).expect("shadow");
        let evening = panel.column_from_values(vec![Some(0.5)]).expect("evening");
        let factor = combine_components(&dawn, &shadow, &evening).expect("factor");

        assert_eq!(factor.values(), &[Some(2.5)]);
    }

    fn sample_panel(values: Vec<Option<f64>>, name: &str) -> DailyPanel {
        let len = values.len();
        let table = Table::new(BTreeMap::from([
            (
                "trade_date".to_string(),
                ColumnData::I32((0..len).map(|_| Some(20260101)).collect()),
            ),
            (
                "ts_code".to_string(),
                ColumnData::Utf8((0..len).map(|idx| Some(format!("s{idx}"))).collect()),
            ),
            (name.to_string(), ColumnData::F64(values)),
        ]))
        .expect("table");
        let context = FactorContext {
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            start_date: 20260101,
            end_date: 20260101,
            load_start_date: 20260101,
            load_dates: vec![20260101],
            target_dates: vec![20260101],
        };
        DailyPanel::from_table(&table, &context).expect("panel")
    }
}

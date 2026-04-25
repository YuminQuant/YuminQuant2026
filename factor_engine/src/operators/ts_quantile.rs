use super::utils;

pub fn ts_quantile(
    values: &[Option<f64>],
    window: usize,
    min_periods: usize,
    quantile: f64,
) -> Vec<Option<f64>> {
    utils::rolling_unary(values, window, min_periods, |window_values| {
        utils::quantile(window_values, quantile)
    })
}

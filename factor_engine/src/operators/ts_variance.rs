use super::utils;

pub fn ts_variance(values: &[Option<f64>], window: usize, min_periods: usize) -> Vec<Option<f64>> {
    utils::rolling_unary(values, window, min_periods, |window_values| {
        Some(utils::variance(window_values))
    })
}

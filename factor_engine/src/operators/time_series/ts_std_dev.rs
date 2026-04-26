use super::utils;

pub fn ts_std_dev(values: &[Option<f64>], window: usize, min_periods: usize) -> Vec<Option<f64>> {
    utils::rolling_unary(values, window, min_periods, |window_values| {
        Some(utils::std_dev(window_values))
    })
}

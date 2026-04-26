use super::utils;

pub fn ts_prod(values: &[Option<f64>], window: usize, min_periods: usize) -> Vec<Option<f64>> {
    utils::rolling_unary(values, window, min_periods, |window_values| {
        Some(window_values.iter().product::<f64>())
    })
}

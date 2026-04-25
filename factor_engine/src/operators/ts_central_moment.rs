use super::utils;

pub fn ts_central_moment(
    values: &[Option<f64>],
    window: usize,
    min_periods: usize,
    order: u32,
) -> Vec<Option<f64>> {
    utils::rolling_unary(values, window, min_periods, |window_values| {
        Some(utils::central_moment(window_values, order))
    })
}

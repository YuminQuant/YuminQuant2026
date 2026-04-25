use super::utils;

pub fn ts_rank(values: &[Option<f64>], window: usize, min_periods: usize) -> Vec<Option<f64>> {
    utils::rolling_unary(values, window, min_periods, |window_values| {
        utils::ranks(window_values).last().copied()
    })
}

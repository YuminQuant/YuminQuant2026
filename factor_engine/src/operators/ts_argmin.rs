use super::utils;

pub fn ts_argmin(values: &[Option<f64>], window: usize, min_periods: usize) -> Vec<Option<f64>> {
    utils::rolling_extreme(values, window, min_periods, false, true)
}

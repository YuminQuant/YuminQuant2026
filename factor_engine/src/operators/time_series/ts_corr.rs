use super::utils;

pub fn ts_corr(
    left: &[Option<f64>],
    right: &[Option<f64>],
    window: usize,
    min_periods: usize,
) -> Vec<Option<f64>> {
    utils::rolling_binary(left, right, window, min_periods, utils::pearson_corr)
}

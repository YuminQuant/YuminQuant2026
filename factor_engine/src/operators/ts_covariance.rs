use super::utils;

pub fn ts_covariance(
    left: &[Option<f64>],
    right: &[Option<f64>],
    window: usize,
    min_periods: usize,
) -> Vec<Option<f64>> {
    utils::rolling_binary(
        left,
        right,
        window,
        min_periods,
        |left_values, right_values| Some(utils::covariance(left_values, right_values)),
    )
}

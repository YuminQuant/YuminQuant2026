use super::utils;

pub fn ts_regression(
    y: &[Option<f64>],
    x: &[Option<f64>],
    window: usize,
    min_periods: usize,
) -> Vec<Option<f64>> {
    utils::rolling_binary(y, x, window, min_periods, |y_values, x_values| {
        let variance_x = utils::variance(x_values);
        (variance_x > f64::EPSILON).then_some(utils::covariance(y_values, x_values) / variance_x)
    })
}

use super::utils;

pub fn ts_skew(values: &[Option<f64>], window: usize, min_periods: usize) -> Vec<Option<f64>> {
    utils::rolling_unary(values, window, min_periods, |window_values| {
        let std_dev = utils::std_dev(window_values);
        (std_dev > f64::EPSILON)
            .then_some(utils::central_moment(window_values, 3) / std_dev.powi(3))
    })
}

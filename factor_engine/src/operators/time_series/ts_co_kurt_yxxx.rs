use super::utils;

pub fn ts_co_kurt_yxxx(
    y: &[Option<f64>],
    x: &[Option<f64>],
    window: usize,
    min_periods: usize,
) -> Vec<Option<f64>> {
    utils::rolling_binary(y, x, window, min_periods, |y_values, x_values| {
        let mean_y = utils::mean(y_values);
        let mean_x = utils::mean(x_values);
        let std_y = utils::std_dev(y_values);
        let std_x = utils::std_dev(x_values);
        let denominator = std_y * std_x.powi(3);
        if denominator <= f64::EPSILON {
            return None;
        }

        let moment = y_values
            .iter()
            .zip(x_values.iter())
            .map(|(y, x)| (y - mean_y) * (x - mean_x).powi(3))
            .sum::<f64>()
            / y_values.len() as f64;
        Some(moment / denominator)
    })
}

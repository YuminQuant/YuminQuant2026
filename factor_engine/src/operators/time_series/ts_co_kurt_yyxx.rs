use super::utils;

pub fn ts_co_kurt_yyxx(
    y: &[Option<f64>],
    x: &[Option<f64>],
    window: usize,
    min_periods: usize,
) -> Vec<Option<f64>> {
    utils::rolling_binary(y, x, window, min_periods, |y_values, x_values| {
        let mean_y = utils::mean(y_values);
        let mean_x = utils::mean(x_values);
        let variance_y = utils::variance(y_values);
        let variance_x = utils::variance(x_values);
        let denominator = variance_y * variance_x;
        if denominator <= f64::EPSILON {
            return None;
        }

        let moment = y_values
            .iter()
            .zip(x_values.iter())
            .map(|(y, x)| (y - mean_y).powi(2) * (x - mean_x).powi(2))
            .sum::<f64>()
            / y_values.len() as f64;
        Some(moment / denominator)
    })
}

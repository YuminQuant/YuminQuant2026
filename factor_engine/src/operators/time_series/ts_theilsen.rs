use super::utils;

pub fn ts_theilsen(
    y: &[Option<f64>],
    x: &[Option<f64>],
    window: usize,
    min_periods: usize,
) -> Vec<Option<f64>> {
    utils::rolling_binary(y, x, window, min_periods, |y_values, x_values| {
        let mut slopes = Vec::new();
        for left in 0..x_values.len() {
            for right in left + 1..x_values.len() {
                let x_diff = x_values[right] - x_values[left];
                if x_diff.abs() <= f64::EPSILON {
                    continue;
                }
                slopes.push((y_values[right] - y_values[left]) / x_diff);
            }
        }
        utils::median(&slopes)
    })
}

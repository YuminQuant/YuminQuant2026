use super::utils;

pub fn ts_entropy(values: &[Option<f64>], window: usize, min_periods: usize) -> Vec<Option<f64>> {
    utils::rolling_unary(values, window, min_periods, |window_values| {
        let total_abs = window_values.iter().map(|value| value.abs()).sum::<f64>();
        if total_abs <= f64::EPSILON {
            return None;
        }

        Some(
            window_values
                .iter()
                .map(|value| value.abs() / total_abs)
                .filter(|probability| *probability > f64::EPSILON)
                .map(|probability| -probability * probability.ln())
                .sum::<f64>(),
        )
    })
}

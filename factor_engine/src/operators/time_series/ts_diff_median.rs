use crate::operators::ts_median;

use super::utils;

pub fn ts_diff_median(
    values: &[Option<f64>],
    window: usize,
    min_periods: usize,
) -> Vec<Option<f64>> {
    let median = ts_median(values, window, min_periods);
    values
        .iter()
        .zip(median.iter())
        .map(
            |(value, median)| match (utils::valid_value(*value), median) {
                (Some(value), Some(median)) => Some(value - median),
                _ => None,
            },
        )
        .collect()
}

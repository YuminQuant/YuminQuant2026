use crate::operators::ts_max;

use super::utils;

pub fn ts_diff_max(values: &[Option<f64>], window: usize, min_periods: usize) -> Vec<Option<f64>> {
    let max = ts_max(values, window, min_periods);
    values
        .iter()
        .zip(max.iter())
        .map(|(value, max)| match (utils::valid_value(*value), max) {
            (Some(value), Some(max)) => Some(value - max),
            _ => None,
        })
        .collect()
}

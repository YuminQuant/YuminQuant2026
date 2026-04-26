use crate::operators::ts_min;

use super::utils;

pub fn ts_diff_min(values: &[Option<f64>], window: usize, min_periods: usize) -> Vec<Option<f64>> {
    let min = ts_min(values, window, min_periods);
    values
        .iter()
        .zip(min.iter())
        .map(|(value, min)| match (utils::valid_value(*value), min) {
            (Some(value), Some(min)) => Some(value - min),
            _ => None,
        })
        .collect()
}

use crate::operators::ts_mean;

use super::utils;

pub fn ts_diff_avg(values: &[Option<f64>], window: usize, min_periods: usize) -> Vec<Option<f64>> {
    let mean = ts_mean(values, window, min_periods);
    values
        .iter()
        .zip(mean.iter())
        .map(|(value, mean)| match (utils::valid_value(*value), mean) {
            (Some(value), Some(mean)) => Some(value - mean),
            _ => None,
        })
        .collect()
}

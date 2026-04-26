use crate::operators::{ts_mean, ts_std_dev};

use super::utils;

pub fn ts_zscore(values: &[Option<f64>], window: usize, min_periods: usize) -> Vec<Option<f64>> {
    let mean = ts_mean(values, window, min_periods);
    let std_dev = ts_std_dev(values, window, min_periods);
    values
        .iter()
        .zip(mean.iter())
        .zip(std_dev.iter())
        .map(
            |((value, mean), std_dev)| match (utils::valid_value(*value), mean, std_dev) {
                (Some(value), Some(mean), Some(std_dev)) if std_dev.abs() > f64::EPSILON => {
                    Some((value - mean) / std_dev)
                }
                _ => None,
            },
        )
        .collect()
}

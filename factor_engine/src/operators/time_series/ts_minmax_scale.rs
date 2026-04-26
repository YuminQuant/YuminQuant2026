use crate::operators::{ts_max, ts_min};

use super::utils;

pub fn ts_minmax_scale(
    values: &[Option<f64>],
    window: usize,
    min_periods: usize,
) -> Vec<Option<f64>> {
    let min = ts_min(values, window, min_periods);
    let max = ts_max(values, window, min_periods);
    values
        .iter()
        .zip(min.iter())
        .zip(max.iter())
        .map(
            |((value, min), max)| match (utils::valid_value(*value), min, max) {
                (Some(value), Some(min), Some(max)) if (max - min).abs() > f64::EPSILON => {
                    Some((value - min) / (max - min))
                }
                _ => None,
            },
        )
        .collect()
}

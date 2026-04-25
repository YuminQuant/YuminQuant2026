use crate::operators::ts_delay;

use super::utils;

pub fn ts_diff(values: &[Option<f64>], periods: usize) -> Vec<Option<f64>> {
    let delayed = ts_delay(values, periods);
    values
        .iter()
        .zip(delayed.iter())
        .map(|(current, previous)| {
            match (utils::valid_value(*current), utils::valid_value(*previous)) {
                (Some(current), Some(previous)) => Some(current - previous),
                _ => None,
            }
        })
        .collect()
}

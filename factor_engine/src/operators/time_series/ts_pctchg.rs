use crate::operators::ts_delay;

use super::utils;

pub fn ts_pctchg(values: &[Option<f64>], periods: usize) -> Vec<Option<f64>> {
    let delayed = ts_delay(values, periods);
    values
        .iter()
        .zip(delayed.iter())
        .map(|(current, previous)| {
            match (utils::valid_value(*current), utils::valid_value(*previous)) {
                (Some(current), Some(previous)) if previous.abs() > f64::EPSILON => {
                    Some(current / previous - 1.0)
                }
                _ => None,
            }
        })
        .collect()
}

pub fn ts_pct_chg(values: &[Option<f64>], periods: usize) -> Vec<Option<f64>> {
    ts_pctchg(values, periods)
}

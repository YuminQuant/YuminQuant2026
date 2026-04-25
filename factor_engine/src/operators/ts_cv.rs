use crate::operators::{ts_mean, ts_std_dev};

pub fn ts_cv(values: &[Option<f64>], window: usize, min_periods: usize) -> Vec<Option<f64>> {
    let mean = ts_mean(values, window, min_periods);
    let std_dev = ts_std_dev(values, window, min_periods);
    mean.iter()
        .zip(std_dev.iter())
        .map(|(mean, std_dev)| match (mean, std_dev) {
            (Some(mean), Some(std_dev)) if mean.abs() > f64::EPSILON => Some(std_dev / mean),
            _ => None,
        })
        .collect()
}

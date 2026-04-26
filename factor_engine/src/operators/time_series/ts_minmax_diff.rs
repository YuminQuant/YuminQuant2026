use crate::operators::{ts_max, ts_min};

pub fn ts_minmax_diff(
    values: &[Option<f64>],
    window: usize,
    min_periods: usize,
) -> Vec<Option<f64>> {
    let min = ts_min(values, window, min_periods);
    let max = ts_max(values, window, min_periods);
    min.iter()
        .zip(max.iter())
        .map(|(min, max)| match (min, max) {
            (Some(min), Some(max)) => Some(max - min),
            _ => None,
        })
        .collect()
}

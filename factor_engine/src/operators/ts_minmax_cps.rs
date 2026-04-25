use crate::operators::ts_minmax_scale;

pub fn ts_minmax_cps(
    values: &[Option<f64>],
    window: usize,
    min_periods: usize,
) -> Vec<Option<f64>> {
    ts_minmax_scale(values, window, min_periods)
        .into_iter()
        .map(|value| value.map(|value| value * 2.0 - 1.0))
        .collect()
}

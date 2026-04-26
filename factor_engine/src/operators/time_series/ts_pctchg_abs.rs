use crate::operators::ts_pctchg;

pub fn ts_pctchg_abs(values: &[Option<f64>], periods: usize) -> Vec<Option<f64>> {
    ts_pctchg(values, periods)
        .into_iter()
        .map(|value| value.map(f64::abs))
        .collect()
}

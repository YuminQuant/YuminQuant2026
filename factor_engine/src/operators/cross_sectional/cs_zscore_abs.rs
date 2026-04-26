use super::cs_zscore::cs_zscore;

pub fn cs_zscore_abs(values: &[Option<f64>]) -> Vec<Option<f64>> {
    cs_zscore(values)
        .into_iter()
        .map(|value| value.map(f64::abs))
        .collect()
}

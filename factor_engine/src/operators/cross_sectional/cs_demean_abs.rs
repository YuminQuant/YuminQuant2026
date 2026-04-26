use super::cs_demean::cs_demean;

pub fn cs_demean_abs(values: &[Option<f64>]) -> Vec<Option<f64>> {
    cs_demean(values)
        .into_iter()
        .map(|value| value.map(f64::abs))
        .collect()
}

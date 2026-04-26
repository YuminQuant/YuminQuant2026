use super::cs_utils;

pub fn cs_mean(values: &[Option<f64>]) -> Vec<Option<f64>> {
    vec![cs_utils::mean(values); values.len()]
}

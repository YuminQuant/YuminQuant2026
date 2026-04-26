use super::cs_utils;

pub fn cs_min(values: &[Option<f64>]) -> Vec<Option<f64>> {
    vec![cs_utils::min(values); values.len()]
}

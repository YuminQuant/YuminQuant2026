use super::cs_utils;

pub fn cs_max(values: &[Option<f64>]) -> Vec<Option<f64>> {
    vec![cs_utils::max(values); values.len()]
}

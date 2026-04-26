use super::cs_utils;

pub fn cs_sum(values: &[Option<f64>]) -> Vec<Option<f64>> {
    vec![cs_utils::sum(values); values.len()]
}

use super::cs_utils;

pub fn cs_median(values: &[Option<f64>]) -> Vec<Option<f64>> {
    vec![cs_utils::median(values); values.len()]
}

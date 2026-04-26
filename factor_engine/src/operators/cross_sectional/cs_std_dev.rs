use super::cs_utils;

pub fn cs_std_dev(values: &[Option<f64>]) -> Vec<Option<f64>> {
    vec![cs_utils::std_dev(values); values.len()]
}

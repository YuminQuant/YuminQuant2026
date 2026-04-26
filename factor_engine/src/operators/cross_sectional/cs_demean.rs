use super::cs_utils;

pub fn cs_demean(values: &[Option<f64>]) -> Vec<Option<f64>> {
    cs_utils::demean(values)
}

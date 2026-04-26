use super::cs_utils;

pub fn cs_zscore(values: &[Option<f64>]) -> Vec<Option<f64>> {
    cs_utils::zscore(values)
}

use super::cs_utils;

pub fn cs_zscore_by_group(values: &[Option<f64>], groups: &[Option<String>]) -> Vec<Option<f64>> {
    cs_utils::transform_by_group(values, groups, cs_utils::zscore)
}

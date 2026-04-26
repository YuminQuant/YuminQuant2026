use super::cs_utils;

pub fn cs_pctrank_by_group(values: &[Option<f64>], groups: &[Option<String>]) -> Vec<Option<f64>> {
    cs_utils::transform_by_group(values, groups, |values| cs_utils::pctranks(values, true))
}

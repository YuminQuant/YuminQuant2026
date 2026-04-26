use super::cs_utils;

pub fn cs_fill_median_by_group(
    values: &[Option<f64>],
    groups: &[Option<String>],
) -> Vec<Option<f64>> {
    cs_utils::fill_stat_by_group(values, groups, cs_utils::median)
}

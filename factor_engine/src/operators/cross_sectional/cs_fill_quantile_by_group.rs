use super::cs_utils;

pub fn cs_fill_quantile_by_group(
    values: &[Option<f64>],
    groups: &[Option<String>],
    quantile: f64,
) -> Vec<Option<f64>> {
    cs_utils::fill_stat_by_group(values, groups, |values| {
        cs_utils::quantile(values, quantile)
    })
}

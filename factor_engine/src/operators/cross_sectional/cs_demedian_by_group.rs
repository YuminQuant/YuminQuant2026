use super::cs_utils;

pub fn cs_demedian_by_group(values: &[Option<f64>], groups: &[Option<String>]) -> Vec<Option<f64>> {
    cs_utils::transform_by_group(values, groups, |values| {
        let Some(median) = cs_utils::median(values) else {
            return vec![None; values.len()];
        };
        values
            .iter()
            .map(|value| cs_utils::clean(*value).map(|value| value - median))
            .collect()
    })
}

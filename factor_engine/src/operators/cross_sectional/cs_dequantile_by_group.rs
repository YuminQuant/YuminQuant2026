use super::cs_utils;

pub fn cs_dequantile_by_group(
    values: &[Option<f64>],
    groups: &[Option<String>],
    quantile: f64,
) -> Vec<Option<f64>> {
    cs_utils::transform_by_group(values, groups, |values| {
        let Some(quantile_value) = cs_utils::quantile(values, quantile) else {
            return vec![None; values.len()];
        };
        values
            .iter()
            .map(|value| cs_utils::clean(*value).map(|value| value - quantile_value))
            .collect()
    })
}

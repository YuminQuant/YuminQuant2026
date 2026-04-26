use super::cs_utils;

pub fn cs_minmax_scale(values: &[Option<f64>]) -> Vec<Option<f64>> {
    let Some(min_value) = cs_utils::min(values) else {
        return vec![None; values.len()];
    };
    let Some(max_value) = cs_utils::max(values) else {
        return vec![None; values.len()];
    };
    let range = max_value - min_value;
    if range.abs() <= f64::EPSILON {
        return vec![None; values.len()];
    }
    values
        .iter()
        .map(|value| cs_utils::clean(*value).map(|value| (value - min_value) / range))
        .collect()
}

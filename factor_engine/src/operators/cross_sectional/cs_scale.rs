use super::cs_utils;

pub fn cs_scale(values: &[Option<f64>]) -> Vec<Option<f64>> {
    let denominator = values
        .iter()
        .filter_map(|value| cs_utils::clean(*value))
        .map(f64::abs)
        .sum::<f64>();
    if denominator.abs() <= f64::EPSILON {
        return vec![None; values.len()];
    }

    values
        .iter()
        .map(|value| cs_utils::clean(*value).map(|value| value / denominator))
        .collect()
}

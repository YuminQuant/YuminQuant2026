use super::cs_utils;

pub fn cs_winsorize(values: &[Option<f64>], upper_quantile: f64) -> Vec<Option<f64>> {
    let Some(cap) = cs_utils::quantile(values, upper_quantile) else {
        return vec![None; values.len()];
    };
    values
        .iter()
        .map(|value| cs_utils::clean(*value).map(|value| value.min(cap)))
        .collect()
}

pub fn cs_winsorize95(values: &[Option<f64>]) -> Vec<Option<f64>> {
    cs_winsorize(values, 0.95)
}

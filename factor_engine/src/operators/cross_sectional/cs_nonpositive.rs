use super::cs_utils;
use super::cs_zscore::cs_zscore;

pub fn cs_nonpositive(values: &[Option<f64>]) -> Vec<Option<f64>> {
    let zscore = cs_zscore(values);
    let Some(max_value) = cs_utils::max(&zscore) else {
        return vec![None; values.len()];
    };
    zscore
        .into_iter()
        .map(|value| value.map(|value| value - max_value))
        .collect()
}

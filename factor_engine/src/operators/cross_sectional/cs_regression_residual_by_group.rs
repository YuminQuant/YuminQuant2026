use super::cs_utils;

pub fn cs_regression_residual_by_group(
    y: &[Option<f64>],
    x: &[Option<f64>],
    groups: &[Option<String>],
) -> Vec<Option<f64>> {
    let mut output = vec![None; y.len()];
    for indices in cs_utils::groups(groups).values() {
        let grouped_y = cs_utils::values_at(y, indices);
        let grouped_x = cs_utils::values_at(x, indices);
        let residual = cs_utils::regression_residual(&grouped_y, &grouped_x);
        for (group_idx, idx) in indices.iter().enumerate() {
            output[*idx] = residual[group_idx];
        }
    }
    output
}

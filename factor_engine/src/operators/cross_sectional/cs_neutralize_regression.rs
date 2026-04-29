use std::collections::BTreeMap;

use super::cs_utils;

#[derive(Clone, Copy, Debug)]
struct Row {
    idx: usize,
    y: f64,
    weight: f64,
}

pub fn cs_neutralize_regression(
    y: &[Option<f64>],
    continuous: &[&[Option<f64>]],
    groups: Option<&[Option<String>]>,
    weights: Option<&[Option<f64>]>,
) -> Vec<Option<f64>> {
    if continuous.iter().any(|values| values.len() != y.len())
        || groups.is_some_and(|values| values.len() != y.len())
        || weights.is_some_and(|values| values.len() != y.len())
    {
        return vec![None; y.len()];
    }

    let mut rows = Vec::new();
    let mut x_rows = Vec::new();
    let mut row_groups = Vec::new();
    for idx in 0..y.len() {
        let Some(y_value) = cs_utils::clean(y[idx]) else {
            continue;
        };
        let mut x_values = Vec::with_capacity(continuous.len());
        let mut valid = true;
        for values in continuous {
            let Some(value) = cs_utils::clean(values[idx]) else {
                valid = false;
                break;
            };
            x_values.push(value);
        }
        if !valid {
            continue;
        }
        let weight = match weights {
            Some(weights) => match cs_utils::clean(weights[idx]) {
                Some(weight) if weight > 0.0 => weight,
                _ => continue,
            },
            None => 1.0,
        };
        let group = match groups {
            Some(groups) => match &groups[idx] {
                Some(group) if !group.is_empty() => Some(group.clone()),
                _ => continue,
            },
            None => None,
        };
        rows.push(Row {
            idx,
            y: y_value,
            weight,
        });
        x_rows.push(x_values);
        row_groups.push(group);
    }

    if rows.is_empty() {
        return vec![None; y.len()];
    }

    let mut residual_y = rows.iter().map(|row| row.y).collect::<Vec<_>>();
    let mut residual_x = x_rows.clone();
    if groups.is_some() {
        demean_by_group(
            &rows,
            &x_rows,
            &row_groups,
            &mut residual_y,
            &mut residual_x,
        );
    }

    if continuous.is_empty() {
        let mut output = vec![None; y.len()];
        for (row_idx, row) in rows.iter().enumerate() {
            output[row.idx] = Some(residual_y[row_idx]);
        }
        return output;
    }

    let include_intercept = groups.is_none();
    let Some(beta) = weighted_least_squares(&residual_y, &residual_x, &rows, include_intercept)
    else {
        return vec![None; y.len()];
    };

    let mut output = vec![None; y.len()];
    for (row_idx, row) in rows.iter().enumerate() {
        let mut fitted = 0.0;
        let mut beta_offset = 0;
        if include_intercept {
            fitted += beta[0];
            beta_offset = 1;
        }
        for (x_idx, x_value) in residual_x[row_idx].iter().enumerate() {
            fitted += beta[beta_offset + x_idx] * x_value;
        }
        output[row.idx] = Some(residual_y[row_idx] - fitted);
    }
    output
}

fn demean_by_group(
    rows: &[Row],
    x_rows: &[Vec<f64>],
    row_groups: &[Option<String>],
    residual_y: &mut [f64],
    residual_x: &mut [Vec<f64>],
) {
    let factor_count = x_rows.first().map(Vec::len).unwrap_or(0);
    let mut grouped = BTreeMap::<String, (f64, f64, Vec<f64>)>::new();
    for (row_idx, row) in rows.iter().enumerate() {
        let Some(group) = &row_groups[row_idx] else {
            continue;
        };
        let entry = grouped
            .entry(group.clone())
            .or_insert_with(|| (0.0, 0.0, vec![0.0; factor_count]));
        entry.0 += row.weight;
        entry.1 += row.weight * row.y;
        for (x_idx, x_value) in x_rows[row_idx].iter().enumerate() {
            entry.2[x_idx] += row.weight * x_value;
        }
    }

    for (row_idx, row) in rows.iter().enumerate() {
        let Some(group) = &row_groups[row_idx] else {
            continue;
        };
        let Some((sum_weight, sum_y, sum_x)) = grouped.get(group) else {
            continue;
        };
        if sum_weight.abs() <= f64::EPSILON {
            continue;
        }
        residual_y[row_idx] = row.y - sum_y / sum_weight;
        for (x_idx, x_value) in x_rows[row_idx].iter().enumerate() {
            residual_x[row_idx][x_idx] = x_value - sum_x[x_idx] / sum_weight;
        }
    }
}

fn weighted_least_squares(
    y: &[f64],
    x: &[Vec<f64>],
    rows: &[Row],
    include_intercept: bool,
) -> Option<Vec<f64>> {
    let factor_count = x.first().map(Vec::len).unwrap_or(0);
    let parameter_count = factor_count + usize::from(include_intercept);
    if parameter_count == 0 || y.len() < parameter_count {
        return None;
    }

    let mut xtwx = vec![vec![0.0; parameter_count]; parameter_count];
    let mut xtwy = vec![0.0; parameter_count];
    for row_idx in 0..y.len() {
        let mut design = Vec::with_capacity(parameter_count);
        if include_intercept {
            design.push(1.0);
        }
        design.extend(x[row_idx].iter().copied());
        let weight = rows[row_idx].weight;
        for i in 0..parameter_count {
            xtwy[i] += weight * design[i] * y[row_idx];
            for j in 0..parameter_count {
                xtwx[i][j] += weight * design[i] * design[j];
            }
        }
    }

    solve_linear_system(xtwx, xtwy)
}

fn solve_linear_system(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    for col in 0..n {
        let pivot =
            (col..n).max_by(|left, right| a[*left][col].abs().total_cmp(&a[*right][col].abs()))?;
        if a[pivot][col].abs() <= 1e-12 {
            return None;
        }
        if pivot != col {
            a.swap(pivot, col);
            b.swap(pivot, col);
        }
        let pivot_value = a[col][col];
        for j in col..n {
            a[col][j] /= pivot_value;
        }
        b[col] /= pivot_value;
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = a[row][col];
            if factor.abs() <= f64::EPSILON {
                continue;
            }
            for j in col..n {
                a[row][j] -= factor * a[col][j];
            }
            b[row] -= factor * b[col];
        }
    }
    Some(b)
}

#[cfg(test)]
mod tests {
    use super::cs_neutralize_regression;

    #[test]
    fn neutralizes_continuous_exposure() {
        let y = vec![Some(1.0), Some(2.0), Some(3.0), Some(4.0)];
        let x = vec![Some(10.0), Some(20.0), Some(30.0), Some(40.0)];
        let residual = cs_neutralize_regression(&y, &[&x], None, None);

        for value in residual {
            assert!(value.unwrap().abs() < 1e-10);
        }
    }

    #[test]
    fn neutralizes_group_and_continuous_exposure() {
        let y = vec![Some(1.0), Some(2.0), Some(10.0), Some(11.0)];
        let x = vec![Some(1.0), Some(2.0), Some(1.0), Some(2.0)];
        let groups = vec![
            Some("a".to_string()),
            Some("a".to_string()),
            Some("b".to_string()),
            Some("b".to_string()),
        ];
        let residual = cs_neutralize_regression(&y, &[&x], Some(&groups), None);

        for value in residual {
            assert!(value.unwrap().abs() < 1e-10);
        }
    }
}

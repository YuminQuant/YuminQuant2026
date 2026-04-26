use std::collections::BTreeMap;

pub fn clean(value: Option<f64>) -> Option<f64> {
    value.filter(|value| !value.is_nan())
}

pub fn valid_count(values: &[Option<f64>]) -> usize {
    values
        .iter()
        .filter(|value| clean(**value).is_some())
        .count()
}

pub fn valid_values(values: &[Option<f64>]) -> Vec<f64> {
    values.iter().filter_map(|value| clean(*value)).collect()
}

pub fn mean(values: &[Option<f64>]) -> Option<f64> {
    let values = valid_values(values);
    mean_f64(&values)
}

pub fn sum(values: &[Option<f64>]) -> Option<f64> {
    let values = valid_values(values);
    (!values.is_empty()).then(|| values.iter().sum())
}

pub fn min(values: &[Option<f64>]) -> Option<f64> {
    valid_values(values).into_iter().reduce(f64::min)
}

pub fn max(values: &[Option<f64>]) -> Option<f64> {
    valid_values(values).into_iter().reduce(f64::max)
}

pub fn median(values: &[Option<f64>]) -> Option<f64> {
    quantile(values, 0.5)
}

pub fn quantile(values: &[Option<f64>], q: f64) -> Option<f64> {
    quantile_f64(valid_values(values), q)
}

pub fn std_dev(values: &[Option<f64>]) -> Option<f64> {
    let values = valid_values(values);
    let mean = mean_f64(&values)?;
    let variance = values
        .iter()
        .map(|value| {
            let diff = value - mean;
            diff * diff
        })
        .sum::<f64>()
        / values.len() as f64;
    Some(variance.sqrt())
}

pub fn ranks(values: &[Option<f64>], ascending: bool) -> Vec<Option<f64>> {
    let mut pairs = values
        .iter()
        .enumerate()
        .filter_map(|(idx, value)| clean(*value).map(|value| (idx, value)))
        .collect::<Vec<_>>();
    pairs.sort_by(|left, right| {
        let value_order = if ascending {
            left.1.total_cmp(&right.1)
        } else {
            right.1.total_cmp(&left.1)
        };
        value_order.then_with(|| left.0.cmp(&right.0))
    });

    let mut output = vec![None; values.len()];
    for (rank_idx, (idx, _)) in pairs.into_iter().enumerate() {
        output[idx] = Some(rank_idx as f64 + 1.0);
    }
    output
}

pub fn pctranks(values: &[Option<f64>], ascending: bool) -> Vec<Option<f64>> {
    let count = valid_count(values);
    if count < 2 {
        return vec![None; values.len()];
    }
    let denominator = count as f64 - 1.0;
    ranks(values, ascending)
        .into_iter()
        .map(|rank| rank.map(|rank| (rank - 1.0) / denominator))
        .collect()
}

pub fn map_binary<F>(left: &[Option<f64>], right: &[Option<f64>], f: F) -> Vec<Option<f64>>
where
    F: Fn(f64, f64) -> Option<f64>,
{
    left.iter()
        .zip(right)
        .map(|(left, right)| match (clean(*left), clean(*right)) {
            (Some(left), Some(right)) => f(left, right),
            _ => None,
        })
        .collect()
}

pub fn groups(groups: &[Option<String>]) -> BTreeMap<String, Vec<usize>> {
    let mut output = BTreeMap::new();
    for (idx, group) in groups.iter().enumerate() {
        if let Some(group) = group {
            if !group.is_empty() {
                output
                    .entry(group.clone())
                    .or_insert_with(Vec::new)
                    .push(idx);
            }
        }
    }
    output
}

pub fn values_at(values: &[Option<f64>], indices: &[usize]) -> Vec<Option<f64>> {
    indices.iter().map(|idx| values[*idx]).collect()
}

pub fn fill_stat_by_group<F>(
    values: &[Option<f64>],
    group_labels: &[Option<String>],
    stat: F,
) -> Vec<Option<f64>>
where
    F: Fn(&[Option<f64>]) -> Option<f64>,
{
    let mut output = values.iter().map(|value| clean(*value)).collect::<Vec<_>>();
    for indices in groups(group_labels).values() {
        let grouped_values = values_at(values, indices);
        let Some(fill_value) = stat(&grouped_values) else {
            continue;
        };
        for idx in indices {
            if output[*idx].is_none() {
                output[*idx] = Some(fill_value);
            }
        }
    }
    output
}

pub fn transform_by_group<F>(
    values: &[Option<f64>],
    group_labels: &[Option<String>],
    transform: F,
) -> Vec<Option<f64>>
where
    F: Fn(&[Option<f64>]) -> Vec<Option<f64>>,
{
    let mut output = vec![None; values.len()];
    for indices in groups(group_labels).values() {
        let grouped_values = values_at(values, indices);
        let transformed = transform(&grouped_values);
        for (group_idx, idx) in indices.iter().enumerate() {
            output[*idx] = transformed[group_idx];
        }
    }
    output
}

pub fn demean(values: &[Option<f64>]) -> Vec<Option<f64>> {
    let Some(mean) = mean(values) else {
        return vec![None; values.len()];
    };
    values
        .iter()
        .map(|value| clean(*value).map(|value| value - mean))
        .collect()
}

pub fn zscore(values: &[Option<f64>]) -> Vec<Option<f64>> {
    let Some(mean) = mean(values) else {
        return vec![None; values.len()];
    };
    let Some(std_dev) = std_dev(values) else {
        return vec![None; values.len()];
    };
    if std_dev.abs() <= f64::EPSILON {
        return vec![None; values.len()];
    }
    values
        .iter()
        .map(|value| clean(*value).map(|value| (value - mean) / std_dev))
        .collect()
}

pub fn regression_residual(y: &[Option<f64>], x: &[Option<f64>]) -> Vec<Option<f64>> {
    let pairs = y
        .iter()
        .zip(x)
        .enumerate()
        .filter_map(|(idx, (y, x))| match (clean(*y), clean(*x)) {
            (Some(y), Some(x)) => Some((idx, y, x)),
            _ => None,
        })
        .collect::<Vec<_>>();
    if pairs.len() < 2 {
        return vec![None; y.len()];
    }

    let mean_y = pairs.iter().map(|(_, y, _)| *y).sum::<f64>() / pairs.len() as f64;
    let mean_x = pairs.iter().map(|(_, _, x)| *x).sum::<f64>() / pairs.len() as f64;
    let var_x = pairs
        .iter()
        .map(|(_, _, x)| {
            let diff = x - mean_x;
            diff * diff
        })
        .sum::<f64>();
    let cov_xy = pairs
        .iter()
        .map(|(_, y, x)| (x - mean_x) * (y - mean_y))
        .sum::<f64>();
    let slope = if var_x.abs() <= f64::EPSILON {
        0.0
    } else {
        cov_xy / var_x
    };
    let intercept = mean_y - slope * mean_x;

    let mut output = vec![None; y.len()];
    for (idx, y, x) in pairs {
        output[idx] = Some(y - (intercept + slope * x));
    }
    output
}

fn mean_f64(values: &[f64]) -> Option<f64> {
    (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64)
}

fn quantile_f64(mut values: Vec<f64>, q: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(|left, right| left.total_cmp(right));
    let q = q.clamp(0.0, 1.0);
    let pos = (values.len() - 1) as f64 * q;
    let lower = pos.floor() as usize;
    let upper = pos.ceil() as usize;
    if lower == upper {
        return Some(values[lower]);
    }
    let weight = pos - lower as f64;
    Some(values[lower] * (1.0 - weight) + values[upper] * weight)
}

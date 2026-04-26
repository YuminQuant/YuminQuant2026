use std::collections::VecDeque;

pub(crate) fn valid_value(value: Option<f64>) -> Option<f64> {
    value.filter(|value| !value.is_nan())
}

pub(crate) fn effective_min_periods(window: usize, min_periods: usize) -> Option<usize> {
    if window == 0 {
        return None;
    }

    let effective = min_periods.max(1);
    (effective <= window).then_some(effective)
}

fn window_start(end: usize, window: usize) -> usize {
    end.saturating_add(1).saturating_sub(window)
}

pub(crate) fn rolling_unary<F>(
    values: &[Option<f64>],
    window: usize,
    min_periods: usize,
    mut f: F,
) -> Vec<Option<f64>>
where
    F: FnMut(&[f64]) -> Option<f64>,
{
    let mut output = vec![None; values.len()];
    let Some(effective_min_periods) = effective_min_periods(window, min_periods) else {
        return output;
    };

    for (idx, output_value) in output.iter_mut().enumerate() {
        let start = window_start(idx, window);
        let window_values = values[start..=idx]
            .iter()
            .filter_map(|value| valid_value(*value))
            .collect::<Vec<_>>();
        if window_values.len() >= effective_min_periods {
            *output_value = f(&window_values);
        }
    }
    output
}

pub(crate) fn rolling_binary<F>(
    left: &[Option<f64>],
    right: &[Option<f64>],
    window: usize,
    min_periods: usize,
    mut f: F,
) -> Vec<Option<f64>>
where
    F: FnMut(&[f64], &[f64]) -> Option<f64>,
{
    let len = left.len().min(right.len());
    let mut output = vec![None; len];
    let Some(effective_min_periods) = effective_min_periods(window, min_periods) else {
        return output;
    };

    for (idx, output_value) in output.iter_mut().enumerate() {
        let start = window_start(idx, window);
        let mut left_values = Vec::with_capacity(idx + 1 - start);
        let mut right_values = Vec::with_capacity(idx + 1 - start);
        for window_idx in start..=idx {
            let (Some(left_value), Some(right_value)) = (
                valid_value(left[window_idx]),
                valid_value(right[window_idx]),
            ) else {
                continue;
            };
            left_values.push(left_value);
            right_values.push(right_value);
        }

        if left_values.len() >= effective_min_periods {
            *output_value = f(&left_values, &right_values);
        }
    }
    output
}

pub(crate) fn rolling_triple<F>(
    x: &[Option<f64>],
    y: &[Option<f64>],
    z: &[Option<f64>],
    window: usize,
    min_periods: usize,
    mut f: F,
) -> Vec<Option<f64>>
where
    F: FnMut(&[f64], &[f64], &[f64]) -> Option<f64>,
{
    let len = x.len().min(y.len()).min(z.len());
    let mut output = vec![None; len];
    let Some(effective_min_periods) = effective_min_periods(window, min_periods) else {
        return output;
    };

    for (idx, output_value) in output.iter_mut().enumerate() {
        let start = window_start(idx, window);
        let mut x_values = Vec::with_capacity(idx + 1 - start);
        let mut y_values = Vec::with_capacity(idx + 1 - start);
        let mut z_values = Vec::with_capacity(idx + 1 - start);
        for window_idx in start..=idx {
            let (Some(x_value), Some(y_value), Some(z_value)) = (
                valid_value(x[window_idx]),
                valid_value(y[window_idx]),
                valid_value(z[window_idx]),
            ) else {
                continue;
            };
            x_values.push(x_value);
            y_values.push(y_value);
            z_values.push(z_value);
        }

        if x_values.len() >= effective_min_periods {
            *output_value = f(&x_values, &y_values, &z_values);
        }
    }
    output
}

pub(crate) fn rolling_extreme(
    values: &[Option<f64>],
    window: usize,
    min_periods: usize,
    find_max: bool,
    return_position: bool,
) -> Vec<Option<f64>> {
    let mut output = vec![None; values.len()];
    let Some(effective_min_periods) = effective_min_periods(window, min_periods) else {
        return output;
    };

    let mut deque: VecDeque<(usize, f64)> = VecDeque::new();
    let mut valid_flags = vec![false; values.len()];
    let mut valid_count = 0usize;

    for idx in 0..values.len() {
        if idx >= window && valid_flags[idx - window] {
            valid_count -= 1;
        }

        let start = window_start(idx, window);
        while deque
            .front()
            .is_some_and(|(value_idx, _)| *value_idx < start)
        {
            deque.pop_front();
        }

        if let Some(value) = valid_value(values[idx]) {
            valid_flags[idx] = true;
            valid_count += 1;
            while let Some((_, back_value)) = deque.back() {
                let should_pop = if find_max {
                    value > *back_value
                } else {
                    value < *back_value
                };
                if !should_pop {
                    break;
                }
                deque.pop_back();
            }
            deque.push_back((idx, value));
        }

        if valid_count >= effective_min_periods {
            if let Some((best_idx, best_value)) = deque.front() {
                output[idx] = if return_position {
                    Some((*best_idx - start) as f64)
                } else {
                    Some(*best_value)
                };
            }
        }
    }

    output
}

pub(crate) fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

pub(crate) fn variance(values: &[f64]) -> f64 {
    let mean = mean(values);
    values
        .iter()
        .map(|value| {
            let diff = value - mean;
            diff * diff
        })
        .sum::<f64>()
        / values.len() as f64
}

pub(crate) fn std_dev(values: &[f64]) -> f64 {
    variance(values).sqrt()
}

pub(crate) fn central_moment(values: &[f64], order: u32) -> f64 {
    if order == 0 {
        return 1.0;
    }

    let mean = mean(values);
    values
        .iter()
        .map(|value| (value - mean).powi(order as i32))
        .sum::<f64>()
        / values.len() as f64
}

pub(crate) fn covariance(left: &[f64], right: &[f64]) -> f64 {
    let left_mean = mean(left);
    let right_mean = mean(right);
    left.iter()
        .zip(right.iter())
        .map(|(left, right)| (left - left_mean) * (right - right_mean))
        .sum::<f64>()
        / left.len() as f64
}

pub(crate) fn pearson_corr(left: &[f64], right: &[f64]) -> Option<f64> {
    let covariance = covariance(left, right);
    let denominator = std_dev(left) * std_dev(right);
    (denominator > f64::EPSILON).then_some(covariance / denominator)
}

pub(crate) fn quantile(values: &[f64], quantile: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }

    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let q = quantile.clamp(0.0, 1.0);
    let position = q * (sorted.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    if lower == upper {
        return Some(sorted[lower]);
    }

    let weight = position - lower as f64;
    Some(sorted[lower] * (1.0 - weight) + sorted[upper] * weight)
}

pub(crate) fn median(values: &[f64]) -> Option<f64> {
    quantile(values, 0.5)
}

pub(crate) fn ranks(values: &[f64]) -> Vec<f64> {
    let mut indexed = values
        .iter()
        .enumerate()
        .map(|(idx, value)| (idx, *value))
        .collect::<Vec<_>>();
    indexed.sort_by(|left, right| left.1.total_cmp(&right.1));

    let mut ranks = vec![0.0; values.len()];
    let mut idx = 0usize;
    while idx < indexed.len() {
        let start = idx;
        let value = indexed[idx].1;
        while idx + 1 < indexed.len() && indexed[idx + 1].1 == value {
            idx += 1;
        }
        let end = idx;
        let average_rank = (start + end + 2) as f64 / 2.0;
        for item in indexed.iter().take(end + 1).skip(start) {
            ranks[item.0] = average_rank;
        }
        idx += 1;
    }
    ranks
}

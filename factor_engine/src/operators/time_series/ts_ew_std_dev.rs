use super::utils;

pub fn ts_ew_std_dev(
    values: &[Option<f64>],
    window: usize,
    min_periods: usize,
    half_life: f64,
) -> Vec<Option<f64>> {
    let mut output = vec![None; values.len()];
    if window == 0 || min_periods == 0 || min_periods > window || half_life <= 0.0 {
        return output;
    }
    let decay = 0.5_f64.powf(1.0 / half_life);
    let expired_weight = decay.powi(window as i32);

    let mut valid_count = 0usize;
    let mut sum_w = 0.0;
    let mut sum_x = 0.0;
    let mut sum_xx = 0.0;

    for idx in 0..values.len() {
        sum_w *= decay;
        sum_x *= decay;
        sum_xx *= decay;

        if let Some(value) = utils::valid_value(values[idx]) {
            valid_count += 1;
            sum_w += 1.0;
            sum_x += value;
            sum_xx += value * value;
        }

        if idx >= window {
            if let Some(value) = utils::valid_value(values[idx - window]) {
                valid_count -= 1;
                sum_w -= expired_weight;
                sum_x -= expired_weight * value;
                sum_xx -= expired_weight * value * value;
            }
        }

        if valid_count >= min_periods && sum_w > f64::EPSILON {
            let variance = ((sum_xx - sum_x * sum_x / sum_w) / sum_w).max(0.0);
            output[idx] = Some(variance.sqrt());
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use super::ts_ew_std_dev;

    #[test]
    fn ew_std_dev_requires_full_valid_window() {
        let values = vec![Some(1.0), Some(2.0), None, Some(4.0)];
        let output = ts_ew_std_dev(&values, 3, 3, 2.0);

        assert_eq!(output[0], None);
        assert_eq!(output[1], None);
        assert_eq!(output[2], None);
        assert_eq!(output[3], None);
    }
}

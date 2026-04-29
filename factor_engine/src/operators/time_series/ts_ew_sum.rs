use super::utils;

pub fn ts_ew_sum(
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
    let mut weighted_sum = 0.0;

    for idx in 0..values.len() {
        weighted_sum *= decay;

        if let Some(value) = utils::valid_value(values[idx]) {
            valid_count += 1;
            weighted_sum += value;
        }

        if idx >= window {
            if let Some(value) = utils::valid_value(values[idx - window]) {
                valid_count -= 1;
                weighted_sum -= expired_weight * value;
            }
        }

        if valid_count >= min_periods {
            output[idx] = Some(weighted_sum);
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use super::ts_ew_sum;

    #[test]
    fn ew_sum_weights_current_value_most() {
        let values = vec![Some(1.0), Some(2.0), Some(4.0)];
        let output = ts_ew_sum(&values, 3, 3, 1.0);

        assert_eq!(output[0], None);
        assert_eq!(output[1], None);
        assert!((output[2].unwrap() - 5.25).abs() < 1e-10);
    }

    #[test]
    fn ew_sum_requires_min_periods_valid_values() {
        let values = vec![Some(1.0), None, Some(4.0)];
        let output = ts_ew_sum(&values, 3, 3, 1.0);

        assert_eq!(output, vec![None, None, None]);
    }
}

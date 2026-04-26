use super::utils;

pub fn ts_mean(values: &[Option<f64>], window: usize, min_periods: usize) -> Vec<Option<f64>> {
    let mut output = vec![None; values.len()];
    let Some(effective_min_periods) = utils::effective_min_periods(window, min_periods) else {
        return output;
    };

    let mut sum = 0.0;
    let mut valid_count = 0usize;

    for idx in 0..values.len() {
        if let Some(value) = utils::valid_value(values[idx]) {
            sum += value;
            valid_count += 1;
        }

        if idx >= window {
            if let Some(value) = utils::valid_value(values[idx - window]) {
                sum -= value;
                valid_count -= 1;
            }
        }

        if valid_count >= effective_min_periods {
            output[idx] = Some(sum / valid_count as f64);
        }
    }

    output
}

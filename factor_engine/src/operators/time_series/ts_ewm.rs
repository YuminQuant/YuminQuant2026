use super::utils;

pub fn ts_ewm(values: &[Option<f64>], span: usize, min_periods: usize) -> Vec<Option<f64>> {
    let alpha = 2.0 / (span as f64 + 1.0);
    let mut output = vec![None; values.len()];
    let Some(effective_min_periods) = utils::effective_min_periods(span, min_periods) else {
        return output;
    };

    let mut state = None;
    let mut valid_count = 0usize;

    for (idx, value) in values.iter().enumerate() {
        let Some(value) = utils::valid_value(*value) else {
            continue;
        };

        let next = match state {
            Some(previous) => alpha * value + (1.0 - alpha) * previous,
            None => value,
        };
        state = Some(next);
        valid_count += 1;
        if valid_count >= effective_min_periods {
            output[idx] = Some(next);
        }
    }

    output
}

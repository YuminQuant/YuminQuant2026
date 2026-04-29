use super::utils;

pub fn ts_decay_linear(
    values: &[Option<f64>],
    window: usize,
    min_periods: usize,
) -> Vec<Option<f64>> {
    let mut output = vec![None; values.len()];
    let Some(effective_min_periods) = utils::effective_min_periods(window, min_periods) else {
        return output;
    };

    for idx in 0..values.len() {
        let start = idx.saturating_add(1).saturating_sub(window);
        let mut weighted_sum = 0.0;
        let mut weight_sum = 0.0;
        let mut valid_count = 0usize;

        for (position, source_idx) in (start..=idx).enumerate() {
            let Some(value) = utils::valid_value(values[source_idx]) else {
                continue;
            };
            let weight = position as f64 + 1.0;
            weighted_sum += value * weight;
            weight_sum += weight;
            valid_count += 1;
        }

        if valid_count >= effective_min_periods && weight_sum > f64::EPSILON {
            output[idx] = Some(weighted_sum / weight_sum);
        }
    }

    output
}

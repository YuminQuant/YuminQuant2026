/// A normalized exponential average over exactly the most recent `window`
/// bars. A full finite window is required, with no state from earlier bars.
pub fn rolling_ema(values: &[Option<f64>], window: usize) -> Vec<Option<f64>> {
    rolling_exponential_mean(values, window, 1.0 - 2.0 / (window as f64 + 1.0))
}

pub(crate) fn rolling_exponential_mean(
    values: &[Option<f64>],
    window: usize,
    decay: f64,
) -> Vec<Option<f64>> {
    let mut output = vec![None; values.len()];
    if window == 0 || window > values.len() {
        return output;
    }
    let mut weights = Vec::with_capacity(window);
    let mut weight = 1.0;
    for _ in 0..window {
        weights.push(weight);
        weight *= decay;
    }
    let total = weights.iter().sum::<f64>();
    for (out, sample) in output[window - 1..].iter_mut().zip(values.windows(window)) {
        *out = sample
            .iter()
            .rev()
            .zip(&weights)
            .try_fold(0.0, |sum, (&value, weight)| {
                Some(sum + value.filter(|value| value.is_finite())? * weight)
            })
            .map(|sum| sum / total)
            .filter(|value| value.is_finite());
    }
    output
}

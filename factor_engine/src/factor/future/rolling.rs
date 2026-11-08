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

pub(super) fn atr(
    close: &[Option<f64>],
    high: &[Option<f64>],
    low: &[Option<f64>],
    window: usize,
) -> Vec<Option<f64>> {
    let ranges = (0..close.len())
        .map(|idx| {
            let (high, low) = (high[idx]?, low[idx]?);
            let previous = idx.checked_sub(1).and_then(|i| close[i]);
            Some(previous.map_or(high - low, |previous| {
                (high - low)
                    .max((high - previous).abs())
                    .max((low - previous).abs())
            }))
        })
        .collect::<Vec<_>>();
    rolling_exponential_mean(&ranges, window, 1.0 - 1.0 / window as f64)
}

pub(super) fn with_atr(
    atr: Option<f64>,
    operation: impl FnOnce(f64) -> Option<f64>,
) -> Option<f64> {
    let atr = atr?;
    if atr == 0.0 {
        return Some(0.0);
    }
    operation(atr).filter(|value| value.is_finite())
}

//! Finite, causal scoring primitives for soft participation and event weights.

pub(super) fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

pub(super) fn softplus(x: f64) -> f64 {
    x.max(0.0) + (-x.abs()).exp().ln_1p()
}

/// A relative floor supplied by the observation, never an absolute unit-dependent epsilon.
/// Complete flat data is neutral; an observation after a flat baseline remains finite.
pub(super) fn scaled(value: f64, baseline: f64) -> Option<f64> {
    if !value.is_finite() || !baseline.is_finite() || baseline < 0.0 {
        return None;
    }
    let scale = baseline.hypot(0.1 * value);
    Some(if scale == 0.0 { 0.0 } else { value / scale })
}

/// Smooth relative expansion: 1/2 at equal scales, zero for a completely flat pair.
pub(super) fn expansion(current: f64, baseline: f64) -> Option<f64> {
    if current < 0.0 || baseline < 0.0 || !current.is_finite() || !baseline.is_finite() {
        return None;
    }
    let total = current + baseline;
    Some(if total == 0.0 {
        0.0
    } else {
        sigmoid(4.0 * (current - baseline) / total)
    })
}

/// Retain the latest single event inside a finite horizon, with age in trading bars.
/// New events replace the score; absent events use the continuous background score.
/// The linear taper makes the exponential memory vanish at the horizon boundary.
pub(super) fn event_decay(
    scores: &[Option<f64>],
    events: &[Option<bool>],
    horizon: usize,
) -> Vec<Option<f64>> {
    (0..scores.len())
        .map(|t| {
            let background = scores[t]?;
            let start = (t + 1).checked_sub(horizon)?;
            for i in (start..=t).rev() {
                if events[i]? {
                    let age = (t - i) as f64 / horizon as f64;
                    return Some(scores[i]? * (-3.0 * age).exp() * (1.0 - age));
                }
            }
            Some(background)
        })
        .collect()
}

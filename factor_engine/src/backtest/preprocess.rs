use std::collections::BTreeMap;

use crate::backtest::request::NeutralizeSpec;
use crate::operators::{cs_neutralize_regression, cs_pctrank};

#[derive(Clone, Copy, Debug, Default)]
pub struct CoverageStats {
    pub coverage: f64,
    pub inf_rate: f64,
    pub finite_count: usize,
}

pub fn coverage_stats(values: &[Option<f64>]) -> CoverageStats {
    coverage_stats_with_universe(values, None)
}

pub fn coverage_stats_with_universe(
    values: &[Option<f64>],
    universe: Option<&[bool]>,
) -> CoverageStats {
    if values.is_empty() {
        return CoverageStats::default();
    }
    let mut denominator = 0usize;
    let mut present = 0usize;
    let mut inf = 0usize;
    let mut finite = 0usize;
    for (idx, value) in values.iter().enumerate() {
        if universe.is_some_and(|universe| !universe.get(idx).copied().unwrap_or(false)) {
            continue;
        }
        denominator += 1;
        if value.is_some() {
            present += 1;
        }
        if value.is_some_and(|value| value.is_infinite()) {
            inf += 1;
        }
        if value.is_some_and(f64::is_finite) {
            finite += 1;
        }
    }
    if denominator == 0 {
        return CoverageStats::default();
    }
    CoverageStats {
        coverage: present as f64 / denominator as f64,
        inf_rate: inf as f64 / denominator as f64,
        finite_count: finite,
    }
}

pub fn maybe_neutralize(
    values: &[Option<f64>],
    spec: &NeutralizeSpec,
    barra: &[Vec<Option<f64>>],
    groups: Option<&[Option<String>]>,
) -> Vec<Option<f64>> {
    match spec {
        NeutralizeSpec::None => values.to_vec(),
        NeutralizeSpec::Sector => {
            let clean_y = finite_only(values);
            cs_neutralize_regression(&clean_y, &[], groups, None)
        }
        NeutralizeSpec::Barra { .. } => {
            let clean_y = finite_only(values);
            let clean_barra = barra
                .iter()
                .map(|column| finite_only(column))
                .collect::<Vec<_>>();
            let refs = clean_barra.iter().map(Vec::as_slice).collect::<Vec<_>>();
            cs_neutralize_regression(&clean_y, &refs, groups, None)
        }
    }
}

pub fn portfolio_scores(values: &[Option<f64>]) -> Vec<Option<f64>> {
    let ranks = cs_pctrank(values, true);
    zscore(&ranks)
}

pub fn group_assignments(values: &[Option<f64>], group_count: usize) -> Vec<Option<usize>> {
    let mut pairs = values
        .iter()
        .enumerate()
        .filter_map(|(idx, value)| {
            value
                .filter(|value| value.is_finite())
                .map(|value| (idx, value))
        })
        .collect::<Vec<_>>();
    if pairs.is_empty() || group_count == 0 {
        return vec![None; values.len()];
    }
    pairs.sort_by(|left, right| {
        left.1
            .total_cmp(&right.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    let count = pairs.len();
    let mut output = vec![None; values.len()];
    for (rank, (idx, _)) in pairs.into_iter().enumerate() {
        let group = (rank * group_count / count).min(group_count - 1);
        output[idx] = Some(group);
    }
    output
}

pub fn long_short_weights(scores: &[Option<f64>]) -> Vec<f64> {
    let mut weights = vec![0.0; scores.len()];
    let denom = scores
        .iter()
        .filter_map(|value| value.filter(|value| value.is_finite()))
        .map(f64::abs)
        .sum::<f64>();
    if denom.abs() <= f64::EPSILON {
        return weights;
    }
    for (idx, value) in scores.iter().enumerate() {
        if let Some(value) = value.filter(|value| value.is_finite()) {
            weights[idx] = value / denom;
        }
    }
    weights
}

pub fn equal_group_weights(assignments: &[Option<usize>], group_count: usize) -> Vec<Vec<f64>> {
    let mut weights = vec![vec![0.0; assignments.len()]; group_count];
    let mut counts = vec![0usize; group_count];
    for group in assignments.iter().flatten() {
        if *group < group_count {
            counts[*group] += 1;
        }
    }
    for (idx, group) in assignments.iter().enumerate() {
        let Some(group) = group else {
            continue;
        };
        if *group < group_count && counts[*group] > 0 {
            weights[*group][idx] = 1.0 / counts[*group] as f64;
        }
    }
    weights
}

pub fn turnover(previous: Option<&[f64]>, current: &[f64]) -> Option<f64> {
    let previous = previous?;
    if previous.len() != current.len() {
        return None;
    }
    Some(
        previous
            .iter()
            .zip(current)
            .map(|(left, right)| (right - left).abs())
            .sum::<f64>()
            * 0.5,
    )
}

pub fn portfolio_return(weights: &[f64], returns: &[Option<f64>]) -> Option<f64> {
    if weights.len() != returns.len() {
        return None;
    }
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    for (weight, value) in weights.iter().zip(returns) {
        let Some(value) = value.filter(|value| value.is_finite()) else {
            continue;
        };
        if weight.abs() <= f64::EPSILON {
            continue;
        }
        numerator += weight * value;
        denominator += weight.abs();
    }
    (denominator > f64::EPSILON).then_some(numerator / denominator)
}

pub fn zscore(values: &[Option<f64>]) -> Vec<Option<f64>> {
    let finite = values
        .iter()
        .filter_map(|value| value.filter(|value| value.is_finite()))
        .collect::<Vec<_>>();
    if finite.is_empty() {
        return vec![None; values.len()];
    }
    let mean = finite.iter().sum::<f64>() / finite.len() as f64;
    let variance = finite
        .iter()
        .map(|value| {
            let diff = value - mean;
            diff * diff
        })
        .sum::<f64>()
        / finite.len() as f64;
    let std = variance.sqrt();
    if std.abs() <= f64::EPSILON {
        return vec![None; values.len()];
    }
    values
        .iter()
        .map(|value| {
            value
                .filter(|value| value.is_finite())
                .map(|value| (value - mean) / std)
        })
        .collect()
}

pub fn percentile_ranks(values: &[Option<f64>]) -> Vec<Option<f64>> {
    cs_pctrank(values, true)
}

pub fn finite_only(values: &[Option<f64>]) -> Vec<Option<f64>> {
    values
        .iter()
        .map(|value| value.filter(|value| value.is_finite()))
        .collect()
}

pub fn groups_by_code(
    ts_codes: &[String],
    provider: impl Fn(&str) -> Option<String>,
) -> Vec<Option<String>> {
    ts_codes.iter().map(|code| provider(code)).collect()
}

pub fn keyed_values(keys: &[String], values: &[f64]) -> BTreeMap<String, f64> {
    keys.iter().cloned().zip(values.iter().copied()).collect()
}

#[cfg(test)]
mod tests {
    use super::{
        coverage_stats, coverage_stats_with_universe, equal_group_weights, group_assignments,
        long_short_weights, portfolio_return, portfolio_scores,
    };

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1e-10,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn coverage_counts_inf_separately_from_missing() {
        let stats = coverage_stats(&[Some(1.0), Some(f64::INFINITY), None, Some(f64::NAN)]);

        assert_close(stats.coverage, 0.75);
        assert_close(stats.inf_rate, 0.25);
        assert_eq!(stats.finite_count, 1);
    }

    #[test]
    fn coverage_can_use_column_universe_mask() {
        let values = [Some(1.0), None, Some(2.0), None];
        let universe = [true, true, false, false];
        let stats = coverage_stats_with_universe(&values, Some(&universe));

        assert_close(stats.coverage, 0.5);
        assert_close(stats.inf_rate, 0.0);
        assert_eq!(stats.finite_count, 1);
    }

    #[test]
    fn long_short_weights_use_rank_zscore_and_abs_normalization() {
        let scores = portfolio_scores(&[Some(1.0), Some(2.0), Some(3.0)]);
        let weights = long_short_weights(&scores);

        assert_close(weights.iter().map(|value| value.abs()).sum::<f64>(), 1.0);
        assert!(weights[0] < 0.0);
        assert_close(weights[1], 0.0);
        assert!(weights[2] > 0.0);
    }

    #[test]
    fn group_weights_are_equal_inside_each_group() {
        let assignments = group_assignments(&[Some(1.0), Some(2.0), Some(3.0), Some(4.0)], 2);
        let weights = equal_group_weights(&assignments, 2);

        assert_eq!(assignments, vec![Some(0), Some(0), Some(1), Some(1)]);
        assert_close(weights[0][0], 0.5);
        assert_close(weights[0][1], 0.5);
        assert_close(weights[1][2], 0.5);
        assert_close(weights[1][3], 0.5);
    }

    #[test]
    fn portfolio_return_renormalizes_available_weights() {
        let value = portfolio_return(&[0.5, 0.5], &[Some(0.02), None]).unwrap();

        assert_close(value, 0.02);
    }
}

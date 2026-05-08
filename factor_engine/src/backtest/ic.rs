use crate::backtest::preprocess::{coverage_stats, percentile_ranks};

#[derive(Clone, Debug)]
pub struct IcObservation {
    pub factor_id: String,
    pub factor_date: i32,
    pub label_date: i32,
    pub settle_date: Option<i32>,
    pub horizon: Option<usize>,
    pub ic: Option<f64>,
    pub rank_ic: Option<f64>,
    pub pair_count: usize,
    pub coverage: f64,
    pub inf_rate: f64,
}

pub fn compute_ic(
    factor: &[Option<f64>],
    label: &[Option<f64>],
) -> (Option<f64>, Option<f64>, usize) {
    let pair_count = factor
        .iter()
        .zip(label)
        .filter(|(x, y)| x.is_some_and(f64::is_finite) && y.is_some_and(f64::is_finite))
        .count();
    let pearson = pearson_corr(factor, label);
    let factor_rank = percentile_ranks(factor);
    let label_rank = percentile_ranks(label);
    let rank = pearson_corr(&factor_rank, &label_rank);
    (pearson, rank, pair_count)
}

pub fn daily_ic_observation(
    factor_id: &str,
    factor_date: i32,
    label_date: i32,
    settle_date: Option<i32>,
    horizon: Option<usize>,
    factor: &[Option<f64>],
    label: &[Option<f64>],
) -> IcObservation {
    let stats = coverage_stats(factor);
    let (ic, rank_ic, pair_count) = compute_ic(factor, label);
    IcObservation {
        factor_id: factor_id.to_string(),
        factor_date,
        label_date,
        settle_date,
        horizon,
        ic,
        rank_ic,
        pair_count,
        coverage: stats.coverage,
        inf_rate: stats.inf_rate,
    }
}

fn pearson_corr(x: &[Option<f64>], y: &[Option<f64>]) -> Option<f64> {
    if x.len() != y.len() {
        return None;
    }
    let pairs = x
        .iter()
        .zip(y)
        .filter_map(
            |(x, y)| match (x.filter(|v| v.is_finite()), y.filter(|v| v.is_finite())) {
                (Some(x), Some(y)) => Some((x, y)),
                _ => None,
            },
        )
        .collect::<Vec<_>>();
    if pairs.len() < 2 {
        return None;
    }
    let mean_x = pairs.iter().map(|(x, _)| *x).sum::<f64>() / pairs.len() as f64;
    let mean_y = pairs.iter().map(|(_, y)| *y).sum::<f64>() / pairs.len() as f64;
    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;
    for (x, y) in pairs {
        let dx = x - mean_x;
        let dy = y - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }
    if var_x <= f64::EPSILON || var_y <= f64::EPSILON {
        return None;
    }
    Some(cov / (var_x.sqrt() * var_y.sqrt()))
}

#[cfg(test)]
mod tests {
    use super::compute_ic;

    #[test]
    fn rank_ic_can_use_infinite_factor_after_ranking() {
        let factor = vec![Some(1.0), Some(f64::INFINITY), Some(2.0)];
        let label = vec![Some(1.0), Some(3.0), Some(2.0)];
        let (ic, rank_ic, pair_count) = compute_ic(&factor, &label);

        assert!(ic.unwrap() > 0.99);
        assert!(rank_ic.unwrap() > 0.99);
        assert_eq!(pair_count, 2);
    }
}

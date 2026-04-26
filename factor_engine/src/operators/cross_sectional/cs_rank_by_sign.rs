use super::cs_utils;

pub fn cs_rank_by_sign(values: &[Option<f64>]) -> Vec<Option<f64>> {
    let negative = values
        .iter()
        .map(|value| cs_utils::clean(*value).filter(|value| *value < 0.0))
        .collect::<Vec<_>>();
    let positive = values
        .iter()
        .map(|value| cs_utils::clean(*value).filter(|value| *value > 0.0))
        .collect::<Vec<_>>();
    merge_ranks(
        &cs_utils::pctranks(&negative, true),
        &cs_utils::pctranks(&positive, true),
    )
}

fn merge_ranks(left: &[Option<f64>], right: &[Option<f64>]) -> Vec<Option<f64>> {
    left.iter()
        .zip(right)
        .map(|(left, right)| left.or(*right))
        .collect()
}

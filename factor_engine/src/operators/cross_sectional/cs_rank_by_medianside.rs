use super::cs_utils;

pub fn cs_rank_by_medianside(values: &[Option<f64>]) -> Vec<Option<f64>> {
    let Some(median) = cs_utils::median(values) else {
        return vec![None; values.len()];
    };
    let below = values
        .iter()
        .map(|value| cs_utils::clean(*value).filter(|value| *value < median))
        .collect::<Vec<_>>();
    let above = values
        .iter()
        .map(|value| cs_utils::clean(*value).filter(|value| *value > median))
        .collect::<Vec<_>>();
    cs_utils::pctranks(&below, true)
        .into_iter()
        .zip(cs_utils::pctranks(&above, true))
        .map(|(below, above)| below.or(above))
        .collect()
}

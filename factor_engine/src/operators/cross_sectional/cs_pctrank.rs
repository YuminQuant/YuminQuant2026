use super::cs_utils;

pub fn cs_pctrank(values: &[Option<f64>], ascending: bool) -> Vec<Option<f64>> {
    cs_utils::pctranks(values, ascending)
}

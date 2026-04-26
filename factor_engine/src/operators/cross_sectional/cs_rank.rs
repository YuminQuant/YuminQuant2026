use super::cs_utils;

pub fn cs_rank(values: &[Option<f64>], ascending: bool) -> Vec<Option<f64>> {
    cs_utils::ranks(values, ascending)
}

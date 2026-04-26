use super::cs_demean_by_group::cs_demean_by_group;

pub fn cs_neutralize(values: &[Option<f64>], groups: &[Option<String>]) -> Vec<Option<f64>> {
    cs_demean_by_group(values, groups)
}

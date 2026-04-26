use super::cs_pctrank::cs_pctrank;
use super::cs_utils;

pub fn cs_rank_y_add_x(y: &[Option<f64>], x: &[Option<f64>]) -> Vec<Option<f64>> {
    cs_utils::map_binary(&cs_pctrank(y, true), &cs_pctrank(x, true), |y, x| {
        Some(y + x)
    })
}

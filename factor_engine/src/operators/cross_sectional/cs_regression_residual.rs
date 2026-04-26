use super::cs_utils;

pub fn cs_regression_residual(y: &[Option<f64>], x: &[Option<f64>]) -> Vec<Option<f64>> {
    cs_utils::regression_residual(y, x)
}

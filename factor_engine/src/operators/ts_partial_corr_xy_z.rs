use super::utils;

pub fn ts_partial_corr_xy_z(
    x: &[Option<f64>],
    y: &[Option<f64>],
    z: &[Option<f64>],
    window: usize,
    min_periods: usize,
) -> Vec<Option<f64>> {
    utils::rolling_triple(
        x,
        y,
        z,
        window,
        min_periods,
        |x_values, y_values, z_values| {
            let corr_xy = utils::pearson_corr(x_values, y_values)?;
            let corr_xz = utils::pearson_corr(x_values, z_values)?;
            let corr_yz = utils::pearson_corr(y_values, z_values)?;
            let denominator = ((1.0 - corr_xz * corr_xz) * (1.0 - corr_yz * corr_yz)).sqrt();
            (denominator > f64::EPSILON).then_some((corr_xy - corr_xz * corr_yz) / denominator)
        },
    )
}

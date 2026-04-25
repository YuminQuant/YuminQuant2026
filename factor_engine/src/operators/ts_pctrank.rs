use super::utils;

pub fn ts_pctrank(values: &[Option<f64>], window: usize, min_periods: usize) -> Vec<Option<f64>> {
    utils::rolling_unary(values, window, min_periods, |window_values| {
        let rank = utils::ranks(window_values).last().copied()?;
        if window_values.len() == 1 {
            Some(1.0)
        } else {
            Some((rank - 1.0) / (window_values.len() - 1) as f64)
        }
    })
}

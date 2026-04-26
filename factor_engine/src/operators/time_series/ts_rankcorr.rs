use super::utils;

pub fn ts_rankcorr(
    left: &[Option<f64>],
    right: &[Option<f64>],
    window: usize,
    min_periods: usize,
) -> Vec<Option<f64>> {
    utils::rolling_binary(
        left,
        right,
        window,
        min_periods,
        |left_values, right_values| {
            let left_ranks = utils::ranks(left_values);
            let right_ranks = utils::ranks(right_values);
            utils::pearson_corr(&left_ranks, &right_ranks)
        },
    )
}

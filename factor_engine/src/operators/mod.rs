mod utils;

pub mod ts_argmax;
pub mod ts_argmin;
pub mod ts_central_moment;
pub mod ts_co_kurt_yxxx;
pub mod ts_co_kurt_yyxx;
pub mod ts_corr;
pub mod ts_coskew_yxx;
pub mod ts_covariance;
pub mod ts_cv;
pub mod ts_delay;
pub mod ts_diff;
pub mod ts_diff_avg;
pub mod ts_diff_max;
pub mod ts_diff_median;
pub mod ts_diff_min;
pub mod ts_entropy;
pub mod ts_ewm;
pub mod ts_ir;
pub mod ts_kurt;
pub mod ts_max;
pub mod ts_mean;
pub mod ts_median;
pub mod ts_min;
pub mod ts_minmax_cps;
pub mod ts_minmax_diff;
pub mod ts_minmax_scale;
pub mod ts_partial_corr_xy_z;
pub mod ts_pctchg;
pub mod ts_pctchg_abs;
pub mod ts_pctrank;
pub mod ts_prod;
pub mod ts_quantile;
pub mod ts_rank;
pub mod ts_rankcorr;
pub mod ts_regression;
pub mod ts_skew;
pub mod ts_std_dev;
pub mod ts_sum;
pub mod ts_theilsen;
pub mod ts_variance;
pub mod ts_zscore;

pub use ts_argmax::ts_argmax;
pub use ts_argmin::ts_argmin;
pub use ts_central_moment::ts_central_moment;
pub use ts_co_kurt_yxxx::ts_co_kurt_yxxx;
pub use ts_co_kurt_yyxx::ts_co_kurt_yyxx;
pub use ts_corr::ts_corr;
pub use ts_coskew_yxx::ts_coskew_yxx;
pub use ts_covariance::ts_covariance;
pub use ts_cv::ts_cv;
pub use ts_delay::ts_delay;
pub use ts_diff::ts_diff;
pub use ts_diff_avg::ts_diff_avg;
pub use ts_diff_max::ts_diff_max;
pub use ts_diff_median::ts_diff_median;
pub use ts_diff_min::ts_diff_min;
pub use ts_entropy::ts_entropy;
pub use ts_ewm::ts_ewm;
pub use ts_ir::ts_ir;
pub use ts_kurt::ts_kurt;
pub use ts_max::ts_max;
pub use ts_mean::ts_mean;
pub use ts_median::ts_median;
pub use ts_min::ts_min;
pub use ts_minmax_cps::ts_minmax_cps;
pub use ts_minmax_diff::ts_minmax_diff;
pub use ts_minmax_scale::ts_minmax_scale;
pub use ts_partial_corr_xy_z::ts_partial_corr_xy_z;
pub use ts_pctchg::{ts_pct_chg, ts_pctchg};
pub use ts_pctchg_abs::ts_pctchg_abs;
pub use ts_pctrank::ts_pctrank;
pub use ts_prod::ts_prod;
pub use ts_quantile::ts_quantile;
pub use ts_rank::ts_rank;
pub use ts_rankcorr::ts_rankcorr;
pub use ts_regression::ts_regression;
pub use ts_skew::ts_skew;
pub use ts_std_dev::ts_std_dev;
pub use ts_sum::ts_sum;
pub use ts_theilsen::ts_theilsen;
pub use ts_variance::ts_variance;
pub use ts_zscore::ts_zscore;

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_option_close(actual: Option<f64>, expected: Option<f64>) {
        match (actual, expected) {
            (Some(actual), Some(expected)) => assert!((actual - expected).abs() < 1e-10),
            (None, None) => {}
            _ => panic!("expected {:?}, got {:?}", expected, actual),
        }
    }

    #[test]
    fn rolling_mean_skips_nan_with_min_periods() {
        let values = vec![Some(1.0), Some(f64::NAN), Some(3.0)];
        let output = ts_mean(&values, 3, 2);

        assert_eq!(output[0], None);
        assert_eq!(output[1], None);
        assert_option_close(output[2], Some(2.0));
    }

    #[test]
    fn rolling_outputs_none_for_all_missing_windows() {
        let values = vec![None, Some(f64::NAN), None];
        let output = ts_mean(&values, 2, 1);

        assert_eq!(output, vec![None, None, None]);
    }

    #[test]
    fn min_periods_window_matches_strict_old_semantics() {
        let values = vec![Some(1.0), None, Some(3.0)];
        let output = ts_mean(&values, 3, 3);

        assert_eq!(output, vec![None, None, None]);
    }

    #[test]
    fn min_periods_larger_than_window_outputs_none() {
        let values = vec![Some(1.0), Some(2.0), Some(3.0)];
        let output = ts_sum(&values, 2, 3);

        assert_eq!(output, vec![None, None, None]);
    }

    #[test]
    fn rolling_binary_uses_pairwise_non_missing_values() {
        let x = vec![Some(1.0), Some(2.0), Some(f64::NAN), Some(4.0)];
        let y = vec![Some(2.0), Some(4.0), Some(100.0), Some(8.0)];

        let corr = ts_corr(&x, &y, 4, 3);
        let regression = ts_regression(&y, &x, 4, 3);

        assert_eq!(corr[0], None);
        assert_eq!(corr[1], None);
        assert_eq!(corr[2], None);
        assert_option_close(corr[3], Some(1.0));
        assert_option_close(regression[3], Some(2.0));
    }

    #[test]
    fn monotonic_extreme_operators_skip_missing_and_keep_earliest_tie() {
        let values = vec![
            Some(2.0),
            Some(f64::NAN),
            Some(3.0),
            Some(3.0),
            Some(1.0),
            None,
            Some(4.0),
        ];

        assert_eq!(
            ts_max(&values, 3, 1),
            vec![
                Some(2.0),
                Some(2.0),
                Some(3.0),
                Some(3.0),
                Some(3.0),
                Some(3.0),
                Some(4.0),
            ]
        );
        assert_eq!(
            ts_argmax(&values, 3, 1),
            vec![
                Some(0.0),
                Some(0.0),
                Some(2.0),
                Some(1.0),
                Some(0.0),
                Some(0.0),
                Some(2.0),
            ]
        );
        assert_eq!(
            ts_min(&values, 3, 1),
            vec![
                Some(2.0),
                Some(2.0),
                Some(2.0),
                Some(3.0),
                Some(1.0),
                Some(1.0),
                Some(1.0),
            ]
        );
        assert_eq!(
            ts_argmin(&values, 3, 1),
            vec![
                Some(0.0),
                Some(0.0),
                Some(0.0),
                Some(1.0),
                Some(2.0),
                Some(1.0),
                Some(0.0),
            ]
        );
    }
}

pub mod cross_sectional;
pub mod time_series;

pub use cross_sectional::*;
pub use time_series::*;

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

    #[test]
    fn cross_section_rank_is_nan_aware_and_uses_stable_ties() {
        let values = vec![Some(2.0), None, Some(f64::NAN), Some(1.0), Some(2.0)];

        assert_eq!(
            cs_rank(&values, true),
            vec![Some(2.0), None, None, Some(1.0), Some(3.0)]
        );
        assert_eq!(
            cs_pctrank(&values, true),
            vec![Some(0.5), None, None, Some(0.0), Some(1.0)]
        );
        assert_eq!(cs_pctrank(&[Some(1.0)], true), vec![None]);
    }

    #[test]
    fn cross_section_stats_skip_missing_values() {
        let values = vec![Some(1.0), Some(f64::NAN), Some(3.0)];

        assert_eq!(cs_demean(&values), vec![Some(-1.0), None, Some(1.0)]);
        assert_eq!(cs_zscore(&values), vec![Some(-1.0), None, Some(1.0)]);
        assert_eq!(cs_minmax_scale(&values), vec![Some(0.0), None, Some(1.0)]);
        assert_eq!(cs_scale(&values), vec![Some(0.25), None, Some(0.75)]);
        assert_eq!(cs_scale(&[Some(0.0), Some(0.0)]), vec![None, None]);
        assert_eq!(cs_mean(&values), vec![Some(2.0), Some(2.0), Some(2.0)]);
        assert_option_close(cs_winsorize95(&values)[2], Some(2.9));
    }

    #[test]
    fn decay_linear_gives_largest_weight_to_current_value() {
        let values = vec![Some(1.0), Some(2.0), Some(3.0)];
        let output = ts_decay_linear(&values, 3, 3);

        assert_eq!(output[0], None);
        assert_eq!(output[1], None);
        assert_option_close(output[2], Some((1.0 + 4.0 + 9.0) / 6.0));
    }

    #[test]
    fn cross_section_group_operators_work_with_group_labels() {
        let values = vec![Some(1.0), Some(2.0), Some(10.0), None];
        let groups = vec![
            Some("bank".to_string()),
            Some("bank".to_string()),
            Some("tech".to_string()),
            Some("tech".to_string()),
        ];

        assert_eq!(
            cs_neutralize(&values, &groups),
            vec![Some(-0.5), Some(0.5), Some(0.0), None]
        );
        assert_eq!(
            cs_fill_mean_by_group(&values, &groups),
            vec![Some(1.0), Some(2.0), Some(10.0), Some(10.0)]
        );
        assert_eq!(
            cs_pctrank_by_group(&values, &groups),
            vec![Some(0.0), Some(1.0), None, None]
        );
    }

    #[test]
    fn cross_section_regression_residual_uses_pairwise_valid_rows() {
        let y = vec![Some(2.0), Some(4.0), Some(f64::NAN), Some(8.0)];
        let x = vec![Some(1.0), Some(2.0), Some(3.0), Some(4.0)];
        let residual = cs_regression_residual(&y, &x);

        assert_option_close(residual[0], Some(0.0));
        assert_option_close(residual[1], Some(0.0));
        assert_eq!(residual[2], None);
        assert_option_close(residual[3], Some(0.0));
    }
}

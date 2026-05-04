use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::vector::clean;
use crate::factor::Factor;
use crate::operators::{cs_pctrank, ts_mean, ts_std_dev};

const VERSION: &str = "0.2.0";
const WINDOW: usize = 20;
const MIN_PERIODS: usize = 1;

pub struct StockDailyUtr;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyUtr)
}

impl Factor for StockDailyUtr {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "utr".to_string(),
            aliases: vec!["UTR".to_string()],
            name: "UTR".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: [
                "price_volume",
                "turnover",
                "stability",
                "rank",
                "neutralize",
                "barra",
                "size",
                "daily",
                "DWZQ",
            ]
            .iter()
            .map(|value| value.to_string())
            .collect(),
            description: "U-Turnover Rate factor combining SIZE-neutralized 20-day turnover and turnover stability ranks.".to_string(),
            dependencies: vec![
                DataRequest::new(DatasetId::StockDailyBasic, &["turnover_rate_f"]),
                DataRequest::new(DatasetId::StockBarraDaily, &["SIZE"]),
            ],
            intraday_raw_dependencies: Vec::new(),
            lookback: Lookback {
                trading_days: WINDOW - 1,
            },
        }
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let panel = data.daily_panel(DatasetId::StockDailyBasic)?;
        let turnover = panel.column("turnover_rate_f")?;
        let size = panel.column_from_table(data.daily(DatasetId::StockBarraDaily)?, "SIZE")?;

        let turn20 = turnover
            .ts(|values| ts_mean(values, WINDOW, MIN_PERIODS))?
            .cs_neutralize_regression(&[&size], None)?;
        let str = turnover
            .ts(|values| ts_std_dev(values, WINDOW, MIN_PERIODS))?
            .cs_neutralize_regression(&[&size], None)?;
        let factor = turn20.cs_binary(&str, utr_score)?;
        Ok(factor.to_factor_series(self.spec()))
    }
}

fn utr_score(turn20: &[Option<f64>], str: &[Option<f64>]) -> Vec<Option<f64>> {
    let stability_score = cs_pctrank(str, true);
    let stable_indices = valid_partition_indices(turn20, &stability_score, |score| score <= 0.5);
    let unstable_indices = valid_partition_indices(turn20, &stability_score, |score| score > 0.5);
    let stable_turnover_score = subset_pctrank(turn20, &stable_indices, false);
    let unstable_turnover_score = subset_pctrank(turn20, &unstable_indices, true);

    let mut output = vec![None; turn20.len()];
    for idx in stable_indices {
        output[idx] = add_scores(stability_score[idx], stable_turnover_score[idx]);
    }
    for idx in unstable_indices {
        output[idx] = add_scores(stability_score[idx], unstable_turnover_score[idx]);
    }
    output
}

fn valid_partition_indices<F>(
    turn20: &[Option<f64>],
    stability_score: &[Option<f64>],
    predicate: F,
) -> Vec<usize>
where
    F: Fn(f64) -> bool,
{
    turn20
        .iter()
        .zip(stability_score)
        .enumerate()
        .filter_map(|(idx, (turn20, score))| {
            let _ = clean(*turn20)?;
            let score = clean(*score)?;
            predicate(score).then_some(idx)
        })
        .collect()
}

fn subset_pctrank(values: &[Option<f64>], indices: &[usize], ascending: bool) -> Vec<Option<f64>> {
    let subset = indices.iter().map(|idx| values[*idx]).collect::<Vec<_>>();
    let ranked = cs_pctrank(&subset, ascending);
    let mut output = vec![None; values.len()];
    for (subset_idx, idx) in indices.iter().enumerate() {
        output[*idx] = ranked[subset_idx];
    }
    output
}

fn add_scores(left: Option<f64>, right: Option<f64>) -> Option<f64> {
    match (clean(left), clean(right)) {
        (Some(left), Some(right)) => Some(left + right),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: Option<f64>, expected: Option<f64>) {
        match (actual, expected) {
            (Some(actual), Some(expected)) => assert!(
                (actual - expected).abs() < 1e-10,
                "expected {expected}, got {actual}"
            ),
            (None, None) => {}
            _ => panic!("expected {:?}, got {:?}", expected, actual),
        }
    }

    #[test]
    fn utr_scores_stable_half_with_descending_turnover_rank() {
        let turn20 = vec![Some(10.0), Some(20.0), Some(30.0), Some(40.0)];
        let str = vec![Some(1.0), Some(2.0), Some(3.0), Some(4.0)];

        let score = utr_score(&turn20, &str);

        assert_close(score[0], Some(0.0 + 1.0));
        assert_close(score[1], Some(1.0 / 3.0 + 0.0));
        assert_close(score[2], Some(2.0 / 3.0 + 0.0));
        assert_close(score[3], Some(1.0 + 1.0));
    }

    #[test]
    fn utr_keeps_missing_values_missing() {
        let turn20 = vec![Some(10.0), None, Some(30.0), Some(40.0)];
        let str = vec![Some(1.0), Some(2.0), None, Some(4.0)];

        let score = utr_score(&turn20, &str);

        assert!(score[0].is_none());
        assert!(score[1].is_none());
        assert!(score[2].is_none());
        assert!(score[3].is_none());
    }
}

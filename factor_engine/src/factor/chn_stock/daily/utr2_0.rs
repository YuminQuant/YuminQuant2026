use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::vector::clean;
use crate::factor::Factor;
use crate::operators::{ts_mean, ts_std_dev};

const VERSION: &str = "0.2.0";
const WINDOW: usize = 20;
const MIN_PERIODS: usize = 1;

pub struct StockDailyUtr2_0;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyUtr2_0)
}

impl Factor for StockDailyUtr2_0 {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "utr2_0".to_string(),
            aliases: vec!["UTR2.0".to_string(), "UTR2".to_string()],
            name: "UTR2.0".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: [
                "price_volume",
                "turnover",
                "stability",
                "softsign",
                "neutralize",
                "barra",
                "size",
                "daily",
                "DWZQ",
            ]
            .iter()
            .map(|value| value.to_string())
            .collect(),
            description: "New U-Turnover Rate factor using softsign(STR) as a dynamic weight on SIZE-neutralized Turn20.".to_string(),
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
        let turnover = panel
            .column("turnover_rate_f")?
            .map_values(percent_to_decimal);
        let size = panel.column_from_table(data.daily(DatasetId::StockBarraDaily)?, "SIZE")?;

        let turn20 = turnover
            .ts(|values| ts_mean(values, WINDOW, MIN_PERIODS))?
            .cs_neutralize_regression(&[&size], None)?;
        let str = turnover
            .ts(|values| ts_std_dev(values, WINDOW, MIN_PERIODS))?
            .cs_neutralize_regression(&[&size], None)?;
        let factor = str.zip_binary(&turn20, utr2_score)?;
        Ok(factor.to_factor_series(self.spec()))
    }
}

fn percent_to_decimal(value: Option<f64>) -> Option<f64> {
    clean(value).map(|value| value / 100.0)
}

fn softsign(value: f64) -> f64 {
    value / (1.0 + value.abs())
}

fn utr2_score(str: Option<f64>, turn20: Option<f64>) -> Option<f64> {
    match (clean(str), clean(turn20)) {
        (Some(str), Some(turn20)) => Some(str + softsign(str) * turn20),
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
    fn turnover_percent_is_converted_to_decimal_before_softsign_path() {
        assert_close(percent_to_decimal(Some(2.5)), Some(0.025));
    }

    #[test]
    fn softsign_keeps_sign_and_bounds_magnitude_below_one() {
        assert!((softsign(2.0) - 2.0 / 3.0).abs() < 1e-10);
        assert!((softsign(-2.0) + 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn utr2_score_matches_formula() {
        assert_close(
            utr2_score(Some(0.25), Some(0.4)),
            Some(0.25 + 0.25 / 1.25 * 0.4),
        );
        assert_eq!(utr2_score(Some(0.25), None), None);
    }
}

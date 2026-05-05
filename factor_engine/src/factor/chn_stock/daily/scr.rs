use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::vector::clean;
use crate::factor::Factor;
use crate::operators::{ts_delay, ts_std_dev};

const VERSION: &str = "0.4.0";
const CURRENT_WINDOW: usize = 20;
const BASE_WINDOW: usize = 40;
const BASE_DELAY: usize = 20;
const MIN_PERIODS: usize = 1;

pub struct StockDailyScr;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyScr)
}

impl Factor for StockDailyScr {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "scr".to_string(),
            aliases: vec!["SCR".to_string()],
            name: "SCR".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: [
                "price_volume",
                "turnover",
                "stability",
                "change_rate",
                "neutralize",
                "barra",
                "size",
                "daily",
                "DWZQ",
            ]
            .iter()
            .map(|value| value.to_string())
            .collect(),
            description: "The STR Change Rate factor, computed as current 20-day turnover volatility over previous 40-day turnover volatility minus one with relaxed missing-data windows, neutralized by SIZE.".to_string(),
            dependencies: vec![
                DataRequest::new(DatasetId::StockDailyBasic, &["turnover_rate_f"]),
                DataRequest::new(DatasetId::StockBarraDaily, &["SIZE"]),
            ],
            intraday_raw_dependencies: Vec::new(),
            lookback: Lookback {
                trading_days: BASE_WINDOW - 1 + BASE_DELAY,
            },
        }
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let panel = data.daily_panel(DatasetId::StockDailyBasic)?;
        let turnover = panel.column("turnover_rate_f")?;
        let size = panel.column_from_table(data.daily(DatasetId::StockBarraDaily)?, "SIZE")?;

        let current_volatility =
            turnover.ts(|values| ts_std_dev(values, CURRENT_WINDOW, MIN_PERIODS))?;
        let base_volatility = turnover.ts(previous_base_volatility)?;
        let raw = current_volatility.zip_binary(&base_volatility, volatility_change_rate)?;
        let factor = raw.cs_neutralize_regression(&[&size], None)?;
        Ok(factor.to_factor_series(self.spec()))
    }
}

fn previous_base_volatility(values: &[Option<f64>]) -> Vec<Option<f64>> {
    let base = ts_std_dev(values, BASE_WINDOW, MIN_PERIODS);
    ts_delay(&base, BASE_DELAY)
}

fn volatility_change_rate(current: Option<f64>, base: Option<f64>) -> Option<f64> {
    match (clean(current), clean(base)) {
        (Some(current), Some(base)) if base.abs() > f64::EPSILON => Some(current / base - 1.0),
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
    fn previous_base_volatility_uses_prior_forty_days() {
        let mut values = (1..=40).map(|value| Some(value as f64)).collect::<Vec<_>>();
        values.extend((0..20).map(|_| Some(100.0)));

        let base = previous_base_volatility(&values);
        let expected = ts_std_dev(&values[0..40], BASE_WINDOW, MIN_PERIODS)[39];

        assert_close(base[59], expected);
    }

    #[test]
    fn volatility_windows_skip_missing_values_with_min_periods_one() {
        let mut values = (1..=60).map(|value| Some(value as f64)).collect::<Vec<_>>();
        values[50] = None;

        let current = ts_std_dev(&values, CURRENT_WINDOW, MIN_PERIODS);
        let base = previous_base_volatility(&values);

        assert!(current[59].is_some());
        assert!(base[59].is_some());
    }

    #[test]
    fn volatility_change_rate_rejects_zero_base() {
        assert_eq!(volatility_change_rate(Some(1.0), Some(0.0)), None);
        assert_close(volatility_change_rate(Some(3.0), Some(2.0)), Some(0.5));
    }
}

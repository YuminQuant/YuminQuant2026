use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::vector::clean;
use crate::factor::Factor;
use crate::operators::{ts_delay, ts_mean};

const VERSION: &str = "0.4.0";
const BASE_WINDOW: usize = 40;
const BASE_DELAY: usize = 20;
const SIGNAL_WINDOW: usize = 20;
const MIN_PERIODS: usize = 1;

pub struct StockDailyPctTurn20;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyPctTurn20)
}

impl Factor for StockDailyPctTurn20 {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "pct_turn20".to_string(),
            aliases: vec!["PctTurn20".to_string()],
            name: "PctTurn20".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: [
                "price_volume",
                "turnover",
                "liquidity",
                "neutralize",
                "barra",
                "size",
                "daily",
            ]
            .iter()
            .map(|value| value.to_string())
            .collect(),
            description: "20-day mean turnover relative to a single preceding 40-day turnover baseline, neutralized by SIZE.".to_string(),
            dependencies: vec![
                DataRequest::new(DatasetId::StockDailyBasic, &["turnover_rate_f"]),
                DataRequest::new(DatasetId::StockBarraDaily, &["SIZE"]),
            ],
            intraday_raw_dependencies: Vec::new(),
            lookback: Lookback {
                trading_days: (BASE_WINDOW - 1) + BASE_DELAY,
            },
        }
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let panel = data.daily_panel(DatasetId::StockDailyBasic)?;
        let turnover = panel.column("turnover_rate_f")?;
        let size = panel.column_from_table(data.daily(DatasetId::StockBarraDaily)?, "SIZE")?;

        let signal_mean = turnover.ts(|values| ts_mean(values, SIGNAL_WINDOW, MIN_PERIODS))?;
        let base_mean =
            turnover.ts(|values| delayed_rolling_mean(values, BASE_WINDOW, BASE_DELAY))?;
        let raw = signal_mean.zip_binary(&base_mean, relative_change)?;
        let factor = raw.cs_neutralize_regression(&[&size], None)?;
        Ok(factor.to_factor_series(self.spec()))
    }
}

fn delayed_rolling_mean(values: &[Option<f64>], window: usize, delay: usize) -> Vec<Option<f64>> {
    let mean = ts_mean(values, window, MIN_PERIODS);
    ts_delay(&mean, delay)
}

fn relative_change(turnover: Option<f64>, base: Option<f64>) -> Option<f64> {
    match (clean(turnover), clean(base)) {
        (Some(turnover), Some(base)) if base.abs() > f64::EPSILON => Some(turnover / base - 1.0),
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
    fn delayed_rolling_mean_uses_min_periods_one_then_shift() {
        let values = (1..=65).map(|value| Some(value as f64)).collect::<Vec<_>>();
        let base = delayed_rolling_mean(&values, 40, 20);

        assert_eq!(base[19], None);
        assert_close(base[20], Some(1.0));
        assert_close(base[21], Some(1.5));
        assert_close(base[58], Some(20.0));
        assert_close(base[59], Some(20.5));
        assert_close(base[60], Some(21.5));
    }

    #[test]
    fn relative_change_rejects_zero_base() {
        assert_eq!(relative_change(Some(1.0), Some(0.0)), None);
        assert_close(relative_change(Some(12.0), Some(10.0)), Some(0.2));
    }

    #[test]
    fn pct_turn20_uses_signal_mean_over_single_prior_base() {
        let values = (1..=61).map(|value| Some(value as f64)).collect::<Vec<_>>();
        let signal = ts_mean(&values, 20, 1);
        let base = delayed_rolling_mean(&values, 40, 20);
        let raw = signal
            .iter()
            .zip(base.iter())
            .map(|(signal, base)| relative_change(*signal, *base))
            .collect::<Vec<_>>();

        assert_close(raw[58], Some(49.5 / 20.0 - 1.0));
        assert_close(raw[59], Some(50.5 / 20.5 - 1.0));
        assert_close(raw[60], Some(51.5 / 21.5 - 1.0));
    }

    #[test]
    fn pct_turn20_no_longer_averages_daily_relative_bases() {
        let values = (1..=61).map(|value| Some(value as f64)).collect::<Vec<_>>();
        let signal = ts_mean(&values, 20, 1);
        let base = delayed_rolling_mean(&values, 40, 20);
        let new_raw = relative_change(signal[60], base[60]).unwrap();

        let daily_relative = values
            .iter()
            .zip(base.iter())
            .map(|(current, base)| relative_change(*current, *base))
            .collect::<Vec<_>>();
        let old_raw = ts_mean(&daily_relative, 20, 1)[60].unwrap();

        assert!((new_raw - old_raw).abs() > 1e-3);
    }
}

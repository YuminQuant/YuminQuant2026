use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::vector::clean;
use crate::factor::common::PanelColumn;
use crate::factor::Factor;
use crate::operators::{cs_nonnegative, cs_scale, ts_delay, ts_mean, ts_std_dev};

const VERSION: &str = "0.4.0";
const BASE_WINDOW: usize = 40;
const BASE_DELAY: usize = 20;
const SIGNAL_WINDOW: usize = 20;
const MIN_PERIODS: usize = 1;

pub struct StockDailyPctTurn20Turbo;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyPctTurn20Turbo)
}

impl Factor for StockDailyPctTurn20Turbo {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "pct_turn20_turbo".to_string(),
            aliases: vec!["PctTurn20_Turbo".to_string()],
            name: "PctTurn20_Turbo".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: [
                "price_volume",
                "turnover",
                "stability",
                "change_rate",
                "turbo",
                "neutralize",
                "barra",
                "size",
                "daily",
                "DWZQ",
            ]
            .iter()
            .map(|value| value.to_string())
            .collect(),
            description: "Turbo PctTurn20 factor combining SIZE-neutralized 20-day mean turnover relative to a single prior 40-day baseline and direction-aligned GTR after non-negative scaling.".to_string(),
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
        let turnover = panel
            .column("turnover_rate_f")?
            .map_values(percent_to_decimal);
        let size = panel.column_from_table(data.daily(DatasetId::StockBarraDaily)?, "SIZE")?;

        let signal_mean = turnover.ts(|values| ts_mean(values, SIGNAL_WINDOW, MIN_PERIODS))?;
        let base_mean =
            turnover.ts(|values| delayed_rolling_mean(values, BASE_WINDOW, BASE_DELAY))?;
        let pct_turn20 = signal_mean
            .zip_binary(&base_mean, relative_change)?
            .cs_neutralize_regression(&[&size], None)?;
        let growth = turnover.ts(turnover_growth)?;
        let gtr = growth
            .ts(|values| ts_std_dev(values, SIGNAL_WINDOW, MIN_PERIODS))?
            .cs_neutralize_regression(&[&size], None)?
            .map_values(negate);
        let factor = average_pair(&turbo_scale(&pct_turn20)?, &turbo_scale(&gtr)?)?;
        Ok(factor.to_factor_series(self.spec()))
    }
}

fn percent_to_decimal(value: Option<f64>) -> Option<f64> {
    clean(value).map(|value| value / 100.0)
}

fn delayed_rolling_mean(values: &[Option<f64>], window: usize, delay: usize) -> Vec<Option<f64>> {
    let mean = ts_mean(values, window, MIN_PERIODS);
    ts_delay(&mean, delay)
}

fn relative_change(current: Option<f64>, base: Option<f64>) -> Option<f64> {
    match (clean(current), clean(base)) {
        (Some(current), Some(base)) if base.abs() > f64::EPSILON => Some(current / base - 1.0),
        _ => None,
    }
}

fn turnover_growth(values: &[Option<f64>]) -> Vec<Option<f64>> {
    let mut output = vec![None; values.len()];
    for idx in 1..values.len() {
        output[idx] = relative_change(values[idx], values[idx - 1]);
    }
    output
}

fn negate(value: Option<f64>) -> Option<f64> {
    clean(value).map(|value| -value)
}

fn turbo_scale(values: &PanelColumn) -> Result<PanelColumn> {
    values.cs(cs_nonnegative)?.cs(cs_scale)
}

fn average_pair(left: &PanelColumn, right: &PanelColumn) -> Result<PanelColumn> {
    left.zip_binary(right, |left, right| match (clean(left), clean(right)) {
        (Some(left), Some(right)) => Some((left + right) / 2.0),
        _ => None,
    })
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
    fn pct_turn20_turbo_relative_change_has_no_negative_sign() {
        assert_close(relative_change(Some(12.0), Some(10.0)), Some(0.2));
    }

    #[test]
    fn pct_turn20_turbo_uses_signal_mean_over_single_prior_base() {
        let values = (1..=61).map(|value| Some(value as f64)).collect::<Vec<_>>();
        let signal = ts_mean(&values, SIGNAL_WINDOW, MIN_PERIODS);
        let base = delayed_rolling_mean(&values, BASE_WINDOW, BASE_DELAY);
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
    fn pct_turn20_turbo_negates_gtr_direction_before_scaling() {
        assert_close(negate(Some(0.3)), Some(-0.3));
        assert_eq!(negate(None), None);
    }

    #[test]
    fn pct_turn20_turbo_gtr_std_skips_missing_growth_values() {
        let growth = vec![Some(0.1), None, Some(0.3)];
        let std = ts_std_dev(&growth, SIGNAL_WINDOW, MIN_PERIODS);

        assert!(std[2].is_some());
    }
}

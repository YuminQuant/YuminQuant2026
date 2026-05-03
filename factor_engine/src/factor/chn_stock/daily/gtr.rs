use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::vector::clean;
use crate::factor::Factor;
use crate::operators::ts_std_dev;

const VERSION: &str = "0.1.0";
const WINDOW: usize = 20;

pub struct StockDailyGtr;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyGtr)
}

impl Factor for StockDailyGtr {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "gtr".to_string(),
            aliases: vec!["GTR".to_string()],
            name: "GTR".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: [
                "price_volume",
                "turnover",
                "stability",
                "growth_rate",
                "neutralize",
                "barra",
                "size",
                "daily",
                "DWZQ",
            ]
            .iter()
            .map(|value| value.to_string())
            .collect(),
            description: "The Stability of the Growth Rate of Turnover Rate factor, computed as the 20-day standard deviation of turnover growth neutralized by SIZE.".to_string(),
            dependencies: vec![
                DataRequest::new(DatasetId::StockDailyBasic, &["turnover_rate_f"]),
                DataRequest::new(DatasetId::StockBarraDaily, &["SIZE"]),
            ],
            intraday_raw_dependencies: Vec::new(),
            lookback: Lookback {
                trading_days: WINDOW,
            },
        }
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let panel = data.daily_panel(DatasetId::StockDailyBasic)?;
        let turnover = panel
            .column("turnover_rate_f")?
            .map_values(percent_to_decimal);
        let size = panel.column_from_table(data.daily(DatasetId::StockBarraDaily)?, "SIZE")?;

        let growth = turnover.ts(turnover_growth)?;
        let growth_std = growth.ts(|values| ts_std_dev(values, WINDOW, WINDOW))?;
        let factor = growth_std.cs_neutralize_regression(&[&size], None)?;
        Ok(factor.to_factor_series(self.spec()))
    }
}

fn percent_to_decimal(value: Option<f64>) -> Option<f64> {
    clean(value).map(|value| value / 100.0)
}

fn turnover_growth(values: &[Option<f64>]) -> Vec<Option<f64>> {
    let mut output = vec![None; values.len()];
    for idx in 1..values.len() {
        output[idx] = relative_change(values[idx], values[idx - 1]);
    }
    output
}

fn relative_change(current: Option<f64>, previous: Option<f64>) -> Option<f64> {
    match (clean(current), clean(previous)) {
        (Some(current), Some(previous)) if previous.abs() > f64::EPSILON => {
            Some(current / previous - 1.0)
        }
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
    fn gtr_turnover_growth_rejects_zero_previous_turnover() {
        assert_eq!(relative_change(Some(0.2), Some(0.0)), None);
        assert_close(relative_change(Some(0.24), Some(0.2)), Some(0.2));
    }

    #[test]
    fn gtr_turnover_percent_is_converted_to_decimal() {
        assert_close(percent_to_decimal(Some(2.5)), Some(0.025));
    }
}

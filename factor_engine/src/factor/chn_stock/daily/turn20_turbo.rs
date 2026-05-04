use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::vector::clean;
use crate::factor::common::PanelColumn;
use crate::factor::Factor;
use crate::operators::{cs_nonnegative, cs_scale, ts_mean, ts_std_dev};

const VERSION: &str = "0.2.0";
const WINDOW: usize = 20;
const MIN_PERIODS: usize = 1;

pub struct StockDailyTurn20Turbo;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyTurn20Turbo)
}

impl Factor for StockDailyTurn20Turbo {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "turn20_turbo".to_string(),
            aliases: vec!["Turn20_Turbo".to_string()],
            name: "Turn20_Turbo".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: [
                "price_volume",
                "turnover",
                "stability",
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
            description: "Turbo turnover factor combining SIZE-neutralized 20-day average turnover and GTR with relaxed missing-data windows after non-negative scaling.".to_string(),
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

        let turn20 = turnover
            .ts(|values| ts_mean(values, WINDOW, WINDOW))?
            .cs_neutralize_regression(&[&size], None)?;
        let growth = turnover.ts(turnover_growth)?;
        let gtr = growth
            .ts(|values| ts_std_dev(values, WINDOW, MIN_PERIODS))?
            .cs_neutralize_regression(&[&size], None)?;
        let factor = average_pair(&turbo_scale(&turn20)?, &turbo_scale(&gtr)?)?;
        Ok(factor.to_factor_series(self.spec()))
    }
}

fn percent_to_decimal(value: Option<f64>) -> Option<f64> {
    clean(value).map(|value| value / 100.0)
}

fn turnover_growth(values: &[Option<f64>]) -> Vec<Option<f64>> {
    let mut output = vec![None; values.len()];
    for idx in 1..values.len() {
        output[idx] = match (clean(values[idx]), clean(values[idx - 1])) {
            (Some(current), Some(previous)) if previous.abs() > f64::EPSILON => {
                Some(current / previous - 1.0)
            }
            _ => None,
        };
    }
    output
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
    use crate::operators::ts_std_dev;
    use crate::operators::{cs_nonnegative, cs_scale};

    use super::{MIN_PERIODS, WINDOW};

    #[test]
    fn turn20_turbo_transform_order_is_nonnegative_then_scale() {
        let values = vec![Some(-1.0), Some(0.0), Some(1.0)];
        let nonnegative = cs_nonnegative(&values);
        let scaled = cs_scale(&nonnegative);

        assert_eq!(nonnegative[0], Some(0.0));
        assert!(scaled.iter().flatten().all(|value| *value >= 0.0));
        assert!((scaled.iter().flatten().sum::<f64>() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn turn20_turbo_gtr_std_skips_missing_growth_values() {
        let growth = vec![Some(0.1), None, Some(0.3)];
        let std = ts_std_dev(&growth, WINDOW, MIN_PERIODS);

        assert!(std[2].is_some());
    }
}

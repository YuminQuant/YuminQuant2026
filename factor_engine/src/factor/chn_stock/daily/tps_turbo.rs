use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::vector::clean;
use crate::factor::common::PanelColumn;
use crate::factor::Factor;
use crate::operators::{cs_regression_residual, cs_zscore, ts_mean, ts_std_dev};

const VERSION: &str = "0.2.0";
const WINDOW: usize = 20;
const MIN_PERIODS: usize = 1;

pub struct StockDailyTpsTurbo;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyTpsTurbo)
}

impl Factor for StockDailyTpsTurbo {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "tps_turbo".to_string(),
            aliases: vec!["TPS_Turbo".to_string()],
            name: "TPS_Turbo".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: [
                "price_volume",
                "turnover",
                "stability",
                "regression",
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
            description: "Turbo TPS factor combining pure Turn20 and pure GTR with relaxed missing-data windows after cross-sectional residualization, SIZE neutralization, and zscore.".to_string(),
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
        let pure_turnover = turnover.cs_binary(&growth, cs_regression_residual)?;
        let pure_growth = growth.cs_binary(&turnover, cs_regression_residual)?;
        let pure_turn20 = pure_turnover
            .ts(|values| ts_mean(values, WINDOW, WINDOW))?
            .cs_neutralize_regression(&[&size], None)?
            .cs(cs_zscore)?;
        let pure_gtr = pure_growth
            .ts(|values| ts_std_dev(values, WINDOW, MIN_PERIODS))?
            .cs_neutralize_regression(&[&size], None)?
            .cs(cs_zscore)?;
        let factor = average_pair(&pure_turn20, &pure_gtr)?;
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

fn average_pair(left: &PanelColumn, right: &PanelColumn) -> Result<PanelColumn> {
    left.zip_binary(right, |left, right| match (clean(left), clean(right)) {
        (Some(left), Some(right)) => Some((left + right) / 2.0),
        _ => None,
    })
}

#[cfg(test)]
mod tests {
    use super::{MIN_PERIODS, WINDOW};
    use crate::operators::cs_regression_residual;
    use crate::operators::ts_std_dev;

    #[test]
    fn tps_turbo_daily_residual_removes_linear_turnover_growth_effect() {
        let growth = vec![Some(1.0), Some(2.0), Some(3.0)];
        let turnover = vec![Some(3.0), Some(5.0), Some(7.0)];
        let residual = cs_regression_residual(&turnover, &growth);

        assert!(residual.iter().flatten().all(|value| value.abs() < 1e-10));
    }

    #[test]
    fn tps_turbo_pure_gtr_std_skips_missing_growth_values() {
        let growth = vec![Some(0.1), None, Some(0.3)];
        let std = ts_std_dev(&growth, WINDOW, MIN_PERIODS);

        assert!(std[2].is_some());
    }
}

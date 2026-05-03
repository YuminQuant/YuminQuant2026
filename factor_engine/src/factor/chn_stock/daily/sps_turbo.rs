use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::vector::clean;
use crate::factor::common::PanelColumn;
use crate::factor::Factor;
use crate::operators::{cs_regression_residual, cs_zscore, ts_std_dev};

const VERSION: &str = "0.1.0";
const WINDOW: usize = 20;

pub struct StockDailySpsTurbo;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailySpsTurbo)
}

impl Factor for StockDailySpsTurbo {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "sps_turbo".to_string(),
            aliases: vec!["SPS_Turbo".to_string()],
            name: "SPS_Turbo".to_string(),
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
            description: "Turbo SPS factor combining pure STR and pure GTR after cross-sectional residualization, SIZE neutralization, and zscore.".to_string(),
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
        let pure_str = pure_turnover
            .ts(|values| ts_std_dev(values, WINDOW, WINDOW))?
            .cs_neutralize_regression(&[&size], None)?
            .cs(cs_zscore)?;
        let pure_gtr = pure_growth
            .ts(|values| ts_std_dev(values, WINDOW, WINDOW))?
            .cs_neutralize_regression(&[&size], None)?
            .cs(cs_zscore)?;
        let factor = average_pair(&pure_str, &pure_gtr)?;
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
    use super::turnover_growth;

    #[test]
    fn sps_turbo_growth_uses_previous_day_denominator() {
        let growth = turnover_growth(&[Some(0.2), Some(0.25)]);

        assert!((growth[1].unwrap() - 0.25).abs() < 1e-10);
    }
}

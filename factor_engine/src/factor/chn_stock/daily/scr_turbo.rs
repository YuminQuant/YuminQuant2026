use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::vector::clean;
use crate::factor::common::PanelColumn;
use crate::factor::Factor;
use crate::operators::{cs_nonnegative, cs_scale, ts_delay, ts_std_dev};

const VERSION: &str = "0.1.0";
const CURRENT_WINDOW: usize = 20;
const BASE_WINDOW: usize = 40;
const BASE_DELAY: usize = 20;

pub struct StockDailyScrTurbo;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyScrTurbo)
}

impl Factor for StockDailyScrTurbo {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "scr_turbo".to_string(),
            aliases: vec!["SCR_Turbo".to_string()],
            name: "SCR_Turbo".to_string(),
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
            description: "Turbo SCR factor combining SIZE-neutralized turnover stability change and GTR after non-negative scaling.".to_string(),
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
        let turnover = panel
            .column("turnover_rate_f")?
            .map_values(percent_to_decimal);
        let size = panel.column_from_table(data.daily(DatasetId::StockBarraDaily)?, "SIZE")?;

        let current_volatility =
            turnover.ts(|values| ts_std_dev(values, CURRENT_WINDOW, CURRENT_WINDOW))?;
        let base_volatility = turnover.ts(previous_base_volatility)?;
        let scr = current_volatility
            .zip_binary(&base_volatility, relative_change)?
            .cs_neutralize_regression(&[&size], None)?;
        let growth = turnover.ts(turnover_growth)?;
        let gtr = growth
            .ts(|values| ts_std_dev(values, CURRENT_WINDOW, CURRENT_WINDOW))?
            .cs_neutralize_regression(&[&size], None)?;
        let factor = average_pair(&turbo_scale(&scr)?, &turbo_scale(&gtr)?)?;
        Ok(factor.to_factor_series(self.spec()))
    }
}

fn percent_to_decimal(value: Option<f64>) -> Option<f64> {
    clean(value).map(|value| value / 100.0)
}

fn previous_base_volatility(values: &[Option<f64>]) -> Vec<Option<f64>> {
    let base = ts_std_dev(values, BASE_WINDOW, BASE_WINDOW);
    ts_delay(&base, BASE_DELAY)
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
    fn scr_turbo_rejects_zero_base_volatility() {
        assert_eq!(relative_change(Some(1.0), Some(0.0)), None);
        assert_close(relative_change(Some(3.0), Some(2.0)), Some(0.5));
    }
}

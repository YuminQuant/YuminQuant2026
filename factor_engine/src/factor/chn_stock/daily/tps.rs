use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::vector::clean;
use crate::factor::common::PanelColumn;
use crate::factor::Factor;
use crate::operators::{cs_regression_residual, cs_zscore, ts_mean};

const VERSION: &str = "0.1.0";
pub(super) const WINDOW: usize = 20;

pub struct StockDailyTps;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyTps)
}

impl Factor for StockDailyTps {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "tps".to_string(),
            aliases: vec!["TPS".to_string()],
            name: "TPS".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: [
                "price_volume",
                "turnover",
                "price",
                "regression",
                "composite",
                "daily",
                "DWZQ",
            ]
            .iter()
            .map(|value| value.to_string())
            .collect(),
            description: "Turn20 conformed by PLUS, combining pure turnover and pure PLUS after cross-sectional residualization and non-negative shifting.".to_string(),
            dependencies: vec![
                DataRequest::new(
                    DatasetId::StockDailyPv,
                    &["close", "high", "low", "pre_close"],
                ),
                DataRequest::new(DatasetId::StockDailyBasic, &["turnover_rate_f"]),
            ],
            intraday_raw_dependencies: Vec::new(),
            lookback: Lookback {
                trading_days: WINDOW - 1,
            },
        }
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let pv = data.daily_panel(DatasetId::StockDailyPv)?;
        let basic = data.daily_panel(DatasetId::StockDailyBasic)?;
        let close = pv.column("close")?;
        let high = pv.column("high")?;
        let low = pv.column("low")?;
        let pre_close = pv.column("pre_close")?;
        let turnover = basic.column("turnover_rate_f")?;

        let plus = plus_factor(&close, &high, &low, &pre_close)?;
        let turn_deplus20 = turn_deplus(&turnover, &plus)?
            .ts(|values| ts_mean(values, WINDOW, WINDOW))?
            .cs(cs_zscore)?
            .cs(nonnegative_shift)?;
        let plus_deturn20 = plus_deturn(&plus, &turnover)?
            .ts(|values| ts_mean(values, WINDOW, WINDOW))?
            .cs(cs_zscore)?
            .cs(nonnegative_shift)?;
        let factor = multiply_pair(&turn_deplus20, &plus_deturn20)?;
        Ok(factor.to_factor_series(self.spec()))
    }
}

pub(super) fn plus_factor(
    close: &PanelColumn,
    high: &PanelColumn,
    low: &PanelColumn,
    pre_close: &PanelColumn,
) -> Result<PanelColumn> {
    close
        .zip_ternary(high, low, |close, high, low| {
            match (clean(close), clean(high), clean(low)) {
                (Some(close), Some(high), Some(low)) => Some(2.0 * close - high - low),
                _ => None,
            }
        })?
        .zip_binary(pre_close, safe_div)
}

pub(super) fn turn_deplus(turnover: &PanelColumn, plus: &PanelColumn) -> Result<PanelColumn> {
    turnover.cs_binary(plus, cs_regression_residual)
}

pub(super) fn plus_deturn(plus: &PanelColumn, turnover: &PanelColumn) -> Result<PanelColumn> {
    plus.cs_binary(turnover, cs_regression_residual)
}

pub(super) fn nonnegative_shift(values: &[Option<f64>]) -> Vec<Option<f64>> {
    let Some(min_value) = values
        .iter()
        .filter_map(|value| clean(*value))
        .reduce(f64::min)
    else {
        return vec![None; values.len()];
    };
    values
        .iter()
        .map(|value| clean(*value).map(|value| value - min_value))
        .collect()
}

pub(super) fn multiply_pair(left: &PanelColumn, right: &PanelColumn) -> Result<PanelColumn> {
    left.zip_binary(right, |left, right| match (clean(left), clean(right)) {
        (Some(left), Some(right)) => Some(left * right),
        _ => None,
    })
}

fn safe_div(numerator: Option<f64>, denominator: Option<f64>) -> Option<f64> {
    match (clean(numerator), clean(denominator)) {
        (Some(numerator), Some(denominator)) if denominator.abs() > f64::EPSILON => {
            Some(numerator / denominator)
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
    fn safe_div_rejects_zero_denominator() {
        assert_eq!(safe_div(Some(1.0), Some(0.0)), None);
        assert_close(safe_div(Some(6.0), Some(3.0)), Some(2.0));
    }

    #[test]
    fn nonnegative_shift_preserves_order_and_sets_min_to_zero() {
        let shifted = nonnegative_shift(&[Some(-1.5), Some(0.5), None, Some(2.0)]);

        assert_close(shifted[0], Some(0.0));
        assert_close(shifted[1], Some(2.0));
        assert_eq!(shifted[2], None);
        assert_close(shifted[3], Some(3.5));
    }
}

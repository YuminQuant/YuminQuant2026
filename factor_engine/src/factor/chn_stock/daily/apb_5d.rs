use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::stock_daily_ops::neutralize_size_sector;
use crate::factor::common::vector::clean;
use crate::factor::Factor;
use crate::operators::{ts_mean, ts_sum};

const VERSION: &str = "0.1.0";
const PRICE_WINDOW: usize = 5;
const SMOOTH_WINDOW: usize = 20;
const MIN_PERIODS: usize = 1;

pub struct StockDailyApb5d;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyApb5d)
}

impl Factor for StockDailyApb5d {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "apb_5d".to_string(),
            aliases: vec!["APB_5d".to_string(), "APB5D".to_string()],
            name: "APB 5d".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: [
                "price_volume",
                "amount",
                "volume",
                "vwap",
                "neutralize",
                "barra",
                "size",
                "sector",
                "daily",
                "DFZQ",
            ]
            .iter()
            .map(|value| value.to_string())
            .collect(),
            description: "5-day active pressure bias from log(EWAP/VWAP), averaged over 20 days and neutralized by SIZE and SW sector.".to_string(),
            dependencies: vec![
                DataRequest::new(DatasetId::StockDailyPv, &["amount", "vol"]),
                DataRequest::new(DatasetId::StockBarraDaily, &["SIZE"]),
                DataRequest::new(DatasetId::StockSwClassification, &["l1_code"]),
            ],
            intraday_raw_dependencies: Vec::new(),
            lookback: Lookback {
                trading_days: (PRICE_WINDOW - 1) + (SMOOTH_WINDOW - 1),
            },
        }
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let panel = data.daily_panel(DatasetId::StockDailyPv)?;
        let amount = panel.column("amount")?;
        let volume = panel.column("vol")?;

        let daily_vwap_column = amount.zip_binary(&volume, daily_vwap)?;
        let amount_sum5 = amount.ts(|values| ts_sum(values, PRICE_WINDOW, MIN_PERIODS))?;
        let volume_sum5 = volume.ts(|values| ts_sum(values, PRICE_WINDOW, MIN_PERIODS))?;
        let vwap5 = amount_sum5.zip_binary(&volume_sum5, daily_vwap)?;
        let ewap5 = daily_vwap_column.ts(|values| ts_mean(values, PRICE_WINDOW, MIN_PERIODS))?;
        let raw = ewap5.zip_binary(&vwap5, log_ratio)?;
        let smoothed = raw.ts(|values| ts_mean(values, SMOOTH_WINDOW, MIN_PERIODS))?;
        let factor = neutralize_size_sector(&smoothed, panel, data)?;
        Ok(factor.to_factor_series(self.spec()))
    }
}

fn daily_vwap(amount_thousand_yuan: Option<f64>, volume_lots: Option<f64>) -> Option<f64> {
    match (finite(amount_thousand_yuan), finite(volume_lots)) {
        (Some(amount), Some(volume)) if volume.abs() > f64::EPSILON => Some(amount * 10.0 / volume),
        _ => None,
    }
}

fn log_ratio(numerator: Option<f64>, denominator: Option<f64>) -> Option<f64> {
    match (finite(numerator), finite(denominator)) {
        (Some(numerator), Some(denominator))
            if numerator > 0.0 && denominator.abs() > f64::EPSILON =>
        {
            Some((numerator / denominator).ln())
        }
        _ => None,
    }
}

fn finite(value: Option<f64>) -> Option<f64> {
    clean(value).filter(|value| value.is_finite())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: Option<f64>, expected: f64) {
        let actual = actual.expect("expected value");
        assert!(
            (actual - expected).abs() < 1e-12,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn daily_vwap_uses_daily_units() {
        assert_close(daily_vwap(Some(1000.0), Some(100.0)), 100.0);
        assert_eq!(daily_vwap(Some(1000.0), Some(0.0)), None);
    }

    #[test]
    fn apb_raw_uses_log_ewap_over_vwap() {
        assert_close(log_ratio(Some(110.0), Some(100.0)), (1.1_f64).ln());
        assert_eq!(log_ratio(Some(-1.0), Some(100.0)), None);
    }
}

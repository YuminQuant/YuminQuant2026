use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::vector::clean;
use crate::factor::Factor;
use crate::operators::{ts_corr, ts_delay, ts_mean};

const VERSION: &str = "0.1.0";
const WINDOW: usize = 20;
const MIN_PERIODS: usize = 20;

pub struct StockDailyMif;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyMif)
}

impl Factor for StockDailyMif {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "mif".to_string(),
            aliases: vec!["MIF".to_string(), "New_OvernightRet".to_string()],
            name: "MIF".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: [
                "price_volume",
                "return",
                "overnight_return",
                "turnover",
                "correlation",
                "neutralize",
                "barra",
                "size",
                "daily",
                "GSZQ",
            ]
            .iter()
            .map(|value| value.to_string())
            .collect(),
            description: "Market Inefficiency Factor from the 20-day correlation between absolute overnight gaps and prior-day turnover, SIZE-neutralized and orthogonalized against desized overnight gap.".to_string(),
            dependencies: vec![
                DataRequest::new(DatasetId::StockDailyPv, &["open", "pre_close"]),
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
        let panel = data.daily_panel(DatasetId::StockDailyPv)?;
        let open = panel.column("open")?;
        let pre_close = panel.column("pre_close")?;
        let turnover = panel
            .column_from_table(data.daily(DatasetId::StockDailyBasic)?, "turnover_rate_f")?
            .map_values(percent_to_decimal);
        let size = panel.column_from_table(data.daily(DatasetId::StockBarraDaily)?, "SIZE")?;

        let overnight_return = open.zip_binary(&pre_close, ret)?;
        let abs_overnight_return = overnight_return.map_values(abs_value);
        let prior_turnover = turnover.ts(|values| ts_delay(values, 1))?;
        let corr = abs_overnight_return
            .ts_binary(&prior_turnover, |left, right| {
                ts_corr(left, right, WINDOW, MIN_PERIODS)
            })?
            .cs_neutralize_regression(&[&size], None)?;
        let abs_overnight_desize = abs_overnight_return
            .ts(|values| ts_mean(values, WINDOW, MIN_PERIODS))?
            .cs_neutralize_regression(&[&size], None)?;
        let factor = corr.cs_neutralize_regression(&[&abs_overnight_desize], None)?;

        Ok(factor.to_factor_series(self.spec()))
    }
}

fn ret(numerator: Option<f64>, denominator: Option<f64>) -> Option<f64> {
    match (finite(numerator), finite(denominator)) {
        (Some(numerator), Some(denominator)) if denominator.abs() > f64::EPSILON => {
            Some(numerator / denominator - 1.0)
        }
        _ => None,
    }
}

fn abs_value(value: Option<f64>) -> Option<f64> {
    finite(value).map(f64::abs)
}

fn percent_to_decimal(value: Option<f64>) -> Option<f64> {
    finite(value).map(|value| value / 100.0)
}

fn finite(value: Option<f64>) -> Option<f64> {
    clean(value).filter(|value| value.is_finite())
}

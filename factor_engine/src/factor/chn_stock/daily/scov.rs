use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    IntradayDailyRawRequest, Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::chn_stock::daily::sccoiv::LAST30_TURNOVER_RAW_ID;
use crate::factor::common::vector::clean;
use crate::factor::Factor;
use crate::operators::{ts_corr, ts_delay};

const VERSION: &str = "0.1.0";
const WINDOW: usize = 20;

pub struct StockDailyScov;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyScov)
}

impl Factor for StockDailyScov {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "scov".to_string(),
            aliases: vec!["SCOV".to_string()],
            name: "SCOV".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: [
                "price_volume",
                "turnover",
                "price",
                "correlation",
                "smart",
                "intraday",
                "minute_agg",
                "daily",
                "DWZQ",
            ]
            .iter()
            .map(|value| value.to_string())
            .collect(),
            description: "Smart overnight price-volume correlation between overnight price change and previous-day last-half-hour turnover.".to_string(),
            dependencies: vec![DataRequest::new(
                DatasetId::StockDailyPv,
                &["open", "pre_close"],
            )],
            intraday_raw_dependencies: vec![IntradayDailyRawRequest::new(
                LAST30_TURNOVER_RAW_ID,
                WINDOW,
            )],
            lookback: Lookback {
                trading_days: WINDOW,
            },
        }
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let panel = data.intraday_daily_raw_panel(LAST30_TURNOVER_RAW_ID)?;
        let pv_table = data.daily(DatasetId::StockDailyPv)?;
        let open = panel.column_from_table(pv_table, "open")?;
        let pre_close = panel.column_from_table(pv_table, "pre_close")?;
        let oyc = open.zip_binary(&pre_close, subtract)?;
        let y1430v = panel
            .column(LAST30_TURNOVER_RAW_ID)?
            .ts(|values| ts_delay(values, 1))?;
        let factor = oyc.ts_binary(&y1430v, |oyc, y1430v| ts_corr(oyc, y1430v, WINDOW, WINDOW))?;
        Ok(factor.to_factor_series(self.spec()))
    }
}

fn subtract(left: Option<f64>, right: Option<f64>) -> Option<f64> {
    match (clean(left), clean(right)) {
        (Some(left), Some(right)) => Some(left - right),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scov_subtract_requires_both_values() {
        assert_eq!(subtract(Some(10.5), Some(10.0)), Some(0.5));
        assert_eq!(subtract(Some(10.5), None), None);
    }
}

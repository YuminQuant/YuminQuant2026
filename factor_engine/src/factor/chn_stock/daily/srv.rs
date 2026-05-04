use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    IntradayDailyRawRequest, Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::chn_stock::daily::sccoiv::{
    LAST30_TURNOVER_RAW_ID, PM_CO_RAW_ID, PM_SMART_TURNOVER_RAW_ID,
};
use crate::factor::common::vector::clean;
use crate::factor::common::PanelColumn;
use crate::factor::Factor;
use crate::operators::{cs_zscore, ts_corr, ts_delay};

const VERSION: &str = "0.1.0";
const WINDOW: usize = 20;

pub struct StockDailySrv;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailySrv)
}

impl Factor for StockDailySrv {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "srv".to_string(),
            aliases: vec!["SRV".to_string()],
            name: "SRV".to_string(),
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
                "composite",
                "daily",
                "DWZQ",
            ]
            .iter()
            .map(|value| value.to_string())
            .collect(),
            description: "Smart RPV factor combining smart intraday and overnight price-volume correlations with opposite directions.".to_string(),
            dependencies: vec![DataRequest::new(
                DatasetId::StockDailyPv,
                &["open", "pre_close"],
            )],
            intraday_raw_dependencies: vec![
                IntradayDailyRawRequest::new(PM_CO_RAW_ID, WINDOW),
                IntradayDailyRawRequest::new(PM_SMART_TURNOVER_RAW_ID, WINDOW),
                IntradayDailyRawRequest::new(LAST30_TURNOVER_RAW_ID, WINDOW),
            ],
            lookback: Lookback {
                trading_days: WINDOW,
            },
        }
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let panel = data.intraday_daily_raw_panel(PM_CO_RAW_ID)?;
        let pm_co = panel.column(PM_CO_RAW_ID)?;
        let pm_smart_turnover = panel.column(PM_SMART_TURNOVER_RAW_ID)?;
        let sccoiv =
            pm_co.ts_binary(&pm_smart_turnover, |co, sv| ts_corr(co, sv, WINDOW, WINDOW))?;

        let pv_table = data.daily(DatasetId::StockDailyPv)?;
        let open = panel.column_from_table(pv_table, "open")?;
        let pre_close = panel.column_from_table(pv_table, "pre_close")?;
        let oyc = open.zip_binary(&pre_close, subtract)?;
        let y1430v = panel
            .column(LAST30_TURNOVER_RAW_ID)?
            .ts(|values| ts_delay(values, 1))?;
        let scov = oyc.ts_binary(&y1430v, |oyc, y1430v| ts_corr(oyc, y1430v, WINDOW, WINDOW))?;

        let factor = subtract_pair(&sccoiv.cs(cs_zscore)?, &scov.cs(cs_zscore)?)?;
        Ok(factor.to_factor_series(self.spec()))
    }
}

fn subtract(left: Option<f64>, right: Option<f64>) -> Option<f64> {
    match (clean(left), clean(right)) {
        (Some(left), Some(right)) => Some(left - right),
        _ => None,
    }
}

fn subtract_pair(left: &PanelColumn, right: &PanelColumn) -> Result<PanelColumn> {
    left.zip_binary(right, subtract)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn srv_subtract_pair_requires_both_sides() {
        assert_eq!(subtract(Some(1.5), Some(0.5)), Some(1.0));
        assert_eq!(subtract(Some(1.5), None), None);
    }
}

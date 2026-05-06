use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::umr;
use crate::factor::Factor;

const VERSION: &str = "0.1.0";

pub struct StockDailyUmrTr;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyUmrTr)
}

impl Factor for StockDailyUmrTr {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "umr_tr".to_string(),
            aliases: vec!["UMR_TR".to_string(), "TR-Adjusted UMR".to_string()],
            name: "TR-Adjusted UMR".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: tags(),
            description: "Unified momentum and reversal factor adjusted by daily true range risk, neutralized by Barra SIZE and SW sector.".to_string(),
            dependencies: vec![
                DataRequest::new(DatasetId::StockDailyPv, &["high", "low", "close", "pre_close"]),
                umr::market_close_return_request(),
                DataRequest::new(DatasetId::StockBarraDaily, &["SIZE"]),
                DataRequest::new(DatasetId::StockSwClassification, &["l1_code"]),
            ],
            intraday_raw_dependencies: Vec::new(),
            lookback: Lookback {
                trading_days: umr::UMR_LOOKBACK,
            },
        }
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let panel = data.daily_panel(DatasetId::StockDailyPv)?;
        let risk = true_range(&panel)?;
        let ex_ret = umr::excess_return(&panel, data)?;
        let raw = umr::umr_raw(&risk, &ex_ret, true)?;
        let factor = umr::neutralize_size_sector(&raw, &panel, data)?;
        Ok(factor.to_factor_series(self.spec()))
    }
}

fn tags() -> Vec<String> {
    [
        "price_volume",
        "return",
        "momentum",
        "reversal",
        "volatility",
        "neutralize",
        "barra",
        "size",
        "sector",
        "daily",
        "GXZQ",
    ]
    .iter()
    .map(|value| value.to_string())
    .collect()
}

fn true_range(
    panel: &crate::factor::common::DailyPanel,
) -> Result<crate::factor::common::PanelColumn> {
    panel
        .column("high")?
        .zip_ternary(&panel.column("low")?, &panel.column("pre_close")?, tr_value)
}

fn tr_value(high: Option<f64>, low: Option<f64>, pre_close: Option<f64>) -> Option<f64> {
    let (Some(high), Some(low), Some(pre_close)) =
        (umr::finite(high), umr::finite(low), umr::finite(pre_close))
    else {
        return None;
    };
    if pre_close.abs() <= f64::EPSILON {
        return None;
    }
    let tr = (high - low)
        .max((high - pre_close).abs())
        .max((low - pre_close).abs());
    Some(tr / pre_close)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn true_range_uses_largest_intraday_or_gap_range() {
        let value = tr_value(Some(12.0), Some(9.0), Some(10.0));
        assert_eq!(value, Some(0.3));
        let gap = tr_value(Some(13.0), Some(12.0), Some(10.0));
        assert_eq!(gap, Some(0.3));
        assert_eq!(tr_value(Some(1.0), Some(1.0), Some(0.0)), None);
    }
}

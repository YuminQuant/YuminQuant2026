use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::umr;
use crate::factor::Factor;

const VERSION: &str = "0.1.0";

pub struct StockDailyUmrTrIntradayExcess;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyUmrTrIntradayExcess)
}

impl Factor for StockDailyUmrTrIntradayExcess {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "umr_tr_intraday_excess".to_string(),
            aliases: vec![
                "UMR_TR_INTRADAY_EXCESS".to_string(),
                "TR-Adjusted Intraday Excess UMR".to_string(),
            ],
            name: "TR-Adjusted Intraday Excess UMR".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: tags(),
            description: "Unified momentum and reversal factor using TR risk to weight intraday excess return, neutralized by Barra SIZE and SW sector.".to_string(),
            dependencies: vec![
                DataRequest::new(
                    DatasetId::StockDailyPv,
                    &["open", "high", "low", "close", "pre_close"],
                ),
                umr::market_intraday_return_request(),
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
        let intraday_excess = umr::intraday_excess_return(&panel, data)?;
        let raw = umr::umr_raw(&risk, &intraday_excess, true)?;
        let factor = umr::neutralize_size_sector(&raw, &panel, data)?;
        Ok(factor.to_factor_series(self.spec()))
    }
}

fn tags() -> Vec<String> {
    [
        "price_volume",
        "intraday_return",
        "excess_return",
        "volatility",
        "momentum",
        "reversal",
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
    fn market_intraday_request_uses_open_and_close() {
        let request = umr::market_intraday_return_request();
        assert_eq!(
            request.columns,
            vec!["open".to_string(), "close".to_string()]
        );
    }
}

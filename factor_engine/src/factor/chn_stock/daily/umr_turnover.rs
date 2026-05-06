use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::umr;
use crate::factor::Factor;

const VERSION: &str = "0.1.0";

pub struct StockDailyUmrTurnover;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyUmrTurnover)
}

impl Factor for StockDailyUmrTurnover {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "umr_turnover".to_string(),
            aliases: vec![
                "UMR_TURNOVER".to_string(),
                "Turnover-Adjusted UMR".to_string(),
            ],
            name: "Turnover-Adjusted UMR".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: tags(),
            description: "Unified momentum and reversal factor adjusted by daily turnover risk, neutralized by Barra SIZE and SW sector.".to_string(),
            dependencies: vec![
                DataRequest::new(DatasetId::StockDailyPv, &["close", "pre_close"]),
                DataRequest::new(DatasetId::StockDailyBasic, &["turnover_rate_f"]),
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
        let risk = panel
            .column_from_table(data.daily(DatasetId::StockDailyBasic)?, "turnover_rate_f")?
            .map_values(umr::percent_to_decimal);
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
        "turnover",
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn turnover_percent_is_converted_to_decimal() {
        assert_eq!(umr::percent_to_decimal(Some(2.5)), Some(0.025));
    }
}

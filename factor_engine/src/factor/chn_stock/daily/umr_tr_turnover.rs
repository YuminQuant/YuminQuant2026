use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::umr;
use crate::factor::Factor;

const VERSION: &str = "0.1.0";

pub struct StockDailyUmrTrTurnover;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyUmrTrTurnover)
}

impl Factor for StockDailyUmrTrTurnover {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "umr_tr_turnover".to_string(),
            aliases: vec![
                "UMR_TR_TURNOVER".to_string(),
                "TR-Adjusted Turnover UMR".to_string(),
            ],
            name: "TR-Adjusted Turnover UMR".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: tags(),
            description: "Unified momentum and reversal derivative factor using TR risk to weight daily turnover, neutralized by Barra SIZE and SW sector.".to_string(),
            dependencies: vec![
                DataRequest::new(DatasetId::StockDailyPv, &["high", "low", "pre_close"]),
                DataRequest::new(DatasetId::StockDailyBasic, &["turnover_rate_f"]),
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
        let turnover = panel
            .column_from_table(data.daily(DatasetId::StockDailyBasic)?, "turnover_rate_f")?
            .map_values(umr::percent_to_decimal);
        let raw = umr::umr_raw(&risk, &turnover, true)?;
        let factor = umr::neutralize_size_sector(&raw, &panel, data)?;
        Ok(factor.to_factor_series(self.spec()))
    }
}

fn tags() -> Vec<String> {
    [
        "price_volume",
        "turnover",
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
    fn tr_turnover_uses_decimal_turnover_as_weighted_variable() {
        assert_eq!(umr::percent_to_decimal(Some(3.0)), Some(0.03));
    }
}

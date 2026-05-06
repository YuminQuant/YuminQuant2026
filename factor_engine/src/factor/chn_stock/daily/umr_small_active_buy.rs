use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::umr;
use crate::factor::Factor;

const VERSION: &str = "0.1.0";

pub struct StockDailyUmrSmallActiveBuy;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyUmrSmallActiveBuy)
}

impl Factor for StockDailyUmrSmallActiveBuy {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "umr_small_active_buy".to_string(),
            aliases: vec![
                "UMR_SMALL_ACTIVE_BUY".to_string(),
                "Small Active Buy-Adjusted UMR".to_string(),
            ],
            name: "Small Active Buy-Adjusted UMR".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: tags(),
            description: "Unified momentum and reversal factor adjusted by small active buy amount ratio, neutralized by Barra SIZE and SW sector.".to_string(),
            dependencies: vec![
                DataRequest::new(DatasetId::StockDailyPv, &["close", "pre_close", "amount"]),
                DataRequest::new(DatasetId::StockMoneyflow, &["buy_sm_amount"]),
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
        let buy_sm_amount =
            panel.column_from_table(data.daily(DatasetId::StockMoneyflow)?, "buy_sm_amount")?;
        let amount = panel.column("amount")?;
        let risk = buy_sm_amount.zip_binary(&amount, small_active_buy_ratio)?;
        let ex_ret = umr::excess_return(&panel, data)?;
        let raw = umr::umr_raw(&risk, &ex_ret, false)?;
        let factor = umr::neutralize_size_sector(&raw, &panel, data)?;
        Ok(factor.to_factor_series(self.spec()))
    }
}

fn tags() -> Vec<String> {
    [
        "price_volume",
        "return",
        "moneyflow",
        "small_order",
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

fn small_active_buy_ratio(buy_sm_amount: Option<f64>, daily_amount: Option<f64>) -> Option<f64> {
    match (umr::finite(buy_sm_amount), umr::finite(daily_amount)) {
        (Some(buy), Some(amount)) if amount.abs() > f64::EPSILON => Some(buy / (amount / 10.0)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_active_buy_uses_moneyflow_amount_over_daily_amount_in_ten_thousand_yuan() {
        assert_eq!(small_active_buy_ratio(Some(10.0), Some(1000.0)), Some(0.1));
        assert_eq!(small_active_buy_ratio(Some(10.0), Some(0.0)), None);
    }
}

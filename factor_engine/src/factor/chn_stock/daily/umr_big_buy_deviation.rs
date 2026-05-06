use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::umr;
use crate::factor::Factor;

const VERSION: &str = "0.1.0";

pub struct StockDailyUmrBigBuyDeviation;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyUmrBigBuyDeviation)
}

impl Factor for StockDailyUmrBigBuyDeviation {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "umr_big_buy_deviation".to_string(),
            aliases: vec![
                "UMR_BIG_BUY_DEVIATION".to_string(),
                "Big Buy Deviation-Adjusted UMR".to_string(),
            ],
            name: "Big Buy Deviation-Adjusted UMR".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: tags(),
            description: "Unified momentum and reversal factor adjusted by large and extra-large buy VWAP deviation risk, neutralized by Barra SIZE and SW sector.".to_string(),
            dependencies: vec![
                DataRequest::new(DatasetId::StockDailyPv, &["close", "pre_close", "amount", "vol"]),
                DataRequest::new(
                    DatasetId::StockMoneyflow,
                    &["buy_lg_amount", "buy_lg_vol", "buy_elg_amount", "buy_elg_vol"],
                ),
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
        let moneyflow = data.daily(DatasetId::StockMoneyflow)?;
        let amount = panel.column("amount")?;
        let volume = panel.column("vol")?;
        let buy_lg_amount = panel.column_from_table(moneyflow, "buy_lg_amount")?;
        let buy_lg_vol = panel.column_from_table(moneyflow, "buy_lg_vol")?;
        let buy_elg_amount = panel.column_from_table(moneyflow, "buy_elg_amount")?;
        let buy_elg_vol = panel.column_from_table(moneyflow, "buy_elg_vol")?;

        let daily_vwap = amount.zip_binary(&volume, daily_vwap)?;
        let big_vwap = buy_lg_amount.zip_quaternary(
            &buy_lg_vol,
            &buy_elg_amount,
            &buy_elg_vol,
            big_buy_vwap,
        )?;
        let risk = big_vwap.zip_binary(&daily_vwap, deviation)?;
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
        "moneyflow",
        "vwap",
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

fn daily_vwap(amount_thousand_yuan: Option<f64>, volume_lots: Option<f64>) -> Option<f64> {
    match (umr::finite(amount_thousand_yuan), umr::finite(volume_lots)) {
        (Some(amount), Some(volume)) if volume.abs() > f64::EPSILON => Some(amount * 10.0 / volume),
        _ => None,
    }
}

fn big_buy_vwap(
    lg_amount: Option<f64>,
    lg_vol: Option<f64>,
    elg_amount: Option<f64>,
    elg_vol: Option<f64>,
) -> Option<f64> {
    let (Some(lg_amount), Some(lg_vol), Some(elg_amount), Some(elg_vol)) = (
        umr::finite(lg_amount),
        umr::finite(lg_vol),
        umr::finite(elg_amount),
        umr::finite(elg_vol),
    ) else {
        return None;
    };
    let total_vol = lg_vol + elg_vol;
    if total_vol.abs() <= f64::EPSILON {
        return None;
    }
    Some((lg_amount + elg_amount) * 100.0 / total_vol)
}

fn deviation(big_vwap: Option<f64>, daily_vwap: Option<f64>) -> Option<f64> {
    match (umr::finite(big_vwap), umr::finite(daily_vwap)) {
        (Some(big), Some(daily)) if daily.abs() > f64::EPSILON => Some((big - daily).abs() / daily),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vwap_unit_conversions_match_daily_and_moneyflow_units() {
        assert_eq!(daily_vwap(Some(1000.0), Some(100.0)), Some(100.0));
        assert_eq!(deviation(Some(110.0), Some(100.0)), Some(0.1));
    }
}

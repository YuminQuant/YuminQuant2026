use std::collections::BTreeMap;

use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorRowKey, FactorSeries, FactorSpec,
    FactorValue, Frequency, IntradayDailyRawAuxiliaryRequest, IntradayDailyRawRequest,
    IntradayDailyRawSeries, IntradayDailyRawSpec, Lookback,
};
use crate::data::{DataPool, Table};
use crate::error::Result;
use crate::factor::common::umr;
use crate::factor::common::{
    clean_intraday_value, intraday_time_in_range, stock_minute_raw_spec, DailyPanel, PanelColumn,
};
use crate::factor::Factor;

const RAW_ID: &str = "daily_umr_open_close_amount_ratio";
const RAW_VERSION: &str = "0.1.0";
const VERSION: &str = "0.1.0";

pub struct StockDailyUmrOpenCloseAmountRatio;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyUmrOpenCloseAmountRatio)
}

fn raw_spec() -> IntradayDailyRawSpec {
    stock_minute_raw_spec(RAW_ID, RAW_VERSION, &["amount"], 1)
}

impl Factor for StockDailyUmrOpenCloseAmountRatio {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "umr_open_close_amount_ratio".to_string(),
            aliases: vec![
                "UMR_OPEN_CLOSE_AMOUNT_RATIO".to_string(),
                "Open-Close Amount Ratio-Adjusted UMR".to_string(),
            ],
            name: "Open-Close Amount Ratio-Adjusted UMR".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: tags(),
            description: "Unified momentum and reversal factor adjusted by first-30 and last-30 minute amount over free-float market cap, neutralized by Barra SIZE and SW sector.".to_string(),
            dependencies: vec![
                DataRequest::new(DatasetId::StockDailyPv, &["close", "pre_close"]),
                umr::market_close_return_request(),
                DataRequest::new(DatasetId::StockBarraDaily, &["SIZE"]),
                DataRequest::new(DatasetId::StockSwClassification, &["l1_code"]),
            ],
            intraday_raw_dependencies: vec![IntradayDailyRawRequest::new(
                RAW_ID,
                umr::UMR_LOOKBACK,
            )],
            lookback: Lookback {
                trading_days: umr::UMR_LOOKBACK,
            },
        }
    }

    fn intraday_raw_specs(&self) -> Vec<IntradayDailyRawSpec> {
        vec![raw_spec()]
    }

    fn intraday_raw_auxiliary_requirements(
        &self,
        raw_ids: &[String],
    ) -> Vec<IntradayDailyRawAuxiliaryRequest> {
        if raw_ids.iter().any(|raw_id| raw_id == RAW_ID) {
            vec![IntradayDailyRawAuxiliaryRequest::new(
                DataRequest::new(DatasetId::StockDailyBasic, &["circ_mv"]),
                0,
            )]
        } else {
            Vec::new()
        }
    }

    fn minute_compute(
        &self,
        raw_id: &str,
        context: &FactorContext,
        data: &DataPool,
    ) -> Result<Option<IntradayDailyRawSeries>> {
        if raw_id != RAW_ID {
            return Ok(None);
        }
        let mut values = Vec::new();
        for trade_date in &context.target_dates {
            let Some(table) = data.minute(DatasetId::StockMinute1m, *trade_date) else {
                continue;
            };
            let circ_mv = circ_mv_by_code(data.daily(DatasetId::StockDailyBasic)?, *trade_date)?;
            let ts_codes = table.required_utf8("ts_code")?;
            let trade_times = table.required_utf8("trade_time")?;
            let amount = table.required_f64_cast("amount")?;

            let mut sums = BTreeMap::<String, f64>::new();
            for idx in 0..table.len {
                let Some(ts_code) = ts_codes[idx].clone() else {
                    continue;
                };
                let Some(trade_time) = trade_times[idx].as_deref() else {
                    continue;
                };
                if !is_open_close_window(trade_time) {
                    continue;
                }
                let Some(amount) = clean_intraday_value(amount[idx]) else {
                    continue;
                };
                *sums.entry(ts_code).or_default() += amount;
            }

            for (ts_code, amount_sum) in sums {
                values.push(FactorValue {
                    key: FactorRowKey::Daily {
                        trade_date: *trade_date,
                        ts_code: ts_code.clone(),
                    },
                    value: amount_ratio(amount_sum, circ_mv.get(&ts_code).copied().flatten()),
                });
            }
        }
        Ok(Some(IntradayDailyRawSeries {
            spec: raw_spec(),
            values,
        }))
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let panel = data.intraday_daily_raw_panel(RAW_ID)?;
        let risk = panel.column(RAW_ID)?;
        let ex_ret = excess_return_from_raw_panel(&panel, data)?;
        let raw = umr::umr_raw(&risk, &ex_ret, true)?;
        let factor = umr::neutralize_size_sector(&raw, &panel, data)?;
        Ok(factor.to_factor_series(self.spec()))
    }
}

fn tags() -> Vec<String> {
    [
        "price_volume",
        "return",
        "amount",
        "intraday",
        "minute_agg",
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

fn excess_return_from_raw_panel(panel: &DailyPanel, data: &DataPool) -> Result<PanelColumn> {
    let close = panel.column_from_table(data.daily(DatasetId::StockDailyPv)?, "close")?;
    let pre_close = panel.column_from_table(data.daily(DatasetId::StockDailyPv)?, "pre_close")?;
    let stock_ret = close.zip_binary(&pre_close, umr::ret)?;
    let market_ret = umr::expanded_market_return(panel, data, false)?;
    stock_ret.zip_binary(&market_ret, umr::subtract)
}

fn circ_mv_by_code(table: &Table, trade_date: i32) -> Result<BTreeMap<String, Option<f64>>> {
    let trade_dates = table.required_i32("trade_date")?;
    let ts_codes = table.required_utf8("ts_code")?;
    let circ_mv = table.required_f64_cast("circ_mv")?;
    let mut output = BTreeMap::new();
    for idx in 0..table.len {
        if trade_dates[idx] == Some(trade_date) {
            if let Some(ts_code) = ts_codes[idx].clone() {
                output.insert(ts_code, umr::finite(circ_mv[idx]));
            }
        }
    }
    Ok(output)
}

fn is_open_close_window(trade_time: &str) -> bool {
    intraday_time_in_range(trade_time, "09:31:00", "10:00:00")
        || intraday_time_in_range(trade_time, "14:31:00", "15:00:00")
}

fn amount_ratio(amount_yuan: f64, circ_mv_ten_thousand_yuan: Option<f64>) -> Option<f64> {
    let circ_mv = umr::finite(circ_mv_ten_thousand_yuan)?;
    if circ_mv <= 0.0 {
        return None;
    }
    Some(amount_yuan / (circ_mv * 10_000.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn amount_ratio_uses_yuan_over_circ_mv_in_ten_thousand_yuan() {
        assert_eq!(amount_ratio(10_000.0, Some(1.0)), Some(1.0));
        assert_eq!(amount_ratio(1.0, Some(0.0)), None);
    }

    #[test]
    fn open_close_window_keeps_first_and_last_thirty_minutes() {
        assert!(is_open_close_window("09:31:00"));
        assert!(is_open_close_window("10:00:00"));
        assert!(!is_open_close_window("10:01:00"));
        assert!(is_open_close_window("14:31:00"));
        assert!(is_open_close_window("15:00:00"));
    }
}

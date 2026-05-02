use std::collections::{BTreeMap, HashMap};

use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorRowKey, FactorSeries, FactorSpec,
    FactorValue, Frequency, IntradayDailyRawRequest, IntradayDailyRawSeries, IntradayDailyRawSpec,
    Lookback,
};
use crate::data::DataPool;
use crate::error::{err, Result};
use crate::factor::common::vector::clean;
use crate::factor::common::{
    clean_intraday_value, stock_minute_raw_spec, ClassificationLevel, ClassificationMap,
    DailyPanel, PanelColumn,
};
use crate::factor::Factor;
use crate::operators::{cs_zscore, ts_mean, ts_pctchg, ts_std_dev};

const MARKET_INDEX: &str = "000985.CSI";
const RAW_ID: &str = "daily_panic_intraday_volatility";
const RAW_VERSION: &str = "0.1.0";
const VERSION: &str = "0.1.0";
const WINDOW: usize = 20;

pub struct StockDailyGrassTreesPanic;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyGrassTreesPanic)
}

fn raw_spec() -> IntradayDailyRawSpec {
    stock_minute_raw_spec(RAW_ID, RAW_VERSION, &["close"], 1)
}

impl Factor for StockDailyGrassTreesPanic {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "grass_trees_panic".to_string(),
            aliases: Vec::new(),
            name: "Grass Trees Panic".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: [
                "price_volume",
                "return",
                "panic",
                "intraday",
                "moneyflow",
                "composite",
                "neutralize",
                "barra",
                "size",
                "sector",
                "daily",
                "FZZQ",
            ]
            .iter()
            .map(|value| value.to_string())
            .collect(),
            description: "Composite panic factor using decayed market panic, intraday volatility and small-order moneyflow ratio, neutralized by Barra SIZE and SW sector.".to_string(),
            dependencies: vec![
                DataRequest::new(DatasetId::StockDailyPv, &["close", "pre_close", "amount"]),
                DataRequest::new(
                    DatasetId::StockMoneyflow,
                    &["buy_sm_amount", "sell_sm_amount"],
                ),
                DataRequest::index_daily(MARKET_INDEX, &["close", "pre_close"]),
                DataRequest::new(DatasetId::StockBarraDaily, &["SIZE"]),
                DataRequest::new(DatasetId::StockSwClassification, &["l1_code"]),
            ],
            intraday_raw_dependencies: vec![IntradayDailyRawRequest::new(RAW_ID, WINDOW - 1)],
            lookback: Lookback { trading_days: 21 },
        }
    }

    fn intraday_raw_specs(&self) -> Vec<IntradayDailyRawSpec> {
        vec![raw_spec()]
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
            let ts_codes = table.required_utf8("ts_code")?;
            let trade_times = table.required_utf8("trade_time")?;
            let close = table.required_f64_cast("close")?;
            let mut grouped = BTreeMap::<String, Vec<usize>>::new();
            for idx in 0..table.len {
                let Some(ts_code) = ts_codes[idx].clone() else {
                    continue;
                };
                if trade_times[idx].is_none() {
                    continue;
                }
                grouped.entry(ts_code).or_default().push(idx);
            }

            for (ts_code, mut indices) in grouped {
                indices.sort_by(|left, right| trade_times[*left].cmp(&trade_times[*right]));
                values.push(FactorValue {
                    key: FactorRowKey::Daily {
                        trade_date: *trade_date,
                        ts_code,
                    },
                    value: intraday_return_volatility(&indices, &close),
                });
            }
        }

        Ok(Some(IntradayDailyRawSeries {
            spec: raw_spec(),
            values,
        }))
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let sector_map = ClassificationMap::from_table(
            data.daily(DatasetId::StockSwClassification)?,
            ClassificationLevel::Sector,
        )?;
        let panel = data.intraday_daily_raw_panel(RAW_ID)?;
        let pv_table = data.daily(DatasetId::StockDailyPv)?;
        let moneyflow_table = data.daily(DatasetId::StockMoneyflow)?;

        let close = panel.column_from_table(pv_table, "close")?;
        let pre_close = panel.column_from_table(pv_table, "pre_close")?;
        let amount = panel.column_from_table(pv_table, "amount")?;
        let buy_sm_amount = panel.column_from_table(moneyflow_table, "buy_sm_amount")?;
        let sell_sm_amount = panel.column_from_table(moneyflow_table, "sell_sm_amount")?;
        let intraday_volatility = panel.column(RAW_ID)?;
        let size = panel.column_from_table(data.daily(DatasetId::StockBarraDaily)?, "SIZE")?;

        let stock_return = close.zip_binary(&pre_close, ret)?;
        let index_panel = data.index_daily_panel(MARKET_INDEX)?;
        let index_return = index_panel
            .column("close")?
            .zip_binary(&index_panel.column("pre_close")?, ret)?;
        let market_return = expand_index_column(panel, index_panel, &index_return)?;
        let panic = stock_return.zip_binary(&market_return, panic_degree)?;
        let decayed_panic = panic.ts(decayed_positive_panic)?;
        let retail_ratio =
            buy_sm_amount.zip_ternary(&sell_sm_amount, &amount, small_trade_ratio)?;

        let panic_vol_score = decayed_panic.zip_binary(&intraday_volatility, multiply)?;
        let panic_vol_retail_score = panic_vol_score.zip_binary(&retail_ratio, multiply)?;
        let score = panic_vol_retail_score.zip_binary(&stock_return, multiply)?;

        let return_component = score
            .ts(|values| ts_mean(values, WINDOW, 5))?
            .cs(cs_zscore)?;
        let volatility_component = score
            .ts(|values| ts_std_dev(values, WINDOW, 5))?
            .cs(cs_zscore)?;
        let raw_factor = average_pair(&return_component, &volatility_component)?;
        let neutralized = raw_factor.cs_neutralize_regression_by_group(
            &[&size],
            None,
            |trade_date, ts_codes| sector_map.groups_for(trade_date, ts_codes),
        )?;

        Ok(neutralized.to_factor_series(self.spec()))
    }
}

fn expand_index_column(
    stock_panel: &DailyPanel,
    index_panel: &DailyPanel,
    index_column: &PanelColumn,
) -> Result<PanelColumn> {
    let index_instrument_count = index_panel.instruments().len();
    if index_instrument_count == 0 {
        return Err(err("index daily panel has no instruments"));
    }
    let mut by_date = HashMap::new();
    for (date_idx, trade_date) in index_panel.dates().iter().enumerate() {
        by_date.insert(
            *trade_date,
            index_column.values()[date_idx * index_instrument_count],
        );
    }

    let mut values = Vec::with_capacity(stock_panel.shape_len());
    for trade_date in stock_panel.dates() {
        let value = by_date.get(trade_date).copied().unwrap_or(None);
        for _ in stock_panel.instruments() {
            values.push(value);
        }
    }
    stock_panel.column_from_values(values)
}

fn intraday_return_volatility(indices: &[usize], close: &[Option<f64>]) -> Option<f64> {
    let close_series = indices
        .iter()
        .map(|idx| clean_intraday_value(close[*idx]))
        .collect::<Vec<_>>();
    let returns = ts_pctchg(&close_series, 1);
    mean_std(returns.into_iter().filter_map(clean)).map(|(_, std)| std)
}

fn decayed_positive_panic(values: &[Option<f64>]) -> Vec<Option<f64>> {
    let mut output = vec![None; values.len()];
    for idx in 2..values.len() {
        let (Some(current), Some(prev_1), Some(prev_2)) = (
            clean(values[idx]),
            clean(values[idx - 1]),
            clean(values[idx - 2]),
        ) else {
            continue;
        };
        let decayed = current - (prev_1 + prev_2) / 2.0;
        if decayed > 0.0 {
            output[idx] = Some(decayed);
        }
    }
    output
}

fn average_pair(left: &PanelColumn, right: &PanelColumn) -> Result<PanelColumn> {
    left.zip_binary(right, |left, right| match (clean(left), clean(right)) {
        (Some(left), Some(right)) => Some((left + right) / 2.0),
        _ => None,
    })
}

fn ret(numerator: Option<f64>, denominator: Option<f64>) -> Option<f64> {
    match (clean(numerator), clean(denominator)) {
        (Some(numerator), Some(denominator)) if denominator.abs() > f64::EPSILON => {
            Some(numerator / denominator - 1.0)
        }
        _ => None,
    }
}

fn panic_degree(stock_return: Option<f64>, market_return: Option<f64>) -> Option<f64> {
    match (clean(stock_return), clean(market_return)) {
        (Some(stock_return), Some(market_return)) => {
            let denominator = stock_return.abs() + market_return.abs() + 0.1;
            Some((stock_return - market_return).abs() / denominator)
        }
        _ => None,
    }
}

fn small_trade_ratio(
    buy_sm_amount: Option<f64>,
    sell_sm_amount: Option<f64>,
    amount: Option<f64>,
) -> Option<f64> {
    match (
        clean(buy_sm_amount),
        clean(sell_sm_amount),
        clean(amount).map(|value| value / 10.0),
    ) {
        (Some(buy), Some(sell), Some(total_amount)) if total_amount.abs() > f64::EPSILON => {
            Some(((buy + sell) / 2.0) / total_amount)
        }
        _ => None,
    }
}

fn multiply(left: Option<f64>, right: Option<f64>) -> Option<f64> {
    match (clean(left), clean(right)) {
        (Some(left), Some(right)) => Some(left * right),
        _ => None,
    }
}

fn mean_std(values: impl IntoIterator<Item = f64>) -> Option<(f64, f64)> {
    let values = values
        .into_iter()
        .filter(|value| !value.is_nan())
        .collect::<Vec<_>>();
    if values.is_empty() {
        return None;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / values.len() as f64;
    Some((mean, variance.sqrt()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: Option<f64>, expected: Option<f64>) {
        match (actual, expected) {
            (Some(actual), Some(expected)) => assert!((actual - expected).abs() < 1e-10),
            (None, None) => {}
            _ => panic!("expected {:?}, got {:?}", expected, actual),
        }
    }

    #[test]
    fn panic_degree_uses_relative_deviation_over_market_benchmark() {
        assert_close(
            panic_degree(Some(0.03), Some(0.01)),
            Some(0.02 / (0.03_f64.abs() + 0.01_f64.abs() + 0.1)),
        );
        assert_eq!(panic_degree(Some(f64::NAN), Some(0.01)), None);
    }

    #[test]
    fn decayed_positive_panic_keeps_only_positive_surprises() {
        let values = vec![Some(0.2), Some(0.4), Some(0.5), Some(0.2), Some(0.7)];
        let output = decayed_positive_panic(&values);
        assert_eq!(output[0], None);
        assert_eq!(output[1], None);
        assert_close(output[2], Some(0.2));
        assert_eq!(output[3], None);
        assert_close(output[4], Some(0.35));
    }

    #[test]
    fn small_trade_ratio_uses_mean_buy_sell_over_amount_divided_by_ten() {
        assert_close(
            small_trade_ratio(Some(10.0), Some(14.0), Some(120.0)),
            Some(1.0),
        );
        assert_eq!(small_trade_ratio(Some(10.0), Some(14.0), Some(0.0)), None);
    }

    #[test]
    fn intraday_return_volatility_uses_minute_close_returns() {
        let close = vec![Some(10.0), Some(11.0), Some(11.0), Some(12.1)];
        let indices = vec![0, 1, 2, 3];
        let expected_returns = vec![0.1, 0.0, 0.1];
        let mean = expected_returns.iter().sum::<f64>() / expected_returns.len() as f64;
        let expected_std = (expected_returns
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            / expected_returns.len() as f64)
            .sqrt();
        assert_close(
            intraday_return_volatility(&indices, &close),
            Some(expected_std),
        );
    }
}

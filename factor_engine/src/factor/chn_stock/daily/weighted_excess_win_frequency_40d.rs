use std::collections::HashMap;

use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::{err, Result};
use crate::factor::common::vector::clean;
use crate::factor::common::{ClassificationLevel, ClassificationMap, DailyPanel, PanelColumn};
use crate::factor::Factor;
use crate::operators::{cs_neutralize, ts_ew_sum, ts_pctchg};

const MARKET_INDEX: &str = "000985.CSI";

pub struct StockDailyWeightedExcessWinFrequency40d;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyWeightedExcessWinFrequency40d)
}

impl Factor for StockDailyWeightedExcessWinFrequency40d {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "weighted_excess_win_frequency_40d".to_string(),
            aliases: Vec::new(),
            name: "Weighted Excess Win Frequency 40D".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: "0.1.0".to_string(),
            tags: [
                "price_volume",
                "return",
                "excess_return",
                "ewm",
                "neutralize",
                "daily",
                "XNZQ",
            ]
            .iter()
            .map(|value| value.to_string())
            .collect(),
            description:
                "40-day half-life 20 EW sum of excess win flags over 000985.CSI, neutralized by SW sector."
                    .to_string(),
            dependencies: vec![
                DataRequest::new(DatasetId::StockDailyPv, &["close"]),
                DataRequest::new(DatasetId::StockAdjFactor, &["adj_factor"]),
                DataRequest::index_daily(MARKET_INDEX, &["close", "pre_close"]),
                DataRequest::new(DatasetId::StockSwClassification, &["l1_code"]),
            ],
            intraday_raw_dependencies: Vec::new(),
            lookback: Lookback { trading_days: 40 },
        }
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let panel = data.daily_panel(DatasetId::StockDailyPv)?;
        let adj_factor =
            panel.column_from_table(data.daily(DatasetId::StockAdjFactor)?, "adj_factor")?;
        let adj_close = panel.column("close")?.zip_binary(&adj_factor, mul)?;
        let stock_return = adj_close.ts(|values| ts_pctchg(values, 1))?;

        let index_panel = data.index_daily_panel(MARKET_INDEX)?;
        let index_return = index_panel
            .column("close")?
            .zip_binary(&index_panel.column("pre_close")?, ret)?;
        let market_return = expand_index_column(panel, index_panel, &index_return)?;

        let win_flag = stock_return.zip_binary(&market_return, |stock, market| {
            match (clean(stock), clean(market)) {
                (Some(stock), Some(market)) => Some(if stock - market > 0.02 { 1.0 } else { 0.0 }),
                _ => None,
            }
        })?;
        let raw = win_flag.ts(|values| ts_ew_sum(values, 40, 40, 20.0))?;
        let sector_map = ClassificationMap::from_table(
            data.daily(DatasetId::StockSwClassification)?,
            ClassificationLevel::Sector,
        )?;
        let factor = raw.cs_by_group(
            |trade_date, ts_codes| sector_map.groups_for(trade_date, ts_codes),
            cs_neutralize,
        )?;
        Ok(factor.to_factor_series(self.spec()))
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

fn mul(left: Option<f64>, right: Option<f64>) -> Option<f64> {
    match (clean(left), clean(right)) {
        (Some(left), Some(right)) => Some(left * right),
        _ => None,
    }
}

fn ret(close: Option<f64>, pre_close: Option<f64>) -> Option<f64> {
    match (clean(close), clean(pre_close)) {
        (Some(close), Some(pre_close)) if pre_close.abs() > f64::EPSILON => {
            Some(close / pre_close - 1.0)
        }
        _ => None,
    }
}

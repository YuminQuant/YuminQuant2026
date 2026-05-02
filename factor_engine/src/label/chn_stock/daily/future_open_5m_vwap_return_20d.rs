use std::collections::HashMap;

use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, Frequency, LabelSeries, LabelSpec, Lookahead,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::{clean_intraday_value, intraday_time_in_range, PanelColumn};
use crate::label::Label;

pub struct StockDailyFutureOpen5mVwapReturn20d;

pub fn create() -> Box<dyn Label> {
    Box::new(StockDailyFutureOpen5mVwapReturn20d)
}

impl Label for StockDailyFutureOpen5mVwapReturn20d {
    fn spec(&self) -> LabelSpec {
        LabelSpec {
            id: "future_open_5m_vwap_return_20d".to_string(),
            aliases: Vec::new(),
            name: "Stock future 20-day opening 5-minute adjusted VWAP".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: "0.2.0".to_string(),
            tags: [
                "label",
                "future_return",
                "adjusted",
                "vwap",
                "minute_vwap",
                "open_5m_vwap",
                "daily",
            ]
            .iter()
            .map(|value| value.to_string())
            .collect(),
            description:
                "Future adjusted return from t+1 09:31-09:35 VWAP to t+21 09:31-09:35 VWAP."
                    .to_string(),
            dependencies: vec![
                DataRequest::new(DatasetId::StockDailyPv, &["close"]),
                DataRequest::new(DatasetId::StockAdjFactor, &["adj_factor"]),
                DataRequest::new(DatasetId::StockMinute1m, &["amount", "vol"]),
            ],
            lookahead: Lookahead { trading_days: 21 },
        }
    }

    fn compute(&self, context: &FactorContext, data: &DataPool) -> Result<LabelSeries> {
        let adjusted_vwap = adjusted_opening_vwap(data, context, "09:31:00", "09:35:00")?;
        let label = adjusted_vwap.ts(|values| future_return(values, 21))?;
        Ok(label.to_label_series(self.spec()))
    }
}

fn adjusted_opening_vwap(
    data: &DataPool,
    context: &FactorContext,
    start_time: &str,
    end_time: &str,
) -> Result<PanelColumn> {
    let panel = data.daily_panel(DatasetId::StockDailyPv)?;
    let adj_factor =
        panel.column_from_table(data.daily(DatasetId::StockAdjFactor)?, "adj_factor")?;
    let instrument_count = panel.instruments().len();
    let date_index = panel
        .dates()
        .iter()
        .enumerate()
        .map(|(idx, date)| (*date, idx))
        .collect::<HashMap<_, _>>();
    let instrument_index = panel
        .instruments()
        .iter()
        .enumerate()
        .map(|(idx, code)| (code.as_str(), idx))
        .collect::<HashMap<_, _>>();
    let mut values = vec![None; panel.shape_len()];

    for trade_date in &context.load_dates {
        let Some(date_idx) = date_index.get(trade_date).copied() else {
            continue;
        };
        let Some(table) = data.minute(DatasetId::StockMinute1m, *trade_date) else {
            continue;
        };
        let ts_code = table.required_utf8("ts_code")?;
        let trade_time = table.required_utf8("trade_time")?;
        let amount = table.required_f64_cast("amount")?;
        let vol = table.required_f64_cast("vol")?;
        let mut sums: HashMap<&str, (f64, f64)> = HashMap::new();

        for row in 0..table.len {
            let Some(time) = trade_time[row].as_deref() else {
                continue;
            };
            if !intraday_time_in_range(time, start_time, end_time) {
                continue;
            }
            let (Some(amount), Some(vol)) = (
                clean_intraday_value(amount[row]),
                clean_intraday_value(vol[row]),
            ) else {
                continue;
            };
            if vol <= 0.0 {
                continue;
            }
            let Some(code) = ts_code[row].as_deref() else {
                continue;
            };
            let entry = sums.entry(code).or_insert((0.0, 0.0));
            entry.0 += amount;
            entry.1 += vol;
        }

        for (code, (amount_sum, vol_sum)) in sums {
            if vol_sum.abs() <= f64::EPSILON {
                continue;
            }
            let Some(inst_idx) = instrument_index.get(code).copied() else {
                continue;
            };
            values[date_idx * instrument_count + inst_idx] = Some(amount_sum * 10.0 / vol_sum);
        }
    }

    let raw_vwap = panel.column_from_values(values)?;
    raw_vwap.zip_binary(&adj_factor, adjusted_value)
}

fn future_return(values: &[Option<f64>], end_offset: usize) -> Vec<Option<f64>> {
    let mut output = vec![None; values.len()];
    for idx in 0..values.len() {
        let Some(start) = values.get(idx + 1).and_then(|value| clean_value(*value)) else {
            continue;
        };
        if start.abs() <= f64::EPSILON {
            continue;
        }
        let Some(end) = values
            .get(idx + end_offset)
            .and_then(|value| clean_value(*value))
        else {
            continue;
        };
        output[idx] = Some(end / start - 1.0);
    }
    output
}

fn adjusted_value(price: Option<f64>, adj_factor: Option<f64>) -> Option<f64> {
    let (Some(price), Some(adj_factor)) = (clean_value(price), clean_value(adj_factor)) else {
        return None;
    };
    Some(price * adj_factor)
}

fn clean_value(value: Option<f64>) -> Option<f64> {
    value.filter(|value| !value.is_nan())
}

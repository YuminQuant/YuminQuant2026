use std::collections::HashMap;

use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorRowKey, FactorValue, Frequency,
    IntradayDailyRawRequest, IntradayDailyRawSeries, IntradayDailyRawSpec, LabelSeries, LabelSpec,
    Lookahead,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::{
    clean_intraday_value, intraday_time_in_range, minute_vwap_from_amount_vol,
};
use crate::label::Label;

const RAW_ID: &str = "open_10m_vwap";
const RAW_VERSION: &str = "0.2.0";
const START_TIME: &str = "09:31:00";
const END_TIME: &str = "09:40:00";

pub struct StockDailyFutureOpen10mVwapReturn1d;

pub fn create() -> Box<dyn Label> {
    Box::new(StockDailyFutureOpen10mVwapReturn1d)
}

impl Label for StockDailyFutureOpen10mVwapReturn1d {
    fn spec(&self) -> LabelSpec {
        LabelSpec {
            id: "future_open_10m_vwap_return_1d".to_string(),
            aliases: Vec::new(),
            name: "Stock future 1-day opening 10-minute adjusted VWAP".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: "0.4.0".to_string(),
            tags: [
                "label",
                "future_return",
                "adjusted",
                "vwap",
                "minute_vwap",
                "open_10m_vwap",
                "daily",
            ]
            .iter()
            .map(|value| value.to_string())
            .collect(),
            description:
                "Future adjusted return from t+1 09:31-09:40 VWAP to t+2 09:31-09:40 VWAP."
                    .to_string(),
            dependencies: vec![DataRequest::new(DatasetId::StockAdjFactor, &["adj_factor"])],
            lookahead: Lookahead { trading_days: 2 },
        }
    }

    fn intraday_raw_specs(&self) -> Vec<IntradayDailyRawSpec> {
        vec![raw_spec()]
    }

    fn intraday_raw_dependencies(&self) -> Vec<IntradayDailyRawRequest> {
        vec![IntradayDailyRawRequest::new(RAW_ID, 0)]
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
        Ok(Some(compute_opening_vwap_raw(
            raw_spec(),
            context,
            data,
            START_TIME,
            END_TIME,
        )?))
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<LabelSeries> {
        let panel = data.intraday_daily_raw_panel(RAW_ID)?;
        let raw_vwap = panel.column(RAW_ID)?;
        let adj_factor =
            panel.column_from_table(data.daily(DatasetId::StockAdjFactor)?, "adj_factor")?;
        let adjusted_vwap = raw_vwap.zip_binary(&adj_factor, adjusted_value)?;
        let label = adjusted_vwap.ts(|values| future_return(values, 2))?;
        Ok(label.to_label_series(self.spec()))
    }
}

fn raw_spec() -> IntradayDailyRawSpec {
    IntradayDailyRawSpec {
        raw_id: RAW_ID.to_string(),
        version: RAW_VERSION.to_string(),
        asset_class: AssetClass::Stock,
        source_dataset: DatasetId::StockMinute1m,
        columns: vec!["amount".to_string(), "vol".to_string()],
        window_days: 1,
    }
}

fn compute_opening_vwap_raw(
    spec: IntradayDailyRawSpec,
    context: &FactorContext,
    data: &DataPool,
    start_time: &str,
    end_time: &str,
) -> Result<IntradayDailyRawSeries> {
    let mut values = Vec::new();
    for trade_date in &context.target_dates {
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
            values.push(FactorValue {
                key: FactorRowKey::Daily {
                    trade_date: *trade_date,
                    ts_code: code.to_string(),
                },
                value: minute_vwap_from_amount_vol(Some(amount_sum), Some(vol_sum)),
            });
        }
    }

    Ok(IntradayDailyRawSeries { spec, values })
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

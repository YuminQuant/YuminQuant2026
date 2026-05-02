use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, Frequency, LabelSeries, LabelSpec, Lookahead,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::label::Label;

pub struct StockDailyFutureVwapReturn20d;

pub fn create() -> Box<dyn Label> {
    Box::new(StockDailyFutureVwapReturn20d)
}

impl Label for StockDailyFutureVwapReturn20d {
    fn spec(&self) -> LabelSpec {
        LabelSpec {
            id: "future_vwap_return_20d".to_string(),
            aliases: Vec::new(),
            name: "Stock future 20-day adjusted daily VWAP".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: "0.2.0".to_string(),
            tags: ["label", "future_return", "adjusted", "daily_vwap", "daily"]
                .iter()
                .map(|value| value.to_string())
                .collect(),
            description:
                "Future adjusted daily VWAP return from t+1 full-day VWAP to t+21 full-day VWAP."
                    .to_string(),
            dependencies: vec![
                DataRequest::new(DatasetId::StockDailyPv, &["amount", "vol"]),
                DataRequest::new(DatasetId::StockAdjFactor, &["adj_factor"]),
            ],
            lookahead: Lookahead { trading_days: 21 },
        }
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<LabelSeries> {
        let panel = data.daily_panel(DatasetId::StockDailyPv)?;
        let amount = panel.column("amount")?;
        let vol = panel.column("vol")?;
        let vwap = amount.zip_binary(&vol, daily_vwap_value)?;
        let adj_factor =
            panel.column_from_table(data.daily(DatasetId::StockAdjFactor)?, "adj_factor")?;
        let adjusted_vwap = vwap.zip_binary(&adj_factor, adjusted_value)?;
        let label = adjusted_vwap.ts(|values| future_return(values, 21))?;
        Ok(label.to_label_series(self.spec()))
    }
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

fn daily_vwap_value(amount: Option<f64>, vol: Option<f64>) -> Option<f64> {
    let (Some(amount), Some(vol)) = (clean_value(amount), clean_value(vol)) else {
        return None;
    };
    if vol.abs() <= f64::EPSILON {
        return None;
    }
    Some(amount * 10.0 / vol)
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

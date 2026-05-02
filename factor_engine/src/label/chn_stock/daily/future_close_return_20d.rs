use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, Frequency, LabelSeries, LabelSpec, Lookahead,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::label::Label;

pub struct StockDailyFutureCloseReturn20d;

pub fn create() -> Box<dyn Label> {
    Box::new(StockDailyFutureCloseReturn20d)
}

impl Label for StockDailyFutureCloseReturn20d {
    fn spec(&self) -> LabelSpec {
        LabelSpec {
            id: "future_close_return_20d".to_string(),
            aliases: Vec::new(),
            name: "Stock future 20-day adjusted close-to-close".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: "0.2.0".to_string(),
            tags: [
                "label",
                "future_return",
                "adjusted",
                "close_to_close",
                "daily",
            ]
            .iter()
            .map(|value| value.to_string())
            .collect(),
            description: "Future adjusted close-to-close return from t+1 close to t+21 close."
                .to_string(),
            dependencies: vec![
                DataRequest::new(DatasetId::StockDailyPv, &["close"]),
                DataRequest::new(DatasetId::StockAdjFactor, &["adj_factor"]),
            ],
            lookahead: Lookahead { trading_days: 21 },
        }
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<LabelSeries> {
        let panel = data.daily_panel(DatasetId::StockDailyPv)?;
        let close = panel.column("close")?;
        let adj_factor =
            panel.column_from_table(data.daily(DatasetId::StockAdjFactor)?, "adj_factor")?;
        let adjusted_close = close.zip_binary(&adj_factor, adjusted_value)?;
        let label = adjusted_close.ts(|values| future_return(values, 21))?;
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

fn adjusted_value(price: Option<f64>, adj_factor: Option<f64>) -> Option<f64> {
    let (Some(price), Some(adj_factor)) = (clean_value(price), clean_value(adj_factor)) else {
        return None;
    };
    Some(price * adj_factor)
}

fn clean_value(value: Option<f64>) -> Option<f64> {
    value.filter(|value| !value.is_nan())
}

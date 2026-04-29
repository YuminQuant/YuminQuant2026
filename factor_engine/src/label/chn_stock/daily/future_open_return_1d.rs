use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, Frequency, LabelSeries, LabelSpec, Lookahead,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::vector::clean;
use crate::label::Label;

pub struct StockDailyFutureOpenReturn1d;

pub fn create() -> Box<dyn Label> {
    Box::new(StockDailyFutureOpenReturn1d)
}

impl Label for StockDailyFutureOpenReturn1d {
    fn spec(&self) -> LabelSpec {
        LabelSpec {
            id: "future_open_return_1d".to_string(),
            aliases: Vec::new(),
            name: "Stock future 1-day open-to-open return".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: "0.1.0".to_string(),
            tags: ["label", "future_return", "open_to_open", "daily"]
                .iter()
                .map(|value| value.to_string())
                .collect(),
            description: "Future return from t+1 open to t+2 open.".to_string(),
            dependencies: vec![DataRequest::new(DatasetId::StockDailyPv, &["open"])],
            lookahead: Lookahead { trading_days: 2 },
        }
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<LabelSeries> {
        let panel = data.daily_panel(DatasetId::StockDailyPv)?;
        let open = panel.column("open")?;
        let label = open.ts(|values| future_open_return(values, 2))?;
        Ok(label.to_label_series(self.spec()))
    }
}

fn future_open_return(values: &[Option<f64>], end_offset: usize) -> Vec<Option<f64>> {
    let mut output = vec![None; values.len()];
    for idx in 0..values.len() {
        let Some(start) = values.get(idx + 1).and_then(|value| clean(*value)) else {
            continue;
        };
        let Some(end) = values.get(idx + end_offset).and_then(|value| clean(*value)) else {
            continue;
        };
        if start.abs() > f64::EPSILON {
            output[idx] = Some(end / start - 1.0);
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::future_open_return;

    #[test]
    fn computes_t1_open_to_t2_open_return() {
        let values = vec![Some(10.0), Some(11.0), Some(12.1), Some(13.0)];
        let output = future_open_return(&values, 2);

        assert!((output[0].unwrap() - 0.1).abs() < 1e-10);
        assert_eq!(output[2], None);
    }
}

use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, Frequency, IntradayDailyRawRequest,
    LabelSeries, LabelSpec, Lookahead,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::label::Label;

const RAW_ID: &str = "open_10m_vwap";

pub struct StockDailyFutureOpen10mVwapReturn5d;

pub fn create() -> Box<dyn Label> {
    Box::new(StockDailyFutureOpen10mVwapReturn5d)
}

impl Label for StockDailyFutureOpen10mVwapReturn5d {
    fn spec(&self) -> LabelSpec {
        LabelSpec {
            id: "future_open_10m_vwap_return_5d".to_string(),
            aliases: Vec::new(),
            name: "Stock future 5-day opening 10-minute adjusted VWAP".to_string(),
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
                "Future adjusted return from t+1 09:31-09:40 VWAP to t+6 09:31-09:40 VWAP."
                    .to_string(),
            dependencies: vec![DataRequest::new(DatasetId::StockAdjFactor, &["adj_factor"])],
            lookahead: Lookahead { trading_days: 6 },
        }
    }

    fn intraday_raw_dependencies(&self) -> Vec<IntradayDailyRawRequest> {
        vec![IntradayDailyRawRequest::new(RAW_ID, 0)]
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<LabelSeries> {
        let panel = data.intraday_daily_raw_panel(RAW_ID)?;
        let raw_vwap = panel.column(RAW_ID)?;
        let adj_factor =
            panel.column_from_table(data.daily(DatasetId::StockAdjFactor)?, "adj_factor")?;
        let adjusted_vwap = raw_vwap.zip_binary(&adj_factor, adjusted_value)?;
        let label = adjusted_vwap.ts(|values| future_return(values, 6))?;
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

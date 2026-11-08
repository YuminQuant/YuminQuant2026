pub mod future_open_return_10m;
pub mod future_open_return_1m;
pub mod future_open_return_30m;
pub mod future_open_return_5m;

use rayon::prelude::*;

use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorRowKey, FactorValue, Frequency,
    LabelSeries, LabelSpec, Lookahead,
};
use crate::data::DataPool;
use crate::error::{err, Result};
use crate::factor::future::minute_input::{boundary_ratio, intraday_boundary};
use crate::label::Label;

struct OpenReturn(usize);

fn create_open_return(minutes: usize) -> Box<dyn Label> {
    Box::new(OpenReturn(minutes))
}

impl Label for OpenReturn {
    fn spec(&self) -> LabelSpec {
        let minutes = self.0;
        LabelSpec {
            id: format!("future_open_return_{minutes}m"),
            aliases: Vec::new(),
            name: format!("Futures {minutes}-bar open-to-open return"),
            asset_class: AssetClass::Future,
            frequency: Frequency::Minute1,
            version: "0.1.0".into(),
            tags: ["label", "future_return", "open_to_open", "minute"]
                .map(str::to_string)
                .to_vec(),
            description: format!(
                "Forward return in basis points from the open of bar m+1 to the open of bar m+{} within the same trading day. Insufficient remaining bars yield a null label.",
                minutes + 1
            ),
            dependencies: vec![DataRequest::new(DatasetId::FutureMinute1m, &["open", "close"])],
            lookahead: Lookahead { trading_days: 0 },
        }
    }

    fn compute(&self, context: &FactorContext, data: &DataPool) -> Result<LabelSeries> {
        let options = data
            .future_minute_input()
            .ok_or_else(|| err("missing futures minute label input configuration"))?
            .options();
        let adjust = options.adjust_night || options.adjust_morning || options.adjust_noon;
        let days = context.target_dates.par_iter().map(|&date| {
            let table = data.minute(DatasetId::FutureMinute1m, date)
                .ok_or_else(|| err(format!("missing futures minute bars for {date}")))?;
            let day = data.future_minute_day_index(date)
                .ok_or_else(|| err(format!("missing futures minute day index for {date}")))?;
            if table.len == 0 {
                return Ok(Vec::new());
            }
            let times = table.required_utf8("trade_time")?;
            let opens = table.required_f64_cast("open")?;
            let closes = adjust.then(|| table.required_f64_cast("close")).transpose()?;
            let mut rows = Vec::with_capacity(table.len);
            for (code, indices) in &day.groups {
                let time = |row: usize| times[row].as_deref().expect("indexed minute timestamp");
                let exchange = code.split_once('.').map_or("", |(_, exchange)| exchange);
                let mut adjusted_opens = Vec::with_capacity(indices.len());
                let mut previous = None;
                let mut scale = 1.0;
                for &row in indices {
                    if let Some(prior) = previous {
                        if time(prior) == time(row) {
                            return Err(err(format!(
                                "duplicate minute timestamp {} for {code} on {date}", time(row)
                            )));
                        }
                        if let Some(closes) = &closes {
                            if intraday_boundary(options, exchange, time(prior), time(row)) {
                                scale *= boundary_ratio(closes[prior], opens[row]).unwrap_or(1.0);
                            }
                        }
                    }
                    adjusted_opens.push(opens[row].map(|open| open * scale).filter(|open| open.is_finite()));
                    previous = Some(row);
                }
                for (index, &row) in indices.iter().enumerate() {
                    let value = adjusted_opens.get(index + 1).copied().flatten()
                        .filter(|open| open.abs() > f64::EPSILON)
                        .zip(adjusted_opens.get(index + self.0 + 1).copied().flatten())
                        .map(|(start, end)| (end / start - 1.0) * 1e4)
                        .filter(|value| value.is_finite());
                    rows.push(FactorValue {
                        key: FactorRowKey::Minute {
                            trade_date: date,
                            trade_time: time(row).to_string(),
                            ts_code: code.clone(),
                        },
                        value,
                    });
                }
            }
            Ok(rows)
        }).collect::<Result<Vec<_>>>()?;
        Ok(LabelSeries {
            spec: self.spec(),
            values: days.into_iter().flatten().collect(),
        })
    }
}

use std::collections::BTreeMap;

use crate::core::{DatasetId, FactorContext, FactorRowKey, FactorSeries, FactorSpec, FactorValue};
use crate::data::DataPool;
use crate::error::{err, Result};

use super::collect_numeric_columns;

pub struct MinuteSeries {
    pub ts_code: String,
    pub trade_date: i32,
    pub trade_times: Vec<String>,
    columns: BTreeMap<String, Vec<Option<f64>>>,
}

impl MinuteSeries {
    pub fn column(&self, name: &str) -> Result<&[Option<f64>]> {
        self.columns.get(name).map(Vec::as_slice).ok_or_else(|| {
            err(format!(
                "missing minute column {} for {} on {}",
                name, self.ts_code, self.trade_date
            ))
        })
    }
}

pub fn compute_minute_by_instrument<F>(
    spec: FactorSpec,
    context: &FactorContext,
    data_pool: &DataPool,
    dataset: DatasetId,
    mut expr: F,
) -> Result<FactorSeries>
where
    F: FnMut(&MinuteSeries) -> Result<Vec<Option<f64>>>,
{
    let mut values = Vec::new();
    for trade_date in &context.target_dates {
        let Some(table) = data_pool.minute(dataset, *trade_date) else {
            continue;
        };
        let ts_codes = table.required_utf8("ts_code")?;
        let trade_times = table.required_utf8("trade_time")?;
        let trade_dates = table.required_i32("trade_date").ok();
        let value_columns =
            collect_numeric_columns(table, &["trade_date", "trade_time", "ts_code"])?;
        let mut grouped: BTreeMap<String, Vec<usize>> = BTreeMap::new();

        for idx in 0..table.len {
            if let Some(dates) = trade_dates {
                if dates[idx] != Some(*trade_date) {
                    continue;
                }
            }
            let (Some(ts_code), Some(_trade_time)) =
                (ts_codes[idx].clone(), trade_times[idx].clone())
            else {
                continue;
            };
            grouped.entry(ts_code).or_default().push(idx);
        }

        for (ts_code, mut indices) in grouped {
            indices.sort_by(|left, right| {
                trade_times[*left]
                    .as_ref()
                    .expect("grouped minute row has trade_time")
                    .cmp(
                        trade_times[*right]
                            .as_ref()
                            .expect("grouped minute row has trade_time"),
                    )
            });
            let times = indices
                .iter()
                .filter_map(|idx| trade_times[*idx].clone())
                .collect::<Vec<_>>();
            let columns = value_columns
                .iter()
                .map(|(name, column)| {
                    (
                        name.clone(),
                        indices.iter().map(|idx| column[*idx]).collect::<Vec<_>>(),
                    )
                })
                .collect::<BTreeMap<_, _>>();
            let series = MinuteSeries {
                ts_code: ts_code.clone(),
                trade_date: *trade_date,
                trade_times: times,
                columns,
            };
            let computed = expr(&series)?;
            if computed.len() != series.trade_times.len() {
                return Err(err(format!(
                    "factor {} returned {} values for {} on {}, expected {}",
                    spec.id,
                    computed.len(),
                    series.ts_code,
                    series.trade_date,
                    series.trade_times.len()
                )));
            }

            for idx in 0..series.trade_times.len() {
                values.push(FactorValue {
                    key: FactorRowKey::Minute {
                        trade_date: series.trade_date,
                        trade_time: series.trade_times[idx].clone(),
                        ts_code: series.ts_code.clone(),
                    },
                    value: computed[idx],
                });
            }
        }
    }

    Ok(FactorSeries { spec, values })
}

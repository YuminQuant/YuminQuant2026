use std::collections::{BTreeMap, BTreeSet};

use rayon::prelude::*;

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

/// Apply a rolling expression to one column in timestamp order for each contract.
/// Prepend the previous trading day's tail and emit only target dates.
pub fn compute_rolling_minute_by_instrument<F>(
    spec: FactorSpec,
    context: &FactorContext,
    data_pool: &DataPool,
    dataset: DatasetId,
    column: &str,
    history_bars: usize,
    expr: F,
) -> Result<FactorSeries>
where
    F: Fn(&[Option<f64>]) -> Vec<Option<f64>> + Sync,
{
    let targets = context
        .target_dates
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let mut previous = BTreeMap::<String, (String, Vec<Option<f64>>)>::new();
    let mut values = Vec::new();
    for &trade_date in &context.load_dates {
        let Some(table) = data_pool.minute(dataset, trade_date) else {
            previous.clear();
            continue;
        };
        let ts_codes = table.required_utf8("ts_code")?;
        let trade_times = table.required_utf8("trade_time")?;
        let trade_dates = table.required_i32("trade_date")?;
        let input = table.required_f64_cast(column)?;
        let mut grouped = BTreeMap::<&str, Vec<(usize, &str)>>::new();
        for idx in 0..table.len {
            if trade_dates[idx] != Some(trade_date) {
                continue;
            }
            if let (Some(code), Some(time)) =
                (ts_codes[idx].as_deref(), trade_times[idx].as_deref())
            {
                grouped.entry(code).or_default().push((idx, time));
            }
        }

        let emit = targets.contains(&trade_date);
        let compute = |(code, mut rows): (&str, Vec<(usize, &str)>)| {
            rows.sort_by_key(|&(_, time)| time);
            if let Some(pair) = rows.windows(2).find(|pair| pair[0].1 == pair[1].1) {
                return Err(err(format!(
                    "duplicate minute timestamp {} for {code} on {trade_date}",
                    pair[0].1
                )));
            }
            let first_time = rows[0].1;
            let last_time = rows[rows.len() - 1].1.to_string();
            let tail = rows[rows.len().saturating_sub(history_bars)..]
                .iter()
                .map(|&(idx, _)| input[idx])
                .collect::<Vec<_>>();
            let mut output = Vec::new();
            if emit {
                // Both daily slices are sorted. Check the boundary before
                // concatenating so a mispartitioned bar cannot enter the window.
                let history = match previous.get(code) {
                    Some((last_time, history)) if last_time.as_str() < first_time => {
                        history.as_slice()
                    }
                    Some((last_time, _)) => {
                        return Err(err(format!(
                            "warmup overlaps target minutes for {code} on {trade_date}: {last_time} >= {first_time}"
                        )));
                    }
                    None => &[][..],
                };
                let series = history
                    .iter()
                    .copied()
                    .chain(rows.iter().map(|&(idx, _)| input[idx]))
                    .collect::<Vec<_>>();
                let computed = expr(&series);
                if computed.len() != series.len() {
                    return Err(err(format!(
                        "factor {} returned {} values for {} on {}, expected {}",
                        spec.id,
                        computed.len(),
                        code,
                        trade_date,
                        series.len()
                    )));
                }
                output.reserve(rows.len());
                for ((_, time), value) in rows
                    .into_iter()
                    .zip(computed.into_iter().skip(history.len()))
                {
                    output.push(FactorValue {
                        key: FactorRowKey::Minute {
                            trade_date,
                            trade_time: time.to_string(),
                            ts_code: code.to_string(),
                        },
                        value,
                    });
                }
            }
            Ok((code.to_string(), (last_time, tail), output))
        };
        let groups = grouped.into_iter().collect::<Vec<_>>();
        let results = if crate::resources::parallel_work(table.len) && groups.len() > 1 {
            groups
                .into_par_iter()
                .map(compute)
                .collect::<Result<Vec<_>>>()?
        } else {
            groups
                .into_iter()
                .map(compute)
                .collect::<Result<Vec<_>>>()?
        };
        previous.clear();
        for (code, tail, output) in results {
            previous.insert(code, tail);
            values.extend(output);
        }
    }

    Ok(FactorSeries { spec, values })
}

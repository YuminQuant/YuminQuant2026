use std::collections::{BTreeMap, HashSet};

use crate::core::{FactorContext, FactorRowKey, FactorSeries, FactorSpec, FactorValue};
use crate::data::Table;
use crate::error::{err, Result};

use super::collect_numeric_columns;

pub struct DailySeries {
    pub ts_code: String,
    pub dates: Vec<i32>,
    pub(crate) columns: BTreeMap<String, Vec<Option<f64>>>,
}

impl DailySeries {
    pub fn column(&self, name: &str) -> Result<&[Option<f64>]> {
        self.columns.get(name).map(Vec::as_slice).ok_or_else(|| {
            err(format!(
                "missing daily column {} for {}",
                name, self.ts_code
            ))
        })
    }
}

pub fn compute_daily_by_instrument<F>(
    spec: FactorSpec,
    context: &FactorContext,
    table: &Table,
    mut expr: F,
) -> Result<FactorSeries>
where
    F: FnMut(&DailySeries) -> Result<Vec<Option<f64>>>,
{
    let target_dates = context.target_dates.iter().copied().collect::<HashSet<_>>();
    let ts_codes = table.required_utf8("ts_code")?;
    let trade_dates = table.required_i32("trade_date")?;
    let value_columns = collect_numeric_columns(table, &["trade_date", "ts_code"])?;
    let mut grouped: BTreeMap<String, Vec<usize>> = BTreeMap::new();

    for idx in 0..table.len {
        let (Some(ts_code), Some(_trade_date)) = (ts_codes[idx].clone(), trade_dates[idx]) else {
            continue;
        };
        grouped.entry(ts_code).or_default().push(idx);
    }

    let mut values = Vec::new();
    for (ts_code, mut indices) in grouped {
        indices.sort_by_key(|idx| trade_dates[*idx]);
        let dates = indices
            .iter()
            .filter_map(|idx| trade_dates[*idx])
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
        let series = DailySeries {
            ts_code: ts_code.clone(),
            dates,
            columns,
        };
        let computed = expr(&series)?;
        if computed.len() != series.dates.len() {
            return Err(err(format!(
                "factor {} returned {} values for {}, expected {}",
                spec.id,
                computed.len(),
                series.ts_code,
                series.dates.len()
            )));
        }

        for idx in 0..series.dates.len() {
            let trade_date = series.dates[idx];
            if !target_dates.contains(&trade_date) {
                continue;
            }
            values.push(FactorValue {
                key: FactorRowKey::Daily {
                    trade_date,
                    ts_code: series.ts_code.clone(),
                },
                value: computed[idx],
            });
        }
    }

    Ok(FactorSeries { spec, values })
}

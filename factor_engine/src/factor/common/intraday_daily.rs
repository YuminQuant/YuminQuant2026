use std::collections::{BTreeMap, BTreeSet, HashMap};

use crate::core::{DatasetId, FactorContext};
use crate::data::DataPool;
use crate::error::{err, Result};
use rayon::prelude::*;

use super::{collect_numeric_columns, DailyPanel, PanelColumn};

pub struct IntradaySeries {
    pub ts_code: String,
    pub trade_date: i32,
    pub trade_times: Vec<String>,
    columns: BTreeMap<String, Vec<Option<f64>>>,
}

pub struct IntradayWindow<'a> {
    pub ts_code: &'a str,
    pub trade_date: i32,
    pub series: Vec<&'a IntradaySeries>,
}

impl IntradaySeries {
    pub fn column(&self, name: &str) -> Result<&[Option<f64>]> {
        self.columns.get(name).map(Vec::as_slice).ok_or_else(|| {
            err(format!(
                "missing intraday column {} for {} on {}",
                name, self.ts_code, self.trade_date
            ))
        })
    }

    pub fn time_range_mask(&self, start: &str, end: &str) -> Vec<bool> {
        self.trade_times
            .iter()
            .map(|value| time_in_range(value, start, end))
            .collect()
    }

    pub fn time_in_range_at(&self, idx: usize, start: &str, end: &str) -> bool {
        self.trade_times
            .get(idx)
            .is_some_and(|value| time_in_range(value, start, end))
    }
}

pub struct IntradayDailyPanel {
    panel: DailyPanel,
    series: BTreeMap<(i32, String), IntradaySeries>,
}

impl IntradayDailyPanel {
    pub fn from_data_pool(
        context: &FactorContext,
        data_pool: &DataPool,
        dataset: DatasetId,
    ) -> Result<Self> {
        let load_dates = if context.load_dates.is_empty() {
            context.target_dates.as_slice()
        } else {
            context.load_dates.as_slice()
        };
        let mut date_set = load_dates.iter().copied().collect::<BTreeSet<_>>();
        let mut instrument_set = BTreeSet::new();
        let mut series = BTreeMap::new();

        for trade_date in load_dates {
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

            date_set.insert(*trade_date);
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
                instrument_set.insert(ts_code.clone());
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
                series.insert(
                    (*trade_date, ts_code.clone()),
                    IntradaySeries {
                        ts_code,
                        trade_date: *trade_date,
                        trade_times: times,
                        columns,
                    },
                );
            }
        }

        let dates = date_set.into_iter().collect::<Vec<_>>();
        let instruments = instrument_set.into_iter().collect::<Vec<_>>();
        let date_lookup = dates
            .iter()
            .enumerate()
            .map(|(idx, date)| (*date, idx))
            .collect::<HashMap<_, _>>();
        let instrument_lookup = instruments
            .iter()
            .enumerate()
            .map(|(idx, ts_code)| (ts_code.clone(), idx))
            .collect::<HashMap<_, _>>();
        let mut present = vec![false; dates.len() * instruments.len()];
        for (trade_date, ts_code) in series.keys() {
            if let (Some(date_idx), Some(instrument_idx)) =
                (date_lookup.get(trade_date), instrument_lookup.get(ts_code))
            {
                present[*date_idx * instruments.len() + *instrument_idx] = true;
            }
        }
        let panel = DailyPanel::from_index(dates, instruments, &context.target_dates, present)?;

        Ok(Self { panel, series })
    }

    pub fn aggregate_by_instrument<F>(&self, mut expr: F) -> Result<PanelColumn>
    where
        F: FnMut(&IntradaySeries) -> Result<Option<f64>>,
    {
        let mut values = Vec::with_capacity(self.panel.shape_len());
        for trade_date in self.panel.dates() {
            for ts_code in self.panel.instruments() {
                let value = match self.series.get(&(*trade_date, ts_code.clone())) {
                    Some(series) => expr(series)?,
                    None => None,
                };
                values.push(value);
            }
        }
        self.panel.column_from_values(values)
    }

    pub fn aggregate_by_instrument_parallel<F>(&self, expr: F) -> Result<PanelColumn>
    where
        F: Fn(&IntradaySeries) -> Result<Option<f64>> + Send + Sync,
    {
        let instrument_count = self.panel.instruments().len();
        let values = (0..self.panel.shape_len())
            .into_par_iter()
            .map(|offset| {
                let date_idx = offset / instrument_count;
                let instrument_idx = offset % instrument_count;
                let trade_date = self.panel.dates()[date_idx];
                let ts_code = &self.panel.instruments()[instrument_idx];
                match self.series.get(&(trade_date, ts_code.clone())) {
                    Some(series) => expr(series),
                    None => Ok(None),
                }
            })
            .collect::<Result<Vec<_>>>()?;
        self.panel.column_from_values(values)
    }

    pub fn aggregate_target_by_instrument_parallel<F>(&self, expr: F) -> Result<PanelColumn>
    where
        F: Fn(&IntradaySeries) -> Result<Option<f64>> + Send + Sync,
    {
        let instrument_count = self.panel.instruments().len();
        let values = (0..self.panel.shape_len())
            .into_par_iter()
            .map(|offset| {
                let date_idx = offset / instrument_count;
                let instrument_idx = offset % instrument_count;
                let trade_date = self.panel.dates()[date_idx];
                if !self.panel.is_target_date(trade_date) {
                    return Ok(None);
                }
                let ts_code = &self.panel.instruments()[instrument_idx];
                match self.series.get(&(trade_date, ts_code.clone())) {
                    Some(series) => expr(series),
                    None => Ok(None),
                }
            })
            .collect::<Result<Vec<_>>>()?;
        self.panel.column_from_values(values)
    }

    pub fn aggregate_by_instrument_window<F>(
        &self,
        window_days: usize,
        mut expr: F,
    ) -> Result<PanelColumn>
    where
        F: FnMut(&IntradayWindow<'_>) -> Result<Option<f64>>,
    {
        let mut values = Vec::with_capacity(self.panel.shape_len());
        for (date_idx, trade_date) in self.panel.dates().iter().enumerate() {
            for ts_code in self.panel.instruments() {
                let value = if window_days == 0 || date_idx + 1 < window_days {
                    None
                } else {
                    let start_idx = date_idx + 1 - window_days;
                    let mut window_series = Vec::with_capacity(window_days);
                    let mut complete = true;
                    for source_date in &self.panel.dates()[start_idx..=date_idx] {
                        match self.series.get(&(*source_date, ts_code.clone())) {
                            Some(series) => window_series.push(series),
                            None => {
                                complete = false;
                                break;
                            }
                        }
                    }
                    if complete {
                        expr(&IntradayWindow {
                            ts_code,
                            trade_date: *trade_date,
                            series: window_series,
                        })?
                    } else {
                        None
                    }
                };
                values.push(value);
            }
        }
        self.panel.column_from_values(values)
    }

    pub fn aggregate_by_instrument_window_parallel<F>(
        &self,
        window_days: usize,
        expr: F,
    ) -> Result<PanelColumn>
    where
        F: Fn(&IntradayWindow<'_>) -> Result<Option<f64>> + Send + Sync,
    {
        let instrument_count = self.panel.instruments().len();
        let values = (0..self.panel.shape_len())
            .into_par_iter()
            .map(|offset| {
                let date_idx = offset / instrument_count;
                let instrument_idx = offset % instrument_count;
                let trade_date = self.panel.dates()[date_idx];
                let ts_code = &self.panel.instruments()[instrument_idx];
                if window_days == 0 || date_idx + 1 < window_days {
                    return Ok(None);
                }

                let start_idx = date_idx + 1 - window_days;
                let mut window_series = Vec::with_capacity(window_days);
                for source_date in &self.panel.dates()[start_idx..=date_idx] {
                    let Some(series) = self.series.get(&(*source_date, ts_code.clone())) else {
                        return Ok(None);
                    };
                    window_series.push(series);
                }
                expr(&IntradayWindow {
                    ts_code,
                    trade_date,
                    series: window_series,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        self.panel.column_from_values(values)
    }

    pub fn aggregate_target_by_instrument_window_parallel<F>(
        &self,
        window_days: usize,
        expr: F,
    ) -> Result<PanelColumn>
    where
        F: Fn(&IntradayWindow<'_>) -> Result<Option<f64>> + Send + Sync,
    {
        let instrument_count = self.panel.instruments().len();
        let values = (0..self.panel.shape_len())
            .into_par_iter()
            .map(|offset| {
                let date_idx = offset / instrument_count;
                let instrument_idx = offset % instrument_count;
                let trade_date = self.panel.dates()[date_idx];
                if !self.panel.is_target_date(trade_date) {
                    return Ok(None);
                }
                let ts_code = &self.panel.instruments()[instrument_idx];
                if window_days == 0 || date_idx + 1 < window_days {
                    return Ok(None);
                }

                let start_idx = date_idx + 1 - window_days;
                let mut window_series = Vec::with_capacity(window_days);
                for source_date in &self.panel.dates()[start_idx..=date_idx] {
                    let Some(series) = self.series.get(&(*source_date, ts_code.clone())) else {
                        return Ok(None);
                    };
                    window_series.push(series);
                }
                expr(&IntradayWindow {
                    ts_code,
                    trade_date,
                    series: window_series,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        self.panel.column_from_values(values)
    }
}

pub fn intraday_time_in_range(value: &str, start: &str, end: &str) -> bool {
    let Some(time) = normalize_time(value) else {
        return false;
    };
    if time.len() == 5 {
        let start = start.get(..5).unwrap_or(start);
        let end = end.get(..5).unwrap_or(end);
        return time >= start && time <= end;
    }
    time >= start && time <= end
}

fn time_in_range(value: &str, start: &str, end: &str) -> bool {
    intraday_time_in_range(value, start, end)
}

fn normalize_time(value: &str) -> Option<&str> {
    let value = value.trim();
    if value.is_empty() {
        return None;
    }
    let value = value
        .rsplit_once(' ')
        .map(|(_, right)| right)
        .or_else(|| value.rsplit_once('T').map(|(_, right)| right))
        .unwrap_or(value);
    let value = value.trim();
    if value.len() >= 8 {
        return Some(&value[..8]);
    }
    (value.len() == 5).then_some(value)
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, HashMap};

    use crate::core::{
        AssetClass, DatasetId, FactorContext, FactorRowKey, FactorSpec, Frequency, Lookback,
    };
    use crate::data::{ColumnData, DataPool, Table};
    use crate::operators::ts_mean;

    use super::IntradayDailyPanel;

    fn context(load_dates: Vec<i32>, target_dates: Vec<i32>) -> FactorContext {
        FactorContext {
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            start_date: *target_dates.first().unwrap(),
            end_date: *target_dates.last().unwrap(),
            load_start_date: *load_dates.first().unwrap(),
            load_dates,
            target_dates,
        }
    }

    fn minute_table() -> Table {
        Table::new(BTreeMap::from([
            (
                "trade_date".to_string(),
                ColumnData::I32(vec![
                    Some(20260423),
                    Some(20260423),
                    Some(20260424),
                    Some(20260424),
                    Some(20260424),
                ]),
            ),
            (
                "ts_code".to_string(),
                ColumnData::Utf8(vec![
                    Some("000001.SZ".to_string()),
                    Some("000001.SZ".to_string()),
                    Some("000002.SZ".to_string()),
                    Some("000001.SZ".to_string()),
                    Some("000001.SZ".to_string()),
                ]),
            ),
            (
                "trade_time".to_string(),
                ColumnData::Utf8(vec![
                    Some("2026-04-23 09:31:00".to_string()),
                    Some("2026-04-23 09:32:00".to_string()),
                    Some("09:31:00".to_string()),
                    Some("2026-04-24 09:32:00".to_string()),
                    Some("2026-04-24 09:31:00".to_string()),
                ]),
            ),
            (
                "vol".to_string(),
                ColumnData::F64(vec![Some(1.0), Some(3.0), Some(10.0), Some(5.0), Some(1.0)]),
            ),
        ]))
        .expect("valid table")
    }

    fn single_stock_minute_table(trade_date: i32, ts_code: &str, volumes: &[f64]) -> Table {
        let trade_times = (0..volumes.len())
            .map(|idx| Some(format!("09:{:02}:00", 31 + idx)))
            .collect::<Vec<_>>();
        Table::new(BTreeMap::from([
            (
                "trade_date".to_string(),
                ColumnData::I32(vec![Some(trade_date); volumes.len()]),
            ),
            (
                "ts_code".to_string(),
                ColumnData::Utf8(vec![Some(ts_code.to_string()); volumes.len()]),
            ),
            ("trade_time".to_string(), ColumnData::Utf8(trade_times)),
            (
                "vol".to_string(),
                ColumnData::F64(volumes.iter().map(|value| Some(*value)).collect()),
            ),
        ]))
        .expect("valid table")
    }

    fn spec() -> FactorSpec {
        FactorSpec {
            id: "test_intraday_daily".to_string(),
            aliases: Vec::new(),
            name: "test".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: "0.1.0".to_string(),
            tags: Vec::new(),
            description: String::new(),
            dependencies: Vec::new(),
            intraday_raw_dependencies: Vec::new(),
            lookback: Lookback { trading_days: 0 },
        }
    }

    #[test]
    fn time_range_mask_accepts_full_timestamp_and_plain_time() {
        let pool = DataPool::from_minute_tables(HashMap::from([(
            (DatasetId::StockMinute1m, 20260424),
            minute_table(),
        )]));
        let context = context(vec![20260424], vec![20260424]);
        let panel = IntradayDailyPanel::from_data_pool(&context, &pool, DatasetId::StockMinute1m)
            .expect("panel");
        let output = panel
            .aggregate_by_instrument(|series| {
                Ok(Some(
                    series
                        .time_range_mask("09:31:00", "09:31:00")
                        .into_iter()
                        .filter(|value| *value)
                        .count() as f64,
                ))
            })
            .expect("aggregate");

        assert_eq!(output.values(), &[Some(1.0), Some(1.0)]);
    }

    #[test]
    fn aggregate_sorts_minutes_and_outputs_daily_keys() {
        let pool = DataPool::from_minute_tables(HashMap::from([(
            (DatasetId::StockMinute1m, 20260424),
            minute_table(),
        )]));
        let context = context(vec![20260424], vec![20260424]);
        let panel = IntradayDailyPanel::from_data_pool(&context, &pool, DatasetId::StockMinute1m)
            .expect("panel");
        let output = panel
            .aggregate_by_instrument(|series| {
                assert!(series.trade_times.windows(2).all(|pair| pair[0] <= pair[1]));
                Ok(Some(series.column("vol")?.iter().flatten().sum::<f64>()))
            })
            .expect("aggregate")
            .to_factor_series(spec());

        assert_eq!(output.values.len(), 2);
        assert_eq!(
            output.values[0].key,
            FactorRowKey::Daily {
                trade_date: 20260424,
                ts_code: "000001.SZ".to_string()
            }
        );
        assert_eq!(output.values[0].value, Some(6.0));
    }

    #[test]
    fn intraday_daily_output_can_use_daily_ts_warmup() {
        let pool = DataPool::from_minute_tables(HashMap::from([
            ((DatasetId::StockMinute1m, 20260423), minute_table()),
            ((DatasetId::StockMinute1m, 20260424), minute_table()),
        ]));
        let context = context(vec![20260423, 20260424], vec![20260424]);
        let panel = IntradayDailyPanel::from_data_pool(&context, &pool, DatasetId::StockMinute1m)
            .expect("panel");
        let factor = panel
            .aggregate_by_instrument(|series| {
                Ok(Some(series.column("vol")?.iter().flatten().sum()))
            })
            .expect("aggregate")
            .ts(|values| ts_mean(values, 2, 1))
            .expect("ts");

        let output = factor.to_factor_series(spec());
        assert_eq!(output.values.len(), 2);
        assert_eq!(output.values[0].key.trade_date(), 20260424);
    }

    #[test]
    fn aggregate_by_instrument_window_collects_recent_trading_days_for_stock() {
        let pool = DataPool::from_minute_tables(HashMap::from([
            (
                (DatasetId::StockMinute1m, 20260423),
                single_stock_minute_table(20260423, "000001.SZ", &[1.0]),
            ),
            (
                (DatasetId::StockMinute1m, 20260424),
                single_stock_minute_table(20260424, "000001.SZ", &[2.0]),
            ),
        ]));
        let context = context(vec![20260423, 20260424], vec![20260424]);
        let panel = IntradayDailyPanel::from_data_pool(&context, &pool, DatasetId::StockMinute1m)
            .expect("panel");
        let output = panel
            .aggregate_by_instrument_window(2, |window| {
                assert_eq!(window.ts_code, "000001.SZ");
                assert_eq!(window.trade_date, 20260424);
                assert_eq!(window.series.len(), 2);
                Ok(Some(
                    window
                        .series
                        .iter()
                        .map(|series| series.column("vol").unwrap()[0].unwrap())
                        .sum::<f64>(),
                ))
            })
            .expect("aggregate")
            .to_factor_series(spec());

        assert_eq!(output.values.len(), 1);
        assert_eq!(output.values[0].value, Some(3.0));
    }

    #[test]
    fn aggregate_by_instrument_window_outputs_none_when_any_window_day_is_missing() {
        let pool = DataPool::from_minute_tables(HashMap::from([(
            (DatasetId::StockMinute1m, 20260424),
            single_stock_minute_table(20260424, "000001.SZ", &[2.0]),
        )]));
        let context = context(vec![20260423, 20260424], vec![20260424]);
        let panel = IntradayDailyPanel::from_data_pool(&context, &pool, DatasetId::StockMinute1m)
            .expect("panel");
        let output = panel
            .aggregate_by_instrument_window(2, |_| Ok(Some(1.0)))
            .expect("aggregate")
            .to_factor_series(spec());

        assert_eq!(output.values.len(), 1);
        assert_eq!(output.values[0].value, None);
    }
}

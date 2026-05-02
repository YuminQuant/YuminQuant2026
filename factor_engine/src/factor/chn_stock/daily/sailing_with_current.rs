use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap};

use rayon::prelude::*;

use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorRowKey, FactorSeries, FactorSpec,
    FactorValue, Frequency, IntradayDailyRawAuxiliaryRequest, IntradayDailyRawRequest,
    IntradayDailyRawSeries, IntradayDailyRawSpec, Lookback,
};
use crate::data::{DataPool, Table};
use crate::error::Result;
use crate::factor::common::vector::clean;
use crate::factor::common::{
    clean_intraday_value, intraday_time_in_range, stock_minute_raw_spec, ClassificationLevel,
    ClassificationMap, DailyPanel, PanelColumn,
};
use crate::factor::Factor;
use crate::operators::{cs_zscore, ts_mean, ts_std_dev};

pub const FOLLOW_CURRENT_RAW_ID: &str = "daily_follow_current";
pub const LONE_GOOSE_RAW_ID: &str = "daily_lone_goose";

const RAW_VERSION: &str = "0.1.0";
const VERSION: &str = "0.1.1";
const WINDOW: usize = 20;
const CORR_BLOCK_SIZE: usize = 64;

pub struct StockDailySailingWithCurrent;

#[derive(Clone, Debug)]
struct MinuteMatrix {
    times: Vec<String>,
    codes: Vec<String>,
    close: Vec<Option<f64>>,
    amount: Vec<Option<f64>>,
}

impl MinuteMatrix {
    fn time_count(&self) -> usize {
        self.times.len()
    }

    fn code_count(&self) -> usize {
        self.codes.len()
    }

    fn offset(&self, time_idx: usize, code_idx: usize) -> usize {
        time_idx * self.code_count() + code_idx
    }
}

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailySailingWithCurrent)
}

fn raw_spec(raw_id: &str) -> IntradayDailyRawSpec {
    stock_minute_raw_spec(raw_id, RAW_VERSION, &["open", "close", "amount"], 1)
}

impl Factor for StockDailySailingWithCurrent {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "sailing_with_current".to_string(),
            aliases: Vec::new(),
            name: "Sailing With Current".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: [
                "price_volume",
                "return",
                "amount",
                "intraday",
                "minute_agg",
                "correlation",
                "composite",
                "neutralize",
                "barra",
                "size",
                "sector",
                "daily",
                "FZZQ",
            ]
            .iter()
            .map(|value| value.to_string())
            .collect(),
            description: "Composite intraday crowding factor from follow-current Spearman correlation and lone-goose amount correlation, neutralized by Barra SIZE and SW sector.".to_string(),
            dependencies: vec![
                DataRequest::new(DatasetId::StockBarraDaily, &["SIZE"]),
                DataRequest::new(DatasetId::StockSwClassification, &["l1_code"]),
            ],
            intraday_raw_dependencies: vec![
                IntradayDailyRawRequest::new(FOLLOW_CURRENT_RAW_ID, WINDOW - 1),
                IntradayDailyRawRequest::new(LONE_GOOSE_RAW_ID, WINDOW - 1),
            ],
            lookback: Lookback {
                trading_days: WINDOW - 1,
            },
        }
    }

    fn intraday_raw_specs(&self) -> Vec<IntradayDailyRawSpec> {
        vec![raw_spec(FOLLOW_CURRENT_RAW_ID), raw_spec(LONE_GOOSE_RAW_ID)]
    }

    fn intraday_raw_auxiliary_requirements(
        &self,
        raw_ids: &[String],
    ) -> Vec<IntradayDailyRawAuxiliaryRequest> {
        let wants_follow = raw_ids.iter().any(|raw_id| raw_id == FOLLOW_CURRENT_RAW_ID);
        if !wants_follow {
            return Vec::new();
        }
        vec![
            IntradayDailyRawAuxiliaryRequest::new(
                DataRequest::new(DatasetId::StockDailyPv, &["open", "close"]),
                WINDOW - 1,
            ),
            IntradayDailyRawAuxiliaryRequest::new(
                DataRequest::new(DatasetId::StockDailyBasic, &["circ_mv"]),
                0,
            ),
        ]
    }

    fn minute_compute(
        &self,
        raw_id: &str,
        context: &FactorContext,
        data: &DataPool,
    ) -> Result<Option<IntradayDailyRawSeries>> {
        let raw_ids = vec![raw_id.to_string()];
        Ok(self
            .minute_compute_many(&raw_ids, context, data)?
            .into_iter()
            .next())
    }

    fn minute_compute_many(
        &self,
        raw_ids: &[String],
        context: &FactorContext,
        data: &DataPool,
    ) -> Result<Vec<IntradayDailyRawSeries>> {
        let requested = raw_ids.iter().map(String::as_str).collect::<BTreeSet<_>>();
        let wants_follow = requested.contains(FOLLOW_CURRENT_RAW_ID);
        let wants_goose = requested.contains(LONE_GOOSE_RAW_ID);
        if !wants_follow && !wants_goose {
            return Ok(Vec::new());
        }

        let (reasonable_return, circ_mv) = if wants_follow {
            (
                Some(reasonable_return_map(data)?),
                Some(panel_column_map(
                    data.daily_panel(DatasetId::StockDailyBasic)?,
                    &data
                        .daily_panel(DatasetId::StockDailyBasic)?
                        .column("circ_mv")?,
                )),
            )
        } else {
            (None, None)
        };

        let mut follow_values = Vec::new();
        let mut goose_values = Vec::new();
        for trade_date in &context.target_dates {
            let Some(table) = data.minute(DatasetId::StockMinute1m, *trade_date) else {
                continue;
            };
            if wants_follow {
                follow_values.extend(daily_follow_current_values(
                    table,
                    *trade_date,
                    reasonable_return
                        .as_ref()
                        .expect("reasonable return map is loaded"),
                    circ_mv.as_ref().expect("circ_mv map is loaded"),
                )?);
            }
            if wants_goose {
                goose_values.extend(daily_lone_goose_values(table, *trade_date)?);
            }
        }

        let mut output = Vec::new();
        if wants_follow {
            output.push(IntradayDailyRawSeries {
                spec: raw_spec(FOLLOW_CURRENT_RAW_ID),
                values: follow_values,
            });
        }
        if wants_goose {
            output.push(IntradayDailyRawSeries {
                spec: raw_spec(LONE_GOOSE_RAW_ID),
                values: goose_values,
            });
        }
        Ok(output)
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let sector_map = ClassificationMap::from_table(
            data.daily(DatasetId::StockSwClassification)?,
            ClassificationLevel::Sector,
        )?;
        let panel = data.intraday_daily_raw_panel(FOLLOW_CURRENT_RAW_ID)?;
        let size = panel.column_from_table(data.daily(DatasetId::StockBarraDaily)?, "SIZE")?;

        let follow_raw = panel.column(FOLLOW_CURRENT_RAW_ID)?;
        let goose_raw = panel.column(LONE_GOOSE_RAW_ID)?;
        let follow_component = follow_current_component(panel, &follow_raw, WINDOW)?;
        let goose_mean20 = goose_raw.ts(|values| ts_mean(values, WINDOW, 1))?;
        let goose_std20 = goose_raw.ts(|values| ts_std_dev(values, WINDOW, 1))?;
        let goose_component =
            average_pair(&goose_mean20.cs(cs_zscore)?, &goose_std20.cs(cs_zscore)?)?;
        let raw_factor = subtract_pair(&follow_component, &goose_component)?;
        let neutralized = raw_factor.cs_neutralize_regression_by_group(
            &[&size],
            None,
            |trade_date, ts_codes| sector_map.groups_for(trade_date, ts_codes),
        )?;

        Ok(neutralized.to_factor_series(self.spec()))
    }
}

fn reasonable_return_map(data: &DataPool) -> Result<HashMap<(i32, String), Option<f64>>> {
    let panel = data.daily_panel(DatasetId::StockDailyPv)?;
    let intraday_return = panel
        .column("close")?
        .zip_binary(&panel.column("open")?, ret)?;
    let reasonable = intraday_return.ts(|values| ts_mean(values, WINDOW, 1))?;
    Ok(panel_column_map(panel, &reasonable))
}

fn panel_column_map(
    panel: &DailyPanel,
    column: &PanelColumn,
) -> HashMap<(i32, String), Option<f64>> {
    let mut output = HashMap::new();
    let code_count = panel.instruments().len();
    for (date_idx, trade_date) in panel.dates().iter().enumerate() {
        for (code_idx, ts_code) in panel.instruments().iter().enumerate() {
            output.insert(
                (*trade_date, ts_code.clone()),
                column.values()[date_idx * code_count + code_idx],
            );
        }
    }
    output
}

fn daily_follow_current_values(
    table: &Table,
    trade_date: i32,
    reasonable_return: &HashMap<(i32, String), Option<f64>>,
    circ_mv: &HashMap<(i32, String), Option<f64>>,
) -> Result<Vec<FactorValue>> {
    let ts_codes = table.required_utf8("ts_code")?;
    let trade_times = table.required_utf8("trade_time")?;
    let open = table.required_f64_cast("open")?;
    let close = table.required_f64_cast("close")?;
    let amount = table.required_f64_cast("amount")?;
    let mut grouped = BTreeMap::<String, Vec<usize>>::new();
    for idx in 0..table.len {
        let Some(ts_code) = ts_codes[idx].clone() else {
            continue;
        };
        if trade_times[idx].is_none() {
            continue;
        }
        grouped.entry(ts_code).or_default().push(idx);
    }

    let mut output = Vec::new();
    for (ts_code, mut indices) in grouped {
        indices.sort_by(|left, right| trade_times[*left].cmp(&trade_times[*right]));
        let value = follow_current_for_stock(
            &indices,
            trade_times,
            &open,
            &close,
            &amount,
            reasonable_return
                .get(&(trade_date, ts_code.clone()))
                .copied()
                .flatten(),
            circ_mv
                .get(&(trade_date, ts_code.clone()))
                .copied()
                .flatten(),
        );
        output.push(FactorValue {
            key: FactorRowKey::Daily {
                trade_date,
                ts_code,
            },
            value,
        });
    }
    Ok(output)
}

fn follow_current_for_stock(
    indices: &[usize],
    trade_times: &[Option<String>],
    open: &[Option<f64>],
    close: &[Option<f64>],
    amount: &[Option<f64>],
    reasonable_return: Option<f64>,
    circ_mv: Option<f64>,
) -> Option<f64> {
    let reasonable_return = clean(reasonable_return)?;
    let circ_mv = clean(circ_mv)?;
    if circ_mv.abs() <= f64::EPSILON {
        return None;
    }
    let first_idx = *indices.first()?;
    let day_open = clean_intraday_value(open[first_idx])?;
    if day_open.abs() <= f64::EPSILON {
        return None;
    }

    let mut high_amount = 0.0;
    let mut low_amount = 0.0;
    for idx in indices {
        let Some(trade_time) = trade_times[*idx].as_deref() else {
            continue;
        };
        if !intraday_time_in_range(trade_time, "09:31:00", "15:00:00") {
            continue;
        }
        let Some(close) = clean_intraday_value(close[*idx]) else {
            continue;
        };
        let Some(amount) = clean_intraday_value(amount[*idx]) else {
            continue;
        };
        let relative_return = close / day_open - 1.0;
        let amount = amount * 10.0;
        if relative_return > reasonable_return {
            high_amount += amount;
        } else if relative_return < reasonable_return {
            low_amount += amount;
        }
    }

    Some((high_amount - low_amount) / circ_mv)
}

fn daily_lone_goose_values(table: &Table, trade_date: i32) -> Result<Vec<FactorValue>> {
    let matrix = MinuteMatrix::from_table(table)?;
    let values = lone_goose_from_matrix(&matrix);
    Ok(matrix
        .codes
        .iter()
        .cloned()
        .zip(values)
        .map(|(ts_code, value)| FactorValue {
            key: FactorRowKey::Daily {
                trade_date,
                ts_code,
            },
            value,
        })
        .collect())
}

impl MinuteMatrix {
    fn from_table(table: &Table) -> Result<Self> {
        let ts_codes = table.required_utf8("ts_code")?;
        let trade_times = table.required_utf8("trade_time")?;
        let close = table.required_f64_cast("close")?;
        let amount = table.required_f64_cast("amount")?;
        let mut time_set = BTreeSet::new();
        let mut code_set = BTreeSet::new();
        for idx in 0..table.len {
            if let Some(time) = trade_times[idx].clone() {
                time_set.insert(time);
            }
            if let Some(code) = ts_codes[idx].clone() {
                code_set.insert(code);
            }
        }
        let times = time_set.into_iter().collect::<Vec<_>>();
        let codes = code_set.into_iter().collect::<Vec<_>>();
        let time_lookup = times
            .iter()
            .enumerate()
            .map(|(idx, value)| (value.clone(), idx))
            .collect::<HashMap<_, _>>();
        let code_lookup = codes
            .iter()
            .enumerate()
            .map(|(idx, value)| (value.clone(), idx))
            .collect::<HashMap<_, _>>();
        let shape_len = times.len() * codes.len();
        let mut close_values = vec![None; shape_len];
        let mut amount_values = vec![None; shape_len];
        for idx in 0..table.len {
            let (Some(time), Some(code)) = (trade_times[idx].clone(), ts_codes[idx].clone()) else {
                continue;
            };
            let (Some(time_idx), Some(code_idx)) = (
                time_lookup.get(&time).copied(),
                code_lookup.get(&code).copied(),
            ) else {
                continue;
            };
            let offset = time_idx * codes.len() + code_idx;
            close_values[offset] = clean_intraday_value(close[idx]);
            amount_values[offset] = clean_intraday_value(amount[idx]);
        }
        Ok(Self {
            times,
            codes,
            close: close_values,
            amount: amount_values,
        })
    }
}

fn lone_goose_from_matrix(matrix: &MinuteMatrix) -> Vec<Option<f64>> {
    let time_count = matrix.time_count();
    let code_count = matrix.code_count();
    if time_count < 2 || code_count < 2 {
        return vec![None; code_count];
    }

    let mut returns = vec![None; time_count * code_count];
    for code_idx in 0..code_count {
        for time_idx in 1..time_count {
            let current = matrix.close[matrix.offset(time_idx, code_idx)];
            let previous = matrix.close[matrix.offset(time_idx - 1, code_idx)];
            returns[matrix.offset(time_idx, code_idx)] = ret(current, previous);
        }
    }

    let mut divergence = vec![None; time_count];
    for time_idx in 1..time_count {
        divergence[time_idx] = mean_std(
            (0..code_count)
                .filter_map(|code_idx| clean(returns[matrix.offset(time_idx, code_idx)])),
        )
        .map(|(_, std)| std);
    }
    let Some(divergence_mean) = mean(divergence.iter().filter_map(|value| clean(*value))) else {
        return vec![None; code_count];
    };
    let selected_times = divergence
        .iter()
        .enumerate()
        .filter_map(|(time_idx, value)| {
            clean(*value)
                .is_some_and(|value| value < divergence_mean)
                .then_some(time_idx)
        })
        .collect::<Vec<_>>();
    if selected_times.len() < 2 {
        return vec![None; code_count];
    }

    let mut amount_matrix = vec![None; selected_times.len() * code_count];
    for (row_idx, time_idx) in selected_times.iter().enumerate() {
        for code_idx in 0..code_count {
            amount_matrix[row_idx * code_count + code_idx] =
                matrix.amount[matrix.offset(*time_idx, code_idx)];
        }
    }
    mean_abs_column_corr_complete(&amount_matrix, selected_times.len(), code_count)
}

fn follow_current_component(
    panel: &DailyPanel,
    raw: &PanelColumn,
    window: usize,
) -> Result<PanelColumn> {
    let date_count = panel.dates().len();
    let code_count = panel.instruments().len();
    let mut output = vec![None; panel.shape_len()];
    for end_date_idx in 0..date_count {
        if end_date_idx + 1 < window {
            continue;
        }
        let start_date_idx = end_date_idx + 1 - window;
        let mut ranked_matrix = vec![None; window * code_count];
        for code_idx in 0..code_count {
            let mut series = Vec::with_capacity(window);
            let mut complete = true;
            for date_idx in start_date_idx..=end_date_idx {
                let value = raw.values()[date_idx * code_count + code_idx];
                if let Some(value) = clean(value) {
                    series.push(value);
                } else {
                    complete = false;
                    break;
                }
            }
            if !complete {
                continue;
            }
            let ranks = average_ranks(&series);
            for (row_idx, rank) in ranks.into_iter().enumerate() {
                ranked_matrix[row_idx * code_count + code_idx] = Some(rank);
            }
        }
        let correlations = mean_abs_column_corr_complete(&ranked_matrix, window, code_count);
        for code_idx in 0..code_count {
            output[end_date_idx * code_count + code_idx] = correlations[code_idx];
        }
    }
    panel.column_from_values(output)
}

fn mean_abs_column_corr_complete(
    matrix: &[Option<f64>],
    row_count: usize,
    column_count: usize,
) -> Vec<Option<f64>> {
    if row_count < 2 || column_count < 2 {
        return vec![None; column_count];
    }
    let mut original_columns = Vec::new();
    let mut normalized = Vec::new();
    for column_idx in 0..column_count {
        let mut values = Vec::with_capacity(row_count);
        let mut complete = true;
        for row_idx in 0..row_count {
            match clean(matrix[row_idx * column_count + column_idx]) {
                Some(value) => values.push(value),
                None => {
                    complete = false;
                    break;
                }
            }
        }
        if !complete {
            continue;
        }
        let Some((mean, std)) = mean_std(values.iter().copied()) else {
            continue;
        };
        if std <= f64::EPSILON {
            continue;
        }
        original_columns.push(column_idx);
        normalized.push(
            values
                .into_iter()
                .map(|value| (value - mean) / std)
                .collect::<Vec<_>>(),
        );
    }

    let valid_count = normalized.len();
    if valid_count < 2 {
        return vec![None; column_count];
    }
    let chunk_starts = (0..valid_count)
        .step_by(CORR_BLOCK_SIZE)
        .collect::<Vec<_>>();
    let partials = chunk_starts
        .into_par_iter()
        .map(|start| {
            let end = (start + CORR_BLOCK_SIZE).min(valid_count);
            let mut sums = vec![0.0; valid_count];
            let mut counts = vec![0usize; valid_count];
            for left_idx in start..end {
                for right_idx in (left_idx + 1)..valid_count {
                    let dot = normalized[left_idx]
                        .iter()
                        .zip(&normalized[right_idx])
                        .map(|(left, right)| left * right)
                        .sum::<f64>();
                    let corr = dot / row_count as f64;
                    if corr.is_nan() {
                        continue;
                    }
                    let abs_corr = corr.abs();
                    sums[left_idx] += abs_corr;
                    sums[right_idx] += abs_corr;
                    counts[left_idx] += 1;
                    counts[right_idx] += 1;
                }
            }
            (sums, counts)
        })
        .collect::<Vec<_>>();

    let mut sums = vec![0.0; valid_count];
    let mut counts = vec![0usize; valid_count];
    for (partial_sums, partial_counts) in partials {
        for idx in 0..valid_count {
            sums[idx] += partial_sums[idx];
            counts[idx] += partial_counts[idx];
        }
    }

    let mut output = vec![None; column_count];
    for (valid_idx, original_idx) in original_columns.into_iter().enumerate() {
        if counts[valid_idx] > 0 {
            output[original_idx] = Some(sums[valid_idx] / counts[valid_idx] as f64);
        }
    }
    output
}

fn average_ranks(values: &[f64]) -> Vec<f64> {
    let mut indexed = values
        .iter()
        .copied()
        .enumerate()
        .collect::<Vec<(usize, f64)>>();
    indexed.sort_by(|left, right| {
        left.1
            .partial_cmp(&right.1)
            .unwrap_or(Ordering::Equal)
            .then_with(|| left.0.cmp(&right.0))
    });

    let mut ranks = vec![0.0; values.len()];
    let mut start = 0usize;
    while start < indexed.len() {
        let mut end = start + 1;
        while end < indexed.len() && indexed[end].1 == indexed[start].1 {
            end += 1;
        }
        let average_rank = (start as f64 + 1.0 + end as f64) / 2.0;
        for idx in start..end {
            ranks[indexed[idx].0] = average_rank;
        }
        start = end;
    }
    ranks
}

fn average_pair(left: &PanelColumn, right: &PanelColumn) -> Result<PanelColumn> {
    left.zip_binary(right, |left, right| match (clean(left), clean(right)) {
        (Some(left), Some(right)) => Some((left + right) / 2.0),
        _ => None,
    })
}

fn subtract_pair(left: &PanelColumn, right: &PanelColumn) -> Result<PanelColumn> {
    left.zip_binary(right, |left, right| match (clean(left), clean(right)) {
        (Some(left), Some(right)) => Some(left - right),
        _ => None,
    })
}

fn ret(numerator: Option<f64>, denominator: Option<f64>) -> Option<f64> {
    match (clean(numerator), clean(denominator)) {
        (Some(numerator), Some(denominator)) if denominator.abs() > f64::EPSILON => {
            Some(numerator / denominator - 1.0)
        }
        _ => None,
    }
}

fn mean(values: impl IntoIterator<Item = f64>) -> Option<f64> {
    let values = values
        .into_iter()
        .filter(|value| !value.is_nan())
        .collect::<Vec<_>>();
    (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64)
}

fn mean_std(values: impl IntoIterator<Item = f64>) -> Option<(f64, f64)> {
    let values = values
        .into_iter()
        .filter(|value| !value.is_nan())
        .collect::<Vec<_>>();
    if values.is_empty() {
        return None;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / values.len() as f64;
    Some((mean, variance.sqrt()))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use crate::core::{AssetClass, FactorContext};
    use crate::data::ColumnData;

    use super::*;

    fn assert_close(actual: Option<f64>, expected: Option<f64>) {
        match (actual, expected) {
            (Some(actual), Some(expected)) => assert!((actual - expected).abs() < 1e-10),
            (None, None) => {}
            _ => panic!("expected {:?}, got {:?}", expected, actual),
        }
    }

    #[test]
    fn follow_current_uses_amount_times_ten_over_circ_mv() {
        let indices = vec![0, 1, 2];
        let times = vec![
            Some("09:30:00".to_string()),
            Some("09:31:00".to_string()),
            Some("09:32:00".to_string()),
        ];
        let open = vec![Some(10.0), Some(10.0), Some(10.0)];
        let close = vec![Some(10.0), Some(11.0), Some(9.0)];
        let amount = vec![Some(0.0), Some(5.0), Some(2.0)];

        assert_close(
            follow_current_for_stock(
                &indices,
                &times,
                &open,
                &close,
                &amount,
                Some(0.0),
                Some(100.0),
            ),
            Some((5.0 * 10.0 - 2.0 * 10.0) / 100.0),
        );
    }

    #[test]
    fn average_ranks_use_pandas_style_average_ties() {
        assert_eq!(
            average_ranks(&[3.0, 1.0, 1.0, 2.0]),
            vec![4.0, 1.5, 1.5, 3.0]
        );
    }

    #[test]
    fn mean_abs_column_corr_matches_direct_small_matrix() {
        let matrix = vec![
            Some(1.0),
            Some(1.0),
            Some(3.0),
            Some(2.0),
            Some(2.0),
            Some(2.0),
            Some(3.0),
            Some(3.0),
            Some(1.0),
        ];
        let output = mean_abs_column_corr_complete(&matrix, 3, 3);

        assert_close(output[0], Some(1.0));
        assert_close(output[1], Some(1.0));
        assert_close(output[2], Some(1.0));
    }

    #[test]
    fn lone_goose_filters_low_divergence_minutes_and_correlates_amount() {
        let matrix = MinuteMatrix {
            times: vec![
                "09:30:00".to_string(),
                "09:31:00".to_string(),
                "09:32:00".to_string(),
                "09:33:00".to_string(),
            ],
            codes: vec!["a".to_string(), "b".to_string(), "c".to_string()],
            close: vec![
                Some(10.0),
                Some(10.0),
                Some(10.0),
                Some(11.0),
                Some(11.0),
                Some(8.0),
                Some(12.0),
                Some(12.0),
                Some(6.0),
                Some(13.0),
                Some(11.0),
                Some(12.0),
            ],
            amount: vec![
                Some(0.0),
                Some(0.0),
                Some(0.0),
                Some(1.0),
                Some(2.0),
                Some(4.0),
                Some(2.0),
                Some(4.0),
                Some(8.0),
                Some(9.0),
                Some(1.0),
                Some(3.0),
            ],
        };
        let output = lone_goose_from_matrix(&matrix);

        assert!(output.iter().all(Option::is_some));
    }

    #[test]
    fn follow_current_component_uses_twenty_day_spearman_window() {
        let mut trade_dates = Vec::new();
        let mut ts_codes = Vec::new();
        let mut raw = Vec::new();
        for idx in 0..20 {
            trade_dates.push(Some(20260101 + idx));
            ts_codes.push(Some("a".to_string()));
            raw.push(Some(idx as f64));
            trade_dates.push(Some(20260101 + idx));
            ts_codes.push(Some("b".to_string()));
            raw.push(Some((idx * 2) as f64));
            trade_dates.push(Some(20260101 + idx));
            ts_codes.push(Some("c".to_string()));
            raw.push(Some((20 - idx) as f64));
        }
        let table = Table::new(BTreeMap::from([
            ("trade_date".to_string(), ColumnData::I32(trade_dates)),
            ("ts_code".to_string(), ColumnData::Utf8(ts_codes)),
            ("raw".to_string(), ColumnData::F64(raw)),
        ]))
        .expect("table");
        let context = FactorContext {
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            start_date: 20260120,
            end_date: 20260120,
            load_start_date: 20260101,
            load_dates: (0..20).map(|idx| 20260101 + idx).collect(),
            target_dates: vec![20260120],
        };
        let panel = DailyPanel::from_table(&table, &context).expect("panel");
        let raw = panel.column("raw").expect("raw");
        let output = follow_current_component(&panel, &raw, 20).expect("component");

        assert!(output.values().iter().take(57).all(Option::is_none));
        assert_close(output.values()[57], Some(1.0));
        assert_close(output.values()[58], Some(1.0));
        assert_close(output.values()[59], Some(1.0));
    }
}

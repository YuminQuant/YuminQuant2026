use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Arc;

use crate::core::{
    BarraSeries, BarraSpec, FactorContext, FactorRowKey, FactorSeries, FactorSpec, FactorValue,
    IntradayDailyRawSeries, IntradayDailyRawSpec, LabelSeries, LabelSpec,
};
use crate::data::Table;
use crate::error::{err, Result};
use crate::operators::cs_neutralize_regression;

use super::collect_numeric_columns;

#[derive(Clone, Debug, PartialEq, Eq)]
struct DailyPanelIndex {
    dates: Vec<i32>,
    instruments: Vec<String>,
    target_dates: BTreeSet<i32>,
    present: Vec<bool>,
}

impl DailyPanelIndex {
    fn instrument_count(&self) -> usize {
        self.instruments.len()
    }

    fn date_count(&self) -> usize {
        self.dates.len()
    }

    fn offset(&self, date_idx: usize, instrument_idx: usize) -> usize {
        date_idx * self.instrument_count() + instrument_idx
    }
}

#[derive(Clone, Debug)]
pub struct DailyPanel {
    index: Arc<DailyPanelIndex>,
    columns: BTreeMap<String, Arc<Vec<Option<f64>>>>,
}

impl DailyPanel {
    pub fn from_index(
        dates: Vec<i32>,
        instruments: Vec<String>,
        target_dates: &[i32],
        present: Vec<bool>,
    ) -> Result<Self> {
        let expected_len = dates.len() * instruments.len();
        if present.len() != expected_len {
            return Err(err(format!(
                "daily panel present mask has {} values, expected {}",
                present.len(),
                expected_len
            )));
        }
        Ok(Self {
            index: Arc::new(DailyPanelIndex {
                dates,
                instruments,
                target_dates: target_dates.iter().copied().collect(),
                present,
            }),
            columns: BTreeMap::new(),
        })
    }

    pub fn from_table(table: &Table, context: &FactorContext) -> Result<Self> {
        let ts_codes = table.required_utf8("ts_code")?;
        let trade_dates = table.required_i32("trade_date")?;
        let numeric_columns = collect_numeric_columns(table, &["trade_date", "ts_code"])?;

        let mut date_set = BTreeSet::new();
        let mut instrument_set = BTreeSet::new();
        for idx in 0..table.len {
            let (Some(ts_code), Some(trade_date)) = (ts_codes[idx].clone(), trade_dates[idx])
            else {
                continue;
            };
            date_set.insert(trade_date);
            instrument_set.insert(ts_code);
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
        let shape_len = dates.len() * instruments.len();
        let mut present = vec![false; shape_len];
        let mut columns = numeric_columns
            .keys()
            .map(|name| (name.clone(), vec![None; shape_len]))
            .collect::<BTreeMap<_, _>>();

        for idx in 0..table.len {
            let (Some(ts_code), Some(trade_date)) = (ts_codes[idx].clone(), trade_dates[idx])
            else {
                continue;
            };
            let Some(date_idx) = date_lookup.get(&trade_date).copied() else {
                continue;
            };
            let Some(instrument_idx) = instrument_lookup.get(&ts_code).copied() else {
                continue;
            };
            let offset = date_idx * instruments.len() + instrument_idx;
            present[offset] = true;
            for (name, source) in &numeric_columns {
                if let Some(target) = columns.get_mut(name) {
                    target[offset] = source[idx];
                }
            }
        }

        Ok(Self {
            index: Arc::new(DailyPanelIndex {
                dates,
                instruments,
                target_dates: context.target_dates.iter().copied().collect(),
                present,
            }),
            columns: columns
                .into_iter()
                .map(|(name, values)| (name, Arc::new(values)))
                .collect(),
        })
    }

    pub fn from_stock_basic(table: &Table, context: &FactorContext) -> Result<Self> {
        let ts_codes = table.required_utf8("ts_code")?;
        let list_dates = table.required_i32_date_cast("list_date")?;
        let delist_dates = table.required_i32_date_cast("delist_date")?;

        let mut instruments = Vec::new();
        let mut listing_ranges = Vec::new();
        for idx in 0..table.len {
            let (Some(ts_code), Some(list_date)) = (ts_codes[idx].clone(), list_dates[idx]) else {
                continue;
            };
            if !is_a_stock_code(&ts_code) {
                continue;
            }
            instruments.push(ts_code);
            listing_ranges.push((list_date, delist_dates[idx]));
        }
        let mut paired = instruments
            .into_iter()
            .zip(listing_ranges)
            .collect::<Vec<_>>();
        paired.sort_by(|left, right| left.0.cmp(&right.0));
        paired.dedup_by(|left, right| left.0 == right.0);
        let (instruments, listing_ranges): (Vec<_>, Vec<_>) = paired.into_iter().unzip();

        let dates = if context.load_dates.is_empty() {
            context.target_dates.clone()
        } else {
            context.load_dates.clone()
        };
        let mut dates = dates
            .into_iter()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        if dates.is_empty() {
            dates.extend(context.target_dates.iter().copied());
        }
        let mut present = Vec::with_capacity(dates.len() * instruments.len());
        for trade_date in &dates {
            for (list_date, delist_date) in &listing_ranges {
                present.push(
                    *list_date <= *trade_date
                        && delist_date.map_or(true, |delist_date| *trade_date <= delist_date),
                );
            }
        }

        Self::from_index(dates, instruments, &context.target_dates, present)
    }

    pub fn with_target_dates(&self, target_dates: &[i32]) -> Self {
        Self {
            index: Arc::new(DailyPanelIndex {
                dates: self.index.dates.clone(),
                instruments: self.index.instruments.clone(),
                target_dates: target_dates.iter().copied().collect(),
                present: self.index.present.clone(),
            }),
            columns: self.columns.clone(),
        }
    }

    pub fn slice_dates(&self, selected_dates: &[i32]) -> Self {
        let selected = selected_dates.iter().copied().collect::<BTreeSet<_>>();
        let date_indices = self
            .index
            .dates
            .iter()
            .enumerate()
            .filter_map(|(date_idx, date)| selected.contains(date).then_some(date_idx))
            .collect::<Vec<_>>();
        let dates = date_indices
            .iter()
            .map(|date_idx| self.index.dates[*date_idx])
            .collect::<Vec<_>>();
        let instrument_count = self.index.instrument_count();
        let mut present = Vec::with_capacity(dates.len() * instrument_count);
        for date_idx in &date_indices {
            let offset = date_idx * instrument_count;
            present.extend_from_slice(&self.index.present[offset..offset + instrument_count]);
        }
        let columns = self
            .columns
            .iter()
            .map(|(name, values)| {
                let mut sliced = Vec::with_capacity(dates.len() * instrument_count);
                for date_idx in &date_indices {
                    let offset = date_idx * instrument_count;
                    sliced.extend_from_slice(&values[offset..offset + instrument_count]);
                }
                (name.clone(), Arc::new(sliced))
            })
            .collect();
        Self {
            index: Arc::new(DailyPanelIndex {
                dates,
                instruments: self.index.instruments.clone(),
                target_dates: selected_dates.iter().copied().collect(),
                present,
            }),
            columns,
        }
    }

    pub fn column(&self, name: &str) -> Result<PanelColumn> {
        let values = self
            .columns
            .get(name)
            .cloned()
            .ok_or_else(|| err(format!("missing daily panel column {}", name)))?;
        Ok(PanelColumn {
            index: Arc::clone(&self.index),
            values,
        })
    }

    pub fn column_from_table(&self, table: &Table, name: &str) -> Result<PanelColumn> {
        let ts_codes = table.required_utf8("ts_code")?;
        let trade_dates = table.required_i32("trade_date")?;
        let source = table.required_f64_cast(name)?;
        let date_lookup = self
            .index
            .dates
            .iter()
            .enumerate()
            .map(|(idx, date)| (*date, idx))
            .collect::<HashMap<_, _>>();
        let instrument_lookup = self
            .index
            .instruments
            .iter()
            .enumerate()
            .map(|(idx, ts_code)| (ts_code.clone(), idx))
            .collect::<HashMap<_, _>>();
        let mut values = vec![None; self.shape_len()];
        for idx in 0..table.len {
            let (Some(trade_date), Some(ts_code)) = (trade_dates[idx], ts_codes[idx].clone())
            else {
                continue;
            };
            let Some(date_idx) = date_lookup.get(&trade_date).copied() else {
                continue;
            };
            let Some(instrument_idx) = instrument_lookup.get(&ts_code).copied() else {
                continue;
            };
            values[self.index.offset(date_idx, instrument_idx)] = source[idx];
        }
        self.column_from_values(values)
    }

    pub fn has_column(&self, name: &str) -> bool {
        self.columns.contains_key(name)
    }

    pub fn dates(&self) -> &[i32] {
        &self.index.dates
    }

    pub fn is_target_date(&self, trade_date: i32) -> bool {
        self.index.target_dates.contains(&trade_date)
    }

    pub fn instruments(&self) -> &[String] {
        &self.index.instruments
    }

    pub fn shape_len(&self) -> usize {
        self.index.date_count() * self.index.instrument_count()
    }

    pub fn is_present_offset(&self, offset: usize) -> bool {
        self.index.present.get(offset).copied().unwrap_or(false)
    }

    pub fn column_from_values(&self, values: Vec<Option<f64>>) -> Result<PanelColumn> {
        if values.len() != self.shape_len() {
            return Err(err(format!(
                "panel column has {} values, expected {}",
                values.len(),
                self.shape_len()
            )));
        }
        Ok(PanelColumn {
            index: Arc::clone(&self.index),
            values: Arc::new(values),
        })
    }
}

fn is_a_stock_code(ts_code: &str) -> bool {
    let upper = ts_code.to_ascii_uppercase();
    upper.ends_with(".SH") || upper.ends_with(".SZ") || upper.ends_with(".BJ")
}

#[derive(Clone, Debug)]
pub struct PanelColumn {
    index: Arc<DailyPanelIndex>,
    values: Arc<Vec<Option<f64>>>,
}

impl PanelColumn {
    pub fn values(&self) -> &[Option<f64>] {
        &self.values
    }

    pub fn map_values<F>(&self, mut f: F) -> Self
    where
        F: FnMut(Option<f64>) -> Option<f64>,
    {
        self.with_values(self.values.iter().map(|value| f(*value)).collect())
    }

    pub fn zip_binary<F>(&self, other: &Self, mut f: F) -> Result<Self>
    where
        F: FnMut(Option<f64>, Option<f64>) -> Option<f64>,
    {
        self.require_same_index(other)?;
        Ok(self.with_values(
            self.values
                .iter()
                .zip(other.values.iter())
                .map(|(left, right)| f(*left, *right))
                .collect(),
        ))
    }

    pub fn zip_ternary<F>(&self, second: &Self, third: &Self, mut f: F) -> Result<Self>
    where
        F: FnMut(Option<f64>, Option<f64>, Option<f64>) -> Option<f64>,
    {
        self.require_same_index(second)?;
        self.require_same_index(third)?;
        Ok(self.with_values(
            self.values
                .iter()
                .zip(second.values.iter())
                .zip(third.values.iter())
                .map(|((first, second), third)| f(*first, *second, *third))
                .collect(),
        ))
    }

    pub fn zip_quaternary<F>(
        &self,
        second: &Self,
        third: &Self,
        fourth: &Self,
        mut f: F,
    ) -> Result<Self>
    where
        F: FnMut(Option<f64>, Option<f64>, Option<f64>, Option<f64>) -> Option<f64>,
    {
        self.require_same_index(second)?;
        self.require_same_index(third)?;
        self.require_same_index(fourth)?;
        Ok(self.with_values(
            self.values
                .iter()
                .zip(second.values.iter())
                .zip(third.values.iter())
                .zip(fourth.values.iter())
                .map(|(((first, second), third), fourth)| f(*first, *second, *third, *fourth))
                .collect(),
        ))
    }

    pub fn ts<F>(&self, mut f: F) -> Result<Self>
    where
        F: FnMut(&[Option<f64>]) -> Vec<Option<f64>>,
    {
        let mut output = vec![None; self.values.len()];
        for instrument_idx in 0..self.index.instrument_count() {
            let input = self.series_for_instrument(instrument_idx);
            let computed = f(&input);
            self.require_series_len(computed.len(), "ts")?;
            self.write_series_for_instrument(instrument_idx, &computed, &mut output);
        }
        Ok(self.with_values(output))
    }

    pub fn cs<F>(&self, mut f: F) -> Result<Self>
    where
        F: FnMut(&[Option<f64>]) -> Vec<Option<f64>>,
    {
        let mut output = vec![None; self.values.len()];
        for date_idx in 0..self.index.date_count() {
            let input = self.cross_section_for_date(date_idx);
            let computed = f(&input);
            self.require_cross_section_len(computed.len(), "cs")?;
            self.write_cross_section_for_date(date_idx, &computed, &mut output);
        }
        Ok(self.with_values(output))
    }

    pub fn ts_binary<F>(&self, other: &Self, mut f: F) -> Result<Self>
    where
        F: FnMut(&[Option<f64>], &[Option<f64>]) -> Vec<Option<f64>>,
    {
        self.require_same_index(other)?;
        let mut output = vec![None; self.values.len()];
        for instrument_idx in 0..self.index.instrument_count() {
            let left = self.series_for_instrument(instrument_idx);
            let right = other.series_for_instrument(instrument_idx);
            let computed = f(&left, &right);
            self.require_series_len(computed.len(), "ts_binary")?;
            self.write_series_for_instrument(instrument_idx, &computed, &mut output);
        }
        Ok(self.with_values(output))
    }

    pub fn cs_binary<F>(&self, other: &Self, mut f: F) -> Result<Self>
    where
        F: FnMut(&[Option<f64>], &[Option<f64>]) -> Vec<Option<f64>>,
    {
        self.require_same_index(other)?;
        let mut output = vec![None; self.values.len()];
        for date_idx in 0..self.index.date_count() {
            let left = self.cross_section_for_date(date_idx);
            let right = other.cross_section_for_date(date_idx);
            let computed = f(&left, &right);
            self.require_cross_section_len(computed.len(), "cs_binary")?;
            self.write_cross_section_for_date(date_idx, &computed, &mut output);
        }
        Ok(self.with_values(output))
    }

    pub fn ts_ternary<F>(&self, second: &Self, third: &Self, mut f: F) -> Result<Self>
    where
        F: FnMut(&[Option<f64>], &[Option<f64>], &[Option<f64>]) -> Vec<Option<f64>>,
    {
        self.require_same_index(second)?;
        self.require_same_index(third)?;
        let mut output = vec![None; self.values.len()];
        for instrument_idx in 0..self.index.instrument_count() {
            let first = self.series_for_instrument(instrument_idx);
            let second = second.series_for_instrument(instrument_idx);
            let third = third.series_for_instrument(instrument_idx);
            let computed = f(&first, &second, &third);
            self.require_series_len(computed.len(), "ts_ternary")?;
            self.write_series_for_instrument(instrument_idx, &computed, &mut output);
        }
        Ok(self.with_values(output))
    }

    pub fn cs_ternary<F>(&self, second: &Self, third: &Self, mut f: F) -> Result<Self>
    where
        F: FnMut(&[Option<f64>], &[Option<f64>], &[Option<f64>]) -> Vec<Option<f64>>,
    {
        self.require_same_index(second)?;
        self.require_same_index(third)?;
        let mut output = vec![None; self.values.len()];
        for date_idx in 0..self.index.date_count() {
            let first = self.cross_section_for_date(date_idx);
            let second = second.cross_section_for_date(date_idx);
            let third = third.cross_section_for_date(date_idx);
            let computed = f(&first, &second, &third);
            self.require_cross_section_len(computed.len(), "cs_ternary")?;
            self.write_cross_section_for_date(date_idx, &computed, &mut output);
        }
        Ok(self.with_values(output))
    }

    pub fn cs_by_group<GF, F>(&self, mut group_provider: GF, mut f: F) -> Result<Self>
    where
        GF: FnMut(i32, &[String]) -> Vec<Option<String>>,
        F: FnMut(&[Option<f64>], &[Option<String>]) -> Vec<Option<f64>>,
    {
        let mut output = vec![None; self.values.len()];
        for date_idx in 0..self.index.date_count() {
            let input = self.cross_section_for_date(date_idx);
            let groups = group_provider(self.index.dates[date_idx], &self.index.instruments);
            if groups.len() != self.index.instrument_count() {
                return Err(err(format!(
                    "cs_by_group returned {} group labels for date {}, expected {}",
                    groups.len(),
                    self.index.dates[date_idx],
                    self.index.instrument_count()
                )));
            }
            let computed = f(&input, &groups);
            self.require_cross_section_len(computed.len(), "cs_by_group")?;
            self.write_cross_section_for_date(date_idx, &computed, &mut output);
        }
        Ok(self.with_values(output))
    }

    pub fn cs_neutralize_regression(
        &self,
        continuous: &[&Self],
        weights: Option<&Self>,
    ) -> Result<Self> {
        for column in continuous {
            self.require_same_index(column)?;
        }
        if let Some(weights) = weights {
            self.require_same_index(weights)?;
        }

        let mut output = vec![None; self.values.len()];
        for date_idx in 0..self.index.date_count() {
            let y = self.cross_section_for_date(date_idx);
            let continuous_values = continuous
                .iter()
                .map(|column| column.cross_section_for_date(date_idx))
                .collect::<Vec<_>>();
            let continuous_refs = continuous_values
                .iter()
                .map(Vec::as_slice)
                .collect::<Vec<_>>();
            let weight_values = weights.map(|column| column.cross_section_for_date(date_idx));
            let computed =
                cs_neutralize_regression(&y, &continuous_refs, None, weight_values.as_deref());
            self.require_cross_section_len(computed.len(), "cs_neutralize_regression")?;
            self.write_cross_section_for_date(date_idx, &computed, &mut output);
        }
        Ok(self.with_values(output))
    }

    pub fn cs_neutralize_regression_by_group<GF>(
        &self,
        continuous: &[&Self],
        weights: Option<&Self>,
        mut group_provider: GF,
    ) -> Result<Self>
    where
        GF: FnMut(i32, &[String]) -> Vec<Option<String>>,
    {
        for column in continuous {
            self.require_same_index(column)?;
        }
        if let Some(weights) = weights {
            self.require_same_index(weights)?;
        }

        let mut output = vec![None; self.values.len()];
        for date_idx in 0..self.index.date_count() {
            let y = self.cross_section_for_date(date_idx);
            let continuous_values = continuous
                .iter()
                .map(|column| column.cross_section_for_date(date_idx))
                .collect::<Vec<_>>();
            let continuous_refs = continuous_values
                .iter()
                .map(Vec::as_slice)
                .collect::<Vec<_>>();
            let weight_values = weights.map(|column| column.cross_section_for_date(date_idx));
            let groups = group_provider(self.index.dates[date_idx], &self.index.instruments);
            if groups.len() != self.index.instrument_count() {
                return Err(err(format!(
                    "cs_neutralize_regression_by_group returned {} group labels for date {}, expected {}",
                    groups.len(),
                    self.index.dates[date_idx],
                    self.index.instrument_count()
                )));
            }
            let computed = cs_neutralize_regression(
                &y,
                &continuous_refs,
                Some(&groups),
                weight_values.as_deref(),
            );
            self.require_cross_section_len(computed.len(), "cs_neutralize_regression_by_group")?;
            self.write_cross_section_for_date(date_idx, &computed, &mut output);
        }
        Ok(self.with_values(output))
    }

    pub fn to_factor_series(&self, spec: FactorSpec) -> FactorSeries {
        let mut values = Vec::new();
        for (date_idx, trade_date) in self.index.dates.iter().enumerate() {
            if !self.index.target_dates.contains(trade_date) {
                continue;
            }
            for (instrument_idx, ts_code) in self.index.instruments.iter().enumerate() {
                let offset = self.index.offset(date_idx, instrument_idx);
                if !self.index.present[offset] {
                    continue;
                }
                values.push(FactorValue {
                    key: FactorRowKey::Daily {
                        trade_date: *trade_date,
                        ts_code: ts_code.clone(),
                    },
                    value: self.values[offset],
                });
            }
        }
        FactorSeries { spec, values }
    }

    pub fn to_label_series(&self, spec: LabelSpec) -> LabelSeries {
        let mut values = Vec::new();
        for (date_idx, trade_date) in self.index.dates.iter().enumerate() {
            if !self.index.target_dates.contains(trade_date) {
                continue;
            }
            for (instrument_idx, ts_code) in self.index.instruments.iter().enumerate() {
                let offset = self.index.offset(date_idx, instrument_idx);
                if !self.index.present[offset] {
                    continue;
                }
                values.push(FactorValue {
                    key: FactorRowKey::Daily {
                        trade_date: *trade_date,
                        ts_code: ts_code.clone(),
                    },
                    value: self.values[offset],
                });
            }
        }
        LabelSeries { spec, values }
    }

    pub fn to_barra_series(&self, spec: BarraSpec) -> BarraSeries {
        let mut values = Vec::new();
        for (date_idx, trade_date) in self.index.dates.iter().enumerate() {
            if !self.index.target_dates.contains(trade_date) {
                continue;
            }
            for (instrument_idx, ts_code) in self.index.instruments.iter().enumerate() {
                let offset = self.index.offset(date_idx, instrument_idx);
                if !self.index.present[offset] {
                    continue;
                }
                values.push(FactorValue {
                    key: FactorRowKey::Daily {
                        trade_date: *trade_date,
                        ts_code: ts_code.clone(),
                    },
                    value: self.values[offset],
                });
            }
        }
        BarraSeries { spec, values }
    }

    pub fn to_intraday_daily_raw_series(
        &self,
        spec: IntradayDailyRawSpec,
    ) -> IntradayDailyRawSeries {
        let mut values = Vec::new();
        for (date_idx, trade_date) in self.index.dates.iter().enumerate() {
            if !self.index.target_dates.contains(trade_date) {
                continue;
            }
            for (instrument_idx, ts_code) in self.index.instruments.iter().enumerate() {
                let offset = self.index.offset(date_idx, instrument_idx);
                if !self.index.present[offset] {
                    continue;
                }
                values.push(FactorValue {
                    key: FactorRowKey::Daily {
                        trade_date: *trade_date,
                        ts_code: ts_code.clone(),
                    },
                    value: self.values[offset],
                });
            }
        }
        IntradayDailyRawSeries { spec, values }
    }

    fn with_values(&self, values: Vec<Option<f64>>) -> Self {
        Self {
            index: Arc::clone(&self.index),
            values: Arc::new(values),
        }
    }

    fn require_same_index(&self, other: &Self) -> Result<()> {
        if Arc::ptr_eq(&self.index, &other.index) || self.index == other.index {
            return Ok(());
        }
        Err(err(
            "panel columns use different DailyPanel indexes and cannot be combined",
        ))
    }

    fn require_series_len(&self, len: usize, op: &str) -> Result<()> {
        if len == self.index.date_count() {
            return Ok(());
        }
        Err(err(format!(
            "{} returned {} time-series values, expected {}",
            op,
            len,
            self.index.date_count()
        )))
    }

    fn require_cross_section_len(&self, len: usize, op: &str) -> Result<()> {
        if len == self.index.instrument_count() {
            return Ok(());
        }
        Err(err(format!(
            "{} returned {} cross-section values, expected {}",
            op,
            len,
            self.index.instrument_count()
        )))
    }

    fn series_for_instrument(&self, instrument_idx: usize) -> Vec<Option<f64>> {
        (0..self.index.date_count())
            .map(|date_idx| self.values[self.index.offset(date_idx, instrument_idx)])
            .collect()
    }

    fn write_series_for_instrument(
        &self,
        instrument_idx: usize,
        values: &[Option<f64>],
        output: &mut [Option<f64>],
    ) {
        for (date_idx, value) in values.iter().enumerate() {
            output[self.index.offset(date_idx, instrument_idx)] = *value;
        }
    }

    fn cross_section_for_date(&self, date_idx: usize) -> Vec<Option<f64>> {
        (0..self.index.instrument_count())
            .map(|instrument_idx| self.values[self.index.offset(date_idx, instrument_idx)])
            .collect()
    }

    fn write_cross_section_for_date(
        &self,
        date_idx: usize,
        values: &[Option<f64>],
        output: &mut [Option<f64>],
    ) {
        for (instrument_idx, value) in values.iter().enumerate() {
            output[self.index.offset(date_idx, instrument_idx)] = *value;
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use crate::core::{AssetClass, FactorContext, Frequency};
    use crate::data::{ColumnData, Table};
    use crate::operators::{
        cs_neutralize, cs_rank, cs_regression_residual, ts_corr, ts_pctchg, ts_sum,
    };
    use std::sync::Arc;

    use super::DailyPanel;

    fn assert_option_close(actual: Option<f64>, expected: Option<f64>) {
        match (actual, expected) {
            (Some(actual), Some(expected)) => assert!((actual - expected).abs() < 1e-10),
            (None, None) => {}
            _ => panic!("expected {:?}, got {:?}", expected, actual),
        }
    }

    fn context(target_dates: Vec<i32>) -> FactorContext {
        let start_date = *target_dates.first().unwrap();
        let end_date = *target_dates.last().unwrap();
        FactorContext {
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            start_date,
            end_date,
            load_start_date: 20260101,
            load_dates: target_dates.clone(),
            target_dates,
        }
    }

    fn sample_table() -> Table {
        Table::new(BTreeMap::from([
            (
                "trade_date".to_string(),
                ColumnData::I32(vec![
                    Some(20260101),
                    Some(20260102),
                    Some(20260103),
                    Some(20260101),
                    Some(20260103),
                ]),
            ),
            (
                "ts_code".to_string(),
                ColumnData::Utf8(vec![
                    Some("000002.SZ".to_string()),
                    Some("000001.SZ".to_string()),
                    Some("000001.SZ".to_string()),
                    Some("000001.SZ".to_string()),
                    Some("000002.SZ".to_string()),
                ]),
            ),
            (
                "close".to_string(),
                ColumnData::F64(vec![
                    Some(10.0),
                    Some(2.0),
                    Some(4.0),
                    Some(1.0),
                    Some(20.0),
                ]),
            ),
            (
                "vol".to_string(),
                ColumnData::F64(vec![
                    Some(100.0),
                    Some(20.0),
                    Some(40.0),
                    Some(10.0),
                    Some(200.0),
                ]),
            ),
        ]))
        .expect("valid table")
    }

    #[test]
    fn daily_panel_builds_dense_index_and_fills_missing_combinations() {
        let panel =
            DailyPanel::from_table(&sample_table(), &context(vec![20260103])).expect("panel");
        let close = panel.column("close").expect("close");

        assert_eq!(panel.dates(), &[20260101, 20260102, 20260103]);
        assert_eq!(
            panel.instruments(),
            &["000001.SZ".to_string(), "000002.SZ".to_string()]
        );
        assert_eq!(
            close.values(),
            &[
                Some(1.0),
                Some(10.0),
                Some(2.0),
                None,
                Some(4.0),
                Some(20.0)
            ]
        );
    }

    #[test]
    fn daily_panel_with_target_dates_shares_column_storage() {
        let panel = DailyPanel::from_table(
            &sample_table(),
            &context(vec![20260101, 20260102, 20260103]),
        )
        .expect("panel");
        let retargeted = panel.with_target_dates(&[20260103]);
        let original_close = panel.columns.get("close").expect("original close");
        let retargeted_close = retargeted.columns.get("close").expect("retargeted close");
        assert!(Arc::ptr_eq(original_close, retargeted_close));
        assert_eq!(retargeted.column("close").unwrap().values()[5], Some(20.0));
    }

    #[test]
    fn daily_panel_slice_dates_keeps_only_selected_date_rows() {
        let panel = DailyPanel::from_table(
            &sample_table(),
            &context(vec![20260101, 20260102, 20260103]),
        )
        .expect("panel");
        let sliced = panel.slice_dates(&[20260103]);

        assert_eq!(sliced.dates(), &[20260103]);
        assert_eq!(sliced.instruments(), panel.instruments());
        assert_eq!(sliced.shape_len(), panel.instruments().len());
        assert!(sliced.is_target_date(20260103));

        let close = sliced.column("close").expect("close");
        assert_eq!(close.values(), &[Some(4.0), Some(20.0)]);
        assert!(sliced.is_present_offset(0));
        assert!(sliced.is_present_offset(1));
    }

    #[test]
    fn stock_basic_panel_filters_to_a_shares_and_tracks_listing_window() {
        let table = Table::new(BTreeMap::from([
            (
                "ts_code".to_string(),
                ColumnData::Utf8(vec![
                    Some("000001.SZ".to_string()),
                    Some("600000.SH".to_string()),
                    Some("920001.BJ".to_string()),
                    Some("AAPL.US".to_string()),
                ]),
            ),
            (
                "list_date".to_string(),
                ColumnData::I32(vec![
                    Some(20260102),
                    Some(20260101),
                    Some(20260101),
                    Some(20260101),
                ]),
            ),
            (
                "delist_date".to_string(),
                ColumnData::I32(vec![None, Some(20260102), None, None]),
            ),
        ]))
        .expect("stock basic");
        let panel =
            DailyPanel::from_stock_basic(&table, &context(vec![20260101, 20260102, 20260103]))
                .expect("stock universe panel");

        assert_eq!(panel.dates(), &[20260101, 20260102, 20260103]);
        assert_eq!(
            panel.instruments(),
            &[
                "000001.SZ".to_string(),
                "600000.SH".to_string(),
                "920001.BJ".to_string()
            ]
        );
        assert_eq!(
            (0..panel.shape_len())
                .map(|offset| panel.is_present_offset(offset))
                .collect::<Vec<_>>(),
            vec![false, true, true, true, true, true, true, false, true]
        );
    }

    #[test]
    fn panel_supports_unary_ts_and_cs_transforms() {
        let panel =
            DailyPanel::from_table(&sample_table(), &context(vec![20260103])).expect("panel");
        let ranked = panel
            .column("close")
            .expect("close")
            .ts(|values| ts_sum(values, 2, 1))
            .expect("ts")
            .cs(|values| cs_rank(values, true))
            .expect("cs");

        assert_eq!(
            ranked.values(),
            &[
                Some(1.0),
                Some(2.0),
                Some(1.0),
                Some(2.0),
                Some(1.0),
                Some(2.0)
            ]
        );
    }

    #[test]
    fn panel_ts_binary_aligns_columns_for_correlation() {
        let panel =
            DailyPanel::from_table(&sample_table(), &context(vec![20260103])).expect("panel");
        let close = panel.column("close").expect("close");
        let volume = panel.column("vol").expect("vol");
        let corr = volume
            .ts_binary(&close, |volume, close| ts_corr(volume, close, 3, 2))
            .expect("corr");

        assert_eq!(corr.values()[0], None);
        assert_eq!(corr.values()[1], None);
        assert_option_close(corr.values()[2], Some(1.0));
        assert_eq!(corr.values()[3], None);
        assert_option_close(corr.values()[4], Some(1.0));
        assert_option_close(corr.values()[5], Some(1.0));
    }

    #[test]
    fn panel_pointwise_zip_binary_reuses_existing_index() {
        let panel =
            DailyPanel::from_table(&sample_table(), &context(vec![20260103])).expect("panel");
        let close = panel.column("close").expect("close");
        let volume = panel.column("vol").expect("vol");
        let scaled = close
            .zip_binary(&volume, |close, volume| match (close, volume) {
                (Some(close), Some(volume)) => Some(close * volume),
                _ => None,
            })
            .expect("scaled");

        assert_eq!(
            scaled.values(),
            &[
                Some(10.0),
                Some(1000.0),
                Some(40.0),
                None,
                Some(160.0),
                Some(4000.0)
            ]
        );
    }

    #[test]
    fn panel_cs_binary_aligns_columns_for_regression() {
        let panel =
            DailyPanel::from_table(&sample_table(), &context(vec![20260103])).expect("panel");
        let close = panel.column("close").expect("close");
        let volume = panel.column("vol").expect("vol");
        let residual = volume
            .cs_binary(&close, cs_regression_residual)
            .expect("residual");

        assert_eq!(residual.values()[0], Some(0.0));
        assert_eq!(residual.values()[1], Some(0.0));
        assert_eq!(residual.values()[2], None);
        assert_eq!(residual.values()[3], None);
        assert_eq!(residual.values()[4], Some(0.0));
        assert_eq!(residual.values()[5], Some(0.0));
    }

    #[test]
    fn panel_supports_nested_ts_cs_and_cs_ts_flows() {
        let panel =
            DailyPanel::from_table(&sample_table(), &context(vec![20260103])).expect("panel");
        let close = panel.column("close").expect("close");

        let ts_then_cs_then_ts = close
            .ts(|values| ts_sum(values, 2, 1))
            .expect("ts1")
            .cs(|values| cs_rank(values, true))
            .expect("cs")
            .ts(|values| ts_sum(values, 2, 1))
            .expect("ts2");
        let cs_then_ts = close
            .cs(|values| cs_rank(values, true))
            .expect("cs")
            .ts(|values| ts_sum(values, 2, 1))
            .expect("ts");

        assert_eq!(ts_then_cs_then_ts.values()[4], Some(2.0));
        assert_eq!(cs_then_ts.values()[4], Some(2.0));
    }

    #[test]
    fn panel_cs_by_group_neutralizes_valid_groups_and_skips_missing_groups() {
        let panel =
            DailyPanel::from_table(&sample_table(), &context(vec![20260103])).expect("panel");
        let neutralized = panel
            .column("close")
            .expect("close")
            .cs_by_group(
                |date, codes| {
                    codes
                        .iter()
                        .map(|code| match (date, code.as_str()) {
                            (20260103, "000001.SZ") => Some("sector_a".to_string()),
                            (20260103, "000002.SZ") => Some("sector_a".to_string()),
                            _ => None,
                        })
                        .collect()
                },
                cs_neutralize,
            )
            .expect("neutralized");

        assert_eq!(neutralized.values()[0], None);
        assert_eq!(neutralized.values()[1], None);
        assert_eq!(neutralized.values()[4], Some(-8.0));
        assert_eq!(neutralized.values()[5], Some(8.0));
    }

    #[test]
    fn lookback_20_covers_pctchg_1_plus_sum_20_on_target_date() {
        let mut trade_dates = Vec::new();
        let mut ts_codes = Vec::new();
        let mut close = Vec::new();
        for idx in 0..21 {
            trade_dates.push(Some(20260101 + idx));
            ts_codes.push(Some("000001.SZ".to_string()));
            close.push(Some((idx + 1) as f64));
        }
        let table = Table::new(BTreeMap::from([
            ("trade_date".to_string(), ColumnData::I32(trade_dates)),
            ("ts_code".to_string(), ColumnData::Utf8(ts_codes)),
            ("close".to_string(), ColumnData::F64(close)),
        ]))
        .expect("valid table");
        let panel = DailyPanel::from_table(&table, &context(vec![20260121])).expect("panel");
        let factor = panel
            .column("close")
            .expect("close")
            .ts(|values| ts_pctchg(values, 1))
            .expect("pctchg")
            .ts(|values| ts_sum(values, 20, 20))
            .expect("sum");

        assert!(factor.values()[20].is_some());
    }
}

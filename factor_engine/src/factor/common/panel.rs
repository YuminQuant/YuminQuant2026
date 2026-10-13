use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::{Arc, Mutex, Weak};

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
    aligned: Arc<Mutex<HashMap<(usize, usize), AlignedColumn>>>,
}

#[derive(Debug)]
struct AlignedColumn {
    // Keep source identities alive so allocator address reuse cannot hit a stale entry.
    _source: PanelColumn,
    values: Weak<Vec<Option<f64>>>,
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
            aligned: Default::default(),
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
            aligned: Default::default(),
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
        if self
            .index
            .target_dates
            .iter()
            .copied()
            .eq(target_dates.iter().copied())
        {
            return self.clone();
        }
        Self {
            index: Arc::new(DailyPanelIndex {
                dates: self.index.dates.clone(),
                instruments: self.index.instruments.clone(),
                target_dates: target_dates.iter().copied().collect(),
                present: self.index.present.clone(),
            }),
            columns: self.columns.clone(),
            aligned: Arc::clone(&self.aligned),
        }
    }

    pub fn slice_dates(&self, selected_dates: &[i32]) -> Self {
        let selected = selected_dates.iter().copied().collect::<BTreeSet<_>>();
        if self.index.dates.iter().all(|date| selected.contains(date)) {
            return self.with_target_dates(selected_dates);
        }
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
            aligned: Default::default(),
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

    /// Align raw daily data without changing the target universe, mask or output dates.
    pub fn column_from_panel(&self, source: &DailyPanel, name: &str) -> Result<PanelColumn> {
        let source = source.column(name)?;
        if self.index.dates == source.index.dates
            && self.index.instruments == source.index.instruments
        {
            return Ok(PanelColumn {
                index: Arc::clone(&self.index),
                values: source.values,
            });
        }
        let key = (
            Arc::as_ptr(&source.index) as usize,
            Arc::as_ptr(&source.values) as usize,
        );
        if let Some(values) = self
            .aligned
            .lock()
            .unwrap()
            .get(&key)
            .and_then(|entry| entry.values.upgrade())
        {
            return Ok(PanelColumn {
                index: Arc::clone(&self.index),
                values,
            });
        }
        let stocks: HashMap<_, _> = source
            .index
            .instruments
            .iter()
            .enumerate()
            .map(|(i, code)| (code.as_str(), i))
            .collect();
        let mapping: Vec<_> = self
            .index
            .instruments
            .iter()
            .map(|code| stocks.get(code.as_str()).copied())
            .collect();
        let mut values = vec![None; self.shape_len()];
        for (day, date) in self.index.dates.iter().enumerate() {
            let Ok(source_day) = source.index.dates.binary_search(date) else {
                continue;
            };
            let offset = source_day * source.index.instrument_count();
            for (stock, source_stock) in mapping.iter().enumerate() {
                if let Some(source_stock) = source_stock {
                    values[self.index.offset(day, stock)] = source.values[offset + source_stock];
                }
            }
        }
        let mut cache = self.aligned.lock().unwrap();
        cache.retain(|_, entry| entry.values.strong_count() > 0);
        // Another provider may have aligned this column while we were computing it.
        let values = cache
            .get(&key)
            .and_then(|entry| entry.values.upgrade())
            .unwrap_or_else(|| Arc::new(values));
        cache.insert(
            key,
            AlignedColumn {
                _source: source,
                values: Arc::downgrade(&values),
            },
        );
        Ok(PanelColumn {
            index: Arc::clone(&self.index),
            values,
        })
    }

    pub fn has_column(&self, name: &str) -> bool {
        self.columns.contains_key(name)
    }

    pub fn select_columns(&self, names: &BTreeSet<String>) -> Self {
        Self {
            index: Arc::clone(&self.index),
            aligned: Arc::clone(&self.aligned),
            columns: self
                .columns
                .iter()
                .filter(|(name, _)| names.contains(*name))
                .map(|(name, values)| (name.clone(), Arc::clone(values)))
                .collect(),
        }
    }

    pub fn present_row_count(&self) -> usize {
        self.index
            .present
            .iter()
            .filter(|present| **present)
            .count()
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
    let Some((symbol, exchange)) = upper.split_once('.') else {
        return false;
    };
    symbol.len() == 6
        && symbol.bytes().all(|byte| byte.is_ascii_digit())
        && matches!(exchange, "SH" | "SZ" | "BJ")
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

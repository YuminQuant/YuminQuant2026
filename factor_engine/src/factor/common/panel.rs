use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Arc;

use crate::core::{FactorContext, FactorRowKey, FactorSeries, FactorSpec, FactorValue};
use crate::data::Table;
use crate::error::{err, Result};

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
    columns: BTreeMap<String, Vec<Option<f64>>>,
}

impl DailyPanel {
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
            columns,
        })
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

    pub fn dates(&self) -> &[i32] {
        &self.index.dates
    }

    pub fn instruments(&self) -> &[String] {
        &self.index.instruments
    }
}

#[derive(Clone, Debug)]
pub struct PanelColumn {
    index: Arc<DailyPanelIndex>,
    values: Vec<Option<f64>>,
}

impl PanelColumn {
    pub fn values(&self) -> &[Option<f64>] {
        &self.values
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

    fn with_values(&self, values: Vec<Option<f64>>) -> Self {
        Self {
            index: Arc::clone(&self.index),
            values,
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

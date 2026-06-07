use std::collections::{BTreeMap, BTreeSet};

use crate::core::{DatasetId, FactorContext, FactorRowKey, FactorSeries, FactorSpec, FactorValue};
use crate::data::DataPool;
use crate::data::Table;
use crate::error::{err, Result};

use super::{DailyPanel, PanelColumn};

#[derive(Clone, Debug)]
pub struct PitFinancialRecord {
    pub end_date: i32,
    pub disclosure_date: i32,
    pub report_type: i64,
    pub update_flag: i64,
    columns: BTreeMap<String, Option<f64>>,
}

impl PitFinancialRecord {
    pub fn column(&self, name: &str) -> Option<f64> {
        self.columns
            .get(name)
            .copied()
            .flatten()
            .filter(|value| !value.is_nan())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum FinancialStatementDataset {
    Income,
    BalanceSheet,
    CashFlow,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct FinancialRecordMarker {
    pub dataset: FinancialStatementDataset,
    pub end_date: i32,
    pub disclosure_date: i32,
    pub report_type: i64,
    pub update_flag: i64,
}

impl FinancialRecordMarker {
    pub fn from_record(dataset: FinancialStatementDataset, record: &PitFinancialRecord) -> Self {
        Self {
            dataset,
            end_date: record.end_date,
            disclosure_date: record.disclosure_date,
            report_type: record.report_type,
            update_flag: record.update_flag,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct FinancialSyntheticMarker {
    pub key: &'static str,
    pub value: i64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FinancialEventTableKind {
    Statement,
    DividendLtm,
}

#[derive(Clone, Copy, Debug)]
pub struct FinancialEventTable<'a> {
    table: &'a Table,
    kind: FinancialEventTableKind,
}

impl<'a> FinancialEventTable<'a> {
    pub fn statement(table: &'a Table) -> Self {
        Self {
            table,
            kind: FinancialEventTableKind::Statement,
        }
    }

    pub fn dividend_ltm(table: &'a Table) -> Self {
        Self {
            table,
            kind: FinancialEventTableKind::DividendLtm,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct FinancialEventSchedule {
    event_dates: BTreeSet<i32>,
}

impl FinancialEventSchedule {
    pub fn from_tables(tables: &[FinancialEventTable<'_>]) -> Result<Self> {
        let mut event_dates = BTreeSet::new();
        for event_table in tables {
            match event_table.kind {
                FinancialEventTableKind::Statement => {
                    collect_statement_event_dates(event_table.table, &mut event_dates)?;
                }
                FinancialEventTableKind::DividendLtm => {
                    collect_dividend_ltm_event_dates(event_table.table, &mut event_dates)?;
                }
            }
        }
        Ok(Self { event_dates })
    }

    pub fn has_event_after_until(
        &self,
        after_exclusive: Option<i32>,
        until_inclusive: i32,
    ) -> bool {
        let lower = after_exclusive.unwrap_or(i32::MIN);
        self.event_dates
            .range((lower + 1)..=until_inclusive)
            .next()
            .is_some()
    }
}

#[derive(Clone, Debug, Default)]
pub struct EventDrivenCrossSectionCache {
    latest_values: BTreeMap<String, BTreeMap<String, Option<f64>>>,
    last_processed_trade_date: Option<i32>,
}

impl EventDrivenCrossSectionCache {
    pub fn should_recompute(&self, schedule: &FinancialEventSchedule, trade_date: i32) -> bool {
        self.last_processed_trade_date.is_none()
            || schedule.has_event_after_until(self.last_processed_trade_date, trade_date)
    }

    pub fn update_series(&mut self, series: &FactorSeries) {
        let values = self
            .latest_values
            .entry(series.spec.id.clone())
            .or_default();
        for item in &series.values {
            let FactorRowKey::Daily { ts_code, .. } = &item.key else {
                continue;
            };
            values.insert(ts_code.clone(), item.value);
        }
    }

    pub fn replay_series(
        &self,
        spec: FactorSpec,
        panel: &DailyPanel,
        trade_date: i32,
    ) -> FactorSeries {
        let cached = self.latest_values.get(&spec.id);
        let date_idx = panel
            .dates()
            .iter()
            .position(|date| *date == trade_date)
            .unwrap_or(usize::MAX);
        let instrument_count = panel.instruments().len();
        let mut values = Vec::new();
        if date_idx != usize::MAX {
            let offset = date_idx * instrument_count;
            for (instrument_idx, ts_code) in panel.instruments().iter().enumerate() {
                let panel_idx = offset + instrument_idx;
                if !panel.is_present_offset(panel_idx) {
                    continue;
                }
                values.push(FactorValue {
                    key: FactorRowKey::Daily {
                        trade_date,
                        ts_code: ts_code.clone(),
                    },
                    value: cached.and_then(|values| values.get(ts_code).copied().flatten()),
                });
            }
        }
        FactorSeries { spec, values }
    }

    pub fn mark_processed(&mut self, trade_date: i32) {
        self.last_processed_trade_date = Some(trade_date);
    }
}

pub fn compute_financial_event_snapshot_many<F>(
    requested_ids: &[String],
    context: &FactorContext,
    data: &DataPool,
    state: &mut EventDrivenCrossSectionCache,
    schedule: &FinancialEventSchedule,
    specs: &[FactorSpec],
    mut compute_on_event: F,
) -> Result<Vec<FactorSeries>>
where
    F: FnMut(&[String], &FactorContext, &DataPool) -> Result<Vec<FactorSeries>>,
{
    let panel = data.daily_panel(DatasetId::StockDailyPv)?;
    let mut output = Vec::new();
    for trade_date in &context.target_dates {
        if state.should_recompute(schedule, *trade_date) {
            let event_context = single_target_context(context, *trade_date);
            let event_pool = data.with_target_dates(&[*trade_date]);
            let series_list = compute_on_event(requested_ids, &event_context, &event_pool)?;
            for series in &series_list {
                state.update_series(series);
            }
            output.extend(series_list);
        } else {
            for spec in specs {
                output.push(state.replay_series(spec.clone(), panel, *trade_date));
            }
        }
        state.mark_processed(*trade_date);
    }
    Ok(output)
}

fn collect_statement_event_dates(table: &Table, event_dates: &mut BTreeSet<i32>) -> Result<()> {
    if table.len == 0 {
        return Ok(());
    }
    let ann_dates = table.required_i32_date_cast("ann_date")?;
    let f_ann_dates = table.required_i32_date_cast("f_ann_date")?;
    for idx in 0..table.len {
        if let Some(date) = f_ann_dates[idx].or(ann_dates[idx]) {
            event_dates.insert(date);
        }
    }
    Ok(())
}

fn collect_dividend_ltm_event_dates(table: &Table, event_dates: &mut BTreeSet<i32>) -> Result<()> {
    if table.len == 0 {
        return Ok(());
    }
    let ann_dates = table.required_i32_date_cast("ann_date")?;
    let ex_dates = table.required_i32_date_cast("ex_date")?;
    for idx in 0..table.len {
        if let Some(date) = ann_dates[idx] {
            event_dates.insert(date);
        }
        if let Some(date) = ex_dates[idx] {
            event_dates.insert(date);
            event_dates.insert(add_days(add_months(date, 12), 1));
        }
    }
    Ok(())
}

fn single_target_context(context: &FactorContext, trade_date: i32) -> FactorContext {
    FactorContext {
        asset_class: context.asset_class,
        frequency: context.frequency,
        start_date: trade_date,
        end_date: trade_date,
        load_start_date: context.load_start_date,
        load_dates: context.load_dates.clone(),
        target_dates: vec![trade_date],
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FinancialEventMarker {
    records: Vec<FinancialRecordMarker>,
    synthetic: Vec<FinancialSyntheticMarker>,
}

impl FinancialEventMarker {
    pub fn is_empty(&self) -> bool {
        self.records.is_empty() && self.synthetic.is_empty()
    }
}

#[derive(Clone, Debug, Default)]
pub struct FinancialEventMarkerBuilder {
    records: Vec<FinancialRecordMarker>,
    synthetic: Vec<FinancialSyntheticMarker>,
}

impl FinancialEventMarkerBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn include_record(
        &mut self,
        dataset: FinancialStatementDataset,
        record: Option<&PitFinancialRecord>,
    ) -> &mut Self {
        if let Some(record) = record {
            self.records
                .push(FinancialRecordMarker::from_record(dataset, record));
        }
        self
    }

    pub fn include_record_for_end_date(
        &mut self,
        dataset: FinancialStatementDataset,
        data: &PitFinancialData,
        ts_code: &str,
        trade_date: i32,
        end_date: i32,
    ) -> &mut Self {
        self.include_record(
            dataset,
            data.record_for_end_date(ts_code, trade_date, end_date),
        )
    }

    pub fn include_ttm_for_end_date(
        &mut self,
        dataset: FinancialStatementDataset,
        data: &PitFinancialData,
        ts_code: &str,
        trade_date: i32,
        end_date: i32,
    ) -> &mut Self {
        let mut current = Some(end_date);
        for _ in 0..4 {
            let Some(end_date) = current else {
                break;
            };
            self.include_record_for_end_date(dataset, data, ts_code, trade_date, end_date);
            current = previous_quarter_end_date(end_date);
        }
        self
    }

    pub fn include_latest_ttm(
        &mut self,
        dataset: FinancialStatementDataset,
        data: &PitFinancialData,
        ts_code: &str,
        trade_date: i32,
    ) -> &mut Self {
        if let Some(end_date) = data.latest_quarter_end_date(ts_code, trade_date) {
            self.include_ttm_for_end_date(dataset, data, ts_code, trade_date, end_date);
        }
        self
    }

    pub fn include_latest_quarter(
        &mut self,
        dataset: FinancialStatementDataset,
        data: &PitFinancialData,
        ts_code: &str,
        trade_date: i32,
    ) -> &mut Self {
        if let Some(end_date) = data.latest_quarter_end_date(ts_code, trade_date) {
            self.include_record_for_end_date(dataset, data, ts_code, trade_date, end_date);
        }
        self
    }

    pub fn include_latest_annual(
        &mut self,
        dataset: FinancialStatementDataset,
        data: &PitFinancialData,
        ts_code: &str,
        trade_date: i32,
    ) -> &mut Self {
        if let Some(end_date) = data.latest_annual_end_date(ts_code, trade_date) {
            self.include_record_for_end_date(dataset, data, ts_code, trade_date, end_date);
        }
        self
    }

    pub fn include_annual_chain(
        &mut self,
        dataset: FinancialStatementDataset,
        data: &PitFinancialData,
        ts_code: &str,
        trade_date: i32,
        count: usize,
    ) -> &mut Self {
        let Some(anchor) = data.latest_annual_end_date(ts_code, trade_date) else {
            return self;
        };
        let mut year = anchor / 10_000;
        for _ in 0..count {
            let end_date = year * 10_000 + 12_31;
            self.include_record_for_end_date(dataset, data, ts_code, trade_date, end_date);
            year -= 1;
        }
        self
    }

    pub fn include_synthetic(&mut self, key: &'static str, value: i64) -> &mut Self {
        self.synthetic.push(FinancialSyntheticMarker { key, value });
        self
    }

    pub fn build(mut self) -> Option<FinancialEventMarker> {
        self.records.sort();
        self.records.dedup();
        self.synthetic.sort();
        self.synthetic.dedup();
        let marker = FinancialEventMarker {
            records: self.records,
            synthetic: self.synthetic,
        };
        (!marker.is_empty()).then_some(marker)
    }
}

#[derive(Clone, Debug)]
pub struct FinancialStockSnapshotCache<T> {
    last_marker_by_stock: Vec<Option<FinancialEventMarker>>,
    last_snapshot_by_stock: Vec<Option<T>>,
}

impl<T: Clone> FinancialStockSnapshotCache<T> {
    pub fn new(instrument_count: usize) -> Self {
        Self {
            last_marker_by_stock: vec![None; instrument_count],
            last_snapshot_by_stock: vec![None; instrument_count],
        }
    }

    pub fn clear(&mut self, instrument_idx: usize) {
        if instrument_idx < self.last_marker_by_stock.len() {
            self.last_marker_by_stock[instrument_idx] = None;
            self.last_snapshot_by_stock[instrument_idx] = None;
        }
    }

    pub fn get_or_update<F>(
        &mut self,
        instrument_idx: usize,
        marker: Option<FinancialEventMarker>,
        compute: F,
    ) -> Option<T>
    where
        F: FnOnce() -> Option<T>,
    {
        if instrument_idx >= self.last_marker_by_stock.len() {
            return None;
        }
        let Some(marker) = marker else {
            self.clear(instrument_idx);
            return None;
        };
        if self.last_marker_by_stock[instrument_idx].as_ref() == Some(&marker) {
            return self.last_snapshot_by_stock[instrument_idx].clone();
        }
        let snapshot = compute();
        self.last_marker_by_stock[instrument_idx] = Some(marker);
        self.last_snapshot_by_stock[instrument_idx] = snapshot.clone();
        snapshot
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReportTypePreference {
    order: Vec<i64>,
}

impl ReportTypePreference {
    pub fn new(order: &[i64]) -> Self {
        Self {
            order: order.to_vec(),
        }
    }

    pub fn income_single_quarter() -> Self {
        Self::new(&[3, 2])
    }

    pub fn balance_sheet_consolidated() -> Self {
        Self::new(&[1, 4])
    }

    pub fn consolidated() -> Self {
        Self::new(&[1, 4])
    }

    fn contains(&self, report_type: i64) -> bool {
        self.order.contains(&report_type)
    }
}

pub fn cached_financial_stock_panel<T, MarkerFn, ComputeFn, ValueFn>(
    panel: &DailyPanel,
    mut marker_fn: MarkerFn,
    mut compute_fn: ComputeFn,
    mut value_fn: ValueFn,
) -> Result<PanelColumn>
where
    T: Clone,
    MarkerFn: FnMut(i32, &str) -> Option<FinancialEventMarker>,
    ComputeFn: FnMut(i32, &str) -> Option<T>,
    ValueFn: FnMut(&T, i32, &str, usize) -> Option<f64>,
{
    let instrument_count = panel.instruments().len();
    let mut values = vec![None; panel.shape_len()];
    let mut cache = FinancialStockSnapshotCache::<T>::new(instrument_count);
    for (date_idx, trade_date) in panel.dates().iter().copied().enumerate() {
        if !panel.is_target_date(trade_date) {
            continue;
        }
        let date_offset = date_idx * instrument_count;
        for (instrument_idx, ts_code) in panel.instruments().iter().enumerate() {
            let offset = date_offset + instrument_idx;
            let snapshot =
                cache.get_or_update(instrument_idx, marker_fn(trade_date, ts_code), || {
                    compute_fn(trade_date, ts_code)
                });
            if let Some(snapshot) = snapshot.as_ref() {
                values[offset] = value_fn(snapshot, trade_date, ts_code, offset);
            }
        }
    }
    panel.column_from_values(values)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DeadlinePolicy {
    RequiredAfterDeadline,
}

#[derive(Clone, Debug)]
pub struct PitFinancialData {
    preference: ReportTypePreference,
    by_ts_code: BTreeMap<String, BTreeMap<i32, BTreeMap<i64, Vec<PitFinancialRecord>>>>,
}

impl PitFinancialData {
    pub fn from_table(
        table: &Table,
        value_columns: &[&str],
        preference: ReportTypePreference,
    ) -> Result<Self> {
        let ts_codes = table.required_utf8("ts_code")?;
        let ann_dates = table.required_i32_date_cast("ann_date")?;
        let f_ann_dates = table.required_i32_date_cast("f_ann_date")?;
        let end_dates = table.required_i32_date_cast("end_date")?;
        let update_flags = table.required_i64_cast("update_flag")?;
        let report_types = if table.columns.contains_key("report_type") {
            table.required_i64_cast("report_type")?
        } else {
            vec![Some(1); table.len]
        };
        let mut value_data = BTreeMap::new();
        for column in value_columns {
            value_data.insert((*column).to_string(), table.required_f64_cast(column)?);
        }

        let mut by_ts_code: BTreeMap<
            String,
            BTreeMap<i32, BTreeMap<i64, Vec<PitFinancialRecord>>>,
        > = BTreeMap::new();
        for idx in 0..table.len {
            let (Some(ts_code), Some(end_date), Some(disclosure_date), Some(report_type)) = (
                ts_codes[idx].clone(),
                end_dates[idx],
                f_ann_dates[idx].or(ann_dates[idx]),
                report_types[idx],
            ) else {
                continue;
            };
            if !preference.contains(report_type) {
                continue;
            }
            let columns = value_data
                .iter()
                .map(|(name, values)| (name.clone(), values[idx]))
                .collect::<BTreeMap<_, _>>();
            by_ts_code
                .entry(ts_code)
                .or_default()
                .entry(end_date)
                .or_default()
                .entry(report_type)
                .or_default()
                .push(PitFinancialRecord {
                    end_date,
                    disclosure_date,
                    report_type,
                    update_flag: update_flags[idx].unwrap_or(0),
                    columns,
                });
        }

        for by_end_date in by_ts_code.values_mut() {
            for by_report_type in by_end_date.values_mut() {
                for versions in by_report_type.values_mut() {
                    versions.sort_by(|left, right| {
                        right
                            .disclosure_date
                            .cmp(&left.disclosure_date)
                            .then_with(|| right.update_flag.cmp(&left.update_flag))
                    });
                }
            }
        }

        Ok(Self {
            preference,
            by_ts_code,
        })
    }

    pub fn record_for_end_date(
        &self,
        ts_code: &str,
        trade_date: i32,
        end_date: i32,
    ) -> Option<&PitFinancialRecord> {
        let by_report_type = self.by_ts_code.get(ts_code)?.get(&end_date)?;
        for report_type in &self.preference.order {
            let Some(versions) = by_report_type.get(report_type) else {
                continue;
            };
            if let Some(record) = versions
                .iter()
                .find(|record| record.disclosure_date <= trade_date)
            {
                return Some(record);
            }
        }
        None
    }

    pub fn latest_quarter_end_date(&self, ts_code: &str, trade_date: i32) -> Option<i32> {
        let by_end_date = self.by_ts_code.get(ts_code)?;
        for (&end_date, _) in by_end_date.iter().rev() {
            if self
                .record_for_end_date(ts_code, trade_date, end_date)
                .is_some()
            {
                return Some(end_date);
            }
        }
        None
    }

    pub fn ttm_sum_for_end_date(
        &self,
        ts_code: &str,
        trade_date: i32,
        end_date: i32,
        column: &str,
    ) -> Option<f64> {
        let mut current = end_date;
        let mut sum = 0.0;
        for _ in 0..4 {
            let record = self.record_for_end_date(ts_code, trade_date, current)?;
            sum += record.column(column)?;
            current = previous_quarter_end_date(current)?;
        }
        Some(sum)
    }

    pub fn ttm_sum(&self, ts_code: &str, trade_date: i32, column: &str) -> Option<f64> {
        let by_end_date = self.by_ts_code.get(ts_code)?;
        for (&anchor, _) in by_end_date.iter().rev() {
            if let Some(value) = self.ttm_sum_for_end_date(ts_code, trade_date, anchor, column) {
                return Some(value);
            }
        }
        None
    }

    pub fn latest_annual_end_date(&self, ts_code: &str, trade_date: i32) -> Option<i32> {
        let by_end_date = self.by_ts_code.get(ts_code)?;
        for (&end_date, _) in by_end_date.iter().rev() {
            if end_date % 10_000 == 12_31
                && self
                    .record_for_end_date(ts_code, trade_date, end_date)
                    .is_some()
            {
                return Some(end_date);
            }
        }
        None
    }

    pub fn latest_annual_value(&self, ts_code: &str, trade_date: i32, column: &str) -> Option<f64> {
        let end_date = self.latest_annual_end_date(ts_code, trade_date)?;
        self.annual_value_for_end_date(ts_code, trade_date, end_date, column)
    }

    pub fn annual_value_for_end_date(
        &self,
        ts_code: &str,
        trade_date: i32,
        end_date: i32,
        column: &str,
    ) -> Option<f64> {
        self.record_for_end_date(ts_code, trade_date, end_date)?
            .column(column)
    }

    pub fn annual_values(
        &self,
        ts_code: &str,
        trade_date: i32,
        column: &str,
        count: usize,
    ) -> Option<Vec<f64>> {
        let by_end_date = self.by_ts_code.get(ts_code)?;
        for (&anchor, _) in by_end_date.iter().rev() {
            if anchor % 10_000 != 12_31 {
                continue;
            }
            let mut year = anchor / 10_000;
            let mut values = Vec::with_capacity(count);
            let mut valid = true;
            for _ in 0..count {
                let end_date = year * 10_000 + 12_31;
                let Some(value) =
                    self.annual_value_for_end_date(ts_code, trade_date, end_date, column)
                else {
                    valid = false;
                    break;
                };
                values.push(value);
                year -= 1;
            }
            if valid {
                values.reverse();
                return Some(values);
            }
        }
        None
    }

    pub fn quarters<'a>(
        &'a self,
        panel: &'a DailyPanel,
        column: &str,
        count: usize,
        policy: DeadlinePolicy,
    ) -> Result<QuarterMatrix<'a>> {
        let mut values = Vec::with_capacity(panel.shape_len());
        for trade_date in panel.dates() {
            for ts_code in panel.instruments() {
                let row = self
                    .quarter_chain(ts_code, *trade_date, count, policy)
                    .map(|records| {
                        records
                            .into_iter()
                            .map(|record| {
                                Some(QuarterValue {
                                    end_date: record.end_date,
                                    value: record.column(column),
                                })
                            })
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_else(|| vec![None; count]);
                values.push(row);
            }
        }
        Ok(QuarterMatrix {
            panel,
            quarter_count: count,
            values,
        })
    }

    pub fn quarters_like<'a>(
        &'a self,
        panel: &'a DailyPanel,
        column: &str,
        template: &QuarterMatrix<'a>,
    ) -> Result<QuarterMatrix<'a>> {
        template.require_same_panel(panel)?;
        let mut values = Vec::with_capacity(panel.shape_len());
        for (date_idx, trade_date) in panel.dates().iter().enumerate() {
            for (instrument_idx, ts_code) in panel.instruments().iter().enumerate() {
                let offset = panel_offset(panel, date_idx, instrument_idx);
                let row = template.values[offset]
                    .iter()
                    .map(|quarter| {
                        let end_date = quarter.as_ref()?.end_date;
                        Some(QuarterValue {
                            end_date,
                            value: self
                                .record_for_end_date(ts_code, *trade_date, end_date)
                                .and_then(|record| record.column(column)),
                        })
                    })
                    .collect::<Vec<_>>();
                values.push(row);
            }
        }
        Ok(QuarterMatrix {
            panel,
            quarter_count: template.quarter_count,
            values,
        })
    }

    fn quarter_chain(
        &self,
        ts_code: &str,
        trade_date: i32,
        count: usize,
        _policy: DeadlinePolicy,
    ) -> Option<Vec<&PitFinancialRecord>> {
        if count == 0 {
            return Some(Vec::new());
        }
        let by_end_date = self.by_ts_code.get(ts_code)?;
        let required_anchor = required_anchor_end_date(trade_date);
        let latest_possible = latest_possible_end_date(trade_date);

        for (&anchor, _) in by_end_date.iter().rev() {
            if anchor > latest_possible || anchor < required_anchor {
                continue;
            }
            let mut current = anchor;
            let mut records = Vec::with_capacity(count);
            let mut complete = true;
            for _ in 0..count {
                let Some(record) = self.record_for_end_date(ts_code, trade_date, current) else {
                    complete = false;
                    break;
                };
                records.push(record);
                current = previous_quarter_end_date(current)?;
            }
            if complete {
                return Some(records);
            }
        }
        None
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QuarterValue {
    pub end_date: i32,
    pub value: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct QuarterMatrix<'a> {
    panel: &'a DailyPanel,
    quarter_count: usize,
    values: Vec<Vec<Option<QuarterValue>>>,
}

impl<'a> QuarterMatrix<'a> {
    pub fn binary<F>(&self, other: &Self, mut f: F) -> Result<Self>
    where
        F: FnMut(f64, f64) -> Option<f64>,
    {
        self.require_same_shape(other)?;
        let values = self
            .values
            .iter()
            .zip(&other.values)
            .map(|(left_row, right_row)| {
                left_row
                    .iter()
                    .zip(right_row)
                    .map(|(left, right)| match (left, right) {
                        (Some(left), Some(right)) if left.end_date == right.end_date => {
                            Some(QuarterValue {
                                end_date: left.end_date,
                                value: match (left.value, right.value) {
                                    (Some(left), Some(right))
                                        if !left.is_nan() && !right.is_nan() =>
                                    {
                                        f(left, right)
                                    }
                                    _ => None,
                                },
                            })
                        }
                        _ => None,
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        Ok(Self {
            panel: self.panel,
            quarter_count: self.quarter_count,
            values,
        })
    }

    pub fn mean(&self) -> Result<PanelColumn> {
        let values = self
            .values
            .iter()
            .map(|row| {
                if row.len() != self.quarter_count {
                    return None;
                }
                let mut sum = 0.0;
                for quarter in row {
                    let value = quarter.as_ref()?.value?;
                    if value.is_nan() {
                        return None;
                    }
                    sum += value;
                }
                (self.quarter_count > 0).then_some(sum / self.quarter_count as f64)
            })
            .collect::<Vec<_>>();
        self.panel.column_from_values(values)
    }

    fn require_same_panel(&self, panel: &DailyPanel) -> Result<()> {
        if self.panel.dates() == panel.dates() && self.panel.instruments() == panel.instruments() {
            return Ok(());
        }
        Err(err(
            "quarter matrix and panel use different indexes and cannot be combined",
        ))
    }

    fn require_same_shape(&self, other: &Self) -> Result<()> {
        self.require_same_panel(other.panel)?;
        if self.quarter_count == other.quarter_count && self.values.len() == other.values.len() {
            return Ok(());
        }
        Err(err(
            "quarter matrices use different shapes and cannot be combined",
        ))
    }
}

fn panel_offset(panel: &DailyPanel, date_idx: usize, instrument_idx: usize) -> usize {
    date_idx * panel.instruments().len() + instrument_idx
}

fn latest_possible_end_date(trade_date: i32) -> i32 {
    let year = trade_date / 10_000;
    let mmdd = trade_date % 10_000;
    match mmdd {
        1231..=9999 => year * 10_000 + 1231,
        930..=1230 => year * 10_000 + 930,
        630..=929 => year * 10_000 + 630,
        331..=629 => year * 10_000 + 331,
        _ => (year - 1) * 10_000 + 1231,
    }
}

fn required_anchor_end_date(trade_date: i32) -> i32 {
    let year = trade_date / 10_000;
    let mmdd = trade_date % 10_000;
    match mmdd {
        1031..=9999 => year * 10_000 + 930,
        831..=1030 => year * 10_000 + 630,
        430..=830 => year * 10_000 + 331,
        _ => (year - 1) * 10_000 + 930,
    }
}

pub fn previous_quarter_end_date(end_date: i32) -> Option<i32> {
    let year = end_date / 10_000;
    match end_date % 10_000 {
        331 => Some((year - 1) * 10_000 + 1231),
        630 => Some(year * 10_000 + 331),
        930 => Some(year * 10_000 + 630),
        1231 => Some(year * 10_000 + 930),
        _ => None,
    }
}

fn add_months(date: i32, months_delta: i32) -> i32 {
    let (year, month, day) = ymd(date);
    let month_index = year * 12 + month as i32 - 1 + months_delta;
    let new_year = month_index.div_euclid(12);
    let new_month = month_index.rem_euclid(12) + 1;
    let new_day = day.min(days_in_month(new_year, new_month as u32));
    new_year * 10_000 + new_month * 100 + new_day as i32
}

fn add_days(date: i32, days_delta: i32) -> i32 {
    if days_delta == 0 {
        return date;
    }
    let (mut year, mut month, mut day) = ymd(date);
    if days_delta > 0 {
        for _ in 0..days_delta {
            day += 1;
            let days = days_in_month(year, month);
            if day > days {
                day = 1;
                month += 1;
                if month > 12 {
                    month = 1;
                    year += 1;
                }
            }
        }
    } else {
        for _ in days_delta..0 {
            if day > 1 {
                day -= 1;
            } else {
                if month > 1 {
                    month -= 1;
                } else {
                    month = 12;
                    year -= 1;
                }
                day = days_in_month(year, month);
            }
        }
    }
    year * 10_000 + month as i32 * 100 + day as i32
}

fn ymd(date: i32) -> (i32, u32, u32) {
    (
        date / 10_000,
        ((date / 100) % 100) as u32,
        (date % 100) as u32,
    )
}

fn days_in_month(year: i32, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if is_leap_year(year) => 29,
        2 => 28,
        _ => 30,
    }
}

fn is_leap_year(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use crate::core::{
        AssetClass, DataRequest, FactorContext, FactorRowKey, FactorSeries, FactorSpec,
        FactorValue, Frequency, Lookback,
    };
    use crate::data::{ColumnData, Table};

    use super::{latest_possible_end_date, required_anchor_end_date};
    use super::{
        DeadlinePolicy, EventDrivenCrossSectionCache, FinancialEventMarkerBuilder,
        FinancialEventSchedule, FinancialEventTable, FinancialStatementDataset,
        FinancialStockSnapshotCache, PitFinancialData, ReportTypePreference,
    };

    fn financial_table(rows: &[(i32, i32, i64, i64, f64)]) -> Table {
        Table::new(BTreeMap::from([
            (
                "ts_code".to_string(),
                ColumnData::Utf8(rows.iter().map(|_| Some("000001.SZ".to_string())).collect()),
            ),
            (
                "ann_date".to_string(),
                ColumnData::I32(rows.iter().map(|(_, date, _, _, _)| Some(*date)).collect()),
            ),
            (
                "f_ann_date".to_string(),
                ColumnData::I32(rows.iter().map(|(_, date, _, _, _)| Some(*date)).collect()),
            ),
            (
                "end_date".to_string(),
                ColumnData::I32(rows.iter().map(|(end, _, _, _, _)| Some(*end)).collect()),
            ),
            (
                "report_type".to_string(),
                ColumnData::I64(
                    rows.iter()
                        .map(|(_, _, report_type, _, _)| Some(*report_type))
                        .collect(),
                ),
            ),
            (
                "update_flag".to_string(),
                ColumnData::I64(
                    rows.iter()
                        .map(|(_, _, _, update_flag, _)| Some(*update_flag))
                        .collect(),
                ),
            ),
            (
                "value".to_string(),
                ColumnData::F64(
                    rows.iter()
                        .map(|(_, _, _, _, value)| Some(*value))
                        .collect(),
                ),
            ),
        ]))
        .expect("valid table")
    }

    fn panel(dates: &[i32]) -> crate::factor::common::DailyPanel {
        let table = Table::new(BTreeMap::from([
            (
                "trade_date".to_string(),
                ColumnData::I32(dates.iter().map(|date| Some(*date)).collect()),
            ),
            (
                "ts_code".to_string(),
                ColumnData::Utf8(
                    dates
                        .iter()
                        .map(|_| Some("000001.SZ".to_string()))
                        .collect(),
                ),
            ),
            (
                "dummy".to_string(),
                ColumnData::F64(dates.iter().map(|_| Some(1.0)).collect()),
            ),
        ]))
        .expect("valid table");
        let context = FactorContext {
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            start_date: *dates.first().unwrap(),
            end_date: *dates.last().unwrap(),
            load_start_date: *dates.first().unwrap(),
            load_dates: dates.to_vec(),
            target_dates: dates.to_vec(),
        };
        crate::factor::common::DailyPanel::from_table(&table, &context).expect("panel")
    }

    fn financial_table_with_codes(rows: &[(&str, i32, i32, i64, i64, f64)]) -> Table {
        Table::new(BTreeMap::from([
            (
                "ts_code".to_string(),
                ColumnData::Utf8(
                    rows.iter()
                        .map(|(ts_code, _, _, _, _, _)| Some((*ts_code).to_string()))
                        .collect(),
                ),
            ),
            (
                "ann_date".to_string(),
                ColumnData::I32(
                    rows.iter()
                        .map(|(_, _, date, _, _, _)| Some(*date))
                        .collect(),
                ),
            ),
            (
                "f_ann_date".to_string(),
                ColumnData::I32(
                    rows.iter()
                        .map(|(_, _, date, _, _, _)| Some(*date))
                        .collect(),
                ),
            ),
            (
                "end_date".to_string(),
                ColumnData::I32(rows.iter().map(|(_, end, _, _, _, _)| Some(*end)).collect()),
            ),
            (
                "report_type".to_string(),
                ColumnData::I64(
                    rows.iter()
                        .map(|(_, _, _, report_type, _, _)| Some(*report_type))
                        .collect(),
                ),
            ),
            (
                "update_flag".to_string(),
                ColumnData::I64(
                    rows.iter()
                        .map(|(_, _, _, _, update_flag, _)| Some(*update_flag))
                        .collect(),
                ),
            ),
            (
                "value".to_string(),
                ColumnData::F64(
                    rows.iter()
                        .map(|(_, _, _, _, _, value)| Some(*value))
                        .collect(),
                ),
            ),
        ]))
        .expect("valid table")
    }

    fn statutory_disclosure_date(end_date: i32) -> i32 {
        let year = end_date / 10_000;
        match end_date % 10_000 {
            331 => year * 10_000 + 430,
            630 => year * 10_000 + 831,
            930 => year * 10_000 + 1031,
            1231 => (year + 1) * 10_000 + 430,
            _ => end_date,
        }
    }

    fn event_spec(id: &str) -> FactorSpec {
        FactorSpec {
            id: id.to_string(),
            aliases: Vec::new(),
            name: id.to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: "0.1.0".to_string(),
            tags: Vec::new(),
            description: String::new(),
            dependencies: Vec::<DataRequest>::new(),
            intraday_raw_dependencies: Vec::new(),
            lookback: Lookback { trading_days: 0 },
        }
    }

    #[test]
    fn financial_event_schedule_maps_statement_and_dividend_dates() {
        let statement = financial_table(&[(20251231, 20260103, 1, 0, 1.0)]);
        let dividend = Table::new(BTreeMap::from([
            (
                "ann_date".to_string(),
                ColumnData::I32(vec![Some(20250115)]),
            ),
            ("ex_date".to_string(), ColumnData::I32(vec![Some(20250131)])),
        ]))
        .expect("valid dividend table");
        let schedule = FinancialEventSchedule::from_tables(&[
            FinancialEventTable::statement(&statement),
            FinancialEventTable::dividend_ltm(&dividend),
        ])
        .expect("schedule");

        assert!(schedule.has_event_after_until(Some(20260102), 20260105));
        assert!(!schedule.has_event_after_until(Some(20260105), 20260106));
        assert!(schedule.has_event_after_until(Some(20260131), 20260201));
    }

    #[test]
    fn event_driven_cross_section_cache_replays_latest_values_daily() {
        let panel = panel(&[20260105, 20260106]);
        let spec = event_spec("slow_factor");
        let mut cache = EventDrivenCrossSectionCache::default();
        cache.update_series(&FactorSeries {
            spec: spec.clone(),
            values: vec![FactorValue {
                key: FactorRowKey::Daily {
                    trade_date: 20260105,
                    ts_code: "000001.SZ".to_string(),
                },
                value: Some(1.23),
            }],
        });

        let replay = cache.replay_series(spec, &panel, 20260106);

        assert_eq!(replay.values.len(), 1);
        assert_eq!(
            replay.values[0].key,
            FactorRowKey::Daily {
                trade_date: 20260106,
                ts_code: "000001.SZ".to_string(),
            }
        );
        assert_eq!(replay.values[0].value, Some(1.23));
    }

    #[test]
    fn report_type_preference_uses_order_before_fallback() {
        let table = financial_table(&[
            (20250331, 20250420, 1, 0, 1.0),
            (20250331, 20250420, 2, 0, 2.0),
            (20250331, 20250420, 3, 0, 3.0),
            (20250331, 20250420, 4, 0, 4.0),
        ]);
        let income = PitFinancialData::from_table(
            &table,
            &["value"],
            ReportTypePreference::income_single_quarter(),
        )
        .expect("income");
        let balance = PitFinancialData::from_table(
            &table,
            &["value"],
            ReportTypePreference::balance_sheet_consolidated(),
        )
        .expect("balance");

        assert_eq!(
            income
                .record_for_end_date("000001.SZ", 20250501, 20250331)
                .and_then(|record| record.column("value")),
            Some(3.0)
        );
        assert_eq!(
            balance
                .record_for_end_date("000001.SZ", 20250501, 20250331)
                .and_then(|record| record.column("value")),
            Some(1.0)
        );
    }

    #[test]
    fn pit_financial_data_uses_only_disclosed_versions() {
        let table = financial_table(&[
            (20241231, 20250331, 3, 0, 10.0),
            (20241231, 20250430, 3, 1, 12.0),
        ]);
        let data = PitFinancialData::from_table(
            &table,
            &["value"],
            ReportTypePreference::income_single_quarter(),
        )
        .expect("pit data");

        assert_eq!(
            data.record_for_end_date("000001.SZ", 20250401, 20241231)
                .and_then(|record| record.column("value")),
            Some(10.0)
        );
        assert_eq!(
            data.record_for_end_date("000001.SZ", 20250501, 20241231)
                .and_then(|record| record.column("value")),
            Some(12.0)
        );
    }

    #[test]
    fn pit_financial_data_exposes_annual_and_ttm_helpers() {
        let table = financial_table(&[
            (20211231, 20220430, 1, 0, 10.0),
            (20221231, 20230430, 1, 0, 20.0),
            (20230331, 20230430, 1, 0, 1.0),
            (20230630, 20230831, 1, 0, 2.0),
            (20230930, 20231031, 1, 0, 3.0),
            (20231231, 20240430, 1, 0, 4.0),
            (20241231, 20250430, 1, 0, 40.0),
        ]);
        let data =
            PitFinancialData::from_table(&table, &["value"], ReportTypePreference::consolidated())
                .expect("pit data");

        assert_eq!(
            data.latest_annual_end_date("000001.SZ", 20240501),
            Some(20231231)
        );
        assert_eq!(
            data.latest_annual_value("000001.SZ", 20240501, "value"),
            Some(4.0)
        );
        assert_eq!(
            data.annual_value_for_end_date("000001.SZ", 20240501, 20221231, "value"),
            Some(20.0)
        );
        assert_eq!(
            data.annual_values("000001.SZ", 20250501, "value", 3),
            Some(vec![20.0, 4.0, 40.0])
        );
        assert_eq!(data.ttm_sum("000001.SZ", 20240501, "value"), Some(10.0));
    }

    #[test]
    fn stock_snapshot_cache_reuses_until_stock_marker_changes() {
        let table = financial_table_with_codes(&[
            ("000001.SZ", 20241231, 20250331, 1, 0, 10.0),
            ("000001.SZ", 20250331, 20250428, 1, 0, 11.0),
            ("000002.SZ", 20241231, 20250331, 1, 0, 20.0),
        ]);
        let data =
            PitFinancialData::from_table(&table, &["value"], ReportTypePreference::consolidated())
                .expect("pit data");
        let mut cache = FinancialStockSnapshotCache::<f64>::new(2);
        let mut calls = [0usize; 2];

        for trade_date in [20250425, 20250428, 20250429] {
            for (idx, ts_code) in ["000001.SZ", "000002.SZ"].iter().enumerate() {
                let mut builder = FinancialEventMarkerBuilder::new();
                builder.include_latest_quarter(
                    FinancialStatementDataset::Income,
                    &data,
                    ts_code,
                    trade_date,
                );
                let marker = builder.build();
                let value = cache.get_or_update(idx, marker, || {
                    calls[idx] += 1;
                    data.latest_quarter_end_date(ts_code, trade_date)
                        .and_then(|end_date| {
                            data.record_for_end_date(ts_code, trade_date, end_date)
                                .and_then(|record| record.column("value"))
                        })
                });
                assert!(value.is_some());
            }
        }

        assert_eq!(calls, [2, 1]);
    }

    #[test]
    fn stock_snapshot_cache_does_not_reuse_when_marker_is_missing() {
        let mut cache = FinancialStockSnapshotCache::<f64>::new(1);
        let mut calls = 0usize;
        let mut builder = FinancialEventMarkerBuilder::new();
        builder.include_synthetic("event", 1);
        let marker = builder.build();
        assert_eq!(
            cache.get_or_update(0, marker, || {
                calls += 1;
                Some(1.0)
            }),
            Some(1.0)
        );
        assert_eq!(cache.get_or_update(0, None, || Some(2.0)), None);
        let mut builder = FinancialEventMarkerBuilder::new();
        builder.include_synthetic("event", 1);
        assert_eq!(
            cache.get_or_update(0, builder.build(), || {
                calls += 1;
                Some(3.0)
            }),
            Some(3.0)
        );
        assert_eq!(calls, 2);
    }

    #[test]
    fn deadline_policy_falls_back_before_deadline_but_requires_latest_after_deadline() {
        let mut rows = Vec::new();
        for end_date in [
            20231231, 20240331, 20240630, 20240930, 20241231, 20250331, 20250630, 20250930,
            20251231,
        ] {
            rows.push((end_date, statutory_disclosure_date(end_date), 3, 0, 1.0));
        }
        rows.push((20260331, 20260507, 3, 0, 2.0));
        let data = PitFinancialData::from_table(
            &financial_table(&rows),
            &["value"],
            ReportTypePreference::income_single_quarter(),
        )
        .expect("pit data");
        let panel = panel(&[20260429, 20260506, 20260508]);
        let quarters = data
            .quarters(&panel, "value", 8, DeadlinePolicy::RequiredAfterDeadline)
            .expect("quarters");
        let mean = quarters.mean().expect("mean");

        assert_eq!(mean.values()[0], Some(1.0));
        assert_eq!(mean.values()[1], None);
        assert_eq!(mean.values()[2], Some(1.125));
    }

    #[test]
    fn quarter_deadlines_map_to_expected_required_periods() {
        assert_eq!(required_anchor_end_date(20260429), 20250930);
        assert_eq!(required_anchor_end_date(20260430), 20260331);
        assert_eq!(required_anchor_end_date(20260831), 20260630);
        assert_eq!(required_anchor_end_date(20261031), 20260930);
        assert_eq!(latest_possible_end_date(20260429), 20260331);
    }
}

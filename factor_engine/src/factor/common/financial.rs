use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use crate::core::{FactorContext, FactorRowKey, FactorSeries, FactorSpec, FactorValue};
use crate::data::DataPool;
use crate::data::{ColumnData, Table};
use crate::error::Result;

use super::{DailyPanel, PanelColumn};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum FinancialStatementDataset {
    Income,
    BalanceSheet,
    CashFlow,
}

const IMPLEMENTED_DIV_PROC: &str = "\u{5b9e}\u{65bd}";

fn next_index_generation() -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    NEXT.fetch_add(1, Ordering::Relaxed)
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
    pub fn from_fields(
        dataset: FinancialStatementDataset,
        end_date: i32,
        disclosure_date: i32,
        report_type: i64,
        update_flag: i64,
    ) -> Self {
        Self {
            dataset,
            end_date,
            disclosure_date,
            report_type,
            update_flag,
        }
    }

    pub fn from_record_view(
        dataset: FinancialStatementDataset,
        record: &PitFinancialRecordView<'_>,
    ) -> Self {
        Self::from_fields(
            dataset,
            record.end_date(),
            record.disclosure_date(),
            record.report_type(),
            record.update_flag(),
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct FinancialSyntheticMarker {
    pub key: &'static str,
    pub value: i64,
}

#[derive(Clone, Debug, Default)]
pub struct FinancialEventSchedule {
    event_dates: BTreeSet<i32>,
}

impl FinancialEventSchedule {
    pub fn from_pit_readers(readers: &[FinancialPitReader<'_>]) -> Self {
        let mut event_dates = BTreeSet::new();
        for reader in readers {
            collect_pit_reader_event_dates(reader, &mut event_dates);
        }
        Self { event_dates }
    }

    pub fn from_dividend_reader(reader: &DividendReader<'_>) -> Self {
        Self {
            event_dates: reader.index.event_dates.clone(),
        }
    }

    /// Annual sums become effective only after both announcement and ex-date;
    /// unlike LTM sums they do not expire twelve months later.
    pub fn from_annual_dividend_reader(reader: &DividendReader<'_>) -> Self {
        Self {
            event_dates: reader
                .index
                .records
                .iter()
                .filter(|record| record.end_date.is_some())
                .map(|record| record.ann_date.max(record.ex_date))
                .collect(),
        }
    }

    pub fn merge(&mut self, other: Self) {
        self.event_dates.extend(other.event_dates);
    }

    pub fn has_event_after_until(
        &self,
        after_exclusive: Option<i32>,
        until_inclusive: i32,
    ) -> bool {
        let lower = after_exclusive.unwrap_or(i32::MIN);
        if lower >= until_inclusive {
            return false;
        }
        self.event_dates
            .range((lower + 1)..=until_inclusive)
            .next()
            .is_some()
    }
}

fn collect_pit_reader_event_dates(
    reader: &FinancialPitReader<'_>,
    event_dates: &mut BTreeSet<i32>,
) {
    for part in reader.index.leaves() {
        for report_type in &reader.preference.order {
            if let Some(dates) = part.event_dates_by_type.get(report_type) {
                let end = dates.partition_point(|date| {
                    *date <= reader.index.max_disclosure_date.unwrap_or(i32::MAX)
                });
                event_dates.extend(&dates[..end]);
            }
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct EventDrivenCrossSectionCache {
    latest_values: BTreeMap<String, CachedCrossSection>,
    last_processed_trade_date: Option<i32>,
}

#[derive(Clone, Debug, Default)]
struct CachedCrossSection {
    instruments: Vec<String>,
    values: Vec<Option<f64>>,
    lookup: BTreeMap<String, usize>,
}

impl EventDrivenCrossSectionCache {
    pub fn should_recompute(&self, schedule: &FinancialEventSchedule, trade_date: i32) -> bool {
        self.last_processed_trade_date.is_none()
            || schedule.has_event_after_until(self.last_processed_trade_date, trade_date)
    }

    pub fn update_series(&mut self, series: &FactorSeries, panel: &DailyPanel) {
        let instrument_count = panel.instruments().len();
        let instrument_lookup = panel
            .instruments()
            .iter()
            .enumerate()
            .map(|(idx, ts_code)| (ts_code.as_str(), idx))
            .collect::<BTreeMap<_, _>>();
        let cached = self
            .latest_values
            .entry(series.spec.id.clone())
            .or_insert_with(|| CachedCrossSection {
                instruments: panel.instruments().to_vec(),
                values: vec![None; instrument_count],
                lookup: panel
                    .instruments()
                    .iter()
                    .enumerate()
                    .map(|(idx, ts_code)| (ts_code.clone(), idx))
                    .collect(),
            });
        if cached.instruments != panel.instruments() {
            cached.instruments = panel.instruments().to_vec();
            cached.lookup = cached
                .instruments
                .iter()
                .enumerate()
                .map(|(idx, ts_code)| (ts_code.clone(), idx))
                .collect();
            cached.values.clear();
            cached.values.resize(instrument_count, None);
        } else {
            cached.values.fill(None);
        }
        for item in &series.values {
            let FactorRowKey::Daily { ts_code, .. } = &item.key else {
                continue;
            };
            if let Some(instrument_idx) = instrument_lookup.get(ts_code.as_str()).copied() {
                cached.values[instrument_idx] = item.value;
            }
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
                let value = cached.and_then(|cached| {
                    if cached.instruments.get(instrument_idx) == Some(ts_code) {
                        cached.values.get(instrument_idx).copied().flatten()
                    } else {
                        cached.lookup.get(ts_code).and_then(|cached_idx| {
                            cached.values.get(*cached_idx).copied().flatten()
                        })
                    }
                });
                values.push(FactorValue {
                    key: FactorRowKey::Daily {
                        trade_date,
                        ts_code: ts_code.clone(),
                    },
                    value,
                });
            }
        }
        FactorSeries { spec, values }
    }

    pub fn mark_processed(&mut self, trade_date: i32) {
        self.last_processed_trade_date = Some(trade_date);
    }
}

pub fn compute_financial_event_snapshot_streaming_on_panel<F>(
    requested_ids: &[String],
    context: &FactorContext,
    data: &DataPool,
    panel: &DailyPanel,
    state: &mut EventDrivenCrossSectionCache,
    schedule: &FinancialEventSchedule,
    specs: &[FactorSpec],
    mut compute_on_event: F,
) -> Result<Vec<FactorSeries>>
where
    F: FnMut(&[String], &FactorContext, &DataPool) -> Result<Vec<FactorSeries>>,
{
    let mut output_by_factor = specs
        .iter()
        .map(|spec| {
            (
                spec.id.clone(),
                FactorSeries {
                    spec: spec.clone(),
                    values: Vec::new(),
                },
            )
        })
        .collect::<BTreeMap<_, _>>();
    for trade_date in &context.target_dates {
        if state.should_recompute(schedule, *trade_date) {
            let event_context = multi_target_context(context, &[*trade_date]);
            let event_pool = data.slice_dates(&[*trade_date]);
            let event_series_by_id = compute_on_event(requested_ids, &event_context, &event_pool)?
                .into_iter()
                .map(|series| (series.spec.id.clone(), series))
                .collect::<BTreeMap<_, _>>();
            for spec in specs {
                let series = event_series_by_id
                    .get(&spec.id)
                    .cloned()
                    .unwrap_or_else(|| FactorSeries {
                        spec: spec.clone(),
                        values: Vec::new(),
                    });
                state.update_series(&series, panel);
                append_series_values(&mut output_by_factor, series);
            }
        } else {
            for spec in specs {
                let series = state.replay_series(spec.clone(), panel, *trade_date);
                append_series_values(&mut output_by_factor, series);
            }
        }
        state.mark_processed(*trade_date);
    }
    Ok(specs
        .iter()
        .filter_map(|spec| output_by_factor.remove(&spec.id))
        .collect())
}

fn append_series_values(
    output_by_factor: &mut BTreeMap<String, FactorSeries>,
    series: FactorSeries,
) {
    if let Some(output) = output_by_factor.get_mut(&series.spec.id) {
        output.values.extend(series.values);
    }
}

pub fn factor_series_to_panel_column(
    panel: &DailyPanel,
    series: &FactorSeries,
) -> Result<PanelColumn> {
    let date_lookup = panel
        .dates()
        .iter()
        .enumerate()
        .map(|(idx, date)| (*date, idx))
        .collect::<BTreeMap<_, _>>();
    let instrument_lookup = panel
        .instruments()
        .iter()
        .enumerate()
        .map(|(idx, ts_code)| (ts_code.as_str(), idx))
        .collect::<BTreeMap<_, _>>();
    let instrument_count = panel.instruments().len();
    let mut values = vec![None; panel.shape_len()];
    for item in &series.values {
        let FactorRowKey::Daily {
            trade_date,
            ts_code,
        } = &item.key
        else {
            continue;
        };
        let (Some(date_idx), Some(instrument_idx)) = (
            date_lookup.get(trade_date).copied(),
            instrument_lookup.get(ts_code.as_str()).copied(),
        ) else {
            continue;
        };
        values[date_idx * instrument_count + instrument_idx] = item.value;
    }
    panel.column_from_values(values)
}

pub fn financial_event_trade_dates(
    last_processed_trade_date: Option<i32>,
    schedule: &FinancialEventSchedule,
    target_dates: &[i32],
) -> Vec<i32> {
    let mut last_processed = last_processed_trade_date;
    let mut event_trade_dates = Vec::new();
    for trade_date in target_dates {
        let should_recompute =
            last_processed.is_none() || schedule.has_event_after_until(last_processed, *trade_date);
        if should_recompute {
            event_trade_dates.push(*trade_date);
        }
        last_processed = Some(*trade_date);
    }
    event_trade_dates
}

#[derive(Clone, Debug)]
pub struct DividendIndex {
    generation: u64,
    records: Vec<DividendIndexedRecord>,
    by_ts_code: HashMap<String, Vec<usize>>,
    event_dates: BTreeSet<i32>,
    stock_event_dates: HashMap<String, Vec<i32>>,
}

#[derive(Clone, Debug)]
struct DividendIndexedRecord {
    ts_code: String,
    end_date: Option<i32>,
    ann_date: i32,
    ex_date: i32,
    cash_amount: f64,
}

impl DividendIndex {
    pub fn from_table(table: Arc<Table>) -> Result<Self> {
        let ts_codes = table.required_utf8("ts_code")?;
        let ann_dates = table.required_i32_date_cast("ann_date")?;
        let div_proc = table.required_utf8("div_proc")?;
        let cash_div_tax = table.required_f64_cast("cash_div_tax")?;
        let ex_dates = table.required_i32_date_cast("ex_date")?;
        let base_share = table.required_f64_cast("base_share")?;
        let end_dates = table
            .columns
            .contains_key("end_date")
            .then(|| table.required_i32_date_cast("end_date"))
            .transpose()?;

        let mut records = Vec::new();
        let mut event_dates = BTreeSet::new();
        for idx in 0..table.len {
            if !div_proc[idx]
                .as_deref()
                .is_some_and(|value| value.trim() == IMPLEMENTED_DIV_PROC)
            {
                continue;
            }
            let (
                Some(ts_code),
                Some(ann_date),
                Some(ex_date),
                Some(cash_div_tax),
                Some(base_share),
            ) = (
                ts_codes[idx].clone(),
                ann_dates[idx],
                ex_dates[idx],
                clean_f64(cash_div_tax[idx]),
                clean_f64(base_share[idx]).filter(|value| *value > 0.0),
            )
            else {
                continue;
            };
            records.push(DividendIndexedRecord {
                ts_code,
                end_date: end_dates.as_ref().and_then(|dates| dates[idx]),
                ann_date,
                ex_date,
                cash_amount: cash_div_tax * base_share,
            });
            event_dates.insert(ann_date);
            event_dates.insert(ex_date);
            event_dates.insert(add_days(add_months(ex_date, 12), 1));
        }
        let mut by_ts_code = HashMap::<String, Vec<usize>>::new();
        for (idx, record) in records.iter().enumerate() {
            by_ts_code
                .entry(record.ts_code.clone())
                .or_default()
                .push(idx);
        }
        let mut stock_event_dates = HashMap::<String, Vec<i32>>::new();
        for record in &records {
            let mut expiry = add_days(add_months(record.ex_date, 12), 1);
            while add_months(expiry, -12) <= record.ex_date {
                expiry = add_days(expiry, 1);
            }
            stock_event_dates
                .entry(record.ts_code.clone())
                .or_default()
                .extend([
                    record.ann_date,
                    record.ex_date,
                    add_days(add_months(record.ex_date, 12), 1),
                    expiry,
                ]);
        }
        for dates in stock_event_dates.values_mut() {
            dates.sort_unstable();
            dates.dedup();
        }
        Ok(Self {
            records,
            by_ts_code,
            event_dates,
            generation: next_index_generation(),
            stock_event_dates,
        })
    }

    pub fn reader(&self) -> DividendReader<'_> {
        DividendReader { index: self }
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct DividendReader<'a> {
    index: &'a DividendIndex,
}

impl<'a> DividendReader<'a> {
    /// Fiscal-year sums in one pass, preserving the per-stock summation order.
    pub fn implemented_annual_sums_by_stock(
        &self,
        first_year: i32,
        last_year: i32,
        trade_date: i32,
    ) -> BTreeMap<i32, HashMap<&'a str, f64>> {
        let mut years: BTreeMap<_, HashMap<&str, f64>> = (first_year..=last_year)
            .map(|year| (year, HashMap::new()))
            .collect();
        for record in &self.index.records {
            if record.ann_date > trade_date || record.ex_date > trade_date {
                continue;
            }
            if let Some(sums) = record
                .end_date
                .and_then(|end| years.get_mut(&(end / 10000)))
            {
                *sums.entry(record.ts_code.as_str()).or_default() += record.cash_amount;
            }
        }
        years
    }

    /// Implemented cash distributions for a fiscal year, in ten thousand yuan.
    /// Both announcement and ex-date must be known by the requested PIT date.
    pub fn implemented_annual_sum_by_stock(
        &self,
        year: i32,
        trade_date: i32,
    ) -> HashMap<&'a str, f64> {
        let mut sums = HashMap::new();
        for record in &self.index.records {
            if record.end_date.is_some_and(|end| end / 10000 == year)
                && record.ann_date <= trade_date
                && record.ex_date <= trade_date
            {
                *sums.entry(record.ts_code.as_str()).or_default() += record.cash_amount;
            }
        }
        sums
    }

    pub fn implemented_ltm_sum(&self, ts_code: &str, start_date: i32, trade_date: i32) -> f64 {
        self.index
            .by_ts_code
            .get(ts_code)
            .into_iter()
            .flatten()
            .map(|&idx| &self.index.records[idx])
            .filter(|record| {
                record.ann_date <= trade_date
                    && record.ex_date <= trade_date
                    && record.ex_date >= start_date
            })
            .map(|record| record.cash_amount)
            .sum()
    }

    pub fn implemented_ltm_sum_by_stock(
        &self,
        start_date: i32,
        trade_date: i32,
    ) -> HashMap<&'a str, f64> {
        let mut sums = HashMap::new();
        for record in &self.index.records {
            if record.ann_date > trade_date
                || record.ex_date > trade_date
                || record.ex_date < start_date
            {
                continue;
            }
            *sums.entry(record.ts_code.as_str()).or_default() += record.cash_amount;
        }
        sums
    }
}

#[derive(Clone, Debug)]
pub struct MainBusinessIndex {
    generation: u64,
    rows: Vec<MainBusinessIndexedRow>,
    by_ts_end: BTreeMap<String, BTreeMap<i32, Vec<usize>>>,
    sources: Vec<Arc<Table>>,
    industry_fingerprints: BTreeMap<String, BTreeMap<i32, i64>>,
}

#[derive(Clone, Copy, Debug)]
struct MainBusinessIndexedRow {
    source_idx: usize,
    row_idx: usize,
    end_date: i32,
    update_flag: i64,
    sales: Option<f64>,
}

impl MainBusinessIndex {
    pub fn from_table(table: Arc<Table>) -> Result<Self> {
        Self::from_source_tables(vec![table], i32::MAX)
    }

    pub fn from_source_tables(sources: Vec<Arc<Table>>, max_end: i32) -> Result<Self> {
        let mut rows = Vec::new();
        let mut by_ts_end = BTreeMap::<String, BTreeMap<i32, Vec<usize>>>::new();
        for (source_idx, table) in sources.iter().enumerate() {
            let codes = table.required_utf8("ts_code")?;
            let ends = table.required_i32_date_cast("end_date")?;
            let flags = table.required_i64_cast("update_flag")?;
            table.required_utf8("bz_type")?;
            table.required_utf8("bz_item")?;
            let sales = table.required_f64_cast("bz_sales")?;
            for idx in 0..table.len {
                let (Some(code), Some(end_date), Some(update_flag)) =
                    (codes[idx].as_deref(), ends[idx], flags[idx])
                else {
                    continue;
                };
                if end_date > max_end {
                    continue;
                }
                let pos = rows.len();
                rows.push(MainBusinessIndexedRow {
                    source_idx,
                    row_idx: idx,
                    end_date,
                    update_flag,
                    sales: sales[idx],
                });
                by_ts_end
                    .entry(code.to_owned())
                    .or_default()
                    .entry(end_date)
                    .or_default()
                    .push(pos);
            }
        }
        let mut index = Self {
            generation: next_index_generation(),
            sources,
            rows,
            by_ts_end,
            industry_fingerprints: BTreeMap::new(),
        };
        for (code, periods) in &index.by_ts_end {
            let mut fingerprints = BTreeMap::new();
            for &end in periods.keys() {
                if let Some(value) = index
                    .reader()
                    .compute_industry_update0_fingerprint(code, end)
                {
                    fingerprints.insert(end, value);
                }
            }
            index
                .industry_fingerprints
                .insert(code.clone(), fingerprints);
        }
        Ok(index)
    }

    pub fn reader(&self) -> MainBusinessReader<'_> {
        MainBusinessReader { index: self }
    }

    pub fn len(&self) -> usize {
        self.rows.len()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct MainBusinessRecordView<'a> {
    index: &'a MainBusinessIndex,
    row_pos: usize,
}

impl<'a> MainBusinessRecordView<'a> {
    pub fn end_date(&self) -> i32 {
        self.index.rows[self.row_pos].end_date
    }

    pub fn update_flag(&self) -> i64 {
        self.index.rows[self.row_pos].update_flag
    }

    fn string(&self, name: &str) -> Option<&'a str> {
        let row = &self.index.rows[self.row_pos];
        self.index.sources[row.source_idx]
            .required_utf8(name)
            .ok()?
            .get(row.row_idx)?
            .as_deref()
    }

    pub fn bz_type(&self) -> Option<&'a str> {
        self.string("bz_type")
    }
    pub fn bz_item(&self) -> Option<&'a str> {
        self.string("bz_item")
    }
    pub fn bz_sales(&self) -> Option<f64> {
        self.index.rows[self.row_pos].sales
    }
}

#[derive(Clone, Copy, Debug)]
pub struct MainBusinessReader<'a> {
    index: &'a MainBusinessIndex,
}

impl<'a> MainBusinessReader<'a> {
    pub fn latest_industry_update0_end_date(
        &self,
        ts_code: &str,
        max_end_date: i32,
    ) -> Option<i32> {
        self.index
            .by_ts_end
            .get(ts_code)?
            .range(..=max_end_date)
            .rev()
            .find_map(|(end_date, row_positions)| {
                let has_valid_kind = row_positions.iter().copied().any(|row_pos| {
                    let record = MainBusinessRecordView {
                        index: self.index,
                        row_pos,
                    };
                    record.update_flag() == 0
                        && record
                            .bz_type()
                            .is_some_and(|value| value.eq_ignore_ascii_case("I"))
                });
                has_valid_kind.then_some(*end_date)
            })
    }

    pub fn records_for_end_date(
        &self,
        ts_code: &str,
        end_date: i32,
    ) -> Vec<MainBusinessRecordView<'a>> {
        let Some(row_positions) = self
            .index
            .by_ts_end
            .get(ts_code)
            .and_then(|by_end| by_end.get(&end_date))
        else {
            return Vec::new();
        };
        row_positions
            .iter()
            .copied()
            .map(|row_pos| MainBusinessRecordView {
                index: self.index,
                row_pos,
            })
            .collect()
    }

    pub fn industry_update0_records(
        &self,
        ts_code: &str,
        end_date: i32,
    ) -> Vec<MainBusinessRecordView<'a>> {
        self.records_for_end_date(ts_code, end_date)
            .into_iter()
            .filter(|record| {
                record.update_flag() == 0
                    && record
                        .bz_type()
                        .is_some_and(|value| value.eq_ignore_ascii_case("I"))
            })
            .collect()
    }

    pub fn industry_update0_fingerprint(&self, ts_code: &str, end_date: i32) -> Option<i64> {
        self.index
            .industry_fingerprints
            .get(ts_code)?
            .get(&end_date)
            .copied()
    }

    fn compute_industry_update0_fingerprint(&self, ts_code: &str, end_date: i32) -> Option<i64> {
        let mut rows = self
            .industry_update0_records(ts_code, end_date)
            .into_iter()
            .filter_map(|record| {
                let item = record.bz_item()?;
                let sales = record.bz_sales()?;
                Some((item, sales.to_bits(), record.update_flag()))
            })
            .collect::<Vec<_>>();
        if rows.is_empty() {
            return None;
        }
        rows.sort();
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        end_date.hash(&mut hasher);
        rows.hash(&mut hasher);
        Some(i64::from_ne_bytes(hasher.finish().to_ne_bytes()))
    }
}

fn multi_target_context(context: &FactorContext, target_dates: &[i32]) -> FactorContext {
    let start_date = target_dates.first().copied().unwrap_or(context.start_date);
    let end_date = target_dates.last().copied().unwrap_or(context.end_date);
    FactorContext {
        asset_class: context.asset_class,
        frequency: context.frequency,
        start_date,
        end_date,
        load_start_date: context.load_start_date,
        load_dates: context.load_dates.clone(),
        target_dates: target_dates.to_vec(),
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

    pub fn include_record_view(
        &mut self,
        dataset: FinancialStatementDataset,
        record: Option<PitFinancialRecordView<'_>>,
    ) -> &mut Self {
        if let Some(record) = record {
            self.records
                .push(FinancialRecordMarker::from_record_view(dataset, &record));
        }
        self
    }

    pub fn include_reader_record_for_end_date(
        &mut self,
        dataset: FinancialStatementDataset,
        data: &FinancialPitReader<'_>,
        ts_code: &str,
        trade_date: i32,
        end_date: i32,
    ) -> &mut Self {
        self.include_record_view(
            dataset,
            data.record_for_end_date(ts_code, trade_date, end_date),
        )
    }

    pub fn include_reader_ttm_for_end_date(
        &mut self,
        dataset: FinancialStatementDataset,
        data: &FinancialPitReader<'_>,
        ts_code: &str,
        trade_date: i32,
        end_date: i32,
    ) -> &mut Self {
        let mut current = Some(end_date);
        for _ in 0..4 {
            let Some(end_date) = current else {
                break;
            };
            self.include_reader_record_for_end_date(dataset, data, ts_code, trade_date, end_date);
            current = previous_quarter_end_date(end_date);
        }
        self
    }

    pub fn include_reader_latest_ttm(
        &mut self,
        dataset: FinancialStatementDataset,
        data: &FinancialPitReader<'_>,
        ts_code: &str,
        trade_date: i32,
    ) -> &mut Self {
        if let Some(end_date) = data.latest_quarter_end_date(ts_code, trade_date) {
            self.include_reader_ttm_for_end_date(dataset, data, ts_code, trade_date, end_date);
        }
        self
    }

    pub fn include_reader_latest_quarter(
        &mut self,
        dataset: FinancialStatementDataset,
        data: &FinancialPitReader<'_>,
        ts_code: &str,
        trade_date: i32,
    ) -> &mut Self {
        if let Some(end_date) = data.latest_quarter_end_date(ts_code, trade_date) {
            self.include_reader_record_for_end_date(dataset, data, ts_code, trade_date, end_date);
        }
        self
    }

    pub fn include_reader_latest_annual(
        &mut self,
        dataset: FinancialStatementDataset,
        data: &FinancialPitReader<'_>,
        ts_code: &str,
        trade_date: i32,
    ) -> &mut Self {
        if let Some(end_date) = data.latest_annual_end_date(ts_code, trade_date) {
            self.include_reader_record_for_end_date(dataset, data, ts_code, trade_date, end_date);
        }
        self
    }

    pub fn include_reader_annual_chain(
        &mut self,
        dataset: FinancialStatementDataset,
        data: &FinancialPitReader<'_>,
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
            self.include_reader_record_for_end_date(dataset, data, ts_code, trade_date, end_date);
            year -= 1;
        }
        self
    }

    pub fn include_dividend_ltm(
        &mut self,
        reader: &DividendReader<'_>,
        ts_code: &str,
        start_date: i32,
        trade_date: i32,
    ) -> &mut Self {
        let cash = reader.implemented_ltm_sum(ts_code, start_date, trade_date);
        self.include_synthetic("dividend_ltm", f64_marker_value(cash));
        self
    }

    pub fn include_main_business_end_date(
        &mut self,
        reader: &MainBusinessReader<'_>,
        ts_code: &str,
        end_date: i32,
    ) -> &mut Self {
        if let Some(marker) = reader.industry_update0_fingerprint(ts_code, end_date) {
            self.include_synthetic("main_business", marker);
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

#[derive(Clone, Debug)]
pub struct InstrumentAlignedSnapshotCache<T> {
    instruments: Vec<String>,
    cache: FinancialStockSnapshotCache<T>,
    route_signature: Vec<(u64, Vec<i64>)>,
    checked_dates: Vec<Option<i32>>,
}

impl<T: Clone> Default for InstrumentAlignedSnapshotCache<T> {
    fn default() -> Self {
        Self {
            instruments: Vec::new(),
            cache: FinancialStockSnapshotCache::new(0),
            route_signature: Vec::new(),
            checked_dates: Vec::new(),
        }
    }
}

impl<T: Clone> InstrumentAlignedSnapshotCache<T> {
    pub fn align_to(&mut self, instruments: &[String]) {
        if self.instruments == instruments {
            return;
        }
        let old_lookup = self
            .instruments
            .iter()
            .enumerate()
            .map(|(idx, ts_code)| (ts_code.as_str(), idx))
            .collect::<BTreeMap<_, _>>();
        let mut next = FinancialStockSnapshotCache::new(instruments.len());
        for (new_idx, ts_code) in instruments.iter().enumerate() {
            if let Some(old_idx) = old_lookup.get(ts_code.as_str()).copied() {
                next.last_marker_by_stock[new_idx] = self
                    .cache
                    .last_marker_by_stock
                    .get_mut(old_idx)
                    .and_then(Option::take);
                next.last_snapshot_by_stock[new_idx] = self
                    .cache
                    .last_snapshot_by_stock
                    .get_mut(old_idx)
                    .and_then(Option::take);
            }
        }
        self.instruments = instruments.to_vec();
        self.cache = next;
        self.checked_dates = vec![None; instruments.len()];
    }

    pub fn clear(&mut self, instrument_idx: usize) {
        self.cache.clear(instrument_idx);
        if let Some(date) = self.checked_dates.get_mut(instrument_idx) {
            *date = None;
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
        if let Some(date) = self.checked_dates.get_mut(instrument_idx) {
            *date = None;
        }
        self.cache.get_or_update(instrument_idx, marker, compute)
    }
}

/// Explicit dependencies for skipping unchanged stock markers. Formula keys and
/// non-statement sources must be declared; arbitrary calendar formulas stay unrouted.
pub struct FinancialStockEventRoute<'a> {
    readers: Vec<FinancialPitReader<'a>>,
    dividend: Option<DividendReader<'a>>,
    extra_keys: Vec<(u64, Vec<i64>)>,
    query_date: Option<i32>,
}

impl<'a> FinancialStockEventRoute<'a> {
    pub fn new(readers: &[&FinancialPitReader<'a>]) -> Self {
        Self {
            readers: readers.iter().map(|reader| (*reader).clone()).collect(),
            dividend: None,
            extra_keys: Vec::new(),
            query_date: None,
        }
    }

    pub fn with_dividend(mut self, reader: &DividendReader<'a>) -> Self {
        self.dividend = Some(*reader);
        self
    }

    pub fn with_optional_dividend(mut self, reader: Option<&DividendReader<'a>>) -> Self {
        self.dividend = reader.copied();
        self
    }

    pub fn with_main_business(mut self, reader: &MainBusinessReader<'_>) -> Self {
        self.extra_keys.push((reader.index.generation, Vec::new()));
        self
    }

    pub fn with_key(mut self, key: i64) -> Self {
        self.extra_keys.push((0, vec![key]));
        self
    }

    /// Use when the marker queries a historical PIT date rather than the output date.
    pub fn at_date(mut self, date: i32) -> Self {
        self.query_date = Some(date);
        self
    }

    fn signature(&self) -> Vec<(u64, Vec<i64>)> {
        let mut keys: Vec<_> = self
            .readers
            .iter()
            .map(|r| (r.index.generation, r.preference.order.clone()))
            .collect();
        if let Some(reader) = self.dividend {
            keys.push((reader.index.generation, Vec::new()));
        }
        keys.extend(self.extra_keys.iter().cloned());
        keys
    }

    fn changed(&self, code: &str, after: i32, until: i32) -> bool {
        self.readers.iter().any(|reader| {
            let until = until.min(reader.index.max_disclosure_date.unwrap_or(i32::MAX));
            reader.index.leaves().any(|part| {
                part.events_by_stock_type.get(code).is_some_and(|types| {
                    reader.preference.order.iter().any(|kind| {
                        types
                            .get(kind)
                            .is_some_and(|dates| event_between(dates, after, until))
                    })
                })
            })
        }) || self.dividend.is_some_and(|reader| {
            reader
                .index
                .stock_event_dates
                .get(code)
                .is_some_and(|dates| event_between(dates, after, until))
        })
    }
}

fn event_between(dates: &[i32], after: i32, until: i32) -> bool {
    after < until
        && dates
            .get(dates.partition_point(|date| *date <= after))
            .is_some_and(|date| *date <= until)
}

pub fn routed_financial_stock_snapshots_for_date<T, SkipFn, MarkerFn, ComputeFn>(
    panel: &DailyPanel,
    trade_date: i32,
    cache: &mut InstrumentAlignedSnapshotCache<T>,
    route: &FinancialStockEventRoute<'_>,
    mut skip_fn: SkipFn,
    mut marker_fn: MarkerFn,
    mut compute_fn: ComputeFn,
) -> Vec<Option<T>>
where
    T: Clone,
    SkipFn: FnMut(i32, &str, usize) -> bool,
    MarkerFn: FnMut(i32, &str, usize) -> Option<FinancialEventMarker>,
    ComputeFn: FnMut(i32, &str, usize) -> Option<T>,
{
    cache.align_to(panel.instruments());
    let signature = route.signature();
    if cache.route_signature != signature {
        cache.checked_dates.fill(None);
        cache.route_signature = signature;
    }
    let mut output = vec![None; panel.instruments().len()];
    let Some(date_idx) = panel.dates().iter().position(|date| *date == trade_date) else {
        return output;
    };
    let query_date = route.query_date.unwrap_or(trade_date);
    for (idx, code) in panel.instruments().iter().enumerate() {
        let offset = date_idx * output.len() + idx;
        if skip_fn(trade_date, code, offset) {
            cache.clear(idx);
            continue;
        }
        let unchanged = cache.checked_dates[idx]
            .is_some_and(|after| after <= query_date && !route.changed(code, after, query_date));
        output[idx] = if unchanged {
            cache.cache.last_snapshot_by_stock[idx].clone()
        } else {
            cache.get_or_update(idx, marker_fn(trade_date, code, offset), || {
                compute_fn(trade_date, code, offset)
            })
        };
        cache.checked_dates[idx] = Some(query_date);
    }
    output
}

pub fn cached_financial_stock_snapshots_for_date<T, SkipFn, MarkerFn, ComputeFn>(
    panel: &DailyPanel,
    trade_date: i32,
    cache: &mut InstrumentAlignedSnapshotCache<T>,
    mut skip_fn: SkipFn,
    mut marker_fn: MarkerFn,
    mut compute_fn: ComputeFn,
) -> Vec<Option<T>>
where
    T: Clone,
    SkipFn: FnMut(i32, &str, usize) -> bool,
    MarkerFn: FnMut(i32, &str, usize) -> Option<FinancialEventMarker>,
    ComputeFn: FnMut(i32, &str, usize) -> Option<T>,
{
    cache.align_to(panel.instruments());
    let instrument_count = panel.instruments().len();
    let mut output = vec![None; instrument_count];
    let Some(date_idx) = panel.dates().iter().position(|date| *date == trade_date) else {
        return output;
    };
    let date_offset = date_idx * instrument_count;
    for (instrument_idx, ts_code) in panel.instruments().iter().enumerate() {
        let offset = date_offset + instrument_idx;
        if skip_fn(trade_date, ts_code, offset) {
            cache.clear(instrument_idx);
            continue;
        }
        output[instrument_idx] = cache.get_or_update(
            instrument_idx,
            marker_fn(trade_date, ts_code, offset),
            || compute_fn(trade_date, ts_code, offset),
        );
    }
    output
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
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

    pub fn contains(&self, report_type: i64) -> bool {
        self.order.contains(&report_type)
    }
}

#[derive(Clone, Debug)]
pub struct FinancialPitIndex {
    generation: u64,
    periods_by_stock: BTreeMap<String, Vec<i32>>,
    events_by_stock_type: BTreeMap<String, BTreeMap<i64, Vec<i32>>>,
    event_dates_by_type: BTreeMap<i64, Vec<i32>>,
    record_count: usize,
    sources: Vec<Arc<Table>>,
    records: Vec<FinancialIndexedRecord>,
    by_ts_code: BTreeMap<String, Vec<usize>>,
    partitions: Vec<Arc<FinancialPitIndex>>,
    max_disclosure_date: Option<i32>,
}

#[derive(Clone, Copy, Debug)]
struct FinancialIndexedRecord {
    source_idx: usize,
    row_idx: usize,
    end_date: i32,
    disclosure_date: i32,
    report_type: i64,
    update_flag: i64,
}

impl FinancialPitIndex {
    pub fn from_table(table: Arc<Table>) -> Result<Self> {
        Self::from_source_tables(vec![table], None)
    }

    pub fn from_source_tables(
        sources: Vec<Arc<Table>>,
        max_disclosure_date: Option<i32>,
    ) -> Result<Self> {
        let mut records = Vec::new();
        let mut by_ts_code = BTreeMap::<String, Vec<usize>>::new();
        for (source_idx, table) in sources.iter().enumerate() {
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
            for idx in 0..table.len {
                let (Some(ts_code), Some(end_date), Some(disclosure_date), Some(report_type)) = (
                    ts_codes[idx].clone(),
                    end_dates[idx],
                    f_ann_dates[idx].or(ann_dates[idx]),
                    report_types[idx],
                ) else {
                    continue;
                };
                if max_disclosure_date.is_some_and(|max_date| disclosure_date > max_date) {
                    continue;
                }
                let record_idx = records.len();
                records.push(FinancialIndexedRecord {
                    source_idx,
                    row_idx: idx,
                    end_date,
                    disclosure_date,
                    report_type,
                    update_flag: update_flags[idx].unwrap_or(0),
                });
                by_ts_code.entry(ts_code).or_default().push(record_idx);
            }
        }

        for versions in by_ts_code.values_mut() {
            // Stable ties preserve original source/row precedence without per-period trees.
            versions.sort_by(|left, right| {
                let left = records[*left];
                let right = records[*right];
                (left.end_date, left.report_type)
                    .cmp(&(right.end_date, right.report_type))
                    .then_with(|| right.disclosure_date.cmp(&left.disclosure_date))
                    .then_with(|| right.update_flag.cmp(&left.update_flag))
            });
            versions.shrink_to_fit();
        }
        records.shrink_to_fit();

        let periods_by_stock = by_ts_code
            .iter()
            .map(|(code, rows)| {
                let mut ends = rows
                    .iter()
                    .map(|&i| records[i].end_date)
                    .collect::<Vec<_>>();
                ends.dedup();
                ends.shrink_to_fit();
                (code.clone(), ends)
            })
            .collect();
        let mut events_by_stock_type = BTreeMap::<String, BTreeMap<i64, Vec<i32>>>::new();
        for (code, rows) in &by_ts_code {
            let types = events_by_stock_type.entry(code.clone()).or_default();
            for &i in rows {
                let record = &records[i];
                types
                    .entry(record.report_type)
                    .or_default()
                    .push(record.disclosure_date);
            }
            for dates in types.values_mut() {
                dates.sort_unstable();
                dates.dedup();
                dates.shrink_to_fit();
            }
        }
        let mut event_dates_by_type = BTreeMap::<i64, Vec<i32>>::new();
        for types in events_by_stock_type.values() {
            for (&kind, dates) in types {
                event_dates_by_type.entry(kind).or_default().extend(dates);
            }
        }
        for dates in event_dates_by_type.values_mut() {
            dates.sort_unstable();
            dates.dedup();
            dates.shrink_to_fit();
        }
        Ok(Self {
            generation: next_index_generation(),
            periods_by_stock,
            events_by_stock_type,
            event_dates_by_type,
            record_count: records.len(),
            sources,
            records,
            by_ts_code,
            partitions: Vec::new(),
            max_disclosure_date,
        })
    }

    pub fn from_partitions(partitions: Vec<Arc<Self>>, max_date: i32) -> Self {
        assert!(
            partitions.iter().all(|part| part.partitions.is_empty()),
            "annual partitions must be leaf indexes"
        );
        let record_count = partitions
            .iter()
            .flat_map(|part| part.records.iter())
            .filter(|record| record.disclosure_date <= max_date)
            .count();
        let mut periods_by_stock = BTreeMap::<String, Vec<i32>>::new();
        for part in &partitions {
            for (code, rows) in &part.by_ts_code {
                let mut last_end = None;
                for &idx in rows {
                    let record = &part.records[idx];
                    if record.disclosure_date <= max_date && last_end != Some(record.end_date) {
                        periods_by_stock
                            .entry(code.clone())
                            .or_default()
                            .push(record.end_date);
                        last_end = Some(record.end_date);
                    }
                }
            }
        }
        for ends in periods_by_stock.values_mut() {
            ends.sort_unstable();
            ends.dedup();
            ends.shrink_to_fit();
        }
        Self {
            generation: next_index_generation(),
            periods_by_stock,
            events_by_stock_type: BTreeMap::new(),
            event_dates_by_type: BTreeMap::new(),
            record_count,
            sources: Vec::new(),
            records: Vec::new(),
            by_ts_code: BTreeMap::new(),
            partitions,
            max_disclosure_date: Some(max_date),
        }
    }

    fn leaves(&self) -> impl Iterator<Item = &Self> {
        self.partitions
            .iter()
            .map(Arc::as_ref)
            .chain(self.partitions.is_empty().then_some(self))
    }

    fn versions(&self, code: &str, end_date: i32, report_type: i64) -> &[usize] {
        let Some(rows) = self.by_ts_code.get(code) else {
            return &[];
        };
        let key = (end_date, report_type);
        let start = rows
            .partition_point(|&i| (self.records[i].end_date, self.records[i].report_type) < key);
        let end = rows
            .partition_point(|&i| (self.records[i].end_date, self.records[i].report_type) <= key);
        &rows[start..end]
    }

    pub fn reader(&self, preference: ReportTypePreference) -> FinancialPitReader<'_> {
        FinancialPitReader {
            index: self,
            preference,
        }
    }

    pub fn len(&self) -> usize {
        self.record_count
    }
}

#[derive(Clone, Debug)]
pub struct FinancialPitReader<'a> {
    index: &'a FinancialPitIndex,
    preference: ReportTypePreference,
}

impl<'a> FinancialPitReader<'a> {
    /// Distinct disclosure events, including visible revisions, not just the latest version.
    pub fn disclosure_dates_between(
        &self,
        ts_code: &str,
        after_exclusive: i32,
        until_inclusive: i32,
    ) -> Vec<i32> {
        let mut dates = Vec::new();
        let until_inclusive =
            until_inclusive.min(self.index.max_disclosure_date.unwrap_or(i32::MAX));
        if after_exclusive >= until_inclusive {
            return dates;
        }
        for part in self.index.leaves() {
            let Some(types) = part.events_by_stock_type.get(ts_code) else {
                continue;
            };
            for report_type in &self.preference.order {
                let Some(events) = types.get(report_type) else {
                    continue;
                };
                let start = events.partition_point(|date| *date <= after_exclusive);
                let end = events.partition_point(|date| *date <= until_inclusive);
                dates.extend_from_slice(&events[start..end]);
            }
        }
        dates.sort_unstable();
        dates.dedup();
        dates
    }

    pub fn record_for_end_date(
        &self,
        ts_code: &str,
        trade_date: i32,
        end_date: i32,
    ) -> Option<PitFinancialRecordView<'a>> {
        let date = trade_date.min(self.index.max_disclosure_date.unwrap_or(i32::MAX));
        for report_type in &self.preference.order {
            let mut best: Option<PitFinancialRecordView<'a>> = None;
            for part in self.index.leaves() {
                let versions = part.versions(ts_code, end_date, *report_type);
                let visible =
                    versions.partition_point(|&idx| part.records[idx].disclosure_date > date);
                if let Some(&idx) = versions.get(visible) {
                    let record = &part.records[idx];
                    // Keep the earlier source on exact ties, matching the stable merged sort.
                    if best.as_ref().is_none_or(|old| {
                        (record.disclosure_date, record.update_flag)
                            > (old.record.disclosure_date, old.record.update_flag)
                    }) {
                        best = Some(PitFinancialRecordView {
                            table: part.sources[record.source_idx].as_ref(),
                            record,
                        });
                    }
                }
            }
            if best.is_some() {
                return best;
            }
        }
        None
    }

    fn end_dates(&self, ts_code: &str) -> impl DoubleEndedIterator<Item = i32> + '_ {
        self.index
            .periods_by_stock
            .get(ts_code)
            .map(Vec::as_slice)
            .unwrap_or(&[])
            .iter()
            .copied()
    }

    pub fn latest_quarter_end_date(&self, ts_code: &str, trade_date: i32) -> Option<i32> {
        for end_date in self.end_dates(ts_code).into_iter().rev() {
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
        for anchor in self.end_dates(ts_code).into_iter().rev() {
            if let Some(value) = self.ttm_sum_for_end_date(ts_code, trade_date, anchor, column) {
                return Some(value);
            }
        }
        None
    }

    pub fn latest_annual_end_date(&self, ts_code: &str, trade_date: i32) -> Option<i32> {
        for end_date in self.end_dates(ts_code).into_iter().rev() {
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
        for anchor in self.end_dates(ts_code).into_iter().rev() {
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
}

#[derive(Clone, Copy, Debug)]
pub struct PitFinancialRecordView<'a> {
    table: &'a Table,
    record: &'a FinancialIndexedRecord,
}

impl PitFinancialRecordView<'_> {
    pub fn end_date(&self) -> i32 {
        self.record.end_date
    }

    pub fn disclosure_date(&self) -> i32 {
        self.record.disclosure_date
    }

    pub fn report_type(&self) -> i64 {
        self.record.report_type
    }

    pub fn update_flag(&self) -> i64 {
        self.record.update_flag
    }

    pub fn column(&self, name: &str) -> Option<f64> {
        let value = match self.table.columns.get(name)? {
            ColumnData::F64(values) => values.get(self.record.row_idx).copied().flatten(),
            ColumnData::F32(values) => values
                .get(self.record.row_idx)
                .copied()
                .flatten()
                .map(f64::from),
            ColumnData::I64(values) => values
                .get(self.record.row_idx)
                .copied()
                .flatten()
                .map(|value| value as f64),
            ColumnData::I32(values) => values
                .get(self.record.row_idx)
                .copied()
                .flatten()
                .map(f64::from),
            _ => None,
        }?;
        (!value.is_nan()).then_some(value)
    }
}

pub fn cached_financial_stock_snapshots<T, SkipFn, MarkerFn, ComputeFn>(
    panel: &DailyPanel,
    mut skip_fn: SkipFn,
    mut marker_fn: MarkerFn,
    mut compute_fn: ComputeFn,
) -> Vec<Option<T>>
where
    T: Clone,
    SkipFn: FnMut(i32, &str, usize) -> bool,
    MarkerFn: FnMut(i32, &str, usize) -> Option<FinancialEventMarker>,
    ComputeFn: FnMut(i32, &str, usize) -> Option<T>,
{
    let instrument_count = panel.instruments().len();
    let mut snapshots = vec![None; panel.shape_len()];
    let mut cache = FinancialStockSnapshotCache::<T>::new(instrument_count);
    for (date_idx, trade_date) in panel.dates().iter().copied().enumerate() {
        if !panel.is_target_date(trade_date) {
            continue;
        }
        let date_offset = date_idx * instrument_count;
        for (instrument_idx, ts_code) in panel.instruments().iter().enumerate() {
            let offset = date_offset + instrument_idx;
            if skip_fn(trade_date, ts_code, offset) {
                cache.clear(instrument_idx);
                continue;
            }
            snapshots[offset] = cache.get_or_update(
                instrument_idx,
                marker_fn(trade_date, ts_code, offset),
                || compute_fn(trade_date, ts_code, offset),
            );
        }
    }
    snapshots
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

/// Analyst FY anchor shared with the consensus generator; report presence is sufficient.
pub(crate) fn analyst_fiscal_years(
    date: i32,
    code: &str,
    income: &FinancialPitReader<'_>,
) -> [i32; 4] {
    let year = date / 10_000;
    let switched = date % 10_000 >= 501
        || income
            .record_for_end_date(code, date, (year - 1) * 10_000 + 1231)
            .is_some();
    let base = if switched { year - 1 } else { year - 2 };
    [base, base + 1, base + 2, base + 3]
}

pub(crate) fn add_months(date: i32, months_delta: i32) -> i32 {
    let (year, month, day) = ymd(date);
    let month_index = year * 12 + month as i32 - 1 + months_delta;
    let new_year = month_index.div_euclid(12);
    let new_month = month_index.rem_euclid(12) + 1;
    let new_day = day.min(days_in_month(new_year, new_month as u32));
    new_year * 10_000 + new_month * 100 + new_day as i32
}

pub(crate) fn add_days(date: i32, days_delta: i32) -> i32 {
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

fn clean_f64(value: Option<f64>) -> Option<f64> {
    value.filter(|value| value.is_finite())
}

fn f64_marker_value(value: f64) -> i64 {
    i64::from_ne_bytes(value.to_bits().to_ne_bytes())
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

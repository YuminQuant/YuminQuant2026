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

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::Arc;

    use crate::core::{
        AssetClass, DataRequest, FactorContext, FactorRowKey, FactorSeries, FactorSpec,
        FactorValue, Frequency, Lookback,
    };
    use crate::data::{ColumnData, DataPool, Table};

    use super::{
        cached_financial_stock_snapshots, cached_financial_stock_snapshots_for_date,
        compute_financial_event_snapshot_streaming_on_panel, DailyPanel, DividendIndex,
        EventDrivenCrossSectionCache, FinancialEventMarkerBuilder, FinancialEventSchedule,
        FinancialPitIndex, FinancialStatementDataset, FinancialStockSnapshotCache,
        InstrumentAlignedSnapshotCache, ReportTypePreference,
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

    fn panel_for_codes(date: i32, codes: &[&str]) -> crate::factor::common::DailyPanel {
        let table = Table::new(BTreeMap::from([
            (
                "trade_date".to_string(),
                ColumnData::I32(codes.iter().map(|_| Some(date)).collect()),
            ),
            (
                "ts_code".to_string(),
                ColumnData::Utf8(codes.iter().map(|code| Some((*code).to_string())).collect()),
            ),
            (
                "dummy".to_string(),
                ColumnData::F64(codes.iter().map(|_| Some(1.0)).collect()),
            ),
        ]))
        .expect("valid table");
        let context = FactorContext {
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            start_date: date,
            end_date: date,
            load_start_date: date,
            load_dates: vec![date],
            target_dates: vec![date],
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

    fn dividend_table(rows: &[(&str, i32, &str, f64, i32, f64)]) -> Table {
        Table::new(BTreeMap::from([
            (
                "ts_code".to_string(),
                ColumnData::Utf8(
                    rows.iter()
                        .map(|row| Some(row.0.to_string()))
                        .collect::<Vec<_>>(),
                ),
            ),
            (
                "ann_date".to_string(),
                ColumnData::I32(rows.iter().map(|row| Some(row.1)).collect()),
            ),
            (
                "div_proc".to_string(),
                ColumnData::Utf8(
                    rows.iter()
                        .map(|row| Some(row.2.to_string()))
                        .collect::<Vec<_>>(),
                ),
            ),
            (
                "cash_div_tax".to_string(),
                ColumnData::F64(rows.iter().map(|row| Some(row.3)).collect()),
            ),
            (
                "ex_date".to_string(),
                ColumnData::I32(rows.iter().map(|row| Some(row.4)).collect()),
            ),
            (
                "base_share".to_string(),
                ColumnData::F64(rows.iter().map(|row| Some(row.5)).collect()),
            ),
        ]))
        .expect("valid dividend table")
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
    fn dividend_annual_sum_respects_fiscal_year_visibility_and_units() {
        let mut table = dividend_table(&[
            (
                "000001.SZ",
                20240401,
                "\u{5b9e}\u{65bd}",
                0.2,
                20240603,
                100.0,
            ),
            (
                "000001.SZ",
                20250401,
                "\u{5b9e}\u{65bd}",
                0.3,
                20250603,
                100.0,
            ),
        ]);
        table
            .insert(
                "end_date",
                ColumnData::I32(vec![Some(20231231), Some(20241231)]),
            )
            .unwrap();
        let index = DividendIndex::from_table(Arc::new(table)).unwrap();
        let reader = index.reader();
        assert!(reader
            .implemented_annual_sum_by_stock(2023, 20240602)
            .is_empty());
        assert_eq!(
            reader.implemented_annual_sum_by_stock(2023, 20250603)["000001.SZ"],
            20.0
        );
        assert_eq!(
            reader.implemented_annual_sum_by_stock(2024, 20250603)["000001.SZ"],
            30.0
        );
        assert_eq!(
            reader.implemented_ltm_sum("000001.SZ", 20240101, 20250603),
            50.0
        );
    }

    fn factor_context(dates: &[i32]) -> FactorContext {
        FactorContext {
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            start_date: *dates.first().unwrap(),
            end_date: *dates.last().unwrap(),
            load_start_date: *dates.first().unwrap(),
            load_dates: dates.to_vec(),
            target_dates: dates.to_vec(),
        }
    }

    #[test]
    fn financial_event_schedule_maps_statement_and_dividend_dates() {
        let statement = financial_table(&[(20251231, 20260103, 1, 0, 1.0)]);
        let statement_index = FinancialPitIndex::from_table(Arc::new(statement)).expect("index");
        let dividend_index = DividendIndex::from_table(Arc::new(dividend_table(&[(
            "000001.SZ",
            20250115,
            "\u{5b9e}\u{65bd}",
            0.1,
            20250131,
            100.0,
        )])))
        .expect("dividend index");
        let mut schedule = FinancialEventSchedule::from_pit_readers(&[
            statement_index.reader(ReportTypePreference::consolidated())
        ]);
        schedule.merge(FinancialEventSchedule::from_dividend_reader(
            &dividend_index.reader(),
        ));

        assert!(schedule.has_event_after_until(Some(20260102), 20260105));
        assert!(!schedule.has_event_after_until(Some(20260105), 20260106));
        assert!(schedule.has_event_after_until(Some(20260131), 20260201));
    }

    #[test]
    fn financial_event_schedule_empty_or_replayed_range_does_not_panic() {
        let statement = financial_table(&[(20251231, 20260103, 1, 0, 1.0)]);
        let statement_index = FinancialPitIndex::from_table(Arc::new(statement)).expect("index");
        let schedule = FinancialEventSchedule::from_pit_readers(&[
            statement_index.reader(ReportTypePreference::consolidated())
        ]);

        assert!(!schedule.has_event_after_until(Some(20260103), 20260103));
        assert!(!schedule.has_event_after_until(Some(20260104), 20260103));
    }

    #[test]
    fn event_driven_cross_section_cache_replays_latest_values_daily() {
        let panel = panel(&[20260105, 20260106]);
        let spec = event_spec("slow_factor");
        let mut cache = EventDrivenCrossSectionCache::default();
        cache.update_series(
            &FactorSeries {
                spec: spec.clone(),
                values: vec![FactorValue {
                    key: FactorRowKey::Daily {
                        trade_date: 20260105,
                        ts_code: "000001.SZ".to_string(),
                    },
                    value: Some(1.23),
                }],
            },
            &panel,
        );

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
    fn event_driven_cross_section_cache_replays_by_code_when_panel_changes() {
        let old_panel = panel_for_codes(20260105, &["000002.SZ", "000004.SZ"]);
        let new_panel = panel_for_codes(20260106, &["000001.SZ", "000002.SZ", "000004.SZ"]);
        let spec = event_spec("slow_factor");
        let mut cache = EventDrivenCrossSectionCache::default();
        cache.update_series(
            &FactorSeries {
                spec: spec.clone(),
                values: vec![
                    FactorValue {
                        key: FactorRowKey::Daily {
                            trade_date: 20260105,
                            ts_code: "000002.SZ".to_string(),
                        },
                        value: Some(2.0),
                    },
                    FactorValue {
                        key: FactorRowKey::Daily {
                            trade_date: 20260105,
                            ts_code: "000004.SZ".to_string(),
                        },
                        value: Some(4.0),
                    },
                ],
            },
            &old_panel,
        );

        let replay = cache.replay_series(spec, &new_panel, 20260106);
        let values = replay
            .values
            .iter()
            .map(|item| match &item.key {
                FactorRowKey::Daily { ts_code, .. } => (ts_code.as_str(), item.value),
                _ => unreachable!(),
            })
            .collect::<BTreeMap<_, _>>();

        assert_eq!(values.get("000001.SZ"), Some(&None));
        assert_eq!(values.get("000002.SZ"), Some(&Some(2.0)));
        assert_eq!(values.get("000004.SZ"), Some(&Some(4.0)));
    }

    #[test]
    fn financial_event_snapshot_streams_event_dates_and_replays_cache() {
        let dates = vec![20260102, 20260105, 20260106, 20260107, 20260108];
        let context = factor_context(&dates);
        let data = DataPool::default();
        let panel = panel(&dates);
        let statement = financial_table(&[
            (20251231, 20260104, 1, 0, 1.0),
            (20260331, 20260107, 1, 0, 2.0),
        ]);
        let statement_index = FinancialPitIndex::from_table(Arc::new(statement)).expect("index");
        let schedule = FinancialEventSchedule::from_pit_readers(&[
            statement_index.reader(ReportTypePreference::consolidated())
        ]);
        let spec = event_spec("slow_factor");
        let mut state = EventDrivenCrossSectionCache::default();
        let mut call_count = 0usize;
        let mut seen_event_dates = Vec::new();
        let output = compute_financial_event_snapshot_streaming_on_panel(
            &[spec.id.clone()],
            &context,
            &data,
            &panel,
            &mut state,
            &schedule,
            &[spec.clone()],
            |_, event_context, _| {
                call_count += 1;
                seen_event_dates.extend(event_context.target_dates.iter().copied());
                Ok(vec![FactorSeries {
                    spec: spec.clone(),
                    values: event_context
                        .target_dates
                        .iter()
                        .map(|trade_date| FactorValue {
                            key: FactorRowKey::Daily {
                                trade_date: *trade_date,
                                ts_code: "000001.SZ".to_string(),
                            },
                            value: Some(*trade_date as f64),
                        })
                        .collect(),
                }])
            },
        )
        .expect("event snapshot");

        assert_eq!(call_count, 3);
        assert_eq!(seen_event_dates, vec![20260102, 20260105, 20260107]);
        assert_eq!(output.len(), 1);
        assert_eq!(output[0].values.len(), dates.len());

        let by_date = output[0]
            .values
            .iter()
            .map(|value| {
                let FactorRowKey::Daily { trade_date, .. } = &value.key else {
                    unreachable!("daily key")
                };
                (*trade_date, value.value)
            })
            .collect::<BTreeMap<_, _>>();
        assert_eq!(by_date.get(&20260102), Some(&Some(20260102.0)));
        assert_eq!(by_date.get(&20260105), Some(&Some(20260105.0)));
        assert_eq!(by_date.get(&20260106), Some(&Some(20260105.0)));
        assert_eq!(by_date.get(&20260107), Some(&Some(20260107.0)));
        assert_eq!(by_date.get(&20260108), Some(&Some(20260107.0)));
    }

    #[test]
    fn financial_event_snapshot_on_panel_does_not_require_pv_panel() {
        let dates = vec![20260102, 20260105];
        let context = factor_context(&dates);
        let panel = panel(&dates);
        let data = DataPool::default();
        let statement = financial_table(&[(20251231, 20260104, 1, 0, 1.0)]);
        let statement_index = FinancialPitIndex::from_table(Arc::new(statement)).expect("index");
        let schedule = FinancialEventSchedule::from_pit_readers(&[
            statement_index.reader(ReportTypePreference::consolidated())
        ]);
        let spec = event_spec("slow_factor");
        let mut state = EventDrivenCrossSectionCache::default();

        let output = compute_financial_event_snapshot_streaming_on_panel(
            &[spec.id.clone()],
            &context,
            &data,
            &panel,
            &mut state,
            &schedule,
            &[spec.clone()],
            |_, event_context, _| {
                Ok(vec![FactorSeries {
                    spec: spec.clone(),
                    values: event_context
                        .target_dates
                        .iter()
                        .map(|trade_date| FactorValue {
                            key: FactorRowKey::Daily {
                                trade_date: *trade_date,
                                ts_code: "000001.SZ".to_string(),
                            },
                            value: Some(*trade_date as f64),
                        })
                        .collect(),
                }])
            },
        )
        .expect("event snapshot without pv panel");

        assert_eq!(output[0].values.len(), dates.len());
        assert!(output[0].values.iter().all(|value| matches!(
            &value.key,
            FactorRowKey::Daily { ts_code, .. } if ts_code == "000001.SZ"
        )));
    }

    #[test]
    fn report_type_preference_uses_order_before_fallback() {
        let table = financial_table(&[
            (20250331, 20250420, 1, 0, 1.0),
            (20250331, 20250420, 2, 0, 2.0),
            (20250331, 20250420, 3, 0, 3.0),
            (20250331, 20250420, 4, 0, 4.0),
        ]);
        let index = FinancialPitIndex::from_table(Arc::new(table)).expect("index");
        let income = index.reader(ReportTypePreference::income_single_quarter());
        let balance = index.reader(ReportTypePreference::balance_sheet_consolidated());

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
    fn financial_compact_stock_index_separates_periods_types_and_missing_keys() {
        let index = FinancialPitIndex::from_table(Arc::new(financial_table(&[
            (20240331, 20240430, 2, 0, 1.0),
            (20231231, 20240430, 3, 0, 2.0),
            (20240331, 20240501, 3, 0, 3.0),
            (20240331, 20240501, 3, 1, 4.0),
            (20240331, 20240501, 3, 1, 5.0),
            (20240331, 20240429, 1, 0, 6.0),
        ])))
        .unwrap();
        assert_eq!(index.by_ts_code.len(), 1);
        assert_eq!(index.by_ts_code["000001.SZ"].len(), index.records.len());
        assert!(index.versions("absent", 20240331, 3).is_empty());
        assert!(index.versions("000001.SZ", 20240630, 3).is_empty());
        assert!(index.versions("000001.SZ", 20240331, 4).is_empty());
        let reader = index.reader(ReportTypePreference::income_single_quarter());
        assert_eq!(
            reader
                .record_for_end_date("000001.SZ", 20240430, 20240331)
                .unwrap()
                .column("value"),
            Some(1.0)
        );
        assert_eq!(
            reader
                .record_for_end_date("000001.SZ", 20240501, 20240331)
                .unwrap()
                .column("value"),
            Some(4.0)
        );
        assert_eq!(
            reader
                .record_for_end_date("000001.SZ", 20240501, 20231231)
                .unwrap()
                .column("value"),
            Some(2.0)
        );
    }

    #[test]
    fn financial_binary_version_lookup_matches_linear_for_all_boundaries() {
        let index = FinancialPitIndex::from_table(Arc::new(financial_table(&[
            (20241231, 20250401, 2, 0, 1.0),
            (20241231, 20250410, 3, 0, 2.0),
            (20241231, 20250410, 3, 1, 3.0),
            (20241231, 20250410, 3, 1, 4.0),
            (20241231, 20250420, 2, 1, 5.0),
            (20241231, 20250430, 3, 1, 6.0),
        ])))
        .unwrap();
        let reader = index.reader(ReportTypePreference::income_single_quarter());
        for date in [
            20250331, 20250401, 20250409, 20250410, 20250419, 20250420, 20250430, 20250501,
        ] {
            let expected = [3, 2].iter().find_map(|kind| {
                index.by_ts_code["000001.SZ"].iter().copied().find(|&i| {
                    index.records[i].end_date == 20241231
                        && index.records[i].report_type == *kind
                        && index.records[i].disclosure_date <= date
                })
            });
            let actual = reader.record_for_end_date("000001.SZ", date, 20241231);
            assert_eq!(
                actual.map(|v| v.record.row_idx),
                expected.map(|i| index.records[i].row_idx)
            );
        }
    }

    #[test]
    fn financial_dividend_stock_index_matches_full_scan_in_original_order() {
        let index = DividendIndex::from_table(Arc::new(dividend_table(&[
            ("A", 20240101, "\u{5b9e}\u{65bd}", 1e16, 20240201, 1.0),
            ("B", 20240101, "\u{5b9e}\u{65bd}", 7.0, 20240201, 1.0),
            ("A", 20240101, "\u{5b9e}\u{65bd}", -1e16, 20240201, 1.0),
            ("A", 20240101, "\u{5b9e}\u{65bd}", 1.0, 20240201, 1.0),
            ("A", 20250101, "\u{5b9e}\u{65bd}", 2.0, 20250201, 1.0),
            ("A", 20240101, "proposal", 9.0, 20240201, 1.0),
        ])))
        .unwrap();
        assert_eq!(index.by_ts_code["A"].len(), 4);
        for code in ["A", "B", "missing"] {
            for (start, end) in [
                (20240101, 20240131),
                (20240101, 20240201),
                (20240202, 20250201),
                (20240101, 20250201),
            ] {
                let expected: f64 = index
                    .records
                    .iter()
                    .filter(|r| {
                        r.ts_code == code
                            && r.ann_date <= end
                            && r.ex_date <= end
                            && r.ex_date >= start
                    })
                    .map(|r| r.cash_amount)
                    .sum();
                assert_eq!(
                    index
                        .reader()
                        .implemented_ltm_sum(code, start, end)
                        .to_bits(),
                    expected.to_bits()
                );
            }
        }
    }

    #[test]
    fn financial_snapshot_remap_moves_payload_without_clone() {
        #[derive(Debug)]
        struct Payload;
        impl Clone for Payload {
            fn clone(&self) -> Self {
                panic!("remap must move snapshots");
            }
        }
        let mut cache = InstrumentAlignedSnapshotCache::<Payload>::default();
        cache.align_to(&["A".into(), "B".into()]);
        cache.cache.last_snapshot_by_stock[0] = Some(Payload);
        cache.align_to(&["B".into(), "C".into(), "A".into()]);
        assert!(cache.cache.last_snapshot_by_stock[0].is_none());
        assert!(cache.cache.last_snapshot_by_stock[1].is_none());
        assert!(cache.cache.last_snapshot_by_stock[2].is_some());
        cache.align_to(&["B".into()]);
        assert_eq!(cache.cache.last_snapshot_by_stock.len(), 1);
    }

    #[test]
    fn financial_main_business_cached_marker_preserves_owned_string_hash() {
        use std::hash::{Hash, Hasher};
        let table = Table::new(BTreeMap::from([
            (
                "ts_code".into(),
                ColumnData::Utf8(vec![Some("A".into()); 4]),
            ),
            ("end_date".into(), ColumnData::I32(vec![Some(20241231); 4])),
            (
                "update_flag".into(),
                ColumnData::I64(vec![Some(0), Some(0), Some(0), Some(1)]),
            ),
            (
                "bz_type".into(),
                ColumnData::Utf8(vec![
                    Some("I".into()),
                    Some("i".into()),
                    Some("P".into()),
                    Some("I".into()),
                ]),
            ),
            (
                "bz_item".into(),
                ColumnData::Utf8(vec![
                    Some("second".into()),
                    Some("first".into()),
                    Some("excluded".into()),
                    Some("revision".into()),
                ]),
            ),
            (
                "bz_sales".into(),
                ColumnData::F64(vec![Some(2.0), Some(1.0), Some(99.0), Some(88.0)]),
            ),
        ]))
        .unwrap();
        let index = super::MainBusinessIndex::from_table(Arc::new(table)).unwrap();
        let mut rows = vec![
            ("second".to_string(), 2.0f64.to_bits(), 0i64),
            ("first".to_string(), 1.0f64.to_bits(), 0i64),
        ];
        rows.sort();
        let mut hash = std::collections::hash_map::DefaultHasher::new();
        20241231i32.hash(&mut hash);
        rows.hash(&mut hash);
        let expected = Some(i64::from_ne_bytes(hash.finish().to_ne_bytes()));
        assert_eq!(
            index.reader().industry_update0_fingerprint("A", 20241231),
            expected
        );
        assert_eq!(
            index
                .reader()
                .industry_update0_fingerprint("missing", 20241231),
            None
        );
        assert_eq!(
            index.reader().industry_update0_fingerprint("A", 20240930),
            None
        );
    }

    #[test]
    fn financial_pit_index_shares_rows_but_keeps_report_type_preferences_separate() {
        let table = Arc::new(financial_table(&[
            (20250331, 20250420, 1, 0, 1.0),
            (20250331, 20250420, 2, 0, 2.0),
            (20250331, 20250420, 3, 0, 3.0),
            (20250331, 20250420, 4, 0, 4.0),
        ]));
        let index = FinancialPitIndex::from_table(Arc::clone(&table)).expect("pit index");
        let income = index.reader(ReportTypePreference::income_single_quarter());
        let balance = index.reader(ReportTypePreference::balance_sheet_consolidated());

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
    fn financial_annual_partitions_match_merged_versions_and_cutoffs() {
        let first = Arc::new(financial_table(&[
            (20231231, 20240430, 2, 0, 1.0),
            (20240331, 20240430, 2, 0, 2.0),
            (20240630, 20240830, 2, 0, 3.0),
            (20240930, 20241030, 2, 0, 4.0),
            (20231231, 20240430, 1, 0, 10.0),
        ]));
        let second = Arc::new(financial_table(&[
            (20231231, 20250105, 3, 0, 5.0),
            (20231231, 20250105, 3, 1, 6.0),
            (20231231, 20250105, 3, 1, 7.0),
            (20240930, 20250201, 3, 0, 8.0),
        ]));
        let a = Arc::new(FinancialPitIndex::from_table(first.clone()).unwrap());
        let b = Arc::new(FinancialPitIndex::from_table(second.clone()).unwrap());
        for cutoff in [20240429, 20240430, 20241031, 20250105, 20250201] {
            let composite = FinancialPitIndex::from_partitions(vec![a.clone(), b.clone()], cutoff);
            let merged = FinancialPitIndex::from_source_tables(
                vec![first.clone(), second.clone()],
                Some(cutoff),
            )
            .unwrap();
            assert!(composite.records.is_empty());
            assert!(composite.sources.is_empty());
            assert!(Arc::ptr_eq(&composite.partitions[0], &a));
            assert_eq!(composite.len(), merged.len());
            for preference in [
                ReportTypePreference::new(&[3, 2]),
                ReportTypePreference::new(&[1, 4]),
            ] {
                let actual = composite.reader(preference.clone());
                let expected = merged.reader(preference);
                for date in [20240429, 20240430, 20241031, 20250105, 20250201, 20261231] {
                    for end in [20231231, 20240331, 20240630, 20240930] {
                        let value = |r: Option<super::PitFinancialRecordView<'_>>| {
                            r.map(|r| {
                                (
                                    r.disclosure_date(),
                                    r.report_type(),
                                    r.update_flag(),
                                    r.column("value"),
                                )
                            })
                        };
                        assert_eq!(
                            value(actual.record_for_end_date("000001.SZ", date, end)),
                            value(expected.record_for_end_date("000001.SZ", date, end))
                        );
                    }
                    assert_eq!(
                        actual.latest_quarter_end_date("000001.SZ", date),
                        expected.latest_quarter_end_date("000001.SZ", date)
                    );
                    assert_eq!(
                        actual.ttm_sum("000001.SZ", date, "value"),
                        expected.ttm_sum("000001.SZ", date, "value")
                    );
                    for count in [0, 1, 3] {
                        assert_eq!(
                            actual.annual_values("000001.SZ", date, "value", count),
                            expected.annual_values("000001.SZ", date, "value", count)
                        );
                    }
                    assert_eq!(
                        actual.disclosure_dates_between("000001.SZ", 20230101, date),
                        expected.disclosure_dates_between("000001.SZ", 20230101, date)
                    );
                }
            }
        }
        let tie = Arc::new(
            FinancialPitIndex::from_table(Arc::new(financial_table(&[(
                20231231, 20240430, 2, 0, 999.0,
            )])))
            .unwrap(),
        );
        let composite = FinancialPitIndex::from_partitions(vec![a, tie], 20240430);
        assert_eq!(
            composite
                .reader(ReportTypePreference::new(&[2]))
                .record_for_end_date("000001.SZ", 20240430, 20231231)
                .unwrap()
                .column("value"),
            Some(1.0)
        );
    }

    #[test]
    fn financial_stock_route_skips_unchanged_markers_and_handles_reentry_and_rewind() {
        use super::{routed_financial_stock_snapshots_for_date, FinancialStockEventRoute};
        let index = Arc::new(
            FinancialPitIndex::from_table(Arc::new(financial_table_with_codes(&[
                ("A", 20241231, 20250418, 2, 0, 1.0),
                ("B", 20241231, 20250418, 2, 0, 2.0),
                ("A", 20241231, 20250419, 3, 1, 3.0), // Saturday revision
                ("B", 20241231, 20250420, 1, 0, 99.0), // irrelevant report type
            ])))
            .unwrap(),
        );
        let reader = index.reader(ReportTypePreference::new(&[3, 2]));
        let route = FinancialStockEventRoute::new(&[&reader]);
        let mut cache = InstrumentAlignedSnapshotCache::default();
        let mut markers = BTreeMap::<String, usize>::new();
        let mut computes = BTreeMap::<String, usize>::new();
        for (date, skip_b, expected) in [
            (20250418, false, vec![Some(1.0), Some(2.0)]),
            (20250421, false, vec![Some(3.0), Some(2.0)]),
            (20250422, true, vec![Some(3.0), None]),
            (20250423, false, vec![Some(3.0), Some(2.0)]),
            (20250423, false, vec![Some(3.0), Some(2.0)]),
            (20250418, false, vec![Some(1.0), Some(2.0)]),
        ] {
            let panel = panel_for_codes(date, &["A", "B"]);
            let values = routed_financial_stock_snapshots_for_date(
                &panel,
                date,
                &mut cache,
                &route,
                |_, code, _| skip_b && code == "B",
                |date, code, _| {
                    *markers.entry(code.into()).or_default() += 1;
                    let mut marker = FinancialEventMarkerBuilder::new();
                    marker.include_reader_latest_quarter(
                        FinancialStatementDataset::Income,
                        &reader,
                        code,
                        date,
                    );
                    marker.build()
                },
                |date, code, _| {
                    *computes.entry(code.into()).or_default() += 1;
                    reader
                        .record_for_end_date(code, date, 20241231)?
                        .column("value")
                },
            );
            assert_eq!(values, expected);
        }
        assert_eq!(markers["A"], 3);
        assert_eq!(markers["B"], 3); // initial, reentry, rewind (not the other stock's event)
        assert_eq!(computes["A"], 3);
        assert_eq!(computes["B"], 2);
        // A new bounded batch view must revalidate markers, even if the sources are reused.
        let batch = FinancialPitIndex::from_partitions(vec![index], 20250418);
        let reader = batch.reader(ReportTypePreference::new(&[3, 2]));
        let p = panel_for_codes(20250424, &["B", "C", "A"]);
        let values = routed_financial_stock_snapshots_for_date(
            &p,
            20250424,
            &mut cache,
            &FinancialStockEventRoute::new(&[&reader]),
            |_, _, _| false,
            |date, code, _| {
                let mut m = FinancialEventMarkerBuilder::new();
                m.include_reader_latest_quarter(
                    FinancialStatementDataset::Income,
                    &reader,
                    code,
                    date,
                );
                m.build()
            },
            |date, code, _| {
                reader
                    .record_for_end_date(code, date, 20241231)?
                    .column("value")
            },
        );
        for (code, value) in p.instruments().iter().zip(values) {
            assert_eq!(
                value,
                match code.as_str() {
                    "A" => Some(1.0),
                    "B" => Some(2.0),
                    _ => None,
                }
            );
        }
    }

    #[test]
    fn financial_stock_route_source_loss_clears_previously_valid_snapshot() {
        use super::{routed_financial_stock_snapshots_for_date, FinancialStockEventRoute};
        let valid = FinancialPitIndex::from_table(Arc::new(financial_table(&[(
            20240331, 20240430, 2, 0, 7.0,
        )])))
        .unwrap();
        let empty = FinancialPitIndex::from_partitions(Vec::new(), 20250430);
        let p = panel(&[20250429, 20250430]);
        let mut cache = InstrumentAlignedSnapshotCache::default();
        for (date, index, expected) in [(20250429, &valid, Some(7.0)), (20250430, &empty, None)] {
            let reader = index.reader(ReportTypePreference::new(&[2]));
            let values = routed_financial_stock_snapshots_for_date(
                &p,
                date,
                &mut cache,
                &FinancialStockEventRoute::new(&[&reader]),
                |_, _, _| false,
                |date, code, _| {
                    let mut m = FinancialEventMarkerBuilder::new();
                    m.include_reader_latest_quarter(
                        FinancialStatementDataset::Income,
                        &reader,
                        code,
                        date,
                    );
                    m.build()
                },
                |date, code, _| {
                    reader
                        .record_for_end_date(code, date, 20240331)?
                        .column("value")
                },
            );
            assert_eq!(values, vec![expected]);
        }
    }

    #[test]
    fn financial_stock_route_uses_historical_query_clock_and_formula_key() {
        use super::{routed_financial_stock_snapshots_for_date, FinancialStockEventRoute};
        let index = FinancialPitIndex::from_table(Arc::new(financial_table(&[(
            20240331, 20240430, 2, 0, 7.0,
        )])))
        .unwrap();
        let reader = index.reader(ReportTypePreference::new(&[2]));
        let p = panel(&[20250429, 20250430, 20250501]);
        let mut cache = InstrumentAlignedSnapshotCache::default();
        let mut count = 0;
        for (output, query, key) in [
            (20250429, 20240429, 1),
            (20250430, 20240430, 1),
            (20250501, 20240501, 2),
        ] {
            let values = routed_financial_stock_snapshots_for_date(
                &p,
                output,
                &mut cache,
                &FinancialStockEventRoute::new(&[&reader])
                    .at_date(query)
                    .with_key(key),
                |_, _, _| false,
                |_, code, _| {
                    count += 1;
                    let mut m = FinancialEventMarkerBuilder::new();
                    m.include_reader_latest_quarter(
                        FinancialStatementDataset::Income,
                        &reader,
                        code,
                        query,
                    );
                    m.include_synthetic("formula", key);
                    m.build()
                },
                |_, code, _| {
                    reader
                        .record_for_end_date(code, query, 20240331)?
                        .column("value")
                },
            );
            assert_eq!(
                values,
                vec![if query < 20240430 { None } else { Some(7.0) }]
            );
        }
        assert_eq!(count, 3);
    }

    #[test]
    fn financial_stock_route_dividend_expiry_matches_calendar_window_in_leap_year() {
        use super::{
            add_months, routed_financial_stock_snapshots_for_date, FinancialStockEventRoute,
        };
        let index = DividendIndex::from_table(Arc::new(dividend_table(&[(
            "000001.SZ",
            20230220,
            super::IMPLEMENTED_DIV_PROC,
            1.0,
            20230228,
            10.0,
        )])))
        .unwrap();
        let reader = index.reader();
        let dates = [20240227, 20240228, 20240229, 20240301, 20240304];
        let p = panel(&dates);
        let mut cache = InstrumentAlignedSnapshotCache::default();
        let mut markers = 0;
        for date in dates {
            let value = reader.implemented_ltm_sum("000001.SZ", add_months(date, -12), date);
            let result = routed_financial_stock_snapshots_for_date(
                &p,
                date,
                &mut cache,
                &FinancialStockEventRoute::new(&[]).with_dividend(&reader),
                |_, _, _| false,
                |_, _, _| {
                    markers += 1;
                    let mut m = FinancialEventMarkerBuilder::new();
                    m.include_synthetic("cash", super::f64_marker_value(value));
                    m.build()
                },
                |_, _, _| Some(value),
            );
            assert_eq!(result, vec![Some(value)]);
        }
        assert_eq!(markers, 3); // initial, original expiry event, exact leap-year expiry
    }

    #[test]
    fn financial_main_business_multi_sources_borrow_strings_and_match_single_table() {
        let make = |year: i32, names: &[&str]| {
            Arc::new(
                Table::new(BTreeMap::from([
                    (
                        "ts_code".into(),
                        ColumnData::Utf8(vec![Some("A".into()); names.len()]),
                    ),
                    (
                        "end_date".into(),
                        ColumnData::I32(vec![Some(year * 10000 + 1231); names.len()]),
                    ),
                    (
                        "bz_type".into(),
                        ColumnData::Utf8(vec![Some("I".into()); names.len()]),
                    ),
                    (
                        "bz_item".into(),
                        ColumnData::Utf8(names.iter().map(|s| Some((*s).into())).collect()),
                    ),
                    (
                        "bz_sales".into(),
                        ColumnData::F64(vec![Some(10.0); names.len()]),
                    ),
                    (
                        "update_flag".into(),
                        ColumnData::I64(vec![Some(0); names.len()]),
                    ),
                ]))
                .unwrap(),
            )
        };
        let a = make(2023, &["one", "two"]);
        let b = make(2024, &["three"]);
        let index =
            super::MainBusinessIndex::from_source_tables(vec![a.clone(), b.clone()], 20241231)
                .unwrap();
        let mut merged = (*a).clone();
        merged.append(&b).unwrap();
        let reference = super::MainBusinessIndex::from_table(Arc::new(merged)).unwrap();
        for end in [20231231, 20241231] {
            assert_eq!(
                index.reader().industry_update0_fingerprint("A", end),
                reference.reader().industry_update0_fingerprint("A", end)
            );
        }
        let view = super::MainBusinessRecordView {
            index: &index,
            row_pos: 0,
        };
        let original = a.required_utf8("bz_item").unwrap()[0].as_deref().unwrap();
        assert!(std::ptr::eq(
            view.bz_item().unwrap().as_ptr(),
            original.as_ptr()
        ));
        let bounded = super::MainBusinessIndex::from_source_tables(vec![a, b], 20240101).unwrap();
        assert_eq!(bounded.len(), 2);
        assert_eq!(
            bounded.reader().industry_update0_fingerprint("A", 20241231),
            None
        );
    }

    #[test]
    fn financial_event_schedule_from_pit_reader_filters_unrequested_report_types() {
        let table = Arc::new(financial_table(&[
            (20250331, 20250420, 1, 0, 1.0),
            (20250331, 20250421, 9, 0, 9.0),
        ]));
        let index = FinancialPitIndex::from_table(table).expect("pit index");
        let schedule = FinancialEventSchedule::from_pit_readers(&[
            index.reader(ReportTypePreference::balance_sheet_consolidated())
        ]);

        assert!(schedule.has_event_after_until(Some(20250419), 20250420));
        assert!(!schedule.has_event_after_until(Some(20250420), 20250421));
    }

    #[test]
    fn financial_pit_reader_uses_only_disclosed_versions() {
        let table = financial_table(&[
            (20241231, 20250331, 3, 0, 10.0),
            (20241231, 20250430, 3, 1, 12.0),
        ]);
        let index = FinancialPitIndex::from_table(Arc::new(table)).expect("index");
        let data = index.reader(ReportTypePreference::income_single_quarter());

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
    fn financial_pit_reader_keeps_year_end_and_q1_when_disclosed_together() {
        let table = financial_table(&[
            (20241231, 20250430, 3, 0, 12.0),
            (20250331, 20250430, 3, 0, 3.0),
        ]);
        let index = FinancialPitIndex::from_table(Arc::new(table)).expect("index");
        let data = index.reader(ReportTypePreference::income_single_quarter());

        assert_eq!(
            data.latest_quarter_end_date("000001.SZ", 20250430),
            Some(20250331)
        );
        assert_eq!(
            data.record_for_end_date("000001.SZ", 20250430, 20241231)
                .and_then(|record| record.column("value")),
            Some(12.0)
        );
        assert_eq!(
            data.record_for_end_date("000001.SZ", 20250430, 20250331)
                .and_then(|record| record.column("value")),
            Some(3.0)
        );
    }

    #[test]
    fn financial_pit_reader_exposes_annual_and_ttm_helpers() {
        let table = financial_table(&[
            (20211231, 20220430, 1, 0, 10.0),
            (20221231, 20230430, 1, 0, 20.0),
            (20230331, 20230430, 1, 0, 1.0),
            (20230630, 20230831, 1, 0, 2.0),
            (20230930, 20231031, 1, 0, 3.0),
            (20231231, 20240430, 1, 0, 4.0),
            (20241231, 20250430, 1, 0, 40.0),
        ]);
        let index = FinancialPitIndex::from_table(Arc::new(table)).expect("index");
        let data = index.reader(ReportTypePreference::consolidated());

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
        let index = FinancialPitIndex::from_table(Arc::new(table)).expect("index");
        let data = index.reader(ReportTypePreference::consolidated());
        let mut cache = FinancialStockSnapshotCache::<f64>::new(2);
        let mut calls = [0usize; 2];

        for trade_date in [20250425, 20250428, 20250429] {
            for (idx, ts_code) in ["000001.SZ", "000002.SZ"].iter().enumerate() {
                let mut builder = FinancialEventMarkerBuilder::new();
                builder.include_reader_latest_quarter(
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
    fn cached_financial_stock_snapshots_reuses_until_marker_changes() {
        let panel = panel(&[20260105, 20260106, 20260107]);
        let mut calls = 0usize;
        let snapshots = cached_financial_stock_snapshots(
            &panel,
            |_, _, _| false,
            |trade_date, _, _| {
                let mut builder = FinancialEventMarkerBuilder::new();
                let marker_value = if trade_date < 20260107 { 1 } else { 2 };
                builder.include_synthetic("marker", marker_value);
                builder.build()
            },
            |trade_date, _, _| {
                calls += 1;
                Some(trade_date)
            },
        );

        assert_eq!(calls, 2);
        assert_eq!(
            snapshots,
            vec![Some(20260105), Some(20260105), Some(20260107)]
        );
    }

    #[test]
    fn cached_financial_stock_snapshots_skip_clears_cached_snapshot() {
        let panel = panel(&[20260105, 20260106, 20260107]);
        let mut calls = 0usize;
        let snapshots = cached_financial_stock_snapshots(
            &panel,
            |trade_date, _, _| trade_date == 20260106,
            |_, _, _| {
                let mut builder = FinancialEventMarkerBuilder::new();
                builder.include_synthetic("marker", 1);
                builder.build()
            },
            |trade_date, _, _| {
                calls += 1;
                Some(trade_date)
            },
        );

        assert_eq!(calls, 2);
        assert_eq!(snapshots, vec![Some(20260105), None, Some(20260107)]);
    }

    #[test]
    fn instrument_aligned_snapshot_cache_remaps_by_ts_code() {
        let mut cache = InstrumentAlignedSnapshotCache::<i32>::default();
        cache.align_to(&["000001.SZ".to_string(), "000002.SZ".to_string()]);
        let mut marker = FinancialEventMarkerBuilder::new();
        marker.include_synthetic("event", 1);
        let marker = marker.build();
        assert_eq!(
            cache.get_or_update(0, marker.clone(), || Some(10)),
            Some(10)
        );
        assert_eq!(
            cache.get_or_update(1, marker.clone(), || Some(20)),
            Some(20)
        );

        cache.align_to(&["000002.SZ".to_string(), "000001.SZ".to_string()]);
        let mut calls = 0usize;
        assert_eq!(
            cache.get_or_update(0, marker.clone(), || {
                calls += 1;
                Some(99)
            }),
            Some(20)
        );
        assert_eq!(
            cache.get_or_update(1, marker, || {
                calls += 1;
                Some(88)
            }),
            Some(10)
        );
        assert_eq!(calls, 0);
    }

    #[test]
    fn cached_financial_stock_snapshots_for_date_reuses_provider_state() {
        let panel = DailyPanel::from_index(
            vec![20260105, 20260106],
            vec!["000001.SZ".to_string(), "000002.SZ".to_string()],
            &[20260105, 20260106],
            vec![true, true, true, true],
        )
        .unwrap();
        let mut cache = InstrumentAlignedSnapshotCache::<i32>::default();
        let mut calls = [0usize; 2];

        for trade_date in [20260105, 20260106] {
            let snapshots = cached_financial_stock_snapshots_for_date(
                &panel,
                trade_date,
                &mut cache,
                |_, _, _| false,
                |_, _, _| {
                    let mut marker = FinancialEventMarkerBuilder::new();
                    marker.include_synthetic("event", 1);
                    marker.build()
                },
                |_, _, offset| {
                    let instrument_idx = offset % 2;
                    calls[instrument_idx] += 1;
                    Some((instrument_idx + 1) as i32)
                },
            );
            assert_eq!(snapshots, vec![Some(1), Some(2)]);
        }

        assert_eq!(calls, [1, 1]);
    }
}

use std::any::Any;

use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::{err, Result};
use crate::factor::common::financial::previous_quarter_end_date;
use crate::factor::common::stock_daily_ops::{is_bj_stock, neutralize_size_sector_with_inputs};
use crate::factor::common::{
    cached_financial_stock_snapshots_for_date, ClassificationLevel, ClassificationMap,
    FinancialEventMarker, FinancialEventMarkerBuilder, FinancialEventSchedule, FinancialPitReader,
    FinancialStatementDataset, InstrumentAlignedSnapshotCache, ReportTypePreference,
};
use crate::factor::{Factor, FactorUpdatePolicy};

const HISTORY: usize = 8;
const QUARTERS: usize = HISTORY + 5;
const MIN_QUARTERS: usize = 8;
const MIN_PAIRS: usize = 4;
const EPS: f64 = 1e-12;
const PROVIDER: &str = "dfzq_seasonal_surprise";

#[derive(Clone, Copy)]
pub enum Output {
    Sue,
    Sur,
}
impl Output {
    fn index(self) -> usize {
        match self {
            Self::Sue => 0,
            Self::Sur => 1,
        }
    }
    fn id(self) -> &'static str {
        match self {
            Self::Sue => "sue0",
            Self::Sur => "sur0",
        }
    }
    fn column(self) -> &'static str {
        match self {
            Self::Sue => "n_income",
            Self::Sur => "revenue",
        }
    }
}
pub struct SeasonalSurprise(pub Output);
#[derive(Default)]
struct State {
    snapshots: InstrumentAlignedSnapshotCache<[Option<f64>; 2]>,
}

impl Factor for SeasonalSurprise {
    fn spec(&self) -> FactorSpec {
        let mut spec = FactorSpec {
            id: self.0.id().into(), aliases: vec![self.0.id().to_ascii_uppercase()], name: self.0.id().to_ascii_uppercase(),
            asset_class: AssetClass::Stock, frequency: Frequency::Daily, version: "0.1.0".into(),
            tags: ["DFZQ", "fundamental", "financial", "pit", "surprise", "seasonal_random_walk", "neutralize", "size", "sector", "daily"].into_iter().map(str::to_string).collect(),
            description: format!("PIT single-quarter {} seasonal random walk with drift: (X_t-X_t-4-mean(previous eight YoY differences))/sample_std(previous eight YoY differences). Excludes current difference from estimation; reads 13 consecutive quarter slots, requires >=8 finite quarterly levels, >=4 valid historical YoY pairs, and valid current/prior-year levels. Missing quarters are not compacted or zero-filled; negative levels retained. Regular statements only, no earnings express. Prior-year values use latest PIT revisions visible at the calculation date, not a dedicated current-report comparative field. Shared event-driven stock snapshots; daily SW L1/Barra SIZE neutralization, excludes BJ; no winsorization or final zscore.", self.0.column()),
            dependencies: vec![DataRequest::financial_quarters(DatasetId::StockIncome, &[self.0.column()], QUARTERS), DataRequest::new(DatasetId::StockBarraDaily, &["SIZE"]), DataRequest::new(DatasetId::StockSwClassification, &["l1_code"])],
            intraday_raw_dependencies: vec![], lookback: Lookback { trading_days: 0 },
        };
        if matches!(self.0, Output::Sur) {
            spec.tags.push("deprecated".into());
        }
        spec
    }
    fn compute_provider_key(&self) -> String {
        PROVIDER.into()
    }
    fn update_policy(&self) -> FactorUpdatePolicy {
        FactorUpdatePolicy::FinancialEventStateDailyFast
    }
    fn initial_compute_state(&self, _: &[String]) -> Box<dyn Any + Send> {
        Box::new(State::default())
    }
    fn compute(&self, _: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        compute(&[self.0.id().into()], data, &mut State::default())?
            .pop()
            .ok_or_else(|| err("missing seasonal surprise output"))
    }
    fn compute_many_stateful(
        &self,
        ids: &[String],
        _: &FactorContext,
        data: &DataPool,
        state: &mut (dyn Any + Send),
    ) -> Result<Vec<FactorSeries>> {
        compute(
            ids,
            data,
            state
                .downcast_mut::<State>()
                .ok_or_else(|| err("seasonal surprise state mismatch"))?,
        )
    }
}

fn ends(anchor: i32) -> Option<[i32; QUARTERS]> {
    let mut ends = [anchor; QUARTERS];
    for i in 1..QUARTERS {
        ends[i] = previous_quarter_end_date(ends[i - 1])?;
    }
    Some(ends)
}
fn marker(
    reader: &FinancialPitReader<'_>,
    code: &str,
    date: i32,
    mask: i64,
) -> Option<FinancialEventMarker> {
    let ends = ends(reader.latest_quarter_end_date(code, date)?)?;
    let mut marker = FinancialEventMarkerBuilder::new();
    marker.include_synthetic("requested_outputs", mask);
    for end in ends {
        marker.include_reader_record_for_end_date(
            FinancialStatementDataset::Income,
            reader,
            code,
            date,
            end,
        );
    }
    marker.build()
}
fn snapshot(
    reader: &FinancialPitReader<'_>,
    code: &str,
    date: i32,
    requested: &[Output],
) -> Option<[Option<f64>; 2]> {
    let ends = ends(reader.latest_quarter_end_date(code, date)?)?;
    let records = ends.map(|end| reader.record_for_end_date(code, date, end));
    let mut output = [None; 2];
    for &kind in requested {
        let levels =
            std::array::from_fn(|i| records[i].as_ref().and_then(|r| r.column(kind.column())));
        output[kind.index()] = surprise(&levels);
    }
    Some(output)
}

// Newest first. The current YoY difference must not estimate its own expectation.
fn surprise(levels: &[Option<f64>; QUARTERS]) -> Option<f64> {
    let levels = levels.map(|v| v.filter(|v| v.is_finite()));
    if levels.iter().flatten().count() < MIN_QUARTERS {
        return None;
    }
    let current = levels[0]? - levels[4]?;
    let mut differences = [0.0; HISTORY];
    let mut count = 0;
    for i in 1..=HISTORY {
        if let (Some(now), Some(prior)) = (levels[i], levels[i + 4]) {
            let delta = now - prior;
            if delta.is_finite() {
                differences[count] = delta;
                count += 1;
            }
        }
    }
    if count < MIN_PAIRS {
        return None;
    }
    let differences = &differences[..count];
    let mean = differences.iter().sum::<f64>() / count as f64;
    let sd =
        (differences.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (count - 1) as f64).sqrt();
    if !sd.is_finite() || sd <= EPS {
        return None;
    }
    let result = (current - mean) / sd;
    result.is_finite().then_some(result)
}

fn compute(ids: &[String], data: &DataPool, state: &mut State) -> Result<Vec<FactorSeries>> {
    let requested: Vec<_> = [Output::Sue, Output::Sur]
        .into_iter()
        .filter(|kind| ids.iter().any(|id| id == kind.id()))
        .collect();
    if requested.is_empty() {
        return Ok(vec![]);
    }
    let mask = requested
        .iter()
        .fold(0, |mask, kind| mask | (1 << kind.index()));
    let panel = data.stock_universe_panel()?;
    let reader = data.financial_reader(
        DatasetId::StockIncome,
        ReportTypePreference::income_single_quarter(),
    )?;
    let schedule = FinancialEventSchedule::from_pit_readers(&[reader.clone()]);
    let n = panel.instruments().len();
    let mut values: Vec<_> = requested
        .iter()
        .map(|_| vec![None; panel.shape_len()])
        .collect();
    let mut snapshots = vec![None; n];
    let mut previous_day: Option<usize> = None;
    for (day, &date) in panel.dates().iter().enumerate() {
        if !panel.is_target_date(date) {
            continue;
        }
        let presence_changed = previous_day.is_some_and(|prev| {
            (0..n).any(|i| {
                panel.is_present_offset(prev * n + i) != panel.is_present_offset(day * n + i)
            })
        });
        if previous_day.is_none()
            || presence_changed
            || schedule.has_event_after_until(previous_day.map(|prev| panel.dates()[prev]), date)
        {
            snapshots = cached_financial_stock_snapshots_for_date(
                &panel,
                date,
                &mut state.snapshots,
                |_, code, offset| is_bj_stock(code) || !panel.is_present_offset(offset),
                |date, code, _| marker(&reader, code, date, mask),
                |date, code, _| snapshot(&reader, code, date, &requested),
            );
        }
        for (i, snapshot) in snapshots.iter().enumerate() {
            if !panel.is_present_offset(day * n + i) {
                continue;
            }
            for (j, &kind) in requested.iter().enumerate() {
                values[j][day * n + i] = snapshot.and_then(|v| v[kind.index()]);
            }
        }
        previous_day = Some(day);
    }
    let size = panel.column_from_table(data.daily(DatasetId::StockBarraDaily)?, "SIZE")?;
    let sector = ClassificationMap::from_table(
        data.daily(DatasetId::StockSwClassification)?,
        ClassificationLevel::Sector,
    )?;
    requested
        .into_iter()
        .zip(values)
        .map(|(kind, values)| {
            let raw = panel.column_from_values(values)?;
            Ok(
                neutralize_size_sector_with_inputs(&raw, &panel, &size, &sector)?
                    .to_factor_series(SeasonalSurprise(kind).spec()),
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{ColumnData, Table};
    use std::collections::{BTreeMap, HashMap};

    #[test]
    fn seasonal_surprise_excludes_current_change_and_uses_sample_std() {
        let mut levels = std::array::from_fn(|i| Some((i * i) as f64));
        let history: Vec<_> = (1..=8)
            .map(|i| (i * i) as f64 - ((i + 4) * (i + 4)) as f64)
            .collect();
        let mean = history.iter().sum::<f64>() / 8.0;
        let sd = (history.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / 7.0).sqrt();
        let expected = (-16.0 - mean) / sd;
        assert!((surprise(&levels).unwrap() - expected).abs() < 1e-12);
        levels[0] = Some(100.0);
        assert!((surprise(&levels).unwrap() - expected - 100.0 / sd).abs() < 1e-12);
        let negative = levels.map(|v| v.map(|x| -x));
        assert!((surprise(&negative).unwrap() + surprise(&levels).unwrap()).abs() < 1e-12);
    }

    #[test]
    fn seasonal_surprise_missing_pairs_preserve_quarter_positions() {
        let full = std::array::from_fn(|i| Some((i * i) as f64));
        let mut partial = full;
        partial[8..].fill(None);
        assert_eq!(surprise(&partial), None); // Eight levels but only three historical pairs.
        partial[8] = full[8];
        assert!(surprise(&partial).is_some());
        // Exactly eight levels can still supply four pairs; no compacting missing quarters.
        partial = full;
        for i in [3, 7, 9, 11, 12] {
            partial[i] = None;
        }
        assert_eq!(partial.iter().flatten().count(), 8);
        assert!(surprise(&partial).is_some());
        partial[0] = None;
        assert_eq!(surprise(&partial), None);
        assert_eq!(surprise(&[Some(1.0); QUARTERS]), None);
        assert_eq!(surprise(&[Some(f64::NAN); QUARTERS]), None);
        assert_eq!(surprise(&std::array::from_fn(|i| Some(i as f64))), None);
    }

    fn fixture(targets: Vec<i32>, revision: f64, omit_first: bool) -> DataPool {
        let dates = vec![20250506, 20250507, 20250602];
        let codes: Vec<_> = (0..16)
            .map(|s| {
                Some(if s == 15 {
                    "430001.BJ".into()
                } else {
                    format!("{:06}.SZ", s + 1)
                })
            })
            .collect();
        let mut basic = Table::new(BTreeMap::from([
            ("ts_code".into(), ColumnData::Utf8(codes.clone())),
            (
                "list_date".into(),
                ColumnData::I32(
                    (0..16)
                        .map(|s| Some(if s == 12 { 20250507 } else { 20100101 }))
                        .collect(),
                ),
            ),
            ("delist_date".into(), ColumnData::I32(vec![None; 16])),
        ]))
        .unwrap();
        if omit_first {
            basic = basic.take(&(1..16).rev().collect::<Vec<_>>()).unwrap();
        }
        let sector = Table::new(BTreeMap::from([
            ("ts_code".into(), ColumnData::Utf8(codes.clone())),
            ("in_date".into(), ColumnData::I32(vec![Some(20100101); 16])),
            ("out_date".into(), ColumnData::I32(vec![None; 16])),
            (
                "l1_code".into(),
                ColumnData::Utf8((0..16).map(|s| Some((s / 8).to_string())).collect()),
            ),
        ]))
        .unwrap();
        let size = Table::new(BTreeMap::from([
            (
                "ts_code".into(),
                ColumnData::Utf8(
                    dates
                        .iter()
                        .flat_map(|_| codes.iter().rev().cloned())
                        .collect(),
                ),
            ),
            (
                "trade_date".into(),
                ColumnData::I32(
                    dates
                        .iter()
                        .flat_map(|date| vec![Some(*date); 16])
                        .collect(),
                ),
            ),
            (
                "SIZE".into(),
                ColumnData::F64(
                    dates
                        .iter()
                        .flat_map(|date| {
                            (0..16).rev().map(move |s| {
                                (s != 13).then_some(
                                    s as f64
                                        + if *date == 20250507 {
                                            (s as f64).sin()
                                        } else {
                                            0.0
                                        },
                                )
                            })
                        })
                        .collect(),
                ),
            ),
        ]))
        .unwrap();
        let ends = ends(20250331).unwrap();
        let rows: Vec<_> = (0..16)
            .flat_map(|s| (0..QUARTERS).map(move |q| (s, q, false)))
            .chain([(0, 8, true)])
            .collect();
        let mut columns = BTreeMap::from([
            (
                "ts_code".into(),
                ColumnData::Utf8(rows.iter().map(|(s, _, _)| codes[*s].clone()).collect()),
            ),
            (
                "end_date".into(),
                ColumnData::I32(rows.iter().map(|(_, q, _)| Some(ends[*q])).collect()),
            ),
            (
                "ann_date".into(),
                ColumnData::I32(
                    rows.iter()
                        .map(|(_, _, r)| Some(if *r { 20250602 } else { 20250420 }))
                        .collect(),
                ),
            ),
            (
                "f_ann_date".into(),
                ColumnData::I32(
                    rows.iter()
                        .map(|(_, _, r)| Some(if *r { 20250602 } else { 20250420 }))
                        .collect(),
                ),
            ),
            (
                "report_type".into(),
                ColumnData::I64(vec![Some(2); rows.len()]),
            ),
            (
                "update_flag".into(),
                ColumnData::I64(rows.iter().map(|(_, _, r)| Some(i64::from(*r))).collect()),
            ),
        ]);
        for kind in [Output::Sue, Output::Sur] {
            columns.insert(
                kind.column().into(),
                ColumnData::F64(
                    rows.iter()
                        .map(|(s, q, revised)| {
                            Some(if *revised {
                                revision
                            } else if *s == 14 && kind.index() == 0 {
                                7.0
                            } else {
                                100.0
                                    + ((*s + 1) as f64)
                                        * ((*q + 1) as f64).powi(2 + kind.index() as i32)
                                    + (((*s + 1) * (*q + 1)) as f64).sin() * 20.0
                            })
                        })
                        .collect(),
                ),
            );
        }
        let context = FactorContext {
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            start_date: targets[0],
            end_date: *targets.last().unwrap(),
            load_start_date: dates[0],
            load_dates: dates,
            target_dates: targets,
        };
        DataPool::from_daily_tables_for_test(
            HashMap::from([
                (DatasetId::StockBasic, basic),
                (DatasetId::StockSwClassification, sector),
                (DatasetId::StockBarraDaily, size),
                (DatasetId::StockIncome, Table::new(columns).unwrap()),
            ]),
            &context,
        )
        .unwrap()
    }

    #[test]
    fn seasonal_surprise_pit_batches_requested_outputs_and_masks() {
        let ids = vec!["sue0".into(), "sur0".into()];
        let dates = vec![20250506, 20250507, 20250602];
        let both = compute(
            &ids,
            &fixture(dates.clone(), 5000.0, false),
            &mut State::default(),
        )
        .unwrap();
        assert_eq!(both.len(), 2);
        for output in &both {
            assert!(output.values.iter().any(|v| v.value.is_some()));
            for value in &output.values {
                if let crate::core::FactorRowKey::Daily { ts_code, .. } = &value.key {
                    if ["000014.SZ", "430001.BJ"].contains(&ts_code.as_str()) {
                        assert!(value.value.is_none());
                    }
                    if ts_code == "000015.SZ" && output.spec.id == "sue0" {
                        assert!(value.value.is_none());
                    }
                }
            }
        }
        let mut state = State::default();
        for date in dates {
            let single = compute(&ids, &fixture(vec![date], 5000.0, false), &mut state).unwrap();
            for (a, b) in single.iter().zip(&both) {
                assert_eq!(
                    a.values
                        .iter()
                        .map(|v| (&v.key, v.value))
                        .collect::<Vec<_>>(),
                    b.values
                        .iter()
                        .filter(|v| v.key.trade_date() == date)
                        .map(|v| (&v.key, v.value))
                        .collect::<Vec<_>>()
                );
            }
        }
        let changed = compute(
            &ids,
            &fixture(vec![20250506, 20250507, 20250602], 50000.0, false),
            &mut State::default(),
        )
        .unwrap();
        for (before, after) in both.iter().zip(changed) {
            assert!(before
                .values
                .iter()
                .zip(&after.values)
                .filter(|(v, _)| v.key.trade_date() < 20250602)
                .all(|(a, b)| a.value == b.value));
            assert!(before
                .values
                .iter()
                .zip(&after.values)
                .any(|(a, b)| a.key.trade_date() == 20250602 && a.value != b.value));
        }
        let data = fixture(vec![20250506], 5000.0, false);
        let mut switched = State::default();
        assert_eq!(
            compute(&["sue0".into()], &data, &mut switched)
                .unwrap()
                .len(),
            1
        );
        let sur = compute(&["sur0".into()], &data, &mut switched).unwrap();
        assert_eq!(sur.len(), 1);
        assert_eq!(sur[0].spec.id, "sur0");
        assert_eq!(
            sur[0].values.iter().map(|v| v.value).collect::<Vec<_>>(),
            both[1]
                .values
                .iter()
                .filter(|v| v.key.trade_date() == 20250506)
                .map(|v| v.value)
                .collect::<Vec<_>>()
        );
        let changed_panel = fixture(vec![20250602], 5000.0, true);
        let cached = compute(&ids, &changed_panel, &mut state).unwrap();
        let fresh = compute(&ids, &changed_panel, &mut State::default()).unwrap();
        for (a, b) in cached.iter().zip(fresh) {
            assert_eq!(
                a.values
                    .iter()
                    .map(|v| (&v.key, v.value))
                    .collect::<Vec<_>>(),
                b.values
                    .iter()
                    .map(|v| (&v.key, v.value))
                    .collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn seasonal_surprise_metadata_and_provider() {
        let a = SeasonalSurprise(Output::Sue);
        let b = SeasonalSurprise(Output::Sur);
        assert_eq!(a.compute_provider_key(), b.compute_provider_key());
        assert_eq!(
            a.update_policy(),
            FactorUpdatePolicy::FinancialEventStateDailyFast
        );
        for kind in [Output::Sue, Output::Sur] {
            let spec = SeasonalSurprise(kind).spec();
            assert_eq!(
                spec.tags.contains(&"deprecated".into()),
                matches!(kind, Output::Sur)
            );
            assert!(
                spec.tags.contains(&"DFZQ".into()) && spec.tags.contains(&"fundamental".into())
            );
            assert_eq!(
                spec.dependencies[0].columns,
                vec![kind.column().to_string()]
            );
            assert_eq!(spec.dependencies[0].financial_quarters, Some(13));
            assert!(!spec
                .dependencies
                .iter()
                .any(|r| r.dataset == DatasetId::StockDailyPv));
        }
    }
}

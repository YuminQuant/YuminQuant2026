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
    routed_financial_stock_snapshots_for_date, FinancialStockEventRoute, ClassificationLevel, ClassificationMap,
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
            snapshots = routed_financial_stock_snapshots_for_date(
                &panel,
                date,
                &mut state.snapshots,
                &FinancialStockEventRoute::new(&[&reader]).with_key(mask),
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

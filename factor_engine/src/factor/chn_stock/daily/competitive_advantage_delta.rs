use std::any::Any;
use std::collections::BTreeMap;

use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::{err, Result};
use crate::factor::common::financial::previous_quarter_end_date;
use crate::factor::common::stock_daily_ops::{is_bj_stock, neutralize_size_sector};
use crate::factor::common::{
    cached_financial_stock_snapshots_for_date, compute_financial_event_snapshot_streaming_on_panel,
    factor_series_to_panel_column, ClassificationLevel, ClassificationMap, DailyPanel,
    EventDrivenCrossSectionCache, FinancialEventMarker, FinancialEventMarkerBuilder,
    FinancialEventSchedule, FinancialPitReader, FinancialStatementDataset,
    InstrumentAlignedSnapshotCache, PanelColumn, ReportTypePreference,
};
use crate::factor::{Factor, FactorUpdatePolicy};

const VERSION: &str = "0.1.0";
const RAW_ID: &str = "__competitive_advantage_delta_raw";
const FINANCIAL_QUARTERS: usize = 6;
const EPS: f64 = 1e-12;

const PROFIT_COLUMN: &str = "n_income";
const REVENUE_COLUMN: &str = "revenue";
const ASSETS_COLUMN: &str = "total_assets";

pub struct StockDailyCompetitiveAdvantageDelta;

#[derive(Default)]
struct CompetitiveAdvantageState {
    raw_cache: EventDrivenCrossSectionCache,
    snapshot_cache: InstrumentAlignedSnapshotCache<CompetitiveAdvantageSnapshot>,
}

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyCompetitiveAdvantageDelta)
}

impl Factor for StockDailyCompetitiveAdvantageDelta {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "competitive_advantage_delta".to_string(),
            aliases: vec![
                "Improved Competitive Advantage".to_string(),
                "Composite Rank Delta".to_string(),
            ],
            name: "competitive_advantage_delta".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: tags(),
            description: "XYZQ improved competitive advantage factor. It builds PIT single-quarter net margin and asset turnover, computes SW level-1 industry percentile ranks for current and year-ago quarters, takes the equal-weight composite-rank YoY delta, replays raw values between financial events, and finally neutralizes by Barra SIZE and SW sector.".to_string(),
            dependencies: vec![
                DataRequest::financial_quarters(
                    DatasetId::StockIncome,
                    &[PROFIT_COLUMN, REVENUE_COLUMN],
                    FINANCIAL_QUARTERS,
                ),
                DataRequest::financial_quarters(
                    DatasetId::StockBalanceSheet,
                    &[ASSETS_COLUMN],
                    FINANCIAL_QUARTERS,
                ),
                DataRequest::new(DatasetId::StockSwClassification, &["l1_code"]),
                DataRequest::new(DatasetId::StockBarraDaily, &["SIZE"]),
            ],
            intraday_raw_dependencies: Vec::new(),
            lookback: Lookback { trading_days: 0 },
        }
    }

    fn update_policy(&self) -> FactorUpdatePolicy {
        FactorUpdatePolicy::FinancialEventSnapshot
    }

    fn initial_compute_state(&self, _requested_ids: &[String]) -> Box<dyn Any + Send> {
        Box::new(CompetitiveAdvantageState::default())
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let mut snapshot_cache = InstrumentAlignedSnapshotCache::default();
        self.compute_with_snapshot_cache(data, &mut snapshot_cache)
    }

    fn compute_many_stateful(
        &self,
        requested_ids: &[String],
        context: &FactorContext,
        data: &DataPool,
        state: &mut (dyn Any + Send),
    ) -> Result<Vec<FactorSeries>> {
        if requested_ids
            .iter()
            .all(|id| id != "competitive_advantage_delta")
        {
            return Ok(Vec::new());
        }
        let state = state
            .downcast_mut::<CompetitiveAdvantageState>()
            .ok_or_else(|| err("competitive_advantage_delta received incompatible state"))?;
        let income = data.financial_reader(
            DatasetId::StockIncome,
            ReportTypePreference::income_single_quarter(),
        )?;
        let balance = data.financial_reader(
            DatasetId::StockBalanceSheet,
            ReportTypePreference::balance_sheet_consolidated(),
        )?;
        let sector_map = ClassificationMap::from_table(
            data.daily(DatasetId::StockSwClassification)?,
            ClassificationLevel::Sector,
        )?;
        let schedule = FinancialEventSchedule::from_pit_readers(&[income.clone(), balance.clone()]);
        let raw_specs = [raw_spec()];
        let panel = data.stock_universe_panel()?;
        let raw_series = compute_financial_event_snapshot_streaming_on_panel(
            requested_ids,
            context,
            data,
            panel,
            &mut state.raw_cache,
            &schedule,
            &raw_specs,
            |_, _, data| {
                self.compute_raw_with_prepared_inputs(
                    data,
                    &income,
                    &balance,
                    &sector_map,
                    &mut state.snapshot_cache,
                )
                .map(|series| vec![series])
            },
        )?;
        self.finalize_raw_series(data, raw_series)
            .map(|series| vec![series])
    }
}

impl StockDailyCompetitiveAdvantageDelta {
    fn compute_with_snapshot_cache(
        &self,
        data: &DataPool,
        snapshot_cache: &mut InstrumentAlignedSnapshotCache<CompetitiveAdvantageSnapshot>,
    ) -> Result<FactorSeries> {
        let income = data.financial_reader(
            DatasetId::StockIncome,
            ReportTypePreference::income_single_quarter(),
        )?;
        let balance = data.financial_reader(
            DatasetId::StockBalanceSheet,
            ReportTypePreference::balance_sheet_consolidated(),
        )?;
        let sector_map = ClassificationMap::from_table(
            data.daily(DatasetId::StockSwClassification)?,
            ClassificationLevel::Sector,
        )?;
        let raw_series = vec![self.compute_raw_with_prepared_inputs(
            data,
            &income,
            &balance,
            &sector_map,
            snapshot_cache,
        )?];
        self.finalize_raw_series(data, raw_series)
    }

    fn compute_raw_with_prepared_inputs(
        &self,
        data: &DataPool,
        income: &FinancialPitReader<'_>,
        balance: &FinancialPitReader<'_>,
        sector_map: &ClassificationMap,
        snapshot_cache: &mut InstrumentAlignedSnapshotCache<CompetitiveAdvantageSnapshot>,
    ) -> Result<FactorSeries> {
        let panel = data.stock_universe_panel()?;
        let raw =
            competitive_advantage_raw_column(panel, income, balance, sector_map, snapshot_cache)?;
        Ok(raw.to_factor_series(raw_spec()))
    }

    fn finalize_raw_series(
        &self,
        data: &DataPool,
        raw_series: Vec<FactorSeries>,
    ) -> Result<FactorSeries> {
        let panel = data.stock_universe_panel()?;
        let series = raw_series
            .into_iter()
            .find(|series| series.spec.id == RAW_ID)
            .ok_or_else(|| err("missing competitive_advantage_delta raw series"))?;
        let raw = factor_series_to_panel_column(panel, &series)?;
        let neutralized = neutralize_size_sector(&raw, panel, data)?;
        Ok(neutralized.to_factor_series(self.spec()))
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct CompetitiveAdvantageSnapshot {
    margin: f64,
    turnover: f64,
    margin_yoy: f64,
    turnover_yoy: f64,
}

fn competitive_advantage_raw_column(
    panel: &DailyPanel,
    income: &FinancialPitReader<'_>,
    balance: &FinancialPitReader<'_>,
    sector_map: &ClassificationMap,
    cache: &mut InstrumentAlignedSnapshotCache<CompetitiveAdvantageSnapshot>,
) -> Result<PanelColumn> {
    let instrument_count = panel.instruments().len();
    let mut values = vec![None; panel.shape_len()];

    for (date_idx, trade_date) in panel.dates().iter().copied().enumerate() {
        if !panel.is_target_date(trade_date) {
            continue;
        }
        let snapshots = cached_financial_stock_snapshots_for_date(
            panel,
            trade_date,
            cache,
            |_, ts_code, offset| {
                is_bj_stock(ts_code)
                    || !panel.is_present_offset(offset)
                    || sector_map.group_for(trade_date, ts_code).is_none()
            },
            |trade_date, ts_code, _| {
                competitive_advantage_marker(ts_code, trade_date, income, balance)
            },
            |trade_date, ts_code, _| {
                competitive_advantage_snapshot(ts_code, trade_date, income, balance)
            },
        );
        let date_offset = date_idx * instrument_count;
        let mut groups = BTreeMap::<String, Vec<(usize, CompetitiveAdvantageSnapshot)>>::new();
        for (instrument_idx, ts_code) in panel.instruments().iter().enumerate() {
            let offset = date_offset + instrument_idx;
            if is_bj_stock(ts_code) || !panel.is_present_offset(offset) {
                continue;
            }
            let Some(snapshot) = snapshots[instrument_idx] else {
                continue;
            };
            let Some(group) = sector_map.group_for(trade_date, ts_code) else {
                continue;
            };
            groups
                .entry(group.to_string())
                .or_default()
                .push((offset, snapshot));
        }
        for rows in groups.values() {
            let margin = pctrank(rows, |snapshot| snapshot.margin);
            let turnover = pctrank(rows, |snapshot| snapshot.turnover);
            let margin_yoy = pctrank(rows, |snapshot| snapshot.margin_yoy);
            let turnover_yoy = pctrank(rows, |snapshot| snapshot.turnover_yoy);
            for idx in 0..rows.len() {
                let current = 0.5 * margin[idx] + 0.5 * turnover[idx];
                let previous = 0.5 * margin_yoy[idx] + 0.5 * turnover_yoy[idx];
                values[rows[idx].0] = Some(current - previous);
            }
        }
    }

    panel.column_from_values(values)
}

fn competitive_advantage_marker(
    ts_code: &str,
    trade_date: i32,
    income: &FinancialPitReader<'_>,
    balance: &FinancialPitReader<'_>,
) -> Option<FinancialEventMarker> {
    let end_t = income.latest_quarter_end_date(ts_code, trade_date)?;
    let end_t1 = previous_quarter_end_date(end_t)?;
    let end_t4 = quarter_back(end_t, 4)?;
    let end_t5 = previous_quarter_end_date(end_t4)?;
    let mut builder = FinancialEventMarkerBuilder::new();
    for end_date in [end_t, end_t1, end_t4, end_t5] {
        builder.include_reader_record_for_end_date(
            FinancialStatementDataset::Income,
            income,
            ts_code,
            trade_date,
            end_date,
        );
        builder.include_reader_record_for_end_date(
            FinancialStatementDataset::BalanceSheet,
            balance,
            ts_code,
            trade_date,
            end_date,
        );
    }
    builder.build()
}

fn competitive_advantage_snapshot(
    ts_code: &str,
    trade_date: i32,
    income: &FinancialPitReader<'_>,
    balance: &FinancialPitReader<'_>,
) -> Option<CompetitiveAdvantageSnapshot> {
    let end_t = income.latest_quarter_end_date(ts_code, trade_date)?;
    let end_t1 = previous_quarter_end_date(end_t)?;
    let end_t4 = quarter_back(end_t, 4)?;
    let end_t5 = previous_quarter_end_date(end_t4)?;
    let current =
        profitability_turnover_for_period(ts_code, trade_date, end_t, end_t1, income, balance)?;
    let previous =
        profitability_turnover_for_period(ts_code, trade_date, end_t4, end_t5, income, balance)?;
    Some(CompetitiveAdvantageSnapshot {
        margin: current.0,
        turnover: current.1,
        margin_yoy: previous.0,
        turnover_yoy: previous.1,
    })
}

fn profitability_turnover_for_period(
    ts_code: &str,
    trade_date: i32,
    end_date: i32,
    prev_end_date: i32,
    income: &FinancialPitReader<'_>,
    balance: &FinancialPitReader<'_>,
) -> Option<(f64, f64)> {
    let income_record = income.record_for_end_date(ts_code, trade_date, end_date)?;
    let balance_record = balance.record_for_end_date(ts_code, trade_date, end_date)?;
    let prev_balance_record = balance.record_for_end_date(ts_code, trade_date, prev_end_date)?;
    let profit = clean(income_record.column(PROFIT_COLUMN))?;
    let revenue = clean(income_record.column(REVENUE_COLUMN)).filter(|value| value.abs() > EPS)?;
    let assets = clean(balance_record.column(ASSETS_COLUMN)).filter(|value| *value > 0.0)?;
    let prev_assets =
        clean(prev_balance_record.column(ASSETS_COLUMN)).filter(|value| *value > 0.0)?;
    let avg_assets = 0.5 * (assets + prev_assets);
    if avg_assets <= EPS {
        return None;
    }
    let margin = profit / revenue;
    let turnover = revenue / avg_assets;
    (margin.is_finite() && turnover.is_finite()).then_some((margin, turnover))
}

fn pctrank<F>(rows: &[(usize, CompetitiveAdvantageSnapshot)], value_fn: F) -> Vec<f64>
where
    F: Fn(CompetitiveAdvantageSnapshot) -> f64,
{
    let n = rows.len();
    let mut pairs = rows
        .iter()
        .enumerate()
        .map(|(idx, (_, snapshot))| (idx, value_fn(*snapshot)))
        .collect::<Vec<_>>();
    pairs.sort_by(|left, right| left.1.total_cmp(&right.1).then(left.0.cmp(&right.0)));
    let mut output = vec![0.0; n];
    let mut start = 0usize;
    while start < pairs.len() {
        let mut end = start + 1;
        while end < pairs.len() && pairs[end].1 == pairs[start].1 {
            end += 1;
        }
        let avg_one_based_rank = (start + 1 + end) as f64 / 2.0;
        let rank = avg_one_based_rank / n as f64;
        for idx in start..end {
            output[pairs[idx].0] = rank;
        }
        start = end;
    }
    output
}

fn quarter_back(mut end_date: i32, count: usize) -> Option<i32> {
    for _ in 0..count {
        end_date = previous_quarter_end_date(end_date)?;
    }
    Some(end_date)
}

fn clean(value: Option<f64>) -> Option<f64> {
    value.filter(|value| value.is_finite())
}

fn raw_spec() -> FactorSpec {
    FactorSpec {
        id: RAW_ID.to_string(),
        aliases: Vec::new(),
        name: RAW_ID.to_string(),
        asset_class: AssetClass::Stock,
        frequency: Frequency::Daily,
        version: VERSION.to_string(),
        tags: vec!["internal".to_string(), "financial_raw".to_string()],
        description: "Internal competitive_advantage_delta raw series.".to_string(),
        dependencies: Vec::new(),
        intraday_raw_dependencies: Vec::new(),
        lookback: Lookback { trading_days: 0 },
    }
}

fn tags() -> Vec<String> {
    [
        "XYZQ",
        "financial",
        "fundamental",
        "pit",
        "profitability",
        "turnover",
        "pctrank",
        "neutralize",
        "barra",
        "size",
        "sector",
        "daily",
    ]
    .iter()
    .map(|value| value.to_string())
    .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pctrank_uses_average_ties_and_one_to_n_scale() {
        let rows = vec![
            (
                0,
                CompetitiveAdvantageSnapshot {
                    margin: 2.0,
                    turnover: 0.0,
                    margin_yoy: 0.0,
                    turnover_yoy: 0.0,
                },
            ),
            (
                1,
                CompetitiveAdvantageSnapshot {
                    margin: 1.0,
                    turnover: 0.0,
                    margin_yoy: 0.0,
                    turnover_yoy: 0.0,
                },
            ),
            (
                2,
                CompetitiveAdvantageSnapshot {
                    margin: 2.0,
                    turnover: 0.0,
                    margin_yoy: 0.0,
                    turnover_yoy: 0.0,
                },
            ),
        ];
        let ranks = pctrank(&rows, |snapshot| snapshot.margin);
        assert_eq!(ranks, vec![2.5 / 3.0, 1.0 / 3.0, 2.5 / 3.0]);
    }

    #[test]
    fn spec_has_xyzq_tag() {
        let spec = StockDailyCompetitiveAdvantageDelta.spec();
        assert_eq!(spec.id, "competitive_advantage_delta");
        assert!(spec.tags.contains(&"XYZQ".to_string()));
    }
}

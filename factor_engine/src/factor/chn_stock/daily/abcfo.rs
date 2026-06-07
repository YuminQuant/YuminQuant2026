use std::any::Any;
use std::collections::BTreeMap;

use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::{DataPool, Table};
use crate::error::{err, Result};
use crate::factor::common::financial::previous_quarter_end_date;
use crate::factor::common::stock_daily_ops::is_bj_stock;
use crate::factor::common::{
    cached_financial_stock_snapshots_for_date, compute_financial_event_snapshot_streaming,
    factor_series_to_panel_column, ClassificationLevel, ClassificationMap, DailyPanel,
    EventDrivenCrossSectionCache, FinancialEventMarker, FinancialEventMarkerBuilder,
    FinancialEventSchedule, FinancialEventTable, FinancialStatementDataset,
    InstrumentAlignedSnapshotCache, PanelColumn, PitFinancialData, ReportTypePreference,
};
use crate::factor::{Factor, FactorUpdatePolicy};
use crate::operators::cs_zscore_by_group;

const VERSION: &str = "0.1.0";
const FINANCIAL_QUARTERS: usize = 8;
const RIDGE_LAMBDA: f64 = 10.0;
const MIN_INDUSTRY_RIDGE_OBS: usize = 3;
const REGRESSOR_COUNT: usize = 6;
const PARAM_COUNT: usize = REGRESSOR_COUNT + 1;
const ABCFO_RAW_ID: &str = "__abcfo_residual_raw";

const CFO_COLUMN: &str = "n_cashflow_act";
const EMPLOYEE_CASH_COLUMN: &str = "c_paid_to_for_empl";
const OTHER_OPERATE_CASH_COLUMN: &str = "c_fr_oth_operate_a";
const REVENUE_COLUMN: &str = "revenue";
const ASSET_COLUMN: &str = "total_assets";

pub struct StockDailyAbcfo;

#[derive(Default)]
struct AbcfoComputeState {
    raw_cache: EventDrivenCrossSectionCache,
    snapshot_cache: InstrumentAlignedSnapshotCache<AbcfoSlowSnapshot>,
}

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyAbcfo)
}

impl Factor for StockDailyAbcfo {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "abcfo".to_string(),
            aliases: vec!["ABCFO".to_string(), "Abnormal Cashflow".to_string()],
            name: "abcfo".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: tags(),
            description: "DBZQ abnormal cashflow factor. It anchors on the latest PIT single-quarter cashflow report, builds scaled cashflow/revenue/employee-cash variables plus listing age, takes SW level-1 industry ridge residuals with lambda=10 and an unpenalized intercept, then standardizes residuals within SW level-1 industries. The final event-driven snapshot is recomputed on financial disclosure events and replayed on non-event trading days.".to_string(),
            dependencies: vec![
                DataRequest::new(DatasetId::StockDailyPv, &["close"]),
                DataRequest::financial_quarters(
                    DatasetId::StockCashFlow,
                    &[CFO_COLUMN, EMPLOYEE_CASH_COLUMN, OTHER_OPERATE_CASH_COLUMN],
                    FINANCIAL_QUARTERS,
                ),
                DataRequest::financial_quarters(
                    DatasetId::StockIncome,
                    &[REVENUE_COLUMN],
                    FINANCIAL_QUARTERS,
                ),
                DataRequest::financial_quarters(
                    DatasetId::StockBalanceSheet,
                    &[ASSET_COLUMN],
                    FINANCIAL_QUARTERS,
                ),
                DataRequest::new(DatasetId::StockBasic, &["list_date"]),
                DataRequest::new(DatasetId::StockSwClassification, &["l1_code"]),
            ],
            intraday_raw_dependencies: Vec::new(),
            lookback: Lookback { trading_days: 0 },
        }
    }

    fn update_policy(&self) -> FactorUpdatePolicy {
        FactorUpdatePolicy::FinancialEventSnapshot
    }

    fn initial_compute_state(&self, _requested_ids: &[String]) -> Box<dyn Any + Send> {
        Box::new(AbcfoComputeState::default())
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
        if requested_ids.iter().all(|id| id != "abcfo") {
            return Ok(Vec::new());
        }
        let state = state
            .downcast_mut::<AbcfoComputeState>()
            .ok_or_else(|| err("abcfo received incompatible event cache state"))?;
        let schedule = FinancialEventSchedule::from_tables(&[
            FinancialEventTable::statement_with_preference(
                data.daily(DatasetId::StockCashFlow)?,
                ReportTypePreference::income_single_quarter(),
            ),
            FinancialEventTable::statement_with_preference(
                data.daily(DatasetId::StockIncome)?,
                ReportTypePreference::income_single_quarter(),
            ),
            FinancialEventTable::statement_with_preference(
                data.daily(DatasetId::StockBalanceSheet)?,
                ReportTypePreference::balance_sheet_consolidated(),
            ),
        ])?;
        let raw_specs = [raw_spec()];
        let raw_cache = &mut state.raw_cache;
        let snapshot_cache = &mut state.snapshot_cache;
        let cashflow = PitFinancialData::from_table(
            data.daily(DatasetId::StockCashFlow)?,
            &[CFO_COLUMN, EMPLOYEE_CASH_COLUMN, OTHER_OPERATE_CASH_COLUMN],
            ReportTypePreference::income_single_quarter(),
        )?;
        let income = PitFinancialData::from_table(
            data.daily(DatasetId::StockIncome)?,
            &[REVENUE_COLUMN],
            ReportTypePreference::income_single_quarter(),
        )?;
        let balance = PitFinancialData::from_table(
            data.daily(DatasetId::StockBalanceSheet)?,
            &[ASSET_COLUMN],
            ReportTypePreference::balance_sheet_consolidated(),
        )?;
        let list_dates = stock_basic_list_dates(data.daily(DatasetId::StockBasic)?)?;
        let sector_map = ClassificationMap::from_table(
            data.daily(DatasetId::StockSwClassification)?,
            ClassificationLevel::Sector,
        )?;
        let raw_series = compute_financial_event_snapshot_streaming(
            requested_ids,
            context,
            data,
            raw_cache,
            &schedule,
            &raw_specs,
            |_, _, data| {
                self.compute_raw_with_prepared_financials(
                    data,
                    &cashflow,
                    &income,
                    &balance,
                    &list_dates,
                    &sector_map,
                    snapshot_cache,
                )
                .map(|series| vec![series])
            },
        )?;
        self.finalize_raw_series(data, raw_series)
            .map(|series| vec![series])
    }
}

impl StockDailyAbcfo {
    fn compute_with_snapshot_cache(
        &self,
        data: &DataPool,
        snapshot_cache: &mut InstrumentAlignedSnapshotCache<AbcfoSlowSnapshot>,
    ) -> Result<FactorSeries> {
        let cashflow = PitFinancialData::from_table(
            data.daily(DatasetId::StockCashFlow)?,
            &[CFO_COLUMN, EMPLOYEE_CASH_COLUMN, OTHER_OPERATE_CASH_COLUMN],
            ReportTypePreference::income_single_quarter(),
        )?;
        let income = PitFinancialData::from_table(
            data.daily(DatasetId::StockIncome)?,
            &[REVENUE_COLUMN],
            ReportTypePreference::income_single_quarter(),
        )?;
        let balance = PitFinancialData::from_table(
            data.daily(DatasetId::StockBalanceSheet)?,
            &[ASSET_COLUMN],
            ReportTypePreference::balance_sheet_consolidated(),
        )?;
        let list_dates = stock_basic_list_dates(data.daily(DatasetId::StockBasic)?)?;
        let sector_map = ClassificationMap::from_table(
            data.daily(DatasetId::StockSwClassification)?,
            ClassificationLevel::Sector,
        )?;

        let raw_series = vec![self.compute_raw_with_prepared_financials(
            data,
            &cashflow,
            &income,
            &balance,
            &list_dates,
            &sector_map,
            snapshot_cache,
        )?];
        self.finalize_raw_series(data, raw_series)
    }

    fn compute_raw_with_prepared_financials(
        &self,
        data: &DataPool,
        cashflow: &PitFinancialData,
        income: &PitFinancialData,
        balance: &PitFinancialData,
        list_dates: &BTreeMap<String, i32>,
        sector_map: &ClassificationMap,
        snapshot_cache: &mut InstrumentAlignedSnapshotCache<AbcfoSlowSnapshot>,
    ) -> Result<FactorSeries> {
        let panel = data.daily_panel(DatasetId::StockDailyPv)?;
        let raw = abcfo_ridge_residual_column(
            &panel,
            &cashflow,
            &income,
            &balance,
            &list_dates,
            sector_map,
            snapshot_cache,
        )?;
        Ok(raw.to_factor_series(raw_spec()))
    }

    fn finalize_raw_series(
        &self,
        data: &DataPool,
        raw_series: Vec<FactorSeries>,
    ) -> Result<FactorSeries> {
        let panel = data.daily_panel(DatasetId::StockDailyPv)?;
        let series = raw_series
            .into_iter()
            .find(|series| series.spec.id == ABCFO_RAW_ID)
            .ok_or_else(|| err("missing abcfo raw series"))?;
        let raw = factor_series_to_panel_column(&panel, &series)?;
        let standardized = industry_zscore(&raw, data)?;
        Ok(standardized.to_factor_series(self.spec()))
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct AbcfoSlowSnapshot {
    cfo: f64,
    assets: f64,
    revenue_t: f64,
    revenue_t1: f64,
    revenue_t2: f64,
    employee_cash: f64,
    other_operate_cash: f64,
    list_date: i32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct AbcfoRow {
    y: f64,
    x: [f64; REGRESSOR_COUNT],
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct RidgeObservation {
    offset: usize,
    row: AbcfoRow,
}

fn tags() -> Vec<String> {
    [
        "DBZQ",
        "financial",
        "fundamental",
        "pit",
        "abnormal_cashflow",
        "ridge",
        "residual",
        "industry_standardize",
        "daily",
    ]
    .iter()
    .map(|value| value.to_string())
    .collect()
}

fn raw_spec() -> FactorSpec {
    FactorSpec {
        id: ABCFO_RAW_ID.to_string(),
        aliases: Vec::new(),
        name: ABCFO_RAW_ID.to_string(),
        asset_class: AssetClass::Stock,
        frequency: Frequency::Daily,
        version: VERSION.to_string(),
        tags: vec!["internal".to_string(), "financial_raw".to_string()],
        description: "Internal abcfo ridge residual raw series.".to_string(),
        dependencies: Vec::new(),
        intraday_raw_dependencies: Vec::new(),
        lookback: Lookback { trading_days: 0 },
    }
}

fn stock_basic_list_dates(table: &Table) -> Result<BTreeMap<String, i32>> {
    let ts_codes = table.required_utf8("ts_code")?;
    let list_dates = table.required_i32_date_cast("list_date")?;
    let mut output = BTreeMap::new();
    for idx in 0..table.len {
        let (Some(ts_code), Some(list_date)) = (ts_codes[idx].clone(), list_dates[idx]) else {
            continue;
        };
        output.insert(ts_code, list_date);
    }
    Ok(output)
}

fn abcfo_ridge_residual_column(
    panel: &DailyPanel,
    cashflow: &PitFinancialData,
    income: &PitFinancialData,
    balance: &PitFinancialData,
    list_dates: &BTreeMap<String, i32>,
    sector_map: &ClassificationMap,
    cache: &mut InstrumentAlignedSnapshotCache<AbcfoSlowSnapshot>,
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
                    || !list_dates.contains_key(ts_code)
            },
            |trade_date, ts_code, _| abcfo_marker(ts_code, trade_date, cashflow, income, balance),
            |trade_date, ts_code, _| {
                let list_date = list_dates.get(ts_code).copied()?;
                abcfo_slow_snapshot_for_stock(
                    ts_code, trade_date, list_date, cashflow, income, balance,
                )
            },
        );
        let date_offset = date_idx * instrument_count;
        let mut observations_by_sector = BTreeMap::<String, Vec<RidgeObservation>>::new();
        for (instrument_idx, ts_code) in panel.instruments().iter().enumerate() {
            let offset = date_offset + instrument_idx;
            if is_bj_stock(ts_code) || !panel.is_present_offset(offset) {
                continue;
            }
            let Some(snapshot) = snapshots[instrument_idx] else {
                continue;
            };
            let Some(row) = abcfo_row_from_snapshot(&snapshot, trade_date) else {
                continue;
            };
            push_grouped_observation(
                &mut observations_by_sector,
                sector_map.group_for(trade_date, ts_code),
                RidgeObservation { offset, row },
            );
        }
        for (offset, residual) in grouped_ridge_residuals(&observations_by_sector) {
            values[offset] = Some(residual);
        }
    }

    panel.column_from_values(values)
}

fn abcfo_marker(
    ts_code: &str,
    trade_date: i32,
    cashflow: &PitFinancialData,
    income: &PitFinancialData,
    balance: &PitFinancialData,
) -> Option<FinancialEventMarker> {
    let end_t = cashflow.latest_quarter_end_date(ts_code, trade_date)?;
    let end_t1 = previous_quarter_end_date(end_t)?;
    let end_t2 = previous_quarter_end_date(end_t1)?;
    let mut builder = FinancialEventMarkerBuilder::new();
    builder.include_record_for_end_date(
        FinancialStatementDataset::CashFlow,
        cashflow,
        ts_code,
        trade_date,
        end_t,
    );
    builder.include_record_for_end_date(
        FinancialStatementDataset::Income,
        income,
        ts_code,
        trade_date,
        end_t,
    );
    builder.include_record_for_end_date(
        FinancialStatementDataset::Income,
        income,
        ts_code,
        trade_date,
        end_t1,
    );
    builder.include_record_for_end_date(
        FinancialStatementDataset::Income,
        income,
        ts_code,
        trade_date,
        end_t2,
    );
    builder.include_record_for_end_date(
        FinancialStatementDataset::BalanceSheet,
        balance,
        ts_code,
        trade_date,
        end_t,
    );
    builder.build()
}

fn abcfo_slow_snapshot_for_stock(
    ts_code: &str,
    trade_date: i32,
    list_date: i32,
    cashflow: &PitFinancialData,
    income: &PitFinancialData,
    balance: &PitFinancialData,
) -> Option<AbcfoSlowSnapshot> {
    let end_t = cashflow.latest_quarter_end_date(ts_code, trade_date)?;
    let end_t1 = previous_quarter_end_date(end_t)?;
    let end_t2 = previous_quarter_end_date(end_t1)?;
    let cash_t = cashflow.record_for_end_date(ts_code, trade_date, end_t)?;
    let income_t = income.record_for_end_date(ts_code, trade_date, end_t)?;
    let income_t1 = income.record_for_end_date(ts_code, trade_date, end_t1)?;
    let income_t2 = income.record_for_end_date(ts_code, trade_date, end_t2)?;
    let balance_t = balance.record_for_end_date(ts_code, trade_date, end_t)?;

    let cfo = clean(cash_t.column(CFO_COLUMN))?;
    let employee_cash = clean(cash_t.column(EMPLOYEE_CASH_COLUMN))?;
    let other_operate_cash = clean(cash_t.column(OTHER_OPERATE_CASH_COLUMN)).unwrap_or(0.0);
    let revenue_t = clean(income_t.column(REVENUE_COLUMN))?;
    let revenue_t1 = clean(income_t1.column(REVENUE_COLUMN))?;
    let revenue_t2 = clean(income_t2.column(REVENUE_COLUMN))?;
    let assets = clean(balance_t.column(ASSET_COLUMN)).filter(|value| *value > 0.0)?;
    Some(AbcfoSlowSnapshot {
        cfo,
        assets,
        revenue_t,
        revenue_t1,
        revenue_t2,
        employee_cash,
        other_operate_cash,
        list_date,
    })
}

fn abcfo_row_from_snapshot(snapshot: &AbcfoSlowSnapshot, trade_date: i32) -> Option<AbcfoRow> {
    let age = listing_age_years(trade_date, snapshot.list_date)?;
    abcfo_row_from_values(
        snapshot.cfo,
        snapshot.assets,
        snapshot.revenue_t,
        snapshot.revenue_t1,
        snapshot.revenue_t2,
        snapshot.employee_cash,
        snapshot.other_operate_cash,
        age,
    )
}

fn abcfo_row_from_values(
    cfo: f64,
    assets: f64,
    revenue_t: f64,
    revenue_t1: f64,
    revenue_t2: f64,
    employee_cash: f64,
    other_operate_cash: f64,
    age: f64,
) -> Option<AbcfoRow> {
    if !assets.is_finite() || assets <= 0.0 {
        return None;
    }
    let dd = if revenue_t < revenue_t1 { 1.0 } else { 0.0 };
    let inv_assets = 1.0 / assets;
    let row = AbcfoRow {
        y: cfo * inv_assets,
        x: [
            revenue_t * inv_assets,
            (revenue_t1 - revenue_t2) * inv_assets,
            employee_cash * inv_assets,
            other_operate_cash * inv_assets,
            (revenue_t - revenue_t1) * inv_assets * dd,
            age,
        ],
    };
    row_is_finite(&row).then_some(row)
}

fn ridge_residuals(observations: &[RidgeObservation]) -> Vec<(usize, f64)> {
    if observations.len() < MIN_INDUSTRY_RIDGE_OBS {
        return Vec::new();
    }
    let Some(beta) = ridge_beta(observations) else {
        return Vec::new();
    };
    observations
        .iter()
        .filter_map(|observation| {
            let fitted = beta[0]
                + observation
                    .row
                    .x
                    .iter()
                    .enumerate()
                    .map(|(idx, x)| beta[idx + 1] * x)
                    .sum::<f64>();
            let residual = observation.row.y - fitted;
            residual
                .is_finite()
                .then_some((observation.offset, residual))
        })
        .collect()
}

fn push_grouped_observation(
    observations_by_sector: &mut BTreeMap<String, Vec<RidgeObservation>>,
    group: Option<&str>,
    observation: RidgeObservation,
) {
    let Some(group) = group.filter(|group| !group.is_empty()) else {
        return;
    };
    observations_by_sector
        .entry(group.to_string())
        .or_default()
        .push(observation);
}

fn grouped_ridge_residuals(
    observations_by_sector: &BTreeMap<String, Vec<RidgeObservation>>,
) -> Vec<(usize, f64)> {
    let mut output = Vec::new();
    for observations in observations_by_sector.values() {
        output.extend(ridge_residuals(observations));
    }
    output
}

fn ridge_beta(observations: &[RidgeObservation]) -> Option<[f64; PARAM_COUNT]> {
    let mut xtx = [[0.0; PARAM_COUNT]; PARAM_COUNT];
    let mut xty = [0.0; PARAM_COUNT];

    for observation in observations {
        let mut row = [0.0; PARAM_COUNT];
        row[0] = 1.0;
        row[1..].copy_from_slice(&observation.row.x);
        for i in 0..PARAM_COUNT {
            xty[i] += row[i] * observation.row.y;
            for j in 0..PARAM_COUNT {
                xtx[i][j] += row[i] * row[j];
            }
        }
    }
    for (idx, row) in xtx.iter_mut().enumerate().skip(1) {
        row[idx] += RIDGE_LAMBDA;
    }
    solve_linear_system(xtx, xty)
}

fn solve_linear_system(
    mut a: [[f64; PARAM_COUNT]; PARAM_COUNT],
    mut b: [f64; PARAM_COUNT],
) -> Option<[f64; PARAM_COUNT]> {
    for pivot_idx in 0..PARAM_COUNT {
        let mut pivot_row = pivot_idx;
        let mut pivot_abs = a[pivot_idx][pivot_idx].abs();
        for (row_idx, row) in a.iter().enumerate().skip(pivot_idx + 1) {
            let candidate = row[pivot_idx].abs();
            if candidate > pivot_abs {
                pivot_abs = candidate;
                pivot_row = row_idx;
            }
        }
        if pivot_abs <= 1e-12 {
            return None;
        }
        if pivot_row != pivot_idx {
            a.swap(pivot_idx, pivot_row);
            b.swap(pivot_idx, pivot_row);
        }
        let pivot = a[pivot_idx][pivot_idx];
        for col_idx in pivot_idx..PARAM_COUNT {
            a[pivot_idx][col_idx] /= pivot;
        }
        b[pivot_idx] /= pivot;

        for row_idx in 0..PARAM_COUNT {
            if row_idx == pivot_idx {
                continue;
            }
            let factor = a[row_idx][pivot_idx];
            if factor.abs() <= f64::EPSILON {
                continue;
            }
            for col_idx in pivot_idx..PARAM_COUNT {
                a[row_idx][col_idx] -= factor * a[pivot_idx][col_idx];
            }
            b[row_idx] -= factor * b[pivot_idx];
        }
    }
    b.iter().all(|value| value.is_finite()).then_some(b)
}

fn industry_zscore(raw: &PanelColumn, data: &DataPool) -> Result<PanelColumn> {
    let sector_map = ClassificationMap::from_table(
        data.daily(DatasetId::StockSwClassification)?,
        ClassificationLevel::Sector,
    )?;
    raw.cs_by_group(
        |trade_date, ts_codes| sector_map.groups_for(trade_date, ts_codes),
        cs_zscore_by_group,
    )
}

fn row_is_finite(row: &AbcfoRow) -> bool {
    row.y.is_finite() && row.x.iter().all(|value| value.is_finite())
}

fn clean(value: Option<f64>) -> Option<f64> {
    value.filter(|value| value.is_finite())
}

fn listing_age_years(trade_date: i32, list_date: i32) -> Option<f64> {
    if list_date > trade_date {
        return None;
    }
    let trade_days = days_from_ymd_date(trade_date)?;
    let list_days = days_from_ymd_date(list_date)?;
    Some((trade_days - list_days) as f64 / 365.0)
}

fn days_from_ymd_date(date: i32) -> Option<i64> {
    let year = date / 10_000;
    let month = (date / 100) % 100;
    let day = date % 100;
    if !(1..=12).contains(&month) || !(1..=31).contains(&day) {
        return None;
    }
    Some(days_from_civil(year, month as u32, day as u32))
}

fn days_from_civil(year: i32, month: u32, day: u32) -> i64 {
    let mut year = i64::from(year);
    let month = i64::from(month);
    let day = i64::from(day);
    year -= (month <= 2) as i64;
    let era = if year >= 0 { year } else { year - 399 } / 400;
    let yoe = year - era * 400;
    let month_prime = month + if month > 2 { -3 } else { 9 };
    let doy = (153 * month_prime + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(left: f64, right: f64) {
        assert!(
            (left - right).abs() < 1e-9,
            "left={left}, right={right}, diff={}",
            (left - right).abs()
        );
    }

    #[test]
    fn abcfo_row_builds_scaled_variables_and_down_dummy() {
        let row =
            abcfo_row_from_values(20.0, 100.0, 90.0, 100.0, 80.0, 5.0, 3.0, 4.0).expect("row");
        assert_close(row.y, 0.2);
        assert_close(row.x[0], 0.9);
        assert_close(row.x[1], 0.2);
        assert_close(row.x[2], 0.05);
        assert_close(row.x[3], 0.03);
        assert_close(row.x[4], -0.1);
        assert_close(row.x[5], 4.0);

        let row =
            abcfo_row_from_values(20.0, 100.0, 110.0, 100.0, 80.0, 5.0, 3.0, 4.0).expect("row");
        assert_close(row.x[4], 0.0);
    }

    #[test]
    fn abcfo_row_rejects_nonpositive_assets() {
        assert!(abcfo_row_from_values(20.0, 0.0, 90.0, 100.0, 80.0, 5.0, 3.0, 4.0).is_none());
        assert!(abcfo_row_from_values(20.0, -1.0, 90.0, 100.0, 80.0, 5.0, 3.0, 4.0).is_none());
    }

    #[test]
    fn ridge_beta_does_not_penalize_intercept() {
        let observations = (0..12)
            .map(|idx| {
                let x = idx as f64;
                RidgeObservation {
                    offset: idx,
                    row: AbcfoRow {
                        y: 2.0 + 0.5 * x,
                        x: [x, 0.0, 0.0, 0.0, 0.0, 0.0],
                    },
                }
            })
            .collect::<Vec<_>>();
        let beta = ridge_beta(&observations).expect("beta");
        assert!(beta[0] > 2.0);
        assert!(beta[1] < 0.5);
        let residuals = ridge_residuals(&observations);
        assert_eq!(residuals.len(), observations.len());
    }

    #[test]
    fn ridge_residuals_require_minimum_observations() {
        let observations = (0..2)
            .map(|idx| RidgeObservation {
                offset: idx,
                row: AbcfoRow {
                    y: idx as f64,
                    x: [idx as f64, 0.0, 0.0, 0.0, 0.0, 1.0],
                },
            })
            .collect::<Vec<_>>();
        assert!(ridge_residuals(&observations).is_empty());
    }

    #[test]
    fn abcfo_grouped_ridge_skips_unclassified_observations() {
        let mut grouped = BTreeMap::<String, Vec<RidgeObservation>>::new();
        push_grouped_observation(
            &mut grouped,
            None,
            RidgeObservation {
                offset: 0,
                row: AbcfoRow {
                    y: 1.0,
                    x: [0.0; REGRESSOR_COUNT],
                },
            },
        );
        assert!(grouped.is_empty());
        assert!(grouped_ridge_residuals(&grouped).is_empty());
    }

    #[test]
    fn abcfo_grouped_ridge_requires_more_than_two_industry_observations() {
        let mut grouped = BTreeMap::<String, Vec<RidgeObservation>>::new();
        for idx in 0..2 {
            push_grouped_observation(
                &mut grouped,
                Some("801010"),
                RidgeObservation {
                    offset: idx,
                    row: AbcfoRow {
                        y: idx as f64,
                        x: [0.0; REGRESSOR_COUNT],
                    },
                },
            );
        }
        assert!(grouped_ridge_residuals(&grouped).is_empty());
    }

    #[test]
    fn abcfo_grouped_ridge_uses_industry_specific_regressions() {
        let mut grouped = BTreeMap::<String, Vec<RidgeObservation>>::new();
        for (offset, y) in [(0, 10.0), (1, 11.0), (2, 12.0)] {
            push_grouped_observation(
                &mut grouped,
                Some("801010"),
                RidgeObservation {
                    offset,
                    row: AbcfoRow {
                        y,
                        x: [0.0; REGRESSOR_COUNT],
                    },
                },
            );
        }
        for (offset, y) in [(3, 100.0), (4, 101.0), (5, 102.0)] {
            push_grouped_observation(
                &mut grouped,
                Some("801020"),
                RidgeObservation {
                    offset,
                    row: AbcfoRow {
                        y,
                        x: [0.0; REGRESSOR_COUNT],
                    },
                },
            );
        }

        let residuals = grouped_ridge_residuals(&grouped)
            .into_iter()
            .collect::<BTreeMap<_, _>>();
        assert_close(*residuals.get(&0).expect("offset 0"), -1.0);
        assert_close(*residuals.get(&1).expect("offset 1"), 0.0);
        assert_close(*residuals.get(&2).expect("offset 2"), 1.0);
        assert_close(*residuals.get(&3).expect("offset 3"), -1.0);
        assert_close(*residuals.get(&4).expect("offset 4"), 0.0);
        assert_close(*residuals.get(&5).expect("offset 5"), 1.0);
    }

    #[test]
    fn listing_age_uses_natural_days_over_365() {
        assert_close(
            listing_age_years(20250101, 20240101).expect("age"),
            366.0 / 365.0,
        );
        assert!(listing_age_years(20240101, 20250101).is_none());
    }

    #[test]
    fn abcfo_metadata_has_dbzq_tags() {
        let spec = StockDailyAbcfo.spec();
        assert_eq!(spec.id, "abcfo");
        assert!(spec.tags.iter().any(|tag| tag == "DBZQ"));
        assert!(spec.tags.iter().any(|tag| tag == "financial"));
        assert!(spec.tags.iter().any(|tag| tag == "industry_standardize"));
        assert_eq!(
            StockDailyAbcfo.update_policy(),
            FactorUpdatePolicy::FinancialEventSnapshot
        );
    }
}

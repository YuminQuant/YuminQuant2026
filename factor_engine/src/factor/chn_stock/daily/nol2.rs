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
    cached_financial_stock_snapshots_for_date, compute_financial_event_snapshot_streaming,
    factor_series_to_panel_column, ClassificationLevel, ClassificationMap, DailyPanel,
    EventDrivenCrossSectionCache, FinancialEventMarker, FinancialEventMarkerBuilder,
    FinancialEventSchedule, FinancialPitReader, FinancialStatementDataset,
    InstrumentAlignedSnapshotCache, PanelColumn, PitFinancialRecordView, ReportTypePreference,
};
use crate::factor::{Factor, FactorUpdatePolicy};

const VERSION: &str = "0.1.0";
const NOL2_RAW_ID: &str = "__nol2_residual_raw";
const FINANCIAL_QUARTERS: usize = 2;
const PARAM_COUNT: usize = 3;
const MIN_INDUSTRY_OLS_OBS: usize = 3;

const ACCOUNTS_PAY_COLUMN: &str = "accounts_pay";
const ADV_RECEIPTS_COLUMN: &str = "adv_receipts";
const CONTRACT_LIAB_COLUMN: &str = "contract_liab";
const PAYROLL_PAYABLE_COLUMN: &str = "payroll_payable";
const PREPAYMENT_COLUMN: &str = "prepayment";
const CONTRACT_ASSETS_COLUMN: &str = "contract_assets";
const ACCOUNTS_RECEIV_BILL_COLUMN: &str = "accounts_receiv_bill";
const ASSET_COLUMN: &str = "total_assets";
const REVENUE_COLUMN: &str = "revenue";

pub struct StockDailyNol2;

#[derive(Default)]
struct Nol2ComputeState {
    raw_cache: EventDrivenCrossSectionCache,
    snapshot_cache: InstrumentAlignedSnapshotCache<Nol2Snapshot>,
}

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyNol2)
}

impl Factor for StockDailyNol2 {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "nol2".to_string(),
            aliases: vec!["NOL2".to_string(), "Net Operating Liability 2".to_string()],
            name: "nol2".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: tags(),
            description: "DBZQ net operating liability factor 2. It uses PIT consolidated balance-sheet operating assets/liabilities and single-quarter revenue, runs SW level-1 industry OLS residuals for NOL on current and previous revenue scaled by assets, replays raw residuals between financial events, and finally neutralizes by Barra SIZE and SW sector.".to_string(),
            dependencies: vec![
                DataRequest::new(DatasetId::StockDailyPv, &["close"]),
                DataRequest::financial_quarters(
                    DatasetId::StockBalanceSheet,
                    &[
                        ACCOUNTS_PAY_COLUMN,
                        ADV_RECEIPTS_COLUMN,
                        CONTRACT_LIAB_COLUMN,
                        PAYROLL_PAYABLE_COLUMN,
                        PREPAYMENT_COLUMN,
                        CONTRACT_ASSETS_COLUMN,
                        ACCOUNTS_RECEIV_BILL_COLUMN,
                        ASSET_COLUMN,
                    ],
                    FINANCIAL_QUARTERS,
                ),
                DataRequest::financial_quarters(
                    DatasetId::StockIncome,
                    &[REVENUE_COLUMN],
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
        Box::new(Nol2ComputeState::default())
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
        if requested_ids.iter().all(|id| id != "nol2") {
            return Ok(Vec::new());
        }
        let state = state
            .downcast_mut::<Nol2ComputeState>()
            .ok_or_else(|| err("nol2 received incompatible event cache state"))?;
        let balance = data.financial_reader(
            DatasetId::StockBalanceSheet,
            ReportTypePreference::balance_sheet_consolidated(),
        )?;
        let income = data.financial_reader(
            DatasetId::StockIncome,
            ReportTypePreference::income_single_quarter(),
        )?;
        let schedule = FinancialEventSchedule::from_pit_readers(&[balance.clone(), income.clone()]);
        let sector_map = ClassificationMap::from_table(
            data.daily(DatasetId::StockSwClassification)?,
            ClassificationLevel::Sector,
        )?;
        let raw_specs = [raw_spec()];
        let raw_series = compute_financial_event_snapshot_streaming(
            requested_ids,
            context,
            data,
            &mut state.raw_cache,
            &schedule,
            &raw_specs,
            |_, _, data| {
                self.compute_raw_with_prepared_financials(
                    data,
                    &balance,
                    &income,
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

impl StockDailyNol2 {
    fn compute_with_snapshot_cache(
        &self,
        data: &DataPool,
        snapshot_cache: &mut InstrumentAlignedSnapshotCache<Nol2Snapshot>,
    ) -> Result<FactorSeries> {
        let balance = data.financial_reader(
            DatasetId::StockBalanceSheet,
            ReportTypePreference::balance_sheet_consolidated(),
        )?;
        let income = data.financial_reader(
            DatasetId::StockIncome,
            ReportTypePreference::income_single_quarter(),
        )?;
        let sector_map = ClassificationMap::from_table(
            data.daily(DatasetId::StockSwClassification)?,
            ClassificationLevel::Sector,
        )?;
        let raw_series = vec![self.compute_raw_with_prepared_financials(
            data,
            &balance,
            &income,
            &sector_map,
            snapshot_cache,
        )?];
        self.finalize_raw_series(data, raw_series)
    }

    fn compute_raw_with_prepared_financials(
        &self,
        data: &DataPool,
        balance: &FinancialPitReader<'_>,
        income: &FinancialPitReader<'_>,
        sector_map: &ClassificationMap,
        snapshot_cache: &mut InstrumentAlignedSnapshotCache<Nol2Snapshot>,
    ) -> Result<FactorSeries> {
        let panel = data.daily_panel(DatasetId::StockDailyPv)?;
        let raw = nol2_residual_column(&panel, balance, income, sector_map, snapshot_cache)?;
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
            .find(|series| series.spec.id == NOL2_RAW_ID)
            .ok_or_else(|| err("missing nol2 raw series"))?;
        let raw = factor_series_to_panel_column(&panel, &series)?;
        let neutralized = neutralize_size_sector(&raw, &panel, data)?;
        Ok(neutralized.to_factor_series(self.spec()))
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Nol2Snapshot {
    nol: f64,
    revenue_scaled: f64,
    prev_revenue_scaled: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct NolObservation {
    offset: usize,
    y: f64,
    x: [f64; 2],
}

fn nol2_residual_column(
    panel: &DailyPanel,
    balance: &FinancialPitReader<'_>,
    income: &FinancialPitReader<'_>,
    sector_map: &ClassificationMap,
    cache: &mut InstrumentAlignedSnapshotCache<Nol2Snapshot>,
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
            |trade_date, ts_code, _| nol2_marker(ts_code, trade_date, balance, income),
            |trade_date, ts_code, _| nol2_snapshot_for_stock(ts_code, trade_date, balance, income),
        );
        let date_offset = date_idx * instrument_count;
        let mut observations_by_sector = BTreeMap::<String, Vec<NolObservation>>::new();
        for (instrument_idx, ts_code) in panel.instruments().iter().enumerate() {
            let offset = date_offset + instrument_idx;
            if is_bj_stock(ts_code) || !panel.is_present_offset(offset) {
                continue;
            }
            let Some(snapshot) = snapshots[instrument_idx] else {
                continue;
            };
            push_grouped_observation(
                &mut observations_by_sector,
                sector_map.group_for(trade_date, ts_code),
                NolObservation {
                    offset,
                    y: snapshot.nol,
                    x: [snapshot.revenue_scaled, snapshot.prev_revenue_scaled],
                },
            );
        }
        for (offset, residual) in grouped_ols_residuals(&observations_by_sector) {
            values[offset] = Some(residual);
        }
    }

    panel.column_from_values(values)
}

fn nol2_marker(
    ts_code: &str,
    trade_date: i32,
    balance: &FinancialPitReader<'_>,
    income: &FinancialPitReader<'_>,
) -> Option<FinancialEventMarker> {
    let end_t = balance.latest_quarter_end_date(ts_code, trade_date)?;
    let end_t1 = previous_quarter_end_date(end_t)?;
    let mut builder = FinancialEventMarkerBuilder::new();
    builder.include_reader_record_for_end_date(
        FinancialStatementDataset::BalanceSheet,
        balance,
        ts_code,
        trade_date,
        end_t,
    );
    builder.include_reader_record_for_end_date(
        FinancialStatementDataset::Income,
        income,
        ts_code,
        trade_date,
        end_t,
    );
    builder.include_reader_record_for_end_date(
        FinancialStatementDataset::Income,
        income,
        ts_code,
        trade_date,
        end_t1,
    );
    builder.build()
}

fn nol2_snapshot_for_stock(
    ts_code: &str,
    trade_date: i32,
    balance: &FinancialPitReader<'_>,
    income: &FinancialPitReader<'_>,
) -> Option<Nol2Snapshot> {
    let end_t = balance.latest_quarter_end_date(ts_code, trade_date)?;
    let end_t1 = previous_quarter_end_date(end_t)?;
    let balance_t = balance.record_for_end_date(ts_code, trade_date, end_t)?;
    let income_t = income.record_for_end_date(ts_code, trade_date, end_t)?;
    let income_t1 = income.record_for_end_date(ts_code, trade_date, end_t1)?;
    let assets = clean(balance_t.column(ASSET_COLUMN)).filter(|value| *value > 0.0)?;
    let nol = nol_from_record(balance_t, assets)?;
    let revenue_t = clean(income_t.column(REVENUE_COLUMN))?;
    let revenue_t1 = clean(income_t1.column(REVENUE_COLUMN))?;
    let revenue_scaled = revenue_t / assets;
    let prev_revenue_scaled = revenue_t1 / assets;
    (revenue_scaled.is_finite() && prev_revenue_scaled.is_finite()).then_some(Nol2Snapshot {
        nol,
        revenue_scaled,
        prev_revenue_scaled,
    })
}

fn nol_from_record(balance_t: PitFinancialRecordView<'_>, assets: f64) -> Option<f64> {
    nol_from_values(
        clean_or_zero(balance_t.column(ACCOUNTS_PAY_COLUMN)),
        clean_or_zero(balance_t.column(ADV_RECEIPTS_COLUMN)),
        clean_or_zero(balance_t.column(CONTRACT_LIAB_COLUMN)),
        clean_or_zero(balance_t.column(PAYROLL_PAYABLE_COLUMN)),
        clean_or_zero(balance_t.column(PREPAYMENT_COLUMN)),
        clean_or_zero(balance_t.column(CONTRACT_ASSETS_COLUMN)),
        clean_or_zero(balance_t.column(ACCOUNTS_RECEIV_BILL_COLUMN)),
        assets,
    )
}

fn nol_from_values(
    accounts_pay: f64,
    adv_receipts: f64,
    contract_liab: f64,
    payroll_payable: f64,
    prepayment: f64,
    contract_assets: f64,
    accounts_receiv_bill: f64,
    assets: f64,
) -> Option<f64> {
    if !assets.is_finite() || assets <= 0.0 {
        return None;
    }
    let operating_liability = accounts_pay + adv_receipts + contract_liab + payroll_payable;
    let operating_asset = prepayment + contract_assets + accounts_receiv_bill;
    let value = (operating_liability - operating_asset) / assets;
    value.is_finite().then_some(value)
}

fn push_grouped_observation(
    observations_by_sector: &mut BTreeMap<String, Vec<NolObservation>>,
    group: Option<&str>,
    observation: NolObservation,
) {
    let Some(group) = group.filter(|group| !group.is_empty()) else {
        return;
    };
    if observation.y.is_finite() && observation.x.iter().all(|value| value.is_finite()) {
        observations_by_sector
            .entry(group.to_string())
            .or_default()
            .push(observation);
    }
}

fn grouped_ols_residuals(
    observations_by_sector: &BTreeMap<String, Vec<NolObservation>>,
) -> Vec<(usize, f64)> {
    let mut output = Vec::new();
    for observations in observations_by_sector.values() {
        output.extend(ols_residuals(observations));
    }
    output
}

fn ols_residuals(observations: &[NolObservation]) -> Vec<(usize, f64)> {
    if observations.len() < MIN_INDUSTRY_OLS_OBS {
        return Vec::new();
    }
    let Some(beta) = ols_beta(observations) else {
        return Vec::new();
    };
    observations
        .iter()
        .filter_map(|observation| {
            let fitted = beta[0] + beta[1] * observation.x[0] + beta[2] * observation.x[1];
            let residual = observation.y - fitted;
            residual
                .is_finite()
                .then_some((observation.offset, residual))
        })
        .collect()
}

fn ols_beta(observations: &[NolObservation]) -> Option<[f64; PARAM_COUNT]> {
    let mut xtx = [[0.0; PARAM_COUNT]; PARAM_COUNT];
    let mut xty = [0.0; PARAM_COUNT];

    for observation in observations {
        let row = [1.0, observation.x[0], observation.x[1]];
        for i in 0..PARAM_COUNT {
            xty[i] += row[i] * observation.y;
            for j in 0..PARAM_COUNT {
                xtx[i][j] += row[i] * row[j];
            }
        }
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

fn clean(value: Option<f64>) -> Option<f64> {
    value.filter(|value| value.is_finite())
}

fn clean_or_zero(value: Option<f64>) -> f64 {
    clean(value).unwrap_or(0.0)
}

fn raw_spec() -> FactorSpec {
    FactorSpec {
        id: NOL2_RAW_ID.to_string(),
        aliases: Vec::new(),
        name: NOL2_RAW_ID.to_string(),
        asset_class: AssetClass::Stock,
        frequency: Frequency::Daily,
        version: VERSION.to_string(),
        tags: vec!["internal".to_string(), "financial_raw".to_string()],
        description: "Internal nol2 industry OLS residual raw series.".to_string(),
        dependencies: Vec::new(),
        intraday_raw_dependencies: Vec::new(),
        lookback: Lookback { trading_days: 0 },
    }
}

fn tags() -> Vec<String> {
    [
        "DBZQ",
        "financial",
        "fundamental",
        "pit",
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

    fn assert_close(left: f64, right: f64) {
        assert!(
            (left - right).abs() < 1e-9,
            "left={left}, right={right}, diff={}",
            (left - right).abs()
        );
    }

    #[test]
    fn nol_formula_uses_operating_liability_minus_operating_asset() {
        let value = nol_from_values(10.0, 2.0, 3.0, 4.0, 1.0, 5.0, 7.0, 100.0).expect("nol");
        assert_close(value, 6.0 / 100.0);
    }

    #[test]
    fn nol_rejects_invalid_assets() {
        assert!(nol_from_values(10.0, 2.0, 3.0, 4.0, 1.0, 5.0, 7.0, 0.0).is_none());
        assert!(nol_from_values(10.0, 2.0, 3.0, 4.0, 1.0, 5.0, 7.0, -1.0).is_none());
    }

    #[test]
    fn nol_ols_requires_more_than_two_industry_observations() {
        let observations = (0..2)
            .map(|idx| NolObservation {
                offset: idx,
                y: idx as f64,
                x: [idx as f64, (idx + 1) as f64],
            })
            .collect::<Vec<_>>();
        assert!(ols_residuals(&observations).is_empty());
    }

    #[test]
    fn nol_grouped_ols_skips_unclassified_observations() {
        let mut grouped = BTreeMap::<String, Vec<NolObservation>>::new();
        push_grouped_observation(
            &mut grouped,
            None,
            NolObservation {
                offset: 0,
                y: 1.0,
                x: [0.0, 1.0],
            },
        );
        assert!(grouped.is_empty());
        assert!(grouped_ols_residuals(&grouped).is_empty());
    }

    #[test]
    fn nol_ols_residuals_fit_by_industry_group() {
        let mut grouped = BTreeMap::<String, Vec<NolObservation>>::new();
        let observations = [
            (0, 1.0, [1.0, 0.0]),
            (1, 2.0, [2.0, 1.0]),
            (2, 3.0, [3.0, 1.0]),
            (3, 5.0, [4.0, 2.0]),
        ];
        for (offset, y, x) in observations {
            push_grouped_observation(
                &mut grouped,
                Some("801010"),
                NolObservation { offset, y, x },
            );
        }
        let residuals = grouped_ols_residuals(&grouped);
        assert_eq!(residuals.len(), 4);
        assert!(residuals.iter().all(|(_, value)| value.is_finite()));
    }

    #[test]
    fn nol2_metadata_has_dbzq_tags() {
        let spec = StockDailyNol2.spec();
        assert_eq!(spec.id, "nol2");
        assert!(spec.tags.iter().any(|tag| tag == "DBZQ"));
        assert!(spec.tags.iter().any(|tag| tag == "financial"));
    }
}

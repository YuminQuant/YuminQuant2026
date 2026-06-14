use std::any::Any;

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
    factor_series_to_panel_column, DailyPanel, EventDrivenCrossSectionCache, FinancialEventMarker,
    FinancialEventMarkerBuilder, FinancialEventSchedule, FinancialPitReader,
    FinancialStatementDataset, InstrumentAlignedSnapshotCache, PanelColumn, PitFinancialRecordView,
    ReportTypePreference,
};
use crate::factor::{Factor, FactorUpdatePolicy};

const VERSION: &str = "0.1.1";
const SFLI2_RAW_ID: &str = "__sfli2_raw";
const SFLI_WINDOW: usize = 8;
const CASHFLOW_QUARTERS: usize = 8;
const BALANCE_QUARTERS: usize = 9;
const EPS: f64 = 1e-12;

const CPACF_COLUMN: &str = "c_pay_acq_const_fiolta";
const CFO_COLUMN: &str = "n_cashflow_act";
const CIDF_COLUMN: &str = "n_recp_disp_fiolta";
const LONG_BORROW_COLUMN: &str = "lt_borr";
const EQUITY_COLUMN: &str = "total_hldr_eqy_exc_min_int";
const ASSET_COLUMN: &str = "total_assets";

pub struct StockDailySfli2;

#[derive(Default)]
struct Sfli2ComputeState {
    raw_cache: EventDrivenCrossSectionCache,
    snapshot_cache: InstrumentAlignedSnapshotCache<Sfli2Snapshot>,
}

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailySfli2)
}

impl Factor for StockDailySfli2 {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "sfli2".to_string(),
            aliases: vec!["SFLI2".to_string(), "Short Financing Long Investment 2".to_string()],
            name: "sfli2".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: tags(),
            description: "DBZQ short-financing long-investment factor 2. It uses PIT single-quarter cashflow and consolidated balance-sheet reports to compute eight strict quarterly SFLI observations, forms mean/sample-std, replays raw values between financial events, and finally neutralizes by Barra SIZE and SW sector.".to_string(),
            dependencies: vec![
                DataRequest::financial_quarters(
                    DatasetId::StockCashFlow,
                    &[CPACF_COLUMN, CFO_COLUMN, CIDF_COLUMN],
                    CASHFLOW_QUARTERS,
                ),
                DataRequest::financial_quarters(
                    DatasetId::StockBalanceSheet,
                    &[LONG_BORROW_COLUMN, EQUITY_COLUMN, ASSET_COLUMN],
                    BALANCE_QUARTERS,
                ),
                DataRequest::new(DatasetId::StockBarraDaily, &["SIZE"]),
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
        Box::new(Sfli2ComputeState::default())
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
        if requested_ids.iter().all(|id| id != "sfli2") {
            return Ok(Vec::new());
        }
        let state = state
            .downcast_mut::<Sfli2ComputeState>()
            .ok_or_else(|| err("sfli2 received incompatible event cache state"))?;
        let cashflow = data.financial_reader(
            DatasetId::StockCashFlow,
            ReportTypePreference::income_single_quarter(),
        )?;
        let balance = data.financial_reader(
            DatasetId::StockBalanceSheet,
            ReportTypePreference::balance_sheet_consolidated(),
        )?;
        let schedule =
            FinancialEventSchedule::from_pit_readers(&[cashflow.clone(), balance.clone()]);
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
                self.compute_raw_with_prepared_financials(
                    data,
                    &cashflow,
                    &balance,
                    &mut state.snapshot_cache,
                )
                .map(|series| vec![series])
            },
        )?;
        self.finalize_raw_series(data, raw_series)
            .map(|series| vec![series])
    }
}

impl StockDailySfli2 {
    fn compute_with_snapshot_cache(
        &self,
        data: &DataPool,
        snapshot_cache: &mut InstrumentAlignedSnapshotCache<Sfli2Snapshot>,
    ) -> Result<FactorSeries> {
        let cashflow = data.financial_reader(
            DatasetId::StockCashFlow,
            ReportTypePreference::income_single_quarter(),
        )?;
        let balance = data.financial_reader(
            DatasetId::StockBalanceSheet,
            ReportTypePreference::balance_sheet_consolidated(),
        )?;
        let raw_series = vec![self.compute_raw_with_prepared_financials(
            data,
            &cashflow,
            &balance,
            snapshot_cache,
        )?];
        self.finalize_raw_series(data, raw_series)
    }

    fn compute_raw_with_prepared_financials(
        &self,
        data: &DataPool,
        cashflow: &FinancialPitReader<'_>,
        balance: &FinancialPitReader<'_>,
        snapshot_cache: &mut InstrumentAlignedSnapshotCache<Sfli2Snapshot>,
    ) -> Result<FactorSeries> {
        let panel = data.stock_universe_panel()?;
        let raw = sfli2_raw_column(&panel, cashflow, balance, snapshot_cache)?;
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
            .find(|series| series.spec.id == SFLI2_RAW_ID)
            .ok_or_else(|| err("missing sfli2 raw series"))?;
        let raw = factor_series_to_panel_column(&panel, &series)?;
        let neutralized = neutralize_size_sector(&raw, &panel, data)?;
        Ok(neutralized.to_factor_series(self.spec()))
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Sfli2Snapshot {
    value: f64,
}

fn sfli2_raw_column(
    panel: &DailyPanel,
    cashflow: &FinancialPitReader<'_>,
    balance: &FinancialPitReader<'_>,
    cache: &mut InstrumentAlignedSnapshotCache<Sfli2Snapshot>,
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
            |_, ts_code, offset| is_bj_stock(ts_code) || !panel.is_present_offset(offset),
            |trade_date, ts_code, _| sfli2_marker(ts_code, trade_date, cashflow, balance),
            |trade_date, ts_code, _| {
                sfli2_snapshot_for_stock(ts_code, trade_date, cashflow, balance)
            },
        );
        let date_offset = date_idx * instrument_count;
        for (instrument_idx, snapshot) in snapshots.iter().enumerate() {
            if let Some(snapshot) = snapshot {
                values[date_offset + instrument_idx] = Some(snapshot.value);
            }
        }
    }

    panel.column_from_values(values)
}

fn sfli2_marker(
    ts_code: &str,
    trade_date: i32,
    cashflow: &FinancialPitReader<'_>,
    balance: &FinancialPitReader<'_>,
) -> Option<FinancialEventMarker> {
    let mut current = cashflow.latest_quarter_end_date(ts_code, trade_date)?;
    let mut builder = FinancialEventMarkerBuilder::new();
    for _ in 0..SFLI_WINDOW {
        let prev = previous_quarter_end_date(current)?;
        builder.include_reader_record_for_end_date(
            FinancialStatementDataset::CashFlow,
            cashflow,
            ts_code,
            trade_date,
            current,
        );
        builder.include_reader_record_for_end_date(
            FinancialStatementDataset::BalanceSheet,
            balance,
            ts_code,
            trade_date,
            current,
        );
        builder.include_reader_record_for_end_date(
            FinancialStatementDataset::BalanceSheet,
            balance,
            ts_code,
            trade_date,
            prev,
        );
        current = prev;
    }
    builder.build()
}

fn sfli2_snapshot_for_stock(
    ts_code: &str,
    trade_date: i32,
    cashflow: &FinancialPitReader<'_>,
    balance: &FinancialPitReader<'_>,
) -> Option<Sfli2Snapshot> {
    let mut current = cashflow.latest_quarter_end_date(ts_code, trade_date)?;
    let mut sfli_values = Vec::with_capacity(SFLI_WINDOW);
    for _ in 0..SFLI_WINDOW {
        let prev = previous_quarter_end_date(current)?;
        let cash_t = cashflow.record_for_end_date(ts_code, trade_date, current)?;
        let balance_t = balance.record_for_end_date(ts_code, trade_date, current)?;
        let balance_prev = balance.record_for_end_date(ts_code, trade_date, prev)?;
        let sfli = sfli_for_records(cash_t, balance_t, balance_prev)?;
        sfli_values.push(sfli);
        current = prev;
    }
    let value = mean_over_sample_std(&sfli_values)?;
    Some(Sfli2Snapshot { value })
}

fn sfli_for_records(
    cash_t: PitFinancialRecordView<'_>,
    balance_t: PitFinancialRecordView<'_>,
    balance_prev: PitFinancialRecordView<'_>,
) -> Option<f64> {
    let cpacf = clean_or_zero(cash_t.column(CPACF_COLUMN));
    let cfo = clean_or_zero(cash_t.column(CFO_COLUMN));
    let cidf = clean_or_zero(cash_t.column(CIDF_COLUMN));
    let lb_t = clean_or_zero(balance_t.column(LONG_BORROW_COLUMN));
    let lb_prev = clean_or_zero(balance_prev.column(LONG_BORROW_COLUMN));
    let cs_t = positive_equity(balance_t.column(EQUITY_COLUMN))?;
    let cs_prev = positive_equity(balance_prev.column(EQUITY_COLUMN))?;
    let assets = clean(balance_t.column(ASSET_COLUMN)).filter(|value| *value > 0.0)?;
    sfli_from_values(cpacf, lb_t - lb_prev, cs_t - cs_prev, cfo, cidf, assets)
}

fn positive_equity(value: Option<f64>) -> Option<f64> {
    clean(value).filter(|value| *value > EPS)
}

fn sfli_from_values(
    cpacf: f64,
    delta_lb: f64,
    delta_cs: f64,
    cfo: f64,
    cidf: f64,
    assets: f64,
) -> Option<f64> {
    if !assets.is_finite() || assets <= 0.0 {
        return None;
    }
    let value = (cpacf - (delta_lb + delta_cs + cfo + cidf)) / assets;
    value.is_finite().then_some(value)
}

fn mean_over_sample_std(values: &[f64]) -> Option<f64> {
    if values.len() != SFLI_WINDOW || values.iter().any(|value| !value.is_finite()) {
        return None;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|value| {
            let diff = value - mean;
            diff * diff
        })
        .sum::<f64>()
        / (values.len() - 1) as f64;
    let std = variance.sqrt();
    (std.is_finite() && std > EPS).then_some(mean / std)
}

fn clean(value: Option<f64>) -> Option<f64> {
    value.filter(|value| value.is_finite())
}

fn clean_or_zero(value: Option<f64>) -> f64 {
    clean(value).unwrap_or(0.0)
}

fn raw_spec() -> FactorSpec {
    FactorSpec {
        id: SFLI2_RAW_ID.to_string(),
        aliases: Vec::new(),
        name: SFLI2_RAW_ID.to_string(),
        asset_class: AssetClass::Stock,
        frequency: Frequency::Daily,
        version: VERSION.to_string(),
        tags: vec!["internal".to_string(), "financial_raw".to_string()],
        description: "Internal sfli2 raw series.".to_string(),
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
    fn sfli_formula_uses_delta_debt_delta_equity_and_assets() {
        let value = sfli_from_values(100.0, 10.0, 20.0, 30.0, 5.0, 500.0).expect("sfli");
        assert_close(value, 35.0 / 500.0);
    }

    #[test]
    fn sfli_rejects_invalid_assets() {
        assert!(sfli_from_values(100.0, 10.0, 20.0, 30.0, 5.0, 0.0).is_none());
        assert!(sfli_from_values(100.0, 10.0, 20.0, 30.0, 5.0, -1.0).is_none());
    }

    #[test]
    fn positive_equity_rejects_zero_and_negative_values() {
        assert_eq!(positive_equity(Some(10.0)), Some(10.0));
        assert_eq!(positive_equity(Some(0.0)), None);
        assert_eq!(positive_equity(Some(-1.0)), None);
    }

    #[test]
    fn sfli2_requires_strict_eight_values_and_positive_sample_std() {
        assert!(mean_over_sample_std(&[1.0, 2.0, 3.0]).is_none());
        assert!(mean_over_sample_std(&[1.0; SFLI_WINDOW]).is_none());
        let values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mean = 4.5;
        let std = (42.0_f64 / 7.0).sqrt();
        assert_close(mean_over_sample_std(&values).expect("sfli2"), mean / std);
    }

    #[test]
    fn sfli2_metadata_has_dbzq_tags() {
        let spec = StockDailySfli2.spec();
        assert_eq!(spec.id, "sfli2");
        assert!(spec.tags.iter().any(|tag| tag == "DBZQ"));
        assert!(spec.tags.iter().any(|tag| tag == "financial"));
    }
}

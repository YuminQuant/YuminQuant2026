use std::any::Any;
use std::collections::{BTreeMap, BTreeSet};

use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::{DataPool, Table};
use crate::error::{err, Result};
use crate::factor::common::financial::previous_quarter_end_date;
use crate::factor::common::stock_daily_ops::{is_bj_stock, neutralize_size_sector_with_inputs};
use crate::factor::common::{
    cached_financial_stock_snapshots_for_date, compute_financial_event_snapshot_streaming,
    factor_series_to_panel_column, ClassificationLevel, ClassificationMap, DailyPanel,
    EventDrivenCrossSectionCache, FinancialEventMarker, FinancialEventMarkerBuilder,
    FinancialEventSchedule, FinancialPitReader, FinancialStatementDataset,
    InstrumentAlignedSnapshotCache, PanelColumn, PitFinancialRecordView, ReportTypePreference,
};
use crate::factor::{Factor, FactorUpdatePolicy};

const VERSION: &str = "0.1.0";
const SPECIAL_ROA1_RAW_ID: &str = "__special_roa1_residual_raw";
const SPECIAL_ROA2_RAW_ID: &str = "__special_roa2_residual_raw";
const FINANCIAL_QUARTERS: usize = 5;
const REGRESSOR_COUNT: usize = 7;
const RIDGE_LAMBDA: f64 = 1.0;
const EPS: f64 = 1e-12;

const PROFIT_COLUMN: &str = "n_income";
const OPER_COST_COLUMN: &str = "oper_cost";
const TOTAL_ASSETS_COLUMN: &str = "total_assets";
const TOTAL_LIAB_COLUMN: &str = "total_liab";
const EQUITY_COLUMN: &str = "total_hldr_eqy_exc_min_int";
const CUR_ASSETS_COLUMN: &str = "total_cur_assets";
const CUR_LIAB_COLUMN: &str = "total_cur_liab";
const INTAN_ASSETS_COLUMN: &str = "intan_assets";
const INVENTORIES_COLUMN: &str = "inventories";
const PB_COLUMN: &str = "pb";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RegressionMode {
    BySector,
    BySectorWithIndustryDummies,
}

pub struct StockDailySpecialRoa1;
pub struct StockDailySpecialRoa2;

#[derive(Default)]
struct SpecialRoa2ComputeState {
    raw_cache: EventDrivenCrossSectionCache,
    snapshot_cache: InstrumentAlignedSnapshotCache<SpecialRoa2Snapshot>,
}

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailySpecialRoa2)
}

impl Factor for StockDailySpecialRoa1 {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "special_roa1".to_string(),
            aliases: vec![
                "Special ROA 1".to_string(),
                "Idiosyncratic Profitability 1".to_string(),
            ],
            name: "special_roa1".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: tags_roa1(),
            description: "Deprecated DBZQ special ROA 1 factor. It builds PIT single-quarter ROA from net profit, uses single-quarter operating cost for inventory turnover, standardizes ROA and seven explanatory variables within CITIC level-1 industries, fills missing standardized explanatory variables with zero, then runs ridge regression separately within each CITIC level-1 industry. The ridge lambda is 1 and the intercept is unpenalized. The residual is neutralized against Barra SIZE and CITIC level-1 sector.".to_string(),
            dependencies: vec![
                DataRequest::new(DatasetId::StockDailyPv, &["close"]),
                DataRequest::financial_quarters(
                    DatasetId::StockIncome,
                    &[PROFIT_COLUMN, OPER_COST_COLUMN],
                    FINANCIAL_QUARTERS,
                ),
                DataRequest::financial_quarters(
                    DatasetId::StockBalanceSheet,
                    &[
                        TOTAL_ASSETS_COLUMN,
                        TOTAL_LIAB_COLUMN,
                        EQUITY_COLUMN,
                        CUR_ASSETS_COLUMN,
                        CUR_LIAB_COLUMN,
                        INTAN_ASSETS_COLUMN,
                        INVENTORIES_COLUMN,
                    ],
                    FINANCIAL_QUARTERS,
                ),
                DataRequest::new(DatasetId::StockDailyBasic, &[PB_COLUMN]),
                DataRequest::new(DatasetId::StockBarraDaily, &["SIZE"]),
                DataRequest::new(DatasetId::StockBasic, &["list_date"]),
                DataRequest::new(DatasetId::StockCiClassification, &["l1_code", "l2_code"]),
            ],
            intraday_raw_dependencies: Vec::new(),
            lookback: Lookback { trading_days: 0 },
        }
    }

    fn update_policy(&self) -> FactorUpdatePolicy {
        FactorUpdatePolicy::FinancialEventSnapshot
    }

    fn initial_compute_state(&self, _requested_ids: &[String]) -> Box<dyn Any + Send> {
        Box::new(SpecialRoa2ComputeState::default())
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
        if requested_ids.iter().all(|id| id != "special_roa1") {
            return Ok(Vec::new());
        }
        let state = state
            .downcast_mut::<SpecialRoa2ComputeState>()
            .ok_or_else(|| err("special_roa1 received incompatible event cache state"))?;
        let income = data.financial_reader(
            DatasetId::StockIncome,
            ReportTypePreference::income_single_quarter(),
        )?;
        let balance = data.financial_reader(
            DatasetId::StockBalanceSheet,
            ReportTypePreference::balance_sheet_consolidated(),
        )?;
        let schedule = FinancialEventSchedule::from_pit_readers(&[income.clone(), balance.clone()]);
        let list_dates = stock_basic_list_dates(data.daily(DatasetId::StockBasic)?)?;
        let sector_map = ClassificationMap::from_table(
            data.daily(DatasetId::StockCiClassification)?,
            ClassificationLevel::Sector,
        )?;
        let industry_map = ClassificationMap::from_table(
            data.daily(DatasetId::StockCiClassification)?,
            ClassificationLevel::Industry,
        )?;
        let raw_specs = [raw_spec(SPECIAL_ROA1_RAW_ID)];
        let raw_series = compute_financial_event_snapshot_streaming(
            requested_ids,
            context,
            data,
            &mut state.raw_cache,
            &schedule,
            &raw_specs,
            |_, _, data| {
                self.compute_raw_with_prepared_inputs(
                    data,
                    &income,
                    &balance,
                    &list_dates,
                    &sector_map,
                    &industry_map,
                    &mut state.snapshot_cache,
                )
                .map(|series| vec![series])
            },
        )?;
        self.finalize_raw_series(data, raw_series)
            .map(|series| vec![series])
    }
}

impl StockDailySpecialRoa1 {
    fn compute_with_snapshot_cache(
        &self,
        data: &DataPool,
        snapshot_cache: &mut InstrumentAlignedSnapshotCache<SpecialRoa2Snapshot>,
    ) -> Result<FactorSeries> {
        let income = data.financial_reader(
            DatasetId::StockIncome,
            ReportTypePreference::income_single_quarter(),
        )?;
        let balance = data.financial_reader(
            DatasetId::StockBalanceSheet,
            ReportTypePreference::balance_sheet_consolidated(),
        )?;
        let list_dates = stock_basic_list_dates(data.daily(DatasetId::StockBasic)?)?;
        let sector_map = ClassificationMap::from_table(
            data.daily(DatasetId::StockCiClassification)?,
            ClassificationLevel::Sector,
        )?;
        let industry_map = ClassificationMap::from_table(
            data.daily(DatasetId::StockCiClassification)?,
            ClassificationLevel::Industry,
        )?;
        let raw_series = vec![self.compute_raw_with_prepared_inputs(
            data,
            &income,
            &balance,
            &list_dates,
            &sector_map,
            &industry_map,
            snapshot_cache,
        )?];
        self.finalize_raw_series(data, raw_series)
    }

    fn compute_raw_with_prepared_inputs(
        &self,
        data: &DataPool,
        income: &FinancialPitReader<'_>,
        balance: &FinancialPitReader<'_>,
        list_dates: &BTreeMap<String, i32>,
        sector_map: &ClassificationMap,
        industry_map: &ClassificationMap,
        snapshot_cache: &mut InstrumentAlignedSnapshotCache<SpecialRoa2Snapshot>,
    ) -> Result<FactorSeries> {
        let panel = data.daily_panel(DatasetId::StockDailyPv)?;
        let daily_basic = data.daily(DatasetId::StockDailyBasic)?;
        let pb = panel.column_from_table(daily_basic, PB_COLUMN)?;
        let raw = special_roa2_raw_column(
            &panel,
            &pb,
            income,
            balance,
            list_dates,
            sector_map,
            industry_map,
            snapshot_cache,
            RegressionMode::BySector,
        )?;
        Ok(raw.to_factor_series(raw_spec(SPECIAL_ROA1_RAW_ID)))
    }

    fn finalize_raw_series(
        &self,
        data: &DataPool,
        raw_series: Vec<FactorSeries>,
    ) -> Result<FactorSeries> {
        let panel = data.daily_panel(DatasetId::StockDailyPv)?;
        let series = raw_series
            .into_iter()
            .find(|series| series.spec.id == SPECIAL_ROA1_RAW_ID)
            .ok_or_else(|| err("missing special_roa1 raw series"))?;
        let raw = factor_series_to_panel_column(&panel, &series)?;
        let size = panel.column_from_table(data.daily(DatasetId::StockBarraDaily)?, "SIZE")?;
        let sector_map = ClassificationMap::from_table(
            data.daily(DatasetId::StockCiClassification)?,
            ClassificationLevel::Sector,
        )?;
        let neutralized = neutralize_size_sector_with_inputs(&raw, &panel, &size, &sector_map)?;
        Ok(neutralized.to_factor_series(self.spec()))
    }
}

impl Factor for StockDailySpecialRoa2 {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "special_roa2".to_string(),
            aliases: vec![
                "Special ROA 2".to_string(),
                "Idiosyncratic Profitability 2".to_string(),
            ],
            name: "special_roa2".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: tags(),
            description: "DBZQ special ROA 2 factor. It builds PIT single-quarter ROA from net profit, uses single-quarter operating cost for inventory turnover, standardizes ROA and seven explanatory variables within CITIC level-1 industries, fills missing standardized explanatory variables with zero, then runs ridge regression separately within each CITIC level-1 industry with that sector's CITIC level-2 industry fixed effects. The ridge lambda is 1 and only continuous variables are penalized; intercept and industry dummies are unpenalized. The residual is neutralized against Barra SIZE and CITIC level-1 sector.".to_string(),
            dependencies: vec![
                DataRequest::new(DatasetId::StockDailyPv, &["close"]),
                DataRequest::financial_quarters(
                    DatasetId::StockIncome,
                    &[PROFIT_COLUMN, OPER_COST_COLUMN],
                    FINANCIAL_QUARTERS,
                ),
                DataRequest::financial_quarters(
                    DatasetId::StockBalanceSheet,
                    &[
                        TOTAL_ASSETS_COLUMN,
                        TOTAL_LIAB_COLUMN,
                        EQUITY_COLUMN,
                        CUR_ASSETS_COLUMN,
                        CUR_LIAB_COLUMN,
                        INTAN_ASSETS_COLUMN,
                        INVENTORIES_COLUMN,
                    ],
                    FINANCIAL_QUARTERS,
                ),
                DataRequest::new(DatasetId::StockDailyBasic, &[PB_COLUMN]),
                DataRequest::new(DatasetId::StockBarraDaily, &["SIZE"]),
                DataRequest::new(DatasetId::StockBasic, &["list_date"]),
                DataRequest::new(DatasetId::StockCiClassification, &["l1_code", "l2_code"]),
            ],
            intraday_raw_dependencies: Vec::new(),
            lookback: Lookback { trading_days: 0 },
        }
    }

    fn update_policy(&self) -> FactorUpdatePolicy {
        FactorUpdatePolicy::FinancialEventSnapshot
    }

    fn initial_compute_state(&self, _requested_ids: &[String]) -> Box<dyn Any + Send> {
        Box::new(SpecialRoa2ComputeState::default())
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
        if requested_ids.iter().all(|id| id != "special_roa2") {
            return Ok(Vec::new());
        }
        let state = state
            .downcast_mut::<SpecialRoa2ComputeState>()
            .ok_or_else(|| err("special_roa2 received incompatible event cache state"))?;
        let income = data.financial_reader(
            DatasetId::StockIncome,
            ReportTypePreference::income_single_quarter(),
        )?;
        let balance = data.financial_reader(
            DatasetId::StockBalanceSheet,
            ReportTypePreference::balance_sheet_consolidated(),
        )?;
        let schedule = FinancialEventSchedule::from_pit_readers(&[income.clone(), balance.clone()]);
        let list_dates = stock_basic_list_dates(data.daily(DatasetId::StockBasic)?)?;
        let sector_map = ClassificationMap::from_table(
            data.daily(DatasetId::StockCiClassification)?,
            ClassificationLevel::Sector,
        )?;
        let industry_map = ClassificationMap::from_table(
            data.daily(DatasetId::StockCiClassification)?,
            ClassificationLevel::Industry,
        )?;
        let raw_specs = [raw_spec(SPECIAL_ROA2_RAW_ID)];
        let raw_series = compute_financial_event_snapshot_streaming(
            requested_ids,
            context,
            data,
            &mut state.raw_cache,
            &schedule,
            &raw_specs,
            |_, _, data| {
                self.compute_raw_with_prepared_inputs(
                    data,
                    &income,
                    &balance,
                    &list_dates,
                    &sector_map,
                    &industry_map,
                    &mut state.snapshot_cache,
                )
                .map(|series| vec![series])
            },
        )?;
        self.finalize_raw_series(data, raw_series)
            .map(|series| vec![series])
    }
}

impl StockDailySpecialRoa2 {
    fn compute_with_snapshot_cache(
        &self,
        data: &DataPool,
        snapshot_cache: &mut InstrumentAlignedSnapshotCache<SpecialRoa2Snapshot>,
    ) -> Result<FactorSeries> {
        let income = data.financial_reader(
            DatasetId::StockIncome,
            ReportTypePreference::income_single_quarter(),
        )?;
        let balance = data.financial_reader(
            DatasetId::StockBalanceSheet,
            ReportTypePreference::balance_sheet_consolidated(),
        )?;
        let list_dates = stock_basic_list_dates(data.daily(DatasetId::StockBasic)?)?;
        let sector_map = ClassificationMap::from_table(
            data.daily(DatasetId::StockCiClassification)?,
            ClassificationLevel::Sector,
        )?;
        let industry_map = ClassificationMap::from_table(
            data.daily(DatasetId::StockCiClassification)?,
            ClassificationLevel::Industry,
        )?;
        let raw_series = vec![self.compute_raw_with_prepared_inputs(
            data,
            &income,
            &balance,
            &list_dates,
            &sector_map,
            &industry_map,
            snapshot_cache,
        )?];
        self.finalize_raw_series(data, raw_series)
    }

    fn compute_raw_with_prepared_inputs(
        &self,
        data: &DataPool,
        income: &FinancialPitReader<'_>,
        balance: &FinancialPitReader<'_>,
        list_dates: &BTreeMap<String, i32>,
        sector_map: &ClassificationMap,
        industry_map: &ClassificationMap,
        snapshot_cache: &mut InstrumentAlignedSnapshotCache<SpecialRoa2Snapshot>,
    ) -> Result<FactorSeries> {
        let panel = data.daily_panel(DatasetId::StockDailyPv)?;
        let daily_basic = data.daily(DatasetId::StockDailyBasic)?;
        let pb = panel.column_from_table(daily_basic, PB_COLUMN)?;
        let raw = special_roa2_raw_column(
            &panel,
            &pb,
            income,
            balance,
            list_dates,
            sector_map,
            industry_map,
            snapshot_cache,
            RegressionMode::BySectorWithIndustryDummies,
        )?;
        Ok(raw.to_factor_series(raw_spec(SPECIAL_ROA2_RAW_ID)))
    }

    fn finalize_raw_series(
        &self,
        data: &DataPool,
        raw_series: Vec<FactorSeries>,
    ) -> Result<FactorSeries> {
        let panel = data.daily_panel(DatasetId::StockDailyPv)?;
        let series = raw_series
            .into_iter()
            .find(|series| series.spec.id == SPECIAL_ROA2_RAW_ID)
            .ok_or_else(|| err("missing special_roa2 raw series"))?;
        let raw = factor_series_to_panel_column(&panel, &series)?;
        let size = panel.column_from_table(data.daily(DatasetId::StockBarraDaily)?, "SIZE")?;
        let sector_map = ClassificationMap::from_table(
            data.daily(DatasetId::StockCiClassification)?,
            ClassificationLevel::Sector,
        )?;
        let neutralized = neutralize_size_sector_with_inputs(&raw, &panel, &size, &sector_map)?;
        Ok(neutralized.to_factor_series(self.spec()))
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct SpecialRoa2Snapshot {
    roa: f64,
    debt_to_assets: Option<f64>,
    na_yoy: Option<f64>,
    working_assets_to_assets: Option<f64>,
    intan_to_assets: Option<f64>,
    inventory_turnover: Option<f64>,
}

#[derive(Clone, Debug, PartialEq)]
struct RawObservation {
    offset: usize,
    sector: String,
    industry: String,
    y: f64,
    x: [Option<f64>; REGRESSOR_COUNT],
}

#[derive(Clone, Debug, PartialEq)]
struct RidgeObservation {
    offset: usize,
    sector: String,
    industry: String,
    y: f64,
    x: [f64; REGRESSOR_COUNT],
}

fn special_roa2_raw_column(
    panel: &DailyPanel,
    pb: &PanelColumn,
    income: &FinancialPitReader<'_>,
    balance: &FinancialPitReader<'_>,
    list_dates: &BTreeMap<String, i32>,
    sector_map: &ClassificationMap,
    industry_map: &ClassificationMap,
    cache: &mut InstrumentAlignedSnapshotCache<SpecialRoa2Snapshot>,
    regression_mode: RegressionMode,
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
                    || industry_map.group_for(trade_date, ts_code).is_none()
            },
            |trade_date, ts_code, _| special_roa2_marker(ts_code, trade_date, income, balance),
            |trade_date, ts_code, _| {
                special_roa2_snapshot_for_stock(ts_code, trade_date, income, balance)
            },
        );
        let date_offset = date_idx * instrument_count;
        let mut raw_observations = Vec::new();
        for (instrument_idx, ts_code) in panel.instruments().iter().enumerate() {
            let offset = date_offset + instrument_idx;
            if is_bj_stock(ts_code) || !panel.is_present_offset(offset) {
                continue;
            }
            let Some(snapshot) = snapshots[instrument_idx] else {
                continue;
            };
            let age_log = list_dates
                .get(ts_code)
                .copied()
                .and_then(|list_date| listing_age_log(trade_date, list_date));
            let pb_value = clean(pb.values()[offset]);
            let (Some(sector), Some(industry)) = (
                sector_map.group_for(trade_date, ts_code),
                industry_map.group_for(trade_date, ts_code),
            ) else {
                continue;
            };
            let observation = raw_observation_from_snapshot(
                offset, sector, industry, snapshot, pb_value, age_log,
            );
            if let Some(observation) = observation {
                raw_observations.push(observation);
            }
        }
        let standardized = standardize_observations_by_sector(&raw_observations);
        let residuals = match regression_mode {
            RegressionMode::BySector => ridge_residuals_by_sector(&standardized),
            RegressionMode::BySectorWithIndustryDummies => {
                ridge_residuals_with_industry_dummies(&standardized)
            }
        };
        for (offset, residual) in residuals {
            values[offset] = Some(residual);
        }
    }

    panel.column_from_values(values)
}

fn raw_observation_from_snapshot(
    offset: usize,
    sector: &str,
    industry: &str,
    snapshot: SpecialRoa2Snapshot,
    pb: Option<f64>,
    age_log: Option<f64>,
) -> Option<RawObservation> {
    let y = snapshot.roa;
    let x = [
        snapshot.debt_to_assets,
        snapshot.na_yoy,
        snapshot.working_assets_to_assets,
        pb,
        snapshot.intan_to_assets,
        snapshot.inventory_turnover,
        age_log,
    ];
    if !y.is_finite() || x.iter().flatten().any(|value| !value.is_finite()) {
        return None;
    }
    Some(RawObservation {
        offset,
        sector: sector.to_string(),
        industry: industry.to_string(),
        y,
        x,
    })
}

fn special_roa2_marker(
    ts_code: &str,
    trade_date: i32,
    income: &FinancialPitReader<'_>,
    balance: &FinancialPitReader<'_>,
) -> Option<FinancialEventMarker> {
    let end_t = income.latest_quarter_end_date(ts_code, trade_date)?;
    let end_t1 = previous_quarter_end_date(end_t)?;
    let end_t4 = quarter_lag(end_t, 4)?;
    let mut builder = FinancialEventMarkerBuilder::new();
    builder.include_reader_record_for_end_date(
        FinancialStatementDataset::Income,
        income,
        ts_code,
        trade_date,
        end_t,
    );
    for end_date in [end_t, end_t1, end_t4] {
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

fn special_roa2_snapshot_for_stock(
    ts_code: &str,
    trade_date: i32,
    income: &FinancialPitReader<'_>,
    balance: &FinancialPitReader<'_>,
) -> Option<SpecialRoa2Snapshot> {
    let end_t = income.latest_quarter_end_date(ts_code, trade_date)?;
    let end_t1 = previous_quarter_end_date(end_t)?;
    let end_t4 = quarter_lag(end_t, 4)?;
    let income_t = income.record_for_end_date(ts_code, trade_date, end_t)?;
    let balance_t = balance.record_for_end_date(ts_code, trade_date, end_t)?;
    let balance_t1 = balance.record_for_end_date(ts_code, trade_date, end_t1)?;
    let balance_t4 = balance.record_for_end_date(ts_code, trade_date, end_t4)?;
    special_roa2_snapshot_from_records(income_t, balance_t, balance_t1, balance_t4)
}

fn special_roa2_snapshot_from_records(
    income_t: PitFinancialRecordView<'_>,
    balance_t: PitFinancialRecordView<'_>,
    balance_t1: PitFinancialRecordView<'_>,
    balance_t4: PitFinancialRecordView<'_>,
) -> Option<SpecialRoa2Snapshot> {
    let profit = clean(income_t.column(PROFIT_COLUMN))?;
    let assets_t = clean(balance_t.column(TOTAL_ASSETS_COLUMN)).filter(|value| *value > EPS)?;
    let assets_t1 = clean(balance_t1.column(TOTAL_ASSETS_COLUMN)).filter(|value| *value > EPS)?;
    special_roa2_snapshot_from_values(SpecialRoa2Inputs {
        profit,
        oper_cost: clean(income_t.column(OPER_COST_COLUMN)),
        assets_t,
        assets_t1,
        total_liab: clean(balance_t.column(TOTAL_LIAB_COLUMN)),
        equity_t: clean(balance_t.column(EQUITY_COLUMN)),
        equity_t4: clean(balance_t4.column(EQUITY_COLUMN)),
        cur_assets: clean(balance_t.column(CUR_ASSETS_COLUMN)),
        cur_liab: clean(balance_t.column(CUR_LIAB_COLUMN)),
        intan_assets: clean(balance_t.column(INTAN_ASSETS_COLUMN)),
        inventories_t: clean(balance_t.column(INVENTORIES_COLUMN)),
        inventories_t1: clean(balance_t1.column(INVENTORIES_COLUMN)),
    })
}

#[derive(Clone, Copy, Debug)]
struct SpecialRoa2Inputs {
    profit: f64,
    oper_cost: Option<f64>,
    assets_t: f64,
    assets_t1: f64,
    total_liab: Option<f64>,
    equity_t: Option<f64>,
    equity_t4: Option<f64>,
    cur_assets: Option<f64>,
    cur_liab: Option<f64>,
    intan_assets: Option<f64>,
    inventories_t: Option<f64>,
    inventories_t1: Option<f64>,
}

fn special_roa2_snapshot_from_values(input: SpecialRoa2Inputs) -> Option<SpecialRoa2Snapshot> {
    if input.assets_t <= EPS || input.assets_t1 <= EPS {
        return None;
    }
    let avg_assets = input.assets_t + input.assets_t1;
    if avg_assets <= EPS {
        return None;
    }
    let debt_to_assets = input
        .total_liab
        .map(|total_liab| total_liab / input.assets_t);
    let na_yoy = input
        .equity_t
        .zip(input.equity_t4)
        .and_then(|(equity_t, equity_t4)| {
            (equity_t4.abs() > EPS).then_some((equity_t - equity_t4) / equity_t4.abs())
        });
    let working_assets_to_assets = input
        .cur_assets
        .zip(input.cur_liab)
        .map(|(cur_assets, cur_liab)| (cur_assets - cur_liab) / input.assets_t);
    let intan_to_assets = input
        .intan_assets
        .map(|intan_assets| intan_assets / input.assets_t);
    let inventory_turnover = input
        .oper_cost
        .zip(input.inventories_t.zip(input.inventories_t1))
        .and_then(|(oper_cost, (inventories_t, inventories_t1))| {
            let inventory_base = inventories_t + inventories_t1;
            if inventory_base < -EPS {
                None
            } else if inventory_base.abs() <= EPS {
                Some(0.0)
            } else {
                Some(2.0 * oper_cost / inventory_base)
            }
        });
    let snapshot = SpecialRoa2Snapshot {
        roa: 2.0 * input.profit / avg_assets,
        debt_to_assets,
        na_yoy,
        working_assets_to_assets,
        intan_to_assets,
        inventory_turnover,
    };
    (snapshot.roa.is_finite()
        && [
            snapshot.debt_to_assets,
            snapshot.na_yoy,
            snapshot.working_assets_to_assets,
            snapshot.intan_to_assets,
            snapshot.inventory_turnover,
        ]
        .into_iter()
        .flatten()
        .all(|value| value.is_finite()))
    .then_some(snapshot)
}

fn standardize_observations_by_sector(observations: &[RawObservation]) -> Vec<RidgeObservation> {
    let mut grouped = BTreeMap::<&str, Vec<&RawObservation>>::new();
    for observation in observations {
        grouped
            .entry(observation.sector.as_str())
            .or_default()
            .push(observation);
    }
    let mut output = Vec::new();
    for group in grouped.values() {
        let Some(y_stats) = mean_std(group.iter().map(|observation| observation.y)) else {
            continue;
        };
        let mut x_stats = Vec::with_capacity(REGRESSOR_COUNT);
        for idx in 0..REGRESSOR_COUNT {
            x_stats.push(mean_std(
                group.iter().filter_map(|observation| observation.x[idx]),
            ));
        }
        for observation in group {
            let y = (observation.y - y_stats.0) / y_stats.1;
            let mut x = [0.0; REGRESSOR_COUNT];
            for idx in 0..REGRESSOR_COUNT {
                x[idx] = match (observation.x[idx], x_stats[idx]) {
                    (Some(value), Some(stats)) => (value - stats.0) / stats.1,
                    _ => 0.0,
                };
            }
            let has_signal = x.iter().any(|value| value.abs() > EPS);
            if y.is_finite() && has_signal && x.iter().all(|value| value.is_finite()) {
                output.push(RidgeObservation {
                    offset: observation.offset,
                    sector: observation.sector.clone(),
                    industry: observation.industry.clone(),
                    y,
                    x,
                });
            }
        }
    }
    output
}

fn mean_std<I>(values: I) -> Option<(f64, f64)>
where
    I: Iterator<Item = f64>,
{
    let values = values.collect::<Vec<_>>();
    if values.len() < 2 || values.iter().any(|value| !value.is_finite()) {
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
    (std.is_finite() && std > EPS).then_some((mean, std))
}

fn ridge_residuals_with_industry_dummies(observations: &[RidgeObservation]) -> Vec<(usize, f64)> {
    if observations.is_empty() {
        return Vec::new();
    }
    let mut grouped = BTreeMap::<&str, Vec<&RidgeObservation>>::new();
    for observation in observations {
        grouped
            .entry(observation.sector.as_str())
            .or_default()
            .push(observation);
    }

    let mut output = Vec::new();
    for group in grouped.values() {
        let levels = group
            .iter()
            .map(|observation| observation.industry.clone())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        let dummy_count = levels.len().saturating_sub(1);
        let param_count = 1 + REGRESSOR_COUNT + dummy_count;
        if group.len() < 20.max(param_count + 1) {
            continue;
        }
        let dummy_offsets = levels
            .iter()
            .skip(1)
            .enumerate()
            .map(|(idx, level)| (level.as_str(), 1 + REGRESSOR_COUNT + idx))
            .collect::<BTreeMap<_, _>>();
        let Some(beta) = ridge_beta_with_industry_dummies(group, param_count, &dummy_offsets)
        else {
            continue;
        };
        for observation in group {
            let mut fitted = beta[0];
            for idx in 0..REGRESSOR_COUNT {
                fitted += beta[idx + 1] * observation.x[idx];
            }
            if let Some(dummy_idx) = dummy_offsets.get(observation.industry.as_str()) {
                fitted += beta[*dummy_idx];
            }
            let residual = observation.y - fitted;
            if residual.is_finite() {
                output.push((observation.offset, residual));
            }
        }
    }
    output
}

fn ridge_residuals_by_sector(observations: &[RidgeObservation]) -> Vec<(usize, f64)> {
    let mut grouped = BTreeMap::<&str, Vec<&RidgeObservation>>::new();
    for observation in observations {
        grouped
            .entry(observation.sector.as_str())
            .or_default()
            .push(observation);
    }
    let param_count = 1 + REGRESSOR_COUNT;
    let min_obs = 20.max(param_count + 1);
    let mut output = Vec::new();
    for group in grouped.values() {
        if group.len() < min_obs {
            continue;
        }
        let Some(beta) = ridge_beta_continuous(group) else {
            continue;
        };
        for observation in group {
            let mut fitted = beta[0];
            for idx in 0..REGRESSOR_COUNT {
                fitted += beta[idx + 1] * observation.x[idx];
            }
            let residual = observation.y - fitted;
            if residual.is_finite() {
                output.push((observation.offset, residual));
            }
        }
    }
    output
}

fn ridge_beta_continuous(observations: &[&RidgeObservation]) -> Option<Vec<f64>> {
    let param_count = 1 + REGRESSOR_COUNT;
    let mut xtx = vec![vec![0.0; param_count]; param_count];
    let mut xty = vec![0.0; param_count];
    for observation in observations {
        let mut row = [0.0; REGRESSOR_COUNT + 1];
        row[0] = 1.0;
        row[1..].copy_from_slice(&observation.x);
        for i in 0..param_count {
            xty[i] += row[i] * observation.y;
            for j in 0..param_count {
                xtx[i][j] += row[i] * row[j];
            }
        }
    }
    for idx in 1..=REGRESSOR_COUNT {
        xtx[idx][idx] += RIDGE_LAMBDA;
    }
    solve_linear_system(xtx, xty)
}

fn ridge_beta_with_industry_dummies(
    observations: &[&RidgeObservation],
    param_count: usize,
    dummy_offsets: &BTreeMap<&str, usize>,
) -> Option<Vec<f64>> {
    let mut xtx = vec![vec![0.0; param_count]; param_count];
    let mut xty = vec![0.0; param_count];
    for observation in observations {
        let mut row = vec![0.0; param_count];
        row[0] = 1.0;
        row[1..(REGRESSOR_COUNT + 1)].copy_from_slice(&observation.x);
        if let Some(dummy_idx) = dummy_offsets.get(observation.industry.as_str()) {
            row[*dummy_idx] = 1.0;
        }
        for i in 0..param_count {
            xty[i] += row[i] * observation.y;
            for j in 0..param_count {
                xtx[i][j] += row[i] * row[j];
            }
        }
    }
    for idx in 1..=REGRESSOR_COUNT {
        xtx[idx][idx] += RIDGE_LAMBDA;
    }
    solve_linear_system(xtx, xty)
}

fn solve_linear_system(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    if a.len() != n || a.iter().any(|row| row.len() != n) {
        return None;
    }
    for pivot_idx in 0..n {
        let mut pivot_row = pivot_idx;
        let mut pivot_abs = a[pivot_idx][pivot_idx].abs();
        for (row_idx, row) in a.iter().enumerate().skip(pivot_idx + 1) {
            let candidate = row[pivot_idx].abs();
            if candidate > pivot_abs {
                pivot_abs = candidate;
                pivot_row = row_idx;
            }
        }
        if pivot_abs <= EPS {
            return None;
        }
        if pivot_row != pivot_idx {
            a.swap(pivot_idx, pivot_row);
            b.swap(pivot_idx, pivot_row);
        }
        let pivot = a[pivot_idx][pivot_idx];
        for col_idx in pivot_idx..n {
            a[pivot_idx][col_idx] /= pivot;
        }
        b[pivot_idx] /= pivot;

        for row_idx in 0..n {
            if row_idx == pivot_idx {
                continue;
            }
            let factor = a[row_idx][pivot_idx];
            if factor.abs() <= f64::EPSILON {
                continue;
            }
            for col_idx in pivot_idx..n {
                a[row_idx][col_idx] -= factor * a[pivot_idx][col_idx];
            }
            b[row_idx] -= factor * b[pivot_idx];
        }
    }
    b.iter().all(|value| value.is_finite()).then_some(b)
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

fn listing_age_log(trade_date: i32, list_date: i32) -> Option<f64> {
    if list_date > trade_date {
        return None;
    }
    let trade_days = days_from_ymd_date(trade_date)?;
    let list_days = days_from_ymd_date(list_date)?;
    let years = (trade_days - list_days) as f64 / 365.0;
    (years.is_finite() && years > EPS)
        .then_some(years.ln())
        .filter(|value| value.is_finite())
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

fn quarter_lag(mut end_date: i32, quarters: usize) -> Option<i32> {
    for _ in 0..quarters {
        end_date = previous_quarter_end_date(end_date)?;
    }
    Some(end_date)
}

fn clean(value: Option<f64>) -> Option<f64> {
    value.filter(|value| value.is_finite())
}

fn raw_spec(raw_id: &str) -> FactorSpec {
    FactorSpec {
        id: raw_id.to_string(),
        aliases: Vec::new(),
        name: raw_id.to_string(),
        asset_class: AssetClass::Stock,
        frequency: Frequency::Daily,
        version: VERSION.to_string(),
        tags: vec!["internal".to_string(), "financial_raw".to_string()],
        description: "Internal special ROA ridge residual raw series.".to_string(),
        dependencies: Vec::new(),
        intraday_raw_dependencies: Vec::new(),
        lookback: Lookback { trading_days: 0 },
    }
}

fn tags() -> Vec<String> {
    [
        "DBZQ",
        "deprecated",
        "financial",
        "fundamental",
        "pit",
        "profitability",
        "roa",
        "ridge",
        "residual",
        "industry_dummy",
        "citic",
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

fn tags_roa1() -> Vec<String> {
    [
        "DBZQ",
        "deprecated",
        "financial",
        "fundamental",
        "pit",
        "profitability",
        "roa",
        "ridge",
        "residual",
        "citic",
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
    fn special_roa2_formula_uses_net_profit_and_expected_ratios() {
        let snapshot = special_roa2_snapshot_from_values(SpecialRoa2Inputs {
            profit: 20.0,
            oper_cost: Some(60.0),
            assets_t: 120.0,
            assets_t1: 80.0,
            total_liab: Some(30.0),
            equity_t: Some(50.0),
            equity_t4: Some(40.0),
            cur_assets: Some(70.0),
            cur_liab: Some(20.0),
            intan_assets: Some(6.0),
            inventories_t: Some(10.0),
            inventories_t1: Some(20.0),
        })
        .expect("snapshot");
        assert_close(snapshot.roa, 0.2);
        assert_close(snapshot.debt_to_assets.unwrap(), 0.25);
        assert_close(snapshot.na_yoy.unwrap(), 0.25);
        assert_close(snapshot.working_assets_to_assets.unwrap(), 50.0 / 120.0);
        assert_close(snapshot.intan_to_assets.unwrap(), 0.05);
        assert_close(snapshot.inventory_turnover.unwrap(), 4.0);
    }

    #[test]
    fn special_roa2_rejects_invalid_denominators() {
        assert!(special_roa2_snapshot_from_values(SpecialRoa2Inputs {
            profit: 20.0,
            oper_cost: Some(60.0),
            assets_t: 0.0,
            assets_t1: 80.0,
            total_liab: Some(30.0),
            equity_t: Some(50.0),
            equity_t4: Some(40.0),
            cur_assets: Some(70.0),
            cur_liab: Some(20.0),
            intan_assets: Some(6.0),
            inventories_t: Some(10.0),
            inventories_t1: Some(20.0),
        })
        .is_none());
    }

    #[test]
    fn special_roa2_zero_inventory_sets_turnover_to_zero() {
        let snapshot = special_roa2_snapshot_from_values(SpecialRoa2Inputs {
            profit: 20.0,
            oper_cost: Some(60.0),
            assets_t: 120.0,
            assets_t1: 80.0,
            total_liab: Some(30.0),
            equity_t: Some(50.0),
            equity_t4: Some(40.0),
            cur_assets: Some(70.0),
            cur_liab: Some(20.0),
            intan_assets: Some(6.0),
            inventories_t: Some(0.0),
            inventories_t1: Some(0.0),
        })
        .expect("snapshot");
        assert_close(snapshot.inventory_turnover.unwrap(), 0.0);
    }

    #[test]
    fn listing_age_log_uses_natural_years() {
        let value = listing_age_log(20210101, 20200101).expect("age");
        assert!((value - (366.0_f64 / 365.0).ln()).abs() < 1e-12);
        assert!(listing_age_log(20200101, 20210101).is_none());
    }

    #[test]
    fn sector_standardization_fills_missing_explanatory_values_after_zscore() {
        let rows = vec![
            RawObservation {
                offset: 0,
                sector: "A".to_string(),
                industry: "A1".to_string(),
                y: 1.0,
                x: [
                    Some(1.0),
                    None,
                    Some(3.0),
                    Some(4.0),
                    Some(5.0),
                    Some(6.0),
                    Some(7.0),
                ],
            },
            RawObservation {
                offset: 1,
                sector: "A".to_string(),
                industry: "A2".to_string(),
                y: 2.0,
                x: [
                    Some(2.0),
                    Some(3.0),
                    Some(4.0),
                    Some(5.0),
                    Some(6.0),
                    Some(7.0),
                    Some(8.0),
                ],
            },
        ];
        let standardized = standardize_observations_by_sector(&rows);
        assert_eq!(standardized.len(), 2);
        assert_close(standardized[0].x[1], 0.0);
    }

    #[test]
    fn sector_standardization_skips_all_zero_explanatory_vector() {
        let rows = vec![
            RawObservation {
                offset: 0,
                sector: "A".to_string(),
                industry: "A1".to_string(),
                y: 1.0,
                x: [None; REGRESSOR_COUNT],
            },
            RawObservation {
                offset: 1,
                sector: "A".to_string(),
                industry: "A2".to_string(),
                y: 2.0,
                x: [None; REGRESSOR_COUNT],
            },
        ];
        assert!(standardize_observations_by_sector(&rows).is_empty());
    }

    #[test]
    fn ridge_residuals_leave_unpenalized_industry_fixed_effects_unshrunk() {
        let mut rows = Vec::new();
        for idx in 0..10 {
            rows.push(RidgeObservation {
                offset: idx,
                sector: "S".to_string(),
                industry: "A".to_string(),
                y: 1.0,
                x: [0.0; REGRESSOR_COUNT],
            });
        }
        for idx in 10..20 {
            rows.push(RidgeObservation {
                offset: idx,
                sector: "S".to_string(),
                industry: "B".to_string(),
                y: 3.0,
                x: [0.0; REGRESSOR_COUNT],
            });
        }
        let residuals = ridge_residuals_with_industry_dummies(&rows);
        assert_eq!(residuals.len(), 20);
        for (_, residual) in residuals {
            assert!(residual.abs() < 1e-9, "residual={residual}");
        }
    }

    #[test]
    fn ridge_residuals_with_industry_dummies_remove_continuous_signal() {
        let mut rows = Vec::new();
        for idx in 0..40 {
            let x0 = idx as f64 - 20.0;
            let industry_effect = if idx % 2 == 0 { 0.0 } else { 5.0 };
            let mut x = [0.0; REGRESSOR_COUNT];
            x[0] = x0;
            rows.push(RidgeObservation {
                offset: idx,
                sector: "S".to_string(),
                industry: if idx % 2 == 0 { "A" } else { "B" }.to_string(),
                y: 2.0 * x0 + industry_effect,
                x,
            });
        }
        let residuals = ridge_residuals_with_industry_dummies(&rows);
        assert_eq!(residuals.len(), 40);
        let residual_std = mean_std(residuals.iter().map(|(_, value)| *value))
            .map(|(_, std)| std)
            .unwrap();
        let y_std = mean_std(rows.iter().map(|row| row.y))
            .map(|(_, std)| std)
            .unwrap();
        assert!(
            residual_std < y_std * 0.05,
            "residual_std={residual_std}, y_std={y_std}"
        );
    }

    #[test]
    fn ridge_residuals_by_sector_fit_each_sector_independently() {
        let mut rows = Vec::new();
        for idx in 0..20 {
            rows.push(RidgeObservation {
                offset: idx,
                sector: "S1".to_string(),
                industry: "A".to_string(),
                y: 1.0,
                x: [0.0; REGRESSOR_COUNT],
            });
        }
        for idx in 20..40 {
            rows.push(RidgeObservation {
                offset: idx,
                sector: "S2".to_string(),
                industry: "B".to_string(),
                y: 5.0,
                x: [0.0; REGRESSOR_COUNT],
            });
        }
        let residuals = ridge_residuals_by_sector(&rows);
        assert_eq!(residuals.len(), 40);
        for (_, residual) in residuals {
            assert!(residual.abs() < 1e-9, "residual={residual}");
        }
    }

    #[test]
    fn special_roa1_spec_has_expected_metadata() {
        let spec = StockDailySpecialRoa1.spec();
        assert_eq!(spec.id, "special_roa1");
        assert!(spec.tags.iter().any(|tag| tag == "DBZQ"));
        assert!(spec.tags.iter().any(|tag| tag == "deprecated"));
        assert!(spec.tags.iter().any(|tag| tag == "sector"));
        assert!(spec.tags.iter().any(|tag| tag == "neutralize"));
        assert!(!spec.tags.iter().any(|tag| tag == "industry_dummy"));
        assert_eq!(spec.lookback.trading_days, 0);
    }

    #[test]
    fn special_roa2_spec_has_expected_metadata() {
        let spec = StockDailySpecialRoa2.spec();
        assert_eq!(spec.id, "special_roa2");
        assert!(spec.tags.iter().any(|tag| tag == "DBZQ"));
        assert!(spec.tags.iter().any(|tag| tag == "deprecated"));
        assert!(spec.tags.iter().any(|tag| tag == "industry_dummy"));
        assert!(spec.tags.iter().any(|tag| tag == "neutralize"));
        assert!(spec.tags.iter().any(|tag| tag == "size"));
        assert!(spec.tags.iter().any(|tag| tag == "sector"));
        assert_eq!(spec.lookback.trading_days, 0);
    }
}

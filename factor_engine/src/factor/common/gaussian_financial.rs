use std::any::Any;
use std::collections::HashMap;

use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::{DataPool, Table};
use crate::error::{err, Result};
use crate::factor::common::stock_daily_ops::{
    is_bj_stock, mask_bj, neutralize_size_sector_with_inputs,
};
use crate::factor::common::vector::clean;
use crate::factor::common::{
    cached_financial_stock_snapshots_for_date, compute_financial_event_snapshot_streaming,
    factor_series_to_panel_column, ClassificationLevel, ClassificationMap, DailyPanel,
    EventDrivenCrossSectionCache, FinancialEventMarker, FinancialEventMarkerBuilder,
    FinancialEventSchedule, FinancialEventTable, FinancialPitReader, FinancialStatementDataset,
    InstrumentAlignedSnapshotCache, PanelColumn, ReportTypePreference,
};
use crate::factor::{Factor, FactorUpdatePolicy};
use crate::operators::{cs_neutralize_regression, cs_pctrank};

pub const PROVIDER_KEY: &str = "stock|daily|dfzq_dbzq_gaussian_financial";

const VERSION: &str = "0.1.0";
const LOOKBACK: usize = 252;
const FINANCIAL_QUARTERS: usize = 8;
const GAUSSIAN_P_EPS: f64 = 1e-6;
const IMPLEMENTED_DIV_PROC: &str = "\u{5b9e}\u{65bd}";
const PB_ROE_PB_RAW_ID: &str = "__gaussian_financial_pb_roe_pb_raw";
const PB_ROE_ROE_RAW_ID: &str = "__gaussian_financial_pb_roe_roe_raw";

const INCOME_COLUMNS: [&str; 2] = ["revenue", "n_income_attr_p"];
const BALANCE_COLUMNS: [&str; 1] = ["total_hldr_eqy_exc_min_int"];
const CASHFLOW_COLUMNS: [&str; 2] = ["n_cashflow_act", "c_cash_equ_end_period"];

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum GaussianFinancialOutput {
    EpSq,
    EbTwoVar,
    SpSq,
    CfpSq,
    CashValue,
    DivTtm,
    ProfitYoySq,
    DeltaRoe,
    PbRoeSpread,
}

impl GaussianFinancialOutput {
    pub fn id(self) -> &'static str {
        match self {
            Self::EpSq => "ep_sq_gauss_resid",
            Self::EbTwoVar => "eb_two_var_gauss_resid",
            Self::SpSq => "sp_sq_gauss_resid",
            Self::CfpSq => "cfp_sq_gauss_resid",
            Self::CashValue => "cash_value_gauss_resid",
            Self::DivTtm => "div_ttm_gauss_resid",
            Self::ProfitYoySq => "profit_yoy_sq_gauss_resid",
            Self::DeltaRoe => "delta_roe_gauss_resid",
            Self::PbRoeSpread => "pb_roe_gauss_spread",
        }
    }

    fn alias(self) -> &'static str {
        match self {
            Self::EpSq => "Single Quarter EP Gaussian Residual",
            Self::EbTwoVar => "Two Variable EB Gaussian Residual",
            Self::SpSq => "Single Quarter SP Gaussian Residual",
            Self::CfpSq => "Single Quarter CFP Gaussian Residual",
            Self::CashValue => "Cash Value Gaussian Residual",
            Self::DivTtm => "Dividend TTM Gaussian Residual",
            Self::ProfitYoySq => "Profit YoY SQ Gaussian Residual",
            Self::DeltaRoe => "DeltaROE Gaussian Residual",
            Self::PbRoeSpread => "PB ROE Gaussian Spread",
        }
    }

    fn from_id(id: &str) -> Option<Self> {
        Some(match id {
            "ep_sq_gauss_resid" => Self::EpSq,
            "eb_two_var_gauss_resid" => Self::EbTwoVar,
            "sp_sq_gauss_resid" => Self::SpSq,
            "cfp_sq_gauss_resid" => Self::CfpSq,
            "cash_value_gauss_resid" => Self::CashValue,
            "div_ttm_gauss_resid" => Self::DivTtm,
            "profit_yoy_sq_gauss_resid" => Self::ProfitYoySq,
            "delta_roe_gauss_resid" => Self::DeltaRoe,
            "pb_roe_gauss_spread" => Self::PbRoeSpread,
            _ => return None,
        })
    }
}

pub struct GaussianFinancialFactor {
    kind: GaussianFinancialOutput,
}

#[derive(Default)]
struct GaussianFinancialComputeState {
    raw_cache: EventDrivenCrossSectionCache,
    snapshot_cache: InstrumentAlignedSnapshotCache<FinancialSnapshot>,
}

impl GaussianFinancialFactor {
    pub fn new(kind: GaussianFinancialOutput) -> Self {
        Self { kind }
    }
}

impl Factor for GaussianFinancialFactor {
    fn spec(&self) -> FactorSpec {
        spec(self.kind)
    }

    fn compute_provider_key(&self) -> String {
        PROVIDER_KEY.to_string()
    }

    fn update_policy(&self) -> FactorUpdatePolicy {
        FactorUpdatePolicy::FinancialEventSnapshot
    }

    fn compute(&self, context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let requested = [self.kind.id().to_string()];
        compute_requested(&requested, context, data)?
            .into_iter()
            .find(|series| series.spec.id == self.kind.id())
            .ok_or_else(|| {
                err(format!(
                    "gaussian financial provider did not return {}",
                    self.kind.id()
                ))
            })
    }

    fn compute_many(
        &self,
        requested_ids: &[String],
        context: &FactorContext,
        data: &DataPool,
    ) -> Result<Vec<FactorSeries>> {
        compute_requested(requested_ids, context, data)
    }

    fn initial_compute_state(&self, _requested_ids: &[String]) -> Box<dyn Any + Send> {
        Box::new(GaussianFinancialComputeState::default())
    }

    fn compute_many_stateful(
        &self,
        requested_ids: &[String],
        context: &FactorContext,
        data: &DataPool,
        state: &mut (dyn Any + Send),
    ) -> Result<Vec<FactorSeries>> {
        let state = state
            .downcast_mut::<GaussianFinancialComputeState>()
            .ok_or_else(|| err("gaussian financial provider received incompatible state"))?;
        let mut requested = requested_ids
            .iter()
            .filter_map(|id| GaussianFinancialOutput::from_id(id))
            .collect::<Vec<_>>();
        requested.sort();
        requested.dedup();
        if requested.is_empty() {
            return Ok(Vec::new());
        }
        let needs = FinancialNeeds::from_outputs(&requested);
        let prepared = GaussianFinancialPrepared::from_data(data, needs)?;
        let mut pit_readers = Vec::new();
        if needs.uses_income() {
            pit_readers.push(prepared.income.clone());
        }
        if needs.uses_balance() {
            pit_readers.push(prepared.balance.clone());
        }
        if needs.uses_cashflow() {
            pit_readers.push(prepared.cashflow.clone());
        }
        let mut schedule = FinancialEventSchedule::from_pit_readers(&pit_readers);
        if needs.uses_dividend() {
            schedule.merge(FinancialEventSchedule::from_tables(&[
                FinancialEventTable::dividend_ltm(data.daily(DatasetId::StockDividend)?),
            ])?);
        }
        let raw_specs = raw_specs_for_requested(&requested);
        let raw_cache = &mut state.raw_cache;
        let snapshot_cache = &mut state.snapshot_cache;
        let raw_series = compute_financial_event_snapshot_streaming(
            requested_ids,
            context,
            data,
            raw_cache,
            &schedule,
            &raw_specs,
            |requested_ids, context, data| {
                compute_requested_raw_with_prepared_financials(
                    requested_ids,
                    context,
                    data,
                    &prepared,
                    &requested,
                    snapshot_cache,
                )
            },
        )?;
        finalize_requested_from_raw(&requested, data, raw_series)
    }
}

pub fn spec(kind: GaussianFinancialOutput) -> FactorSpec {
    FactorSpec {
        id: kind.id().to_string(),
        aliases: vec![kind.alias().to_string()],
        name: kind.id().to_string(),
        asset_class: AssetClass::Stock,
        frequency: Frequency::Daily,
        version: VERSION.to_string(),
        tags: tags(kind),
        description: format!(
            "DFZQ/DBZQ financial Gaussian-rank reconstruction factor {}. It Gaussian-rank transforms the financial variables cross-sectionally, takes OLS residuals, applies the requested neutralization rule, and excludes BJ stocks.",
            kind.id()
        ),
        dependencies: vec![
            DataRequest::new(DatasetId::StockDailyPv, &["close"]),
            DataRequest::new(DatasetId::StockDailyBasic, &["total_mv"]),
            DataRequest::financial_quarters(
                DatasetId::StockIncome,
                &INCOME_COLUMNS,
                FINANCIAL_QUARTERS,
            ),
            DataRequest::financial_quarters(
                DatasetId::StockBalanceSheet,
                &BALANCE_COLUMNS,
                FINANCIAL_QUARTERS,
            ),
            DataRequest::financial_quarters(
                DatasetId::StockCashFlow,
                &CASHFLOW_COLUMNS,
                FINANCIAL_QUARTERS,
            ),
            DataRequest::new(
                DatasetId::StockDividend,
                &[
                    "ts_code",
                    "ann_date",
                    "div_proc",
                    "cash_div_tax",
                    "ex_date",
                    "base_share",
                ],
            ),
            DataRequest::new(DatasetId::StockBarraDaily, &["SIZE"]),
            DataRequest::new(DatasetId::StockSwClassification, &["l1_code"]),
        ],
        intraday_raw_dependencies: Vec::new(),
        lookback: Lookback {
            trading_days: LOOKBACK,
        },
    }
}

fn raw_spec(id: &str) -> FactorSpec {
    FactorSpec {
        id: id.to_string(),
        aliases: Vec::new(),
        name: id.to_string(),
        asset_class: AssetClass::Stock,
        frequency: Frequency::Daily,
        version: VERSION.to_string(),
        tags: vec!["internal".to_string(), "financial_raw".to_string()],
        description: format!("Internal Gaussian financial raw series {id}."),
        dependencies: Vec::new(),
        intraday_raw_dependencies: Vec::new(),
        lookback: Lookback { trading_days: 0 },
    }
}

fn raw_specs_for_requested(requested: &[GaussianFinancialOutput]) -> Vec<FactorSpec> {
    let mut specs = Vec::new();
    for kind in requested {
        match kind {
            GaussianFinancialOutput::PbRoeSpread => {
                specs.push(raw_spec(PB_ROE_PB_RAW_ID));
                specs.push(raw_spec(PB_ROE_ROE_RAW_ID));
            }
            _ => specs.push(raw_spec(kind.id())),
        }
    }
    specs
}

fn finalize_requested_from_raw(
    requested: &[GaussianFinancialOutput],
    data: &DataPool,
    raw_series: Vec<FactorSeries>,
) -> Result<Vec<FactorSeries>> {
    let panel = data.daily_panel(DatasetId::StockDailyPv)?;
    let raw_by_id = raw_series
        .into_iter()
        .map(|series| (series.spec.id.clone(), series))
        .collect::<HashMap<_, _>>();
    let sector_map = ClassificationMap::from_table(
        data.daily(DatasetId::StockSwClassification)?,
        ClassificationLevel::Sector,
    )?;
    let size = panel.column_from_table(data.daily(DatasetId::StockBarraDaily)?, "SIZE")?;
    let mut output = Vec::new();
    for kind in requested.iter().copied() {
        let factor = match kind {
            GaussianFinancialOutput::EpSq
            | GaussianFinancialOutput::SpSq
            | GaussianFinancialOutput::CfpSq
            | GaussianFinancialOutput::CashValue
            | GaussianFinancialOutput::DivTtm => {
                let raw = raw_column_by_id(&raw_by_id, kind.id(), panel)?;
                neutralize_sector_only_with_map(&raw, panel, &sector_map)?
            }
            GaussianFinancialOutput::EbTwoVar
            | GaussianFinancialOutput::ProfitYoySq
            | GaussianFinancialOutput::DeltaRoe => {
                let raw = raw_column_by_id(&raw_by_id, kind.id(), panel)?;
                neutralize_size_sector_with_inputs(&raw, panel, &size, &sector_map)?
            }
            GaussianFinancialOutput::PbRoeSpread => {
                let pb_raw = raw_column_by_id(&raw_by_id, PB_ROE_PB_RAW_ID, panel)?;
                let roe_raw = raw_column_by_id(&raw_by_id, PB_ROE_ROE_RAW_ID, panel)?;
                pb_roe_spread_from_raw(&pb_raw, &roe_raw, panel, &size, &sector_map)?
            }
        };
        output.push(mask_bj(&factor, panel)?.to_factor_series(spec(kind)));
    }
    Ok(output)
}

fn raw_column_by_id(
    raw_by_id: &HashMap<String, FactorSeries>,
    id: &str,
    panel: &DailyPanel,
) -> Result<PanelColumn> {
    let series = raw_by_id
        .get(id)
        .ok_or_else(|| err(format!("missing Gaussian financial raw series {id}")))?;
    factor_series_to_panel_column(panel, series)
}

pub fn compute_requested(
    requested_ids: &[String],
    context: &FactorContext,
    data: &DataPool,
) -> Result<Vec<FactorSeries>> {
    let mut snapshot_cache = InstrumentAlignedSnapshotCache::default();
    compute_requested_with_snapshot_cache(requested_ids, context, data, &mut snapshot_cache)
}

fn compute_requested_with_snapshot_cache(
    requested_ids: &[String],
    context: &FactorContext,
    data: &DataPool,
    snapshot_cache: &mut InstrumentAlignedSnapshotCache<FinancialSnapshot>,
) -> Result<Vec<FactorSeries>> {
    let mut requested = requested_ids
        .iter()
        .filter_map(|id| GaussianFinancialOutput::from_id(id))
        .collect::<Vec<_>>();
    requested.sort();
    requested.dedup();
    if requested.is_empty() {
        return Ok(Vec::new());
    }
    let needs = FinancialNeeds::from_outputs(&requested);
    let prepared = GaussianFinancialPrepared::from_data(data, needs)?;
    let raw_series = compute_requested_raw_with_prepared_financials(
        requested_ids,
        context,
        data,
        &prepared,
        &requested,
        snapshot_cache,
    )?;
    finalize_requested_from_raw(&requested, data, raw_series)
}

fn compute_requested_raw_with_prepared_financials(
    _requested_ids: &[String],
    _context: &FactorContext,
    data: &DataPool,
    prepared: &GaussianFinancialPrepared,
    requested: &[GaussianFinancialOutput],
    snapshot_cache: &mut InstrumentAlignedSnapshotCache<FinancialSnapshot>,
) -> Result<Vec<FactorSeries>> {
    if requested.is_empty() {
        return Ok(Vec::new());
    }
    let needs = FinancialNeeds::from_outputs(requested);
    let panel = data.daily_panel(DatasetId::StockDailyPv)?;
    let total_mv = panel.column_from_table(data.daily(DatasetId::StockDailyBasic)?, "total_mv")?;
    let columns = financial_snapshot_columns(
        &panel,
        &total_mv,
        &prepared.income,
        &prepared.balance,
        &prepared.cashflow,
        &prepared.dividends,
        needs,
        snapshot_cache,
    )?;

    let mut raw_cache = GaussianRawCache::default();
    let mut output = Vec::new();
    for kind in requested.iter().copied() {
        match kind {
            GaussianFinancialOutput::EpSq => output.push(
                raw_cache
                    .ep_sq(&columns, &panel)?
                    .to_factor_series(raw_spec(kind.id())),
            ),
            GaussianFinancialOutput::EbTwoVar => output.push(
                raw_cache
                    .eb_two_var(&columns, &panel)?
                    .to_factor_series(raw_spec(kind.id())),
            ),
            GaussianFinancialOutput::SpSq => output.push(
                raw_cache
                    .sp_sq(&columns, &panel)?
                    .to_factor_series(raw_spec(kind.id())),
            ),
            GaussianFinancialOutput::CfpSq => output.push(
                raw_cache
                    .cfp_sq(&columns, &panel)?
                    .to_factor_series(raw_spec(kind.id())),
            ),
            GaussianFinancialOutput::CashValue => output.push(
                raw_cache
                    .cash_value(&columns, &panel)?
                    .to_factor_series(raw_spec(kind.id())),
            ),
            GaussianFinancialOutput::DivTtm => output.push(
                raw_cache
                    .div_ttm(&columns, &panel)?
                    .to_factor_series(raw_spec(kind.id())),
            ),
            GaussianFinancialOutput::ProfitYoySq => output.push(
                raw_cache
                    .profit_yoy_sq(&columns, &panel)?
                    .to_factor_series(raw_spec(kind.id())),
            ),
            GaussianFinancialOutput::DeltaRoe => output.push(
                raw_cache
                    .delta_roe(&columns, &panel)?
                    .to_factor_series(raw_spec(kind.id())),
            ),
            GaussianFinancialOutput::PbRoeSpread => {
                output.push(
                    gaussian_residual(&columns.total_mv_snapshot, &[&columns.book_value])?
                        .to_factor_series(raw_spec(PB_ROE_PB_RAW_ID)),
                );
                output.push(
                    gaussian_residual(&columns.netprofit_sq, &[&columns.book_value])?
                        .to_factor_series(raw_spec(PB_ROE_ROE_RAW_ID)),
                );
            }
        }
    }
    Ok(output)
}

struct GaussianFinancialPrepared<'a> {
    income: FinancialPitReader<'a>,
    balance: FinancialPitReader<'a>,
    cashflow: FinancialPitReader<'a>,
    dividends: Vec<DividendRecord>,
}

impl<'a> GaussianFinancialPrepared<'a> {
    fn from_data(data: &'a DataPool, needs: FinancialNeeds) -> Result<Self> {
        let income = data.financial_reader(
            DatasetId::StockIncome,
            ReportTypePreference::income_single_quarter(),
        )?;
        let balance = data.financial_reader(
            DatasetId::StockBalanceSheet,
            ReportTypePreference::balance_sheet_consolidated(),
        )?;
        let cashflow = data.financial_reader(
            DatasetId::StockCashFlow,
            ReportTypePreference::income_single_quarter(),
        )?;
        let dividends = if needs.uses_dividend() {
            parse_dividend_records(data.daily(DatasetId::StockDividend)?)?
        } else {
            Vec::new()
        };
        Ok(Self {
            income,
            balance,
            cashflow,
            dividends,
        })
    }
}

fn tags(kind: GaussianFinancialOutput) -> Vec<String> {
    let mut tags = [
        "DFZQ",
        "DBZQ",
        "financial",
        "fundamental",
        "pit",
        "gaussian_rank",
        "residual",
        "neutralize",
        "barra",
        "size",
        "sector",
        "daily",
    ]
    .iter()
    .map(|value| value.to_string())
    .collect::<Vec<_>>();
    if kind.is_deprecated() {
        tags.push("deprecated".to_string());
    }
    tags
}

impl GaussianFinancialOutput {
    fn is_deprecated(self) -> bool {
        matches!(self, Self::CfpSq | Self::DeltaRoe | Self::ProfitYoySq)
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct FinancialNeeds {
    total_mv_snapshot: bool,
    income_latest: bool,
    income_yoy: bool,
    income_ttm: bool,
    balance_latest: bool,
    cashflow_latest: bool,
    dividend_ltm: bool,
}

impl FinancialNeeds {
    fn from_outputs(outputs: &[GaussianFinancialOutput]) -> Self {
        let mut needs = Self::default();
        for output in outputs {
            match output {
                GaussianFinancialOutput::EpSq | GaussianFinancialOutput::SpSq => {
                    needs.total_mv_snapshot = true;
                    needs.income_latest = true;
                }
                GaussianFinancialOutput::EbTwoVar => {
                    needs.total_mv_snapshot = true;
                    needs.income_ttm = true;
                    needs.balance_latest = true;
                }
                GaussianFinancialOutput::CfpSq => {
                    needs.total_mv_snapshot = true;
                    needs.cashflow_latest = true;
                }
                GaussianFinancialOutput::CashValue => {
                    needs.total_mv_snapshot = true;
                    needs.cashflow_latest = true;
                }
                GaussianFinancialOutput::DivTtm => {
                    needs.total_mv_snapshot = true;
                    needs.dividend_ltm = true;
                }
                GaussianFinancialOutput::ProfitYoySq => {
                    needs.income_latest = true;
                    needs.income_yoy = true;
                }
                GaussianFinancialOutput::DeltaRoe => {
                    needs.income_latest = true;
                    needs.income_yoy = true;
                    needs.balance_latest = true;
                }
                GaussianFinancialOutput::PbRoeSpread => {
                    needs.total_mv_snapshot = true;
                    needs.income_latest = true;
                    needs.balance_latest = true;
                }
            }
        }
        needs
    }

    fn uses_income(self) -> bool {
        self.income_latest || self.income_yoy || self.income_ttm
    }

    fn uses_balance(self) -> bool {
        self.balance_latest
    }

    fn uses_cashflow(self) -> bool {
        self.cashflow_latest
    }

    fn uses_dividend(self) -> bool {
        self.dividend_ltm
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct FinancialSnapshot {
    total_mv_snapshot: Option<f64>,
    netprofit_ttm: Option<f64>,
    netprofit_sq: Option<f64>,
    netprofit_sq_yoy: Option<f64>,
    revenue_sq: Option<f64>,
    cashflow_sq: Option<f64>,
    book_value: Option<f64>,
    cash_equ_end_period: Option<f64>,
    div_ttm: Option<f64>,
}

struct FinancialSnapshotColumns {
    total_mv_snapshot: PanelColumn,
    netprofit_ttm: PanelColumn,
    netprofit_sq: PanelColumn,
    netprofit_sq_yoy: PanelColumn,
    revenue_sq: PanelColumn,
    cashflow_sq: PanelColumn,
    book_value: PanelColumn,
    cash_equ_end_period: PanelColumn,
    div_ttm: PanelColumn,
    delta_profit_sq_yoy: PanelColumn,
}

fn financial_snapshot_columns(
    panel: &DailyPanel,
    total_mv: &PanelColumn,
    income: &FinancialPitReader<'_>,
    balance: &FinancialPitReader<'_>,
    cashflow: &FinancialPitReader<'_>,
    dividends: &[DividendRecord],
    needs: FinancialNeeds,
    cache: &mut InstrumentAlignedSnapshotCache<FinancialSnapshot>,
) -> Result<FinancialSnapshotColumns> {
    let mut total_mv_snapshot = vec![None; panel.shape_len()];
    let mut netprofit_ttm = vec![None; panel.shape_len()];
    let mut netprofit_sq = vec![None; panel.shape_len()];
    let mut netprofit_sq_yoy = vec![None; panel.shape_len()];
    let mut revenue_sq = vec![None; panel.shape_len()];
    let mut cashflow_sq = vec![None; panel.shape_len()];
    let mut book_value = vec![None; panel.shape_len()];
    let mut cash_equ_end_period = vec![None; panel.shape_len()];
    let mut div_ttm = vec![None; panel.shape_len()];
    let mut delta_profit_sq_yoy = vec![None; panel.shape_len()];
    let dividend_sums_by_date = panel
        .dates()
        .iter()
        .copied()
        .filter(|trade_date| panel.is_target_date(*trade_date))
        .map(|trade_date| {
            (
                trade_date,
                dividend_sum_by_stock(dividends, add_months(trade_date, -12), trade_date),
            )
        })
        .collect::<std::collections::BTreeMap<_, _>>();
    let instrument_count = panel.instruments().len();
    for (date_idx, trade_date) in panel.dates().iter().copied().enumerate() {
        if !panel.is_target_date(trade_date) {
            continue;
        }
        let snapshots = cached_financial_stock_snapshots_for_date(
            panel,
            trade_date,
            cache,
            |_, ts_code, offset| is_bj_stock(ts_code) || !panel.is_present_offset(offset),
            |trade_date, ts_code, _| {
                let cash_dividend = dividend_sums_by_date
                    .get(&trade_date)
                    .and_then(|sum| sum.get(ts_code).copied())
                    .unwrap_or(0.0);
                financial_snapshot_marker(
                    ts_code,
                    trade_date,
                    income,
                    balance,
                    cashflow,
                    cash_dividend,
                    needs,
                )
            },
            |trade_date, ts_code, offset| {
                let cash_dividend = dividend_sums_by_date
                    .get(&trade_date)
                    .and_then(|sum| sum.get(ts_code).copied())
                    .unwrap_or(0.0);
                let total_mv_value = clean(total_mv.values()[offset]).filter(|value| *value > 0.0);
                financial_snapshot_for_stock(
                    ts_code,
                    trade_date,
                    income,
                    balance,
                    cashflow,
                    cash_dividend,
                    total_mv_value,
                    needs,
                )
            },
        );
        let date_offset = date_idx * instrument_count;
        for (instrument_idx, snapshot) in snapshots.into_iter().enumerate() {
            let Some(snapshot) = snapshot else {
                continue;
            };
            let offset = date_offset + instrument_idx;
            total_mv_snapshot[offset] = snapshot.total_mv_snapshot;
            netprofit_ttm[offset] = snapshot.netprofit_ttm;
            netprofit_sq[offset] = snapshot.netprofit_sq;
            netprofit_sq_yoy[offset] = snapshot.netprofit_sq_yoy;
            revenue_sq[offset] = snapshot.revenue_sq;
            cashflow_sq[offset] = snapshot.cashflow_sq;
            book_value[offset] = snapshot.book_value;
            cash_equ_end_period[offset] = snapshot.cash_equ_end_period;
            div_ttm[offset] = snapshot.div_ttm;
            delta_profit_sq_yoy[offset] =
                diff_opt(snapshot.netprofit_sq, snapshot.netprofit_sq_yoy);
        }
    }

    Ok(FinancialSnapshotColumns {
        total_mv_snapshot: panel.column_from_values(total_mv_snapshot)?,
        netprofit_ttm: panel.column_from_values(netprofit_ttm)?,
        netprofit_sq: panel.column_from_values(netprofit_sq)?,
        netprofit_sq_yoy: panel.column_from_values(netprofit_sq_yoy)?,
        revenue_sq: panel.column_from_values(revenue_sq)?,
        cashflow_sq: panel.column_from_values(cashflow_sq)?,
        book_value: panel.column_from_values(book_value)?,
        cash_equ_end_period: panel.column_from_values(cash_equ_end_period)?,
        div_ttm: panel.column_from_values(div_ttm)?,
        delta_profit_sq_yoy: panel.column_from_values(delta_profit_sq_yoy)?,
    })
}

fn financial_snapshot_marker(
    ts_code: &str,
    trade_date: i32,
    income: &FinancialPitReader<'_>,
    balance: &FinancialPitReader<'_>,
    cashflow: &FinancialPitReader<'_>,
    cash_dividend_ltm: f64,
    needs: FinancialNeeds,
) -> Option<FinancialEventMarker> {
    let mut builder = FinancialEventMarkerBuilder::new();
    if needs.income_latest || needs.income_yoy || needs.income_ttm {
        if let Some(end_date) = income.latest_quarter_end_date(ts_code, trade_date) {
            builder.include_reader_record_for_end_date(
                FinancialStatementDataset::Income,
                income,
                ts_code,
                trade_date,
                end_date,
            );
            if needs.income_yoy {
                builder.include_reader_record_for_end_date(
                    FinancialStatementDataset::Income,
                    income,
                    ts_code,
                    trade_date,
                    same_quarter_previous_year(end_date),
                );
            }
            if needs.income_ttm {
                builder.include_reader_ttm_for_end_date(
                    FinancialStatementDataset::Income,
                    income,
                    ts_code,
                    trade_date,
                    end_date,
                );
            }
        }
    }
    if needs.balance_latest {
        builder.include_reader_latest_quarter(
            FinancialStatementDataset::BalanceSheet,
            balance,
            ts_code,
            trade_date,
        );
    }
    if needs.cashflow_latest {
        builder.include_reader_latest_quarter(
            FinancialStatementDataset::CashFlow,
            cashflow,
            ts_code,
            trade_date,
        );
    }
    if needs.dividend_ltm {
        builder.include_synthetic("div_ttm", f64_marker_value(cash_dividend_ltm));
    }
    builder.build()
}

fn financial_snapshot_for_stock(
    ts_code: &str,
    trade_date: i32,
    income: &FinancialPitReader<'_>,
    balance: &FinancialPitReader<'_>,
    cashflow: &FinancialPitReader<'_>,
    cash_dividend_ltm: f64,
    total_mv_value: Option<f64>,
    needs: FinancialNeeds,
) -> Option<FinancialSnapshot> {
    let mut snapshot = FinancialSnapshot {
        total_mv_snapshot: needs.total_mv_snapshot.then_some(total_mv_value).flatten(),
        div_ttm: needs.dividend_ltm.then_some(cash_dividend_ltm),
        ..FinancialSnapshot::default()
    };

    if needs.income_latest || needs.income_yoy || needs.income_ttm {
        if let Some(end_date) = income.latest_quarter_end_date(ts_code, trade_date) {
            if needs.income_latest {
                snapshot.netprofit_sq =
                    financial_value(income, ts_code, trade_date, end_date, "n_income_attr_p");
                snapshot.revenue_sq =
                    financial_value(income, ts_code, trade_date, end_date, "revenue");
            }
            if needs.income_yoy {
                let yoy_end_date = same_quarter_previous_year(end_date);
                snapshot.netprofit_sq_yoy =
                    financial_value(income, ts_code, trade_date, yoy_end_date, "n_income_attr_p");
            }
            if needs.income_ttm {
                snapshot.netprofit_ttm =
                    income.ttm_sum_for_end_date(ts_code, trade_date, end_date, "n_income_attr_p");
            }
        }
    }
    if needs.balance_latest {
        if let Some(end_date) = balance.latest_quarter_end_date(ts_code, trade_date) {
            snapshot.book_value = financial_value(
                balance,
                ts_code,
                trade_date,
                end_date,
                "total_hldr_eqy_exc_min_int",
            );
        }
    }
    if needs.cashflow_latest {
        if let Some(end_date) = cashflow.latest_quarter_end_date(ts_code, trade_date) {
            snapshot.cashflow_sq =
                financial_value(cashflow, ts_code, trade_date, end_date, "n_cashflow_act");
            snapshot.cash_equ_end_period = financial_value(
                cashflow,
                ts_code,
                trade_date,
                end_date,
                "c_cash_equ_end_period",
            );
        }
    }
    Some(snapshot)
}

fn financial_value(
    data: &FinancialPitReader<'_>,
    ts_code: &str,
    trade_date: i32,
    end_date: i32,
    column: &str,
) -> Option<f64> {
    data.record_for_end_date(ts_code, trade_date, end_date)?
        .column(column)
}

#[derive(Default)]
struct GaussianRawCache {
    ep_sq: Option<PanelColumn>,
    eb_two_var: Option<PanelColumn>,
    sp_sq: Option<PanelColumn>,
    cfp_sq: Option<PanelColumn>,
    cash_value: Option<PanelColumn>,
    div_ttm: Option<PanelColumn>,
    profit_yoy_sq: Option<PanelColumn>,
    delta_roe: Option<PanelColumn>,
}

impl GaussianRawCache {
    fn ep_sq<'a>(
        &'a mut self,
        columns: &FinancialSnapshotColumns,
        _panel: &DailyPanel,
    ) -> Result<&'a PanelColumn> {
        if self.ep_sq.is_none() {
            self.ep_sq = Some(gaussian_residual(
                &columns.netprofit_sq,
                &[&columns.total_mv_snapshot],
            )?);
        }
        Ok(self.ep_sq.as_ref().unwrap())
    }

    fn eb_two_var<'a>(
        &'a mut self,
        columns: &FinancialSnapshotColumns,
        panel: &DailyPanel,
    ) -> Result<&'a PanelColumn> {
        if self.eb_two_var.is_none() {
            self.eb_two_var = Some(gaussian_residual_orthogonal_two_x(
                &columns.total_mv_snapshot,
                &columns.netprofit_ttm,
                &columns.book_value,
                panel,
            )?);
        }
        Ok(self.eb_two_var.as_ref().unwrap())
    }

    fn sp_sq<'a>(
        &'a mut self,
        columns: &FinancialSnapshotColumns,
        _panel: &DailyPanel,
    ) -> Result<&'a PanelColumn> {
        if self.sp_sq.is_none() {
            self.sp_sq = Some(gaussian_residual(
                &columns.revenue_sq,
                &[&columns.total_mv_snapshot],
            )?);
        }
        Ok(self.sp_sq.as_ref().unwrap())
    }

    fn cfp_sq<'a>(
        &'a mut self,
        columns: &FinancialSnapshotColumns,
        _panel: &DailyPanel,
    ) -> Result<&'a PanelColumn> {
        if self.cfp_sq.is_none() {
            self.cfp_sq = Some(gaussian_residual(
                &columns.cashflow_sq,
                &[&columns.total_mv_snapshot],
            )?);
        }
        Ok(self.cfp_sq.as_ref().unwrap())
    }

    fn cash_value<'a>(
        &'a mut self,
        columns: &FinancialSnapshotColumns,
        _panel: &DailyPanel,
    ) -> Result<&'a PanelColumn> {
        if self.cash_value.is_none() {
            self.cash_value = Some(gaussian_residual(
                &columns.cash_equ_end_period,
                &[&columns.total_mv_snapshot],
            )?);
        }
        Ok(self.cash_value.as_ref().unwrap())
    }

    fn div_ttm<'a>(
        &'a mut self,
        columns: &FinancialSnapshotColumns,
        _panel: &DailyPanel,
    ) -> Result<&'a PanelColumn> {
        if self.div_ttm.is_none() {
            self.div_ttm = Some(gaussian_residual(
                &columns.div_ttm,
                &[&columns.total_mv_snapshot],
            )?);
        }
        Ok(self.div_ttm.as_ref().unwrap())
    }

    fn profit_yoy_sq<'a>(
        &'a mut self,
        columns: &FinancialSnapshotColumns,
        _panel: &DailyPanel,
    ) -> Result<&'a PanelColumn> {
        if self.profit_yoy_sq.is_none() {
            self.profit_yoy_sq = Some(gaussian_residual(
                &columns.netprofit_sq,
                &[&columns.netprofit_sq_yoy],
            )?);
        }
        Ok(self.profit_yoy_sq.as_ref().unwrap())
    }

    fn delta_roe<'a>(
        &'a mut self,
        columns: &FinancialSnapshotColumns,
        _panel: &DailyPanel,
    ) -> Result<&'a PanelColumn> {
        if self.delta_roe.is_none() {
            self.delta_roe = Some(gaussian_residual(
                &columns.delta_profit_sq_yoy,
                &[&columns.book_value],
            )?);
        }
        Ok(self.delta_roe.as_ref().unwrap())
    }
}

pub fn gaussian_residual(y: &PanelColumn, xs: &[&PanelColumn]) -> Result<PanelColumn> {
    let ranked_y = y.cs(cs_gaussian_rank)?;
    let ranked_xs = xs
        .iter()
        .map(|column| column.cs(cs_gaussian_rank))
        .collect::<Result<Vec<_>>>()?;
    let refs = ranked_xs.iter().collect::<Vec<_>>();
    ranked_y.cs_neutralize_regression(&refs, None)
}

fn gaussian_residual_orthogonal_two_x(
    y: &PanelColumn,
    x1: &PanelColumn,
    x2: &PanelColumn,
    _panel: &DailyPanel,
) -> Result<PanelColumn> {
    let ranked_y = y.cs(cs_gaussian_rank)?;
    let ranked_x1 = x1.cs(cs_gaussian_rank)?;
    let ranked_x2 = x2.cs(cs_gaussian_rank)?;
    ranked_y.cs_ternary(&ranked_x1, &ranked_x2, |y, x1, x2| {
        orthogonal_two_x_residual(y, x1, x2)
    })
}

pub fn cs_gaussian_rank(values: &[Option<f64>]) -> Vec<Option<f64>> {
    let mut pairs = values
        .iter()
        .enumerate()
        .filter_map(|(idx, value)| clean(*value).map(|value| (idx, value)))
        .collect::<Vec<_>>();
    if pairs.len() < 2 {
        return vec![None; values.len()];
    }
    pairs.sort_by(|left, right| {
        left.1
            .total_cmp(&right.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    let n = pairs.len() as f64;
    let mut output = vec![None; values.len()];
    for (rank_idx, (idx, _)) in pairs.into_iter().enumerate() {
        let p = (((rank_idx + 1) as f64 - 0.5) / n).clamp(GAUSSIAN_P_EPS, 1.0 - GAUSSIAN_P_EPS);
        output[idx] = inverse_standard_normal(p);
    }
    output
}

fn orthogonal_two_x_residual(
    y: &[Option<f64>],
    x1: &[Option<f64>],
    x2: &[Option<f64>],
) -> Vec<Option<f64>> {
    let mut dot = 0.0;
    let mut norm = 0.0;
    for idx in 0..y.len() {
        let (Some(_), Some(left), Some(right)) = (clean(y[idx]), clean(x1[idx]), clean(x2[idx]))
        else {
            continue;
        };
        dot += left * right;
        norm += left * left;
    }
    if norm <= f64::EPSILON {
        return vec![None; y.len()];
    }
    let beta = dot / norm;
    let x2_orth = x1
        .iter()
        .zip(x2)
        .map(|(left, right)| match (clean(*left), clean(*right)) {
            (Some(left), Some(right)) => Some(right - beta * left),
            _ => None,
        })
        .collect::<Vec<_>>();
    cs_neutralize_regression(y, &[x1, &x2_orth], None, None)
}

fn neutralize_sector_only_with_map(
    values: &PanelColumn,
    panel: &DailyPanel,
    sector_map: &ClassificationMap,
) -> Result<PanelColumn> {
    let masked = mask_bj(values, panel)?;
    masked.cs_neutralize_regression_by_group(&[], None, |trade_date, ts_codes| {
        sector_map.groups_for(trade_date, ts_codes)
    })
}

fn pb_roe_spread_from_raw(
    pb_raw: &PanelColumn,
    roe_raw: &PanelColumn,
    panel: &DailyPanel,
    size: &PanelColumn,
    sector_map: &ClassificationMap,
) -> Result<PanelColumn> {
    let pb_neutralized = neutralize_size_sector_with_inputs(pb_raw, panel, size, sector_map)?;
    let roe_neutralized = neutralize_size_sector_with_inputs(roe_raw, panel, size, sector_map)?;
    let pb_rank = pb_neutralized.cs(|values| cs_pctrank(values, true))?;
    let roe_rank = roe_neutralized.cs(|values| cs_pctrank(values, true))?;
    pb_rank.zip_binary(&roe_rank, |pb, roe| match (clean(pb), clean(roe)) {
        (Some(pb), Some(roe)) => Some(pb - roe),
        _ => None,
    })
}

fn diff_opt(left: Option<f64>, right: Option<f64>) -> Option<f64> {
    let value = left? - right?;
    value.is_finite().then_some(value)
}

fn same_quarter_previous_year(end_date: i32) -> i32 {
    (end_date / 10_000 - 1) * 10_000 + end_date % 10_000
}

fn f64_marker_value(value: f64) -> i64 {
    i64::from_ne_bytes(value.to_bits().to_ne_bytes())
}

#[derive(Clone, Debug)]
struct DividendRecord {
    ts_code: String,
    ann_date: i32,
    ex_date: i32,
    cash_div_tax: f64,
    base_share: f64,
    implemented: bool,
}

fn parse_dividend_records(table: &Table) -> Result<Vec<DividendRecord>> {
    let ts_codes = table.required_utf8("ts_code")?;
    let ann_dates = table.required_i32_date_cast("ann_date")?;
    let div_proc = table.required_utf8("div_proc")?;
    let cash_div_tax = table.required_f64_cast("cash_div_tax")?;
    let ex_dates = table.required_i32_date_cast("ex_date")?;
    let base_share = table.required_f64_cast("base_share")?;

    let mut records = Vec::new();
    for idx in 0..table.len {
        let (Some(ts_code), Some(ann_date), Some(ex_date), Some(cash_div_tax), Some(base_share)) = (
            ts_codes[idx].clone(),
            ann_dates[idx],
            ex_dates[idx],
            clean(cash_div_tax[idx]),
            clean(base_share[idx]).filter(|value| *value > 0.0),
        ) else {
            continue;
        };
        records.push(DividendRecord {
            ts_code,
            ann_date,
            ex_date,
            cash_div_tax,
            base_share,
            implemented: div_proc[idx]
                .as_deref()
                .is_some_and(|value| value.trim() == IMPLEMENTED_DIV_PROC),
        });
    }
    Ok(records)
}

fn dividend_sum_by_stock(
    records: &[DividendRecord],
    start_date: i32,
    trade_date: i32,
) -> HashMap<&str, f64> {
    let mut sums = HashMap::new();
    for record in records {
        if !record.implemented
            || record.ann_date > trade_date
            || record.ex_date > trade_date
            || record.ex_date < start_date
        {
            continue;
        }
        *sums.entry(record.ts_code.as_str()).or_default() +=
            record.cash_div_tax * record.base_share;
    }
    sums
}

fn add_months(date: i32, months_delta: i32) -> i32 {
    let (year, month, day) = ymd(date);
    let month_index = year * 12 + month as i32 - 1 + months_delta;
    let new_year = month_index.div_euclid(12);
    let new_month = month_index.rem_euclid(12) + 1;
    let new_day = day.min(days_in_month(new_year, new_month as u32));
    new_year * 10_000 + new_month * 100 + new_day as i32
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

// Acklam's rational approximation for the inverse standard normal CDF.
fn inverse_standard_normal(p: f64) -> Option<f64> {
    if !(0.0..=1.0).contains(&p) || p <= 0.0 || p >= 1.0 {
        return None;
    }

    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_69e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];
    const P_LOW: f64 = 0.02425;
    const P_HIGH: f64 = 1.0 - P_LOW;

    let value = if p < P_LOW {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= P_HIGH {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    };
    value.is_finite().then_some(value)
}

#[cfg(test)]
mod tests {
    use crate::factor::common::financial::previous_quarter_end_date;

    use super::*;

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1e-10,
            "actual={actual}, expected={expected}"
        );
    }

    #[test]
    fn gaussian_financial_rank_uses_inverse_normal_midpoint_rank() {
        let ranked = cs_gaussian_rank(&[Some(2.0), Some(1.0), None, Some(3.0)]);

        assert!(ranked[1].unwrap() < 0.0);
        assert_close(ranked[0].unwrap(), 0.0);
        assert!(ranked[3].unwrap() > 0.0);
        assert_eq!(ranked[2], None);
    }

    #[test]
    fn gaussian_financial_rank_single_valid_sample_is_empty() {
        assert_eq!(cs_gaussian_rank(&[None, Some(1.0)]), vec![None, None]);
    }

    #[test]
    fn gaussian_financial_orthogonalizes_second_regressor() {
        let y = vec![Some(-1.0), Some(0.0), Some(1.0), Some(2.0)];
        let x1 = vec![Some(-1.0), Some(0.0), Some(1.0), Some(2.0)];
        let x2 = vec![Some(-2.0), Some(1.0), Some(2.0), Some(5.0)];
        let residual = orthogonal_two_x_residual(&y, &x1, &x2);

        for value in residual {
            assert!(value.unwrap().abs() < 1e-10);
        }
    }

    #[test]
    fn gaussian_financial_dividend_uses_only_implemented_visible_records() {
        let records = vec![
            DividendRecord {
                ts_code: "000001.SZ".to_string(),
                ann_date: 20260101,
                ex_date: 20260301,
                cash_div_tax: 0.2,
                base_share: 100.0,
                implemented: true,
            },
            DividendRecord {
                ts_code: "000001.SZ".to_string(),
                ann_date: 20260101,
                ex_date: 20260302,
                cash_div_tax: 0.3,
                base_share: 100.0,
                implemented: false,
            },
            DividendRecord {
                ts_code: "000001.SZ".to_string(),
                ann_date: 20270101,
                ex_date: 20260301,
                cash_div_tax: 0.4,
                base_share: 100.0,
                implemented: true,
            },
        ];
        let sums = dividend_sum_by_stock(&records, 20250424, 20260424);

        assert_close(*sums.get("000001.SZ").unwrap(), 20.0);
    }

    #[test]
    fn gaussian_financial_specs_have_dfzq_and_dbzq_tags() {
        let spec = spec(GaussianFinancialOutput::EpSq);

        assert!(spec.tags.contains(&"DFZQ".to_string()));
        assert!(spec.tags.contains(&"DBZQ".to_string()));
        assert_eq!(spec.id, "ep_sq_gauss_resid");
    }

    #[test]
    fn gaussian_financial_event_sources_follow_requested_outputs() {
        let ep_needs = FinancialNeeds::from_outputs(&[GaussianFinancialOutput::EpSq]);
        assert!(ep_needs.uses_income());
        assert!(!ep_needs.uses_balance());
        assert!(!ep_needs.uses_cashflow());
        assert!(!ep_needs.uses_dividend());

        let cash_needs = FinancialNeeds::from_outputs(&[GaussianFinancialOutput::CashValue]);
        assert!(!cash_needs.uses_income());
        assert!(!cash_needs.uses_balance());
        assert!(cash_needs.uses_cashflow());
        assert!(!cash_needs.uses_dividend());

        let div_needs = FinancialNeeds::from_outputs(&[GaussianFinancialOutput::DivTtm]);
        assert!(!div_needs.uses_income());
        assert!(!div_needs.uses_balance());
        assert!(!div_needs.uses_cashflow());
        assert!(div_needs.uses_dividend());
    }

    #[test]
    fn gaussian_financial_same_quarter_previous_year_preserves_quarter() {
        assert_eq!(same_quarter_previous_year(20250331), 20240331);
        assert_eq!(same_quarter_previous_year(20251231), 20241231);
        assert_eq!(previous_quarter_end_date(20250331), Some(20241231));
    }
}

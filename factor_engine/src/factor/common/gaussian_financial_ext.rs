use std::any::Any;
use std::collections::HashMap;

use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::{err, Result};
use crate::factor::common::gaussian_financial::gaussian_residual;
use crate::factor::common::stock_daily_ops::{
    is_bj_stock, mask_bj, neutralize_size_sector_with_inputs,
};
use crate::factor::common::vector::clean;
use crate::factor::common::{
    cached_financial_stock_snapshots_for_date, compute_financial_event_snapshot_streaming_on_panel,
    factor_series_to_panel_column, ClassificationLevel, ClassificationMap, DailyPanel,
    EventDrivenCrossSectionCache, FinancialEventMarker, FinancialEventMarkerBuilder,
    FinancialEventSchedule, FinancialPitReader, FinancialStatementDataset,
    InstrumentAlignedSnapshotCache, PanelColumn, ReportTypePreference,
};
use crate::factor::{Factor, FactorUpdatePolicy};

pub const PROVIDER_KEY: &str = "stock|daily|gaussian_financial_ext";

const VERSION: &str = "0.1.0";
const LOOKBACK: usize = 252;
const FINANCIAL_QUARTERS: usize = 8;

const INCOME_COLUMNS: [&str; 9] = [
    "n_income",
    "income_tax",
    "int_exp",
    "t_compr_income",
    "n_oth_income",
    "n_income_attr_p",
    "compr_inc_attr_p",
    "total_cogs",
    "biz_tax_surchg",
];
const BALANCE_COLUMNS: [&str; 8] = [
    "undistr_porfit",
    "total_assets",
    "payroll_payable",
    "oth_pay_total",
    "cip_total",
    "total_cur_liab",
    "surplus_rese",
    "oth_comp_income",
];
const CASHFLOW_COLUMNS: [&str; 4] = [
    "stot_out_inv_act",
    "c_pay_acq_const_fiolta",
    "c_inf_fr_operate_a",
    "c_cash_equ_end_period",
];

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum GaussianFinancialExtOutput {
    EbitYoyChgMv,
    TComprIncomeMv,
    UndistrProfitYoyChgMv,
    NOthIncomeMv,
    TotalAssetsMv,
    InvCashOutMv,
    PayrollPayableOthPayYoy,
    IncomeTaxEbitYoy,
    CapexCipYoy,
    NetprofitComprParent,
    TotalCogsCurLiabYoy,
    BizTaxSurchgAssets,
    SurplusReseIntExpYoy,
    UndistrProfitYoy,
    OperCashInYoy,
    CashEquOthCompIncomeYoy,
}

impl GaussianFinancialExtOutput {
    pub fn id(self) -> &'static str {
        match self {
            Self::EbitYoyChgMv => "ebit_yoy_chg_mv_gauss_resid",
            Self::TComprIncomeMv => "t_compr_income_mv_gauss_resid",
            Self::UndistrProfitYoyChgMv => "undistr_profit_yoy_chg_mv_gauss_resid",
            Self::NOthIncomeMv => "n_oth_income_mv_gauss_resid",
            Self::TotalAssetsMv => "total_assets_mv_gauss_resid",
            Self::InvCashOutMv => "inv_cash_out_mv_gauss_resid",
            Self::PayrollPayableOthPayYoy => "payroll_payable_oth_pay_yoy_gauss_resid",
            Self::IncomeTaxEbitYoy => "income_tax_ebit_yoy_gauss_resid",
            Self::CapexCipYoy => "capex_cip_yoy_gauss_resid",
            Self::NetprofitComprParent => "netprofit_compr_parent_gauss_resid",
            Self::TotalCogsCurLiabYoy => "total_cogs_cur_liab_yoy_gauss_resid",
            Self::BizTaxSurchgAssets => "biz_tax_surchg_assets_gauss_resid",
            Self::SurplusReseIntExpYoy => "surplus_rese_int_exp_yoy_gauss_resid",
            Self::UndistrProfitYoy => "undistr_profit_yoy_gauss_resid",
            Self::OperCashInYoy => "oper_cash_in_yoy_gauss_resid",
            Self::CashEquOthCompIncomeYoy => "cash_equ_oth_comp_income_yoy_gauss_resid",
        }
    }

    fn alias(self) -> &'static str {
        match self {
            Self::EbitYoyChgMv => "EBIT YoY Change Market Cap Gaussian Residual",
            Self::TComprIncomeMv => "Total Comprehensive Income Market Cap Gaussian Residual",
            Self::UndistrProfitYoyChgMv => {
                "Undistributed Profit YoY Change Market Cap Gaussian Residual"
            }
            Self::NOthIncomeMv => "Other Income Market Cap Gaussian Residual",
            Self::TotalAssetsMv => "Total Assets Market Cap Gaussian Residual",
            Self::InvCashOutMv => "Investment Cash Outflow Market Cap Gaussian Residual",
            Self::PayrollPayableOthPayYoy => "Payroll Payable Other Payables YoY Gaussian Residual",
            Self::IncomeTaxEbitYoy => "Income Tax Prior Year EBIT Gaussian Residual",
            Self::CapexCipYoy => "Capex Prior Year CIP Gaussian Residual",
            Self::NetprofitComprParent => {
                "Net Profit Parent Comprehensive Income Parent Gaussian Residual"
            }
            Self::TotalCogsCurLiabYoy => {
                "Total COGS Prior Year Current Liabilities Gaussian Residual"
            }
            Self::BizTaxSurchgAssets => "Business Tax Surcharge Total Assets Gaussian Residual",
            Self::SurplusReseIntExpYoy => {
                "Surplus Reserve Prior Year Interest Expense Gaussian Residual"
            }
            Self::UndistrProfitYoy => "Undistributed Profit YoY Gaussian Residual",
            Self::OperCashInYoy => "Operating Cash Inflow YoY Gaussian Residual",
            Self::CashEquOthCompIncomeYoy => {
                "Cash Equivalent Other Comprehensive Income YoY Gaussian Residual"
            }
        }
    }

    fn from_id(id: &str) -> Option<Self> {
        Some(match id {
            "ebit_yoy_chg_mv_gauss_resid" => Self::EbitYoyChgMv,
            "t_compr_income_mv_gauss_resid" => Self::TComprIncomeMv,
            "undistr_profit_yoy_chg_mv_gauss_resid" => Self::UndistrProfitYoyChgMv,
            "n_oth_income_mv_gauss_resid" => Self::NOthIncomeMv,
            "total_assets_mv_gauss_resid" => Self::TotalAssetsMv,
            "inv_cash_out_mv_gauss_resid" => Self::InvCashOutMv,
            "payroll_payable_oth_pay_yoy_gauss_resid" => Self::PayrollPayableOthPayYoy,
            "income_tax_ebit_yoy_gauss_resid" => Self::IncomeTaxEbitYoy,
            "capex_cip_yoy_gauss_resid" => Self::CapexCipYoy,
            "netprofit_compr_parent_gauss_resid" => Self::NetprofitComprParent,
            "total_cogs_cur_liab_yoy_gauss_resid" => Self::TotalCogsCurLiabYoy,
            "biz_tax_surchg_assets_gauss_resid" => Self::BizTaxSurchgAssets,
            "surplus_rese_int_exp_yoy_gauss_resid" => Self::SurplusReseIntExpYoy,
            "undistr_profit_yoy_gauss_resid" => Self::UndistrProfitYoy,
            "oper_cash_in_yoy_gauss_resid" => Self::OperCashInYoy,
            "cash_equ_oth_comp_income_yoy_gauss_resid" => Self::CashEquOthCompIncomeYoy,
            _ => return None,
        })
    }

    fn uses_total_mv_regression(self) -> bool {
        matches!(
            self,
            Self::EbitYoyChgMv
                | Self::TComprIncomeMv
                | Self::UndistrProfitYoyChgMv
                | Self::NOthIncomeMv
                | Self::TotalAssetsMv
                | Self::InvCashOutMv
        )
    }
}

pub struct GaussianFinancialExtFactor {
    kind: GaussianFinancialExtOutput,
}

#[derive(Default)]
struct GaussianFinancialExtComputeState {
    raw_cache: EventDrivenCrossSectionCache,
    snapshot_cache: InstrumentAlignedSnapshotCache<FinancialExtSnapshot>,
}

impl GaussianFinancialExtFactor {
    pub fn new(kind: GaussianFinancialExtOutput) -> Self {
        Self { kind }
    }
}

impl Factor for GaussianFinancialExtFactor {
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
                    "gaussian financial ext provider did not return {}",
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
        Box::new(GaussianFinancialExtComputeState::default())
    }

    fn compute_many_stateful(
        &self,
        requested_ids: &[String],
        context: &FactorContext,
        data: &DataPool,
        state: &mut (dyn Any + Send),
    ) -> Result<Vec<FactorSeries>> {
        let state = state
            .downcast_mut::<GaussianFinancialExtComputeState>()
            .ok_or_else(|| err("gaussian financial ext provider received incompatible state"))?;
        let mut requested = requested_ids
            .iter()
            .filter_map(|id| GaussianFinancialExtOutput::from_id(id))
            .collect::<Vec<_>>();
        requested.sort();
        requested.dedup();
        if requested.is_empty() {
            return Ok(Vec::new());
        }
        let prepared = GaussianFinancialExtPrepared::from_data(data)?;
        let schedule = FinancialEventSchedule::from_pit_readers(&[
            prepared.income.clone(),
            prepared.balance.clone(),
            prepared.cashflow.clone(),
        ]);
        let raw_specs = raw_specs_for_requested(&requested);
        let panel = data.stock_universe_panel()?;
        let raw_series = compute_financial_event_snapshot_streaming_on_panel(
            requested_ids,
            context,
            data,
            panel,
            &mut state.raw_cache,
            &schedule,
            &raw_specs,
            |requested_ids, context, data| {
                compute_requested_raw_with_prepared_financials(
                    requested_ids,
                    context,
                    data,
                    &prepared,
                    &requested,
                    &mut state.snapshot_cache,
                )
            },
        )?;
        finalize_requested_from_raw(&requested, data, raw_series)
    }
}

pub fn spec(kind: GaussianFinancialExtOutput) -> FactorSpec {
    FactorSpec {
        id: kind.id().to_string(),
        aliases: vec![kind.alias().to_string()],
        name: kind.id().to_string(),
        asset_class: AssetClass::Stock,
        frequency: Frequency::Daily,
        version: VERSION.to_string(),
        tags: tags(),
        description: format!(
            "Gaussian-rank financial regression residual factor {}. It uses PIT single-quarter financial snapshots, Gaussian-rank transforms both sides, runs cross-sectional OLS residualization, and excludes BJ stocks.",
            kind.id()
        ),
        dependencies: vec![
            DataRequest::new(DatasetId::StockDailyBasic, &["total_mv"]),
            DataRequest::financial_quarters(DatasetId::StockIncome, &INCOME_COLUMNS, FINANCIAL_QUARTERS),
            DataRequest::financial_quarters(DatasetId::StockBalanceSheet, &BALANCE_COLUMNS, FINANCIAL_QUARTERS),
            DataRequest::financial_quarters(DatasetId::StockCashFlow, &CASHFLOW_COLUMNS, FINANCIAL_QUARTERS),
            DataRequest::new(DatasetId::StockBarraDaily, &["SIZE"]),
            DataRequest::new(DatasetId::StockSwClassification, &["l1_code"]),
        ],
        intraday_raw_dependencies: Vec::new(),
        lookback: Lookback { trading_days: LOOKBACK },
    }
}

fn tags() -> Vec<String> {
    [
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
    .collect()
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
        description: format!("Internal Gaussian financial ext raw series {id}."),
        dependencies: Vec::new(),
        intraday_raw_dependencies: Vec::new(),
        lookback: Lookback { trading_days: 0 },
    }
}

fn raw_specs_for_requested(requested: &[GaussianFinancialExtOutput]) -> Vec<FactorSpec> {
    requested.iter().map(|kind| raw_spec(kind.id())).collect()
}

fn finalize_requested_from_raw(
    requested: &[GaussianFinancialExtOutput],
    data: &DataPool,
    raw_series: Vec<FactorSeries>,
) -> Result<Vec<FactorSeries>> {
    let panel = data.stock_universe_panel()?;
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
        let raw = raw_column_by_id(&raw_by_id, kind.id(), panel)?;
        let factor = if kind.uses_total_mv_regression() {
            neutralize_sector_only_with_map(&raw, panel, &sector_map)?
        } else {
            let masked = mask_bj(&raw, panel)?;
            neutralize_size_sector_with_inputs(&masked, panel, &size, &sector_map)?
        };
        output.push(factor.to_factor_series(spec(kind)));
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
        .ok_or_else(|| err(format!("gaussian financial ext raw series missing: {id}")))?;
    factor_series_to_panel_column(panel, series)
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
    snapshot_cache: &mut InstrumentAlignedSnapshotCache<FinancialExtSnapshot>,
) -> Result<Vec<FactorSeries>> {
    let mut requested = requested_ids
        .iter()
        .filter_map(|id| GaussianFinancialExtOutput::from_id(id))
        .collect::<Vec<_>>();
    requested.sort();
    requested.dedup();
    if requested.is_empty() {
        return Ok(Vec::new());
    }
    let prepared = GaussianFinancialExtPrepared::from_data(data)?;
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
    prepared: &GaussianFinancialExtPrepared,
    requested: &[GaussianFinancialExtOutput],
    snapshot_cache: &mut InstrumentAlignedSnapshotCache<FinancialExtSnapshot>,
) -> Result<Vec<FactorSeries>> {
    if requested.is_empty() {
        return Ok(Vec::new());
    }
    let panel = data.stock_universe_panel()?;
    let total_mv = panel.column_from_table(data.daily(DatasetId::StockDailyBasic)?, "total_mv")?;
    let columns = financial_ext_snapshot_columns(
        panel,
        &total_mv,
        &prepared.income,
        &prepared.balance,
        &prepared.cashflow,
        snapshot_cache,
    )?;
    let mut output = Vec::new();
    for kind in requested.iter().copied() {
        let raw = match kind {
            GaussianFinancialExtOutput::EbitYoyChgMv => {
                gaussian_residual(&columns.ebit_yoy_change, &[&columns.total_mv_snapshot])?
            }
            GaussianFinancialExtOutput::TComprIncomeMv => {
                gaussian_residual(&columns.t_compr_income, &[&columns.total_mv_snapshot])?
            }
            GaussianFinancialExtOutput::UndistrProfitYoyChgMv => gaussian_residual(
                &columns.undistr_profit_yoy_change,
                &[&columns.total_mv_snapshot],
            )?,
            GaussianFinancialExtOutput::NOthIncomeMv => {
                gaussian_residual(&columns.n_oth_income, &[&columns.total_mv_snapshot])?
            }
            GaussianFinancialExtOutput::TotalAssetsMv => {
                gaussian_residual(&columns.total_assets, &[&columns.total_mv_snapshot])?
            }
            GaussianFinancialExtOutput::InvCashOutMv => {
                gaussian_residual(&columns.inv_cash_out, &[&columns.total_mv_snapshot])?
            }
            GaussianFinancialExtOutput::PayrollPayableOthPayYoy => {
                gaussian_residual(&columns.payroll_payable, &[&columns.oth_pay_total_yoy])?
            }
            GaussianFinancialExtOutput::IncomeTaxEbitYoy => {
                gaussian_residual(&columns.income_tax, &[&columns.ebit_yoy])?
            }
            GaussianFinancialExtOutput::CapexCipYoy => {
                gaussian_residual(&columns.capex, &[&columns.cip_total_yoy])?
            }
            GaussianFinancialExtOutput::NetprofitComprParent => {
                gaussian_residual(&columns.netprofit_parent, &[&columns.compr_income_parent])?
            }
            GaussianFinancialExtOutput::TotalCogsCurLiabYoy => {
                gaussian_residual(&columns.total_cogs, &[&columns.total_cur_liab_yoy])?
            }
            GaussianFinancialExtOutput::BizTaxSurchgAssets => {
                gaussian_residual(&columns.biz_tax_surchg, &[&columns.total_assets])?
            }
            GaussianFinancialExtOutput::SurplusReseIntExpYoy => {
                gaussian_residual(&columns.surplus_rese, &[&columns.int_exp_yoy])?
            }
            GaussianFinancialExtOutput::UndistrProfitYoy => {
                gaussian_residual(&columns.undistr_profit, &[&columns.undistr_profit_yoy])?
            }
            GaussianFinancialExtOutput::OperCashInYoy => {
                gaussian_residual(&columns.oper_cash_in, &[&columns.oper_cash_in_yoy])?
            }
            GaussianFinancialExtOutput::CashEquOthCompIncomeYoy => gaussian_residual(
                &columns.cash_equ_end_period,
                &[&columns.oth_comp_income_yoy],
            )?,
        };
        output.push(raw.to_factor_series(raw_spec(kind.id())));
    }
    Ok(output)
}

struct GaussianFinancialExtPrepared<'a> {
    income: FinancialPitReader<'a>,
    balance: FinancialPitReader<'a>,
    cashflow: FinancialPitReader<'a>,
}

impl<'a> GaussianFinancialExtPrepared<'a> {
    fn from_data(data: &'a DataPool) -> Result<Self> {
        Ok(Self {
            income: data.financial_reader(
                DatasetId::StockIncome,
                ReportTypePreference::income_single_quarter(),
            )?,
            balance: data.financial_reader(
                DatasetId::StockBalanceSheet,
                ReportTypePreference::balance_sheet_consolidated(),
            )?,
            cashflow: data.financial_reader(
                DatasetId::StockCashFlow,
                ReportTypePreference::income_single_quarter(),
            )?,
        })
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct FinancialExtSnapshot {
    total_mv_snapshot: Option<f64>,
    ebit: Option<f64>,
    ebit_yoy: Option<f64>,
    t_compr_income: Option<f64>,
    n_oth_income: Option<f64>,
    income_tax: Option<f64>,
    netprofit_parent: Option<f64>,
    compr_income_parent: Option<f64>,
    total_cogs: Option<f64>,
    biz_tax_surchg: Option<f64>,
    total_assets: Option<f64>,
    undistr_profit: Option<f64>,
    undistr_profit_yoy: Option<f64>,
    payroll_payable: Option<f64>,
    oth_pay_total_yoy: Option<f64>,
    cip_total_yoy: Option<f64>,
    total_cur_liab_yoy: Option<f64>,
    surplus_rese: Option<f64>,
    oth_comp_income_yoy: Option<f64>,
    inv_cash_out: Option<f64>,
    capex: Option<f64>,
    oper_cash_in: Option<f64>,
    oper_cash_in_yoy: Option<f64>,
    cash_equ_end_period: Option<f64>,
    int_exp_yoy: Option<f64>,
}

struct FinancialExtSnapshotColumns {
    total_mv_snapshot: PanelColumn,
    ebit_yoy_change: PanelColumn,
    ebit_yoy: PanelColumn,
    t_compr_income: PanelColumn,
    n_oth_income: PanelColumn,
    income_tax: PanelColumn,
    netprofit_parent: PanelColumn,
    compr_income_parent: PanelColumn,
    total_cogs: PanelColumn,
    biz_tax_surchg: PanelColumn,
    total_assets: PanelColumn,
    undistr_profit: PanelColumn,
    undistr_profit_yoy: PanelColumn,
    undistr_profit_yoy_change: PanelColumn,
    payroll_payable: PanelColumn,
    oth_pay_total_yoy: PanelColumn,
    cip_total_yoy: PanelColumn,
    total_cur_liab_yoy: PanelColumn,
    surplus_rese: PanelColumn,
    oth_comp_income_yoy: PanelColumn,
    inv_cash_out: PanelColumn,
    capex: PanelColumn,
    oper_cash_in: PanelColumn,
    oper_cash_in_yoy: PanelColumn,
    cash_equ_end_period: PanelColumn,
    int_exp_yoy: PanelColumn,
}

fn financial_ext_snapshot_columns(
    panel: &DailyPanel,
    total_mv: &PanelColumn,
    income: &FinancialPitReader<'_>,
    balance: &FinancialPitReader<'_>,
    cashflow: &FinancialPitReader<'_>,
    cache: &mut InstrumentAlignedSnapshotCache<FinancialExtSnapshot>,
) -> Result<FinancialExtSnapshotColumns> {
    let mut snapshots_by_offset = vec![None; panel.shape_len()];
    for trade_date in panel.dates().iter().copied() {
        if !panel.is_target_date(trade_date) {
            continue;
        }
        let total_mv_values = total_mv.values();
        let daily_snapshots = cached_financial_stock_snapshots_for_date(
            panel,
            trade_date,
            cache,
            |_, ts_code, offset| !panel.is_present_offset(offset) || is_bj_stock(ts_code),
            |date, ts_code, _| {
                financial_ext_snapshot_marker(ts_code, date, income, balance, cashflow)
            },
            |date, ts_code, offset| {
                financial_ext_snapshot_for_stock(
                    ts_code,
                    date,
                    income,
                    balance,
                    cashflow,
                    total_mv_values[offset],
                )
            },
        );
        let Some(date_idx) = panel.dates().iter().position(|date| *date == trade_date) else {
            continue;
        };
        let date_offset = date_idx * panel.instruments().len();
        for (instrument_idx, snapshot) in daily_snapshots.into_iter().enumerate() {
            snapshots_by_offset[date_offset + instrument_idx] = snapshot;
        }
    }

    let mut total_mv_snapshot = vec![None; panel.shape_len()];
    let mut ebit_yoy_change = vec![None; panel.shape_len()];
    let mut ebit_yoy = vec![None; panel.shape_len()];
    let mut t_compr_income = vec![None; panel.shape_len()];
    let mut n_oth_income = vec![None; panel.shape_len()];
    let mut income_tax = vec![None; panel.shape_len()];
    let mut netprofit_parent = vec![None; panel.shape_len()];
    let mut compr_income_parent = vec![None; panel.shape_len()];
    let mut total_cogs = vec![None; panel.shape_len()];
    let mut biz_tax_surchg = vec![None; panel.shape_len()];
    let mut total_assets = vec![None; panel.shape_len()];
    let mut undistr_profit = vec![None; panel.shape_len()];
    let mut undistr_profit_yoy = vec![None; panel.shape_len()];
    let mut undistr_profit_yoy_change = vec![None; panel.shape_len()];
    let mut payroll_payable = vec![None; panel.shape_len()];
    let mut oth_pay_total_yoy = vec![None; panel.shape_len()];
    let mut cip_total_yoy = vec![None; panel.shape_len()];
    let mut total_cur_liab_yoy = vec![None; panel.shape_len()];
    let mut surplus_rese = vec![None; panel.shape_len()];
    let mut oth_comp_income_yoy = vec![None; panel.shape_len()];
    let mut inv_cash_out = vec![None; panel.shape_len()];
    let mut capex = vec![None; panel.shape_len()];
    let mut oper_cash_in = vec![None; panel.shape_len()];
    let mut oper_cash_in_yoy = vec![None; panel.shape_len()];
    let mut cash_equ_end_period = vec![None; panel.shape_len()];
    let mut int_exp_yoy = vec![None; panel.shape_len()];

    for (offset, snapshot) in snapshots_by_offset.into_iter().enumerate() {
        let Some(snapshot) = snapshot else {
            continue;
        };
        total_mv_snapshot[offset] = snapshot.total_mv_snapshot;
        ebit_yoy_change[offset] = diff_opt(snapshot.ebit, snapshot.ebit_yoy);
        ebit_yoy[offset] = snapshot.ebit_yoy;
        t_compr_income[offset] = snapshot.t_compr_income;
        n_oth_income[offset] = snapshot.n_oth_income;
        income_tax[offset] = snapshot.income_tax;
        netprofit_parent[offset] = snapshot.netprofit_parent;
        compr_income_parent[offset] = snapshot.compr_income_parent;
        total_cogs[offset] = snapshot.total_cogs;
        biz_tax_surchg[offset] = snapshot.biz_tax_surchg;
        total_assets[offset] = snapshot.total_assets;
        undistr_profit[offset] = snapshot.undistr_profit;
        undistr_profit_yoy[offset] = snapshot.undistr_profit_yoy;
        undistr_profit_yoy_change[offset] =
            diff_opt(snapshot.undistr_profit, snapshot.undistr_profit_yoy);
        payroll_payable[offset] = snapshot.payroll_payable;
        oth_pay_total_yoy[offset] = snapshot.oth_pay_total_yoy;
        cip_total_yoy[offset] = snapshot.cip_total_yoy;
        total_cur_liab_yoy[offset] = snapshot.total_cur_liab_yoy;
        surplus_rese[offset] = snapshot.surplus_rese;
        oth_comp_income_yoy[offset] = snapshot.oth_comp_income_yoy;
        inv_cash_out[offset] = snapshot.inv_cash_out;
        capex[offset] = snapshot.capex;
        oper_cash_in[offset] = snapshot.oper_cash_in;
        oper_cash_in_yoy[offset] = snapshot.oper_cash_in_yoy;
        cash_equ_end_period[offset] = snapshot.cash_equ_end_period;
        int_exp_yoy[offset] = snapshot.int_exp_yoy;
    }

    Ok(FinancialExtSnapshotColumns {
        total_mv_snapshot: panel.column_from_values(total_mv_snapshot)?,
        ebit_yoy_change: panel.column_from_values(ebit_yoy_change)?,
        ebit_yoy: panel.column_from_values(ebit_yoy)?,
        t_compr_income: panel.column_from_values(t_compr_income)?,
        n_oth_income: panel.column_from_values(n_oth_income)?,
        income_tax: panel.column_from_values(income_tax)?,
        netprofit_parent: panel.column_from_values(netprofit_parent)?,
        compr_income_parent: panel.column_from_values(compr_income_parent)?,
        total_cogs: panel.column_from_values(total_cogs)?,
        biz_tax_surchg: panel.column_from_values(biz_tax_surchg)?,
        total_assets: panel.column_from_values(total_assets)?,
        undistr_profit: panel.column_from_values(undistr_profit)?,
        undistr_profit_yoy: panel.column_from_values(undistr_profit_yoy)?,
        undistr_profit_yoy_change: panel.column_from_values(undistr_profit_yoy_change)?,
        payroll_payable: panel.column_from_values(payroll_payable)?,
        oth_pay_total_yoy: panel.column_from_values(oth_pay_total_yoy)?,
        cip_total_yoy: panel.column_from_values(cip_total_yoy)?,
        total_cur_liab_yoy: panel.column_from_values(total_cur_liab_yoy)?,
        surplus_rese: panel.column_from_values(surplus_rese)?,
        oth_comp_income_yoy: panel.column_from_values(oth_comp_income_yoy)?,
        inv_cash_out: panel.column_from_values(inv_cash_out)?,
        capex: panel.column_from_values(capex)?,
        oper_cash_in: panel.column_from_values(oper_cash_in)?,
        oper_cash_in_yoy: panel.column_from_values(oper_cash_in_yoy)?,
        cash_equ_end_period: panel.column_from_values(cash_equ_end_period)?,
        int_exp_yoy: panel.column_from_values(int_exp_yoy)?,
    })
}

fn financial_ext_snapshot_marker(
    ts_code: &str,
    trade_date: i32,
    income: &FinancialPitReader<'_>,
    balance: &FinancialPitReader<'_>,
    cashflow: &FinancialPitReader<'_>,
) -> Option<FinancialEventMarker> {
    let mut builder = FinancialEventMarkerBuilder::new();
    if let Some(end_date) = income.latest_quarter_end_date(ts_code, trade_date) {
        builder.include_reader_record_for_end_date(
            FinancialStatementDataset::Income,
            income,
            ts_code,
            trade_date,
            end_date,
        );
        builder.include_reader_record_for_end_date(
            FinancialStatementDataset::Income,
            income,
            ts_code,
            trade_date,
            same_quarter_previous_year(end_date),
        );
    }
    if let Some(end_date) = balance.latest_quarter_end_date(ts_code, trade_date) {
        builder.include_reader_record_for_end_date(
            FinancialStatementDataset::BalanceSheet,
            balance,
            ts_code,
            trade_date,
            end_date,
        );
        builder.include_reader_record_for_end_date(
            FinancialStatementDataset::BalanceSheet,
            balance,
            ts_code,
            trade_date,
            same_quarter_previous_year(end_date),
        );
    }
    if let Some(end_date) = cashflow.latest_quarter_end_date(ts_code, trade_date) {
        builder.include_reader_record_for_end_date(
            FinancialStatementDataset::CashFlow,
            cashflow,
            ts_code,
            trade_date,
            end_date,
        );
        builder.include_reader_record_for_end_date(
            FinancialStatementDataset::CashFlow,
            cashflow,
            ts_code,
            trade_date,
            same_quarter_previous_year(end_date),
        );
    }
    builder.build()
}

fn financial_ext_snapshot_for_stock(
    ts_code: &str,
    trade_date: i32,
    income: &FinancialPitReader<'_>,
    balance: &FinancialPitReader<'_>,
    cashflow: &FinancialPitReader<'_>,
    total_mv_value: Option<f64>,
) -> Option<FinancialExtSnapshot> {
    let mut snapshot = FinancialExtSnapshot {
        total_mv_snapshot: clean(total_mv_value).filter(|value| *value > 0.0),
        ..FinancialExtSnapshot::default()
    };

    if let Some(end_date) = income.latest_quarter_end_date(ts_code, trade_date) {
        let yoy_end_date = same_quarter_previous_year(end_date);
        snapshot.ebit = derived_ebit(income, ts_code, trade_date, end_date);
        snapshot.ebit_yoy = derived_ebit(income, ts_code, trade_date, yoy_end_date);
        snapshot.t_compr_income =
            financial_value(income, ts_code, trade_date, end_date, "t_compr_income");
        snapshot.n_oth_income =
            financial_value(income, ts_code, trade_date, end_date, "n_oth_income");
        snapshot.income_tax = financial_value(income, ts_code, trade_date, end_date, "income_tax");
        snapshot.netprofit_parent =
            financial_value(income, ts_code, trade_date, end_date, "n_income_attr_p");
        snapshot.compr_income_parent =
            financial_value(income, ts_code, trade_date, end_date, "compr_inc_attr_p");
        snapshot.total_cogs = financial_value(income, ts_code, trade_date, end_date, "total_cogs");
        snapshot.biz_tax_surchg =
            financial_value(income, ts_code, trade_date, end_date, "biz_tax_surchg");
        snapshot.int_exp_yoy =
            financial_value(income, ts_code, trade_date, yoy_end_date, "int_exp");
    }
    if let Some(end_date) = balance.latest_quarter_end_date(ts_code, trade_date) {
        let yoy_end_date = same_quarter_previous_year(end_date);
        snapshot.total_assets =
            financial_value(balance, ts_code, trade_date, end_date, "total_assets");
        snapshot.undistr_profit =
            financial_value(balance, ts_code, trade_date, end_date, "undistr_porfit");
        snapshot.undistr_profit_yoy =
            financial_value(balance, ts_code, trade_date, yoy_end_date, "undistr_porfit");
        snapshot.payroll_payable =
            financial_value(balance, ts_code, trade_date, end_date, "payroll_payable");
        snapshot.oth_pay_total_yoy =
            financial_value(balance, ts_code, trade_date, yoy_end_date, "oth_pay_total");
        snapshot.cip_total_yoy =
            financial_value(balance, ts_code, trade_date, yoy_end_date, "cip_total");
        snapshot.total_cur_liab_yoy =
            financial_value(balance, ts_code, trade_date, yoy_end_date, "total_cur_liab");
        snapshot.surplus_rese =
            financial_value(balance, ts_code, trade_date, end_date, "surplus_rese");
        snapshot.oth_comp_income_yoy = financial_value(
            balance,
            ts_code,
            trade_date,
            yoy_end_date,
            "oth_comp_income",
        );
    }
    if let Some(end_date) = cashflow.latest_quarter_end_date(ts_code, trade_date) {
        let yoy_end_date = same_quarter_previous_year(end_date);
        snapshot.inv_cash_out =
            financial_value(cashflow, ts_code, trade_date, end_date, "stot_out_inv_act");
        snapshot.capex = financial_value(
            cashflow,
            ts_code,
            trade_date,
            end_date,
            "c_pay_acq_const_fiolta",
        );
        snapshot.oper_cash_in = financial_value(
            cashflow,
            ts_code,
            trade_date,
            end_date,
            "c_inf_fr_operate_a",
        );
        snapshot.oper_cash_in_yoy = financial_value(
            cashflow,
            ts_code,
            trade_date,
            yoy_end_date,
            "c_inf_fr_operate_a",
        );
        snapshot.cash_equ_end_period = financial_value(
            cashflow,
            ts_code,
            trade_date,
            end_date,
            "c_cash_equ_end_period",
        );
    }
    Some(snapshot)
}

fn financial_value(
    reader: &FinancialPitReader<'_>,
    ts_code: &str,
    trade_date: i32,
    end_date: i32,
    column: &str,
) -> Option<f64> {
    clean(
        reader
            .record_for_end_date(ts_code, trade_date, end_date)?
            .column(column),
    )
}

fn derived_ebit(
    income: &FinancialPitReader<'_>,
    ts_code: &str,
    trade_date: i32,
    end_date: i32,
) -> Option<f64> {
    let record = income.record_for_end_date(ts_code, trade_date, end_date)?;
    derived_ebit_values(
        record.column("n_income"),
        record.column("income_tax"),
        record.column("int_exp"),
    )
}

fn derived_ebit_values(
    n_income: Option<f64>,
    income_tax: Option<f64>,
    int_exp: Option<f64>,
) -> Option<f64> {
    let value = clean(n_income)? + clean(income_tax).unwrap_or(0.0) + clean(int_exp).unwrap_or(0.0);
    value.is_finite().then_some(value)
}

fn diff_opt(left: Option<f64>, right: Option<f64>) -> Option<f64> {
    let value = left? - right?;
    value.is_finite().then_some(value)
}

fn same_quarter_previous_year(end_date: i32) -> i32 {
    (end_date / 10_000 - 1) * 10_000 + end_date % 10_000
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derived_ebit_uses_net_income_tax_and_interest() {
        assert_eq!(
            derived_ebit_values(Some(100.0), Some(20.0), Some(5.0)),
            Some(125.0)
        );
        assert_eq!(
            derived_ebit_values(Some(100.0), None, Some(5.0)),
            Some(105.0)
        );
        assert_eq!(derived_ebit_values(None, Some(20.0), Some(5.0)), None);
    }

    #[test]
    fn total_mv_regression_outputs_are_sector_only_after_raw_regression() {
        assert!(GaussianFinancialExtOutput::EbitYoyChgMv.uses_total_mv_regression());
        assert!(GaussianFinancialExtOutput::InvCashOutMv.uses_total_mv_regression());
        assert!(!GaussianFinancialExtOutput::IncomeTaxEbitYoy.uses_total_mv_regression());
    }

    #[test]
    fn duplicate_cfp_output_is_not_registered_in_ext_provider() {
        assert!(GaussianFinancialExtOutput::from_id("cfp_sq_gauss_resid").is_none());
    }

    #[test]
    fn same_quarter_previous_year_preserves_quarter_date() {
        assert_eq!(same_quarter_previous_year(20250331), 20240331);
        assert_eq!(same_quarter_previous_year(20251231), 20241231);
    }

    #[test]
    fn ext_specs_keep_fundamental_gaussian_tags() {
        let factor_spec = spec(GaussianFinancialExtOutput::NOthIncomeMv);
        assert_eq!(factor_spec.id, "n_oth_income_mv_gauss_resid");
        assert!(factor_spec.tags.contains(&"fundamental".to_string()));
        assert!(factor_spec.tags.contains(&"gaussian_rank".to_string()));
    }
}

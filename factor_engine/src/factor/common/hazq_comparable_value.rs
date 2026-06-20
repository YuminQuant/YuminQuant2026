use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap, HashMap};

use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::{err, Result};
use crate::factor::common::financial::previous_quarter_end_date;
use crate::factor::common::stock_daily_ops::{is_bj_stock, neutralize_size_sector};
use crate::factor::common::vector::clean;
use crate::factor::common::{
    cached_financial_stock_snapshots_for_date, financial_event_trade_dates, DailyPanel,
    DividendReader, FinancialEventMarker, FinancialEventMarkerBuilder, FinancialEventSchedule,
    FinancialPitReader, FinancialStatementDataset, InstrumentAlignedSnapshotCache, PanelColumn,
    PitFinancialRecordView, ReportTypePreference,
};
use crate::operators::{cs_zscore, ts_zscore};

pub const PROVIDER_KEY: &str = "stock|daily|hazq_comparable_value";

const VERSION: &str = "0.1.0";
const LOOKBACK: usize = 252;
const ZSCORE_MIN_PERIODS: usize = 120;
const FINANCIAL_QUARTERS: usize = 8;
const EPS: f64 = 1e-12;
const SIMILARITY_THRESHOLD: f64 = 0.9;
const TOP_PEER_COUNT: usize = 6;
const BASE_COUNT: usize = 9;
const COMPONENT_COUNT: usize = 13;
const LIFECYCLE_STAGE_COUNT: usize = 5;
const CONTINUOUS_FEATURE_COUNT: usize = 12;
const SLOW_CONTINUOUS_FEATURE_COUNT: usize = CONTINUOUS_FEATURE_COUNT - 1;
const FEATURE_DIM: usize = LIFECYCLE_STAGE_COUNT + CONTINUOUS_FEATURE_COUNT;

const TOTAL_MV_COLUMN: &str = "total_mv";
const PE_TTM_COLUMN: &str = "pe_ttm";
const CONSENSUS_GROWTH_COLUMN: &str = "con_npcgrate_2y_roll";
const CONSENSUS_PE_ROLL_COLUMN: &str = "con_pe_roll";

const REVENUE_COLUMN: &str = "revenue";
const NET_PROFIT_COLUMN: &str = "n_income";
const NET_PROFIT_ATTR_P_COLUMN: &str = "n_income_attr_p";
const INCOME_TAX_COLUMN: &str = "income_tax";
const TOTAL_PROFIT_COLUMN: &str = "total_profit";
const INT_EXP_COLUMN: &str = "int_exp";
const RD_EXP_COLUMN: &str = "rd_exp";

const CFO_COLUMN: &str = "n_cashflow_act";
const CFI_COLUMN: &str = "n_cashflow_inv_act";
const CFF_COLUMN: &str = "n_cash_flows_fnc_act";

const EQUITY_COLUMN: &str = "total_hldr_eqy_exc_min_int";
const TOTAL_ASSETS_COLUMN: &str = "total_assets";
const TOTAL_LIAB_COLUMN: &str = "total_liab";
const MONEY_CAP_COLUMN: &str = "money_cap";
const FIX_ASSETS_COLUMN: &str = "fix_assets";
const ACCOUNTS_RECEIV_COLUMN: &str = "accounts_receiv";
const INVENTORIES_COLUMN: &str = "inventories";
const SHORT_BORROW_COLUMN: &str = "st_borr";
const NON_CURRENT_LIAB_DUE_1Y_COLUMN: &str = "non_cur_liab_due_1y";
const LONG_BORROW_COLUMN: &str = "lt_borr";
const BOND_PAYABLE_COLUMN: &str = "bond_payable";

const INCOME_COLUMNS: [&str; 7] = [
    REVENUE_COLUMN,
    NET_PROFIT_COLUMN,
    NET_PROFIT_ATTR_P_COLUMN,
    INCOME_TAX_COLUMN,
    TOTAL_PROFIT_COLUMN,
    INT_EXP_COLUMN,
    RD_EXP_COLUMN,
];

const CASHFLOW_COLUMNS: [&str; 3] = [CFO_COLUMN, CFI_COLUMN, CFF_COLUMN];

const BALANCE_COLUMNS: [&str; 11] = [
    TOTAL_ASSETS_COLUMN,
    TOTAL_LIAB_COLUMN,
    MONEY_CAP_COLUMN,
    EQUITY_COLUMN,
    FIX_ASSETS_COLUMN,
    ACCOUNTS_RECEIV_COLUMN,
    INVENTORIES_COLUMN,
    SHORT_BORROW_COLUMN,
    NON_CURRENT_LIAB_DUE_1Y_COLUMN,
    LONG_BORROW_COLUMN,
    BOND_PAYABLE_COLUMN,
];

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum HazqComparableBase {
    Bp,
    Dp,
    Ebit2Ev,
    Ep,
    EpQ,
    Ocfp,
    Sales2Ev,
    EpPercentile,
    EpFttm,
}

impl HazqComparableBase {
    pub fn id(self) -> &'static str {
        match self {
            Self::Bp => "bp",
            Self::Dp => "dp",
            Self::Ebit2Ev => "ebit2ev",
            Self::Ep => "ep",
            Self::EpQ => "ep_q",
            Self::Ocfp => "ocfp",
            Self::Sales2Ev => "sales2ev",
            Self::EpPercentile => "ep_percentile",
            Self::EpFttm => "ep_fttm",
        }
    }

    fn alias(self) -> &'static str {
        match self {
            Self::Bp => "BP",
            Self::Dp => "DP",
            Self::Ebit2Ev => "EBIT2EV",
            Self::Ep => "EP",
            Self::EpQ => "EP_Q",
            Self::Ocfp => "OCFP",
            Self::Sales2Ev => "SALES2EV",
            Self::EpPercentile => "EP_PERCENTILE",
            Self::EpFttm => "EP_FTTM",
        }
    }

    fn idx(self) -> usize {
        match self {
            Self::Bp => 0,
            Self::Dp => 1,
            Self::Ebit2Ev => 2,
            Self::Ep => 3,
            Self::EpQ => 4,
            Self::Ocfp => 5,
            Self::Sales2Ev => 6,
            Self::EpPercentile => 7,
            Self::EpFttm => 8,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum HazqComparableComponent {
    Med,
    Avg,
    Weighted,
    Max,
    Min,
    Wgt,
    Wgt2,
    Dst,
    Prm,
    DstZscore,
    PrmZscore,
    GapAvg,
    GapMmm,
}

impl HazqComparableComponent {
    pub fn id(self) -> &'static str {
        match self {
            Self::Med => "med",
            Self::Avg => "avg",
            Self::Weighted => "weighted",
            Self::Max => "max",
            Self::Min => "min",
            Self::Wgt => "wgt",
            Self::Wgt2 => "wgt2",
            Self::Dst => "dst",
            Self::Prm => "prm",
            Self::DstZscore => "dst_zscore",
            Self::PrmZscore => "prm_zscore",
            Self::GapAvg => "gap_avg",
            Self::GapMmm => "gap_mmm",
        }
    }

    fn alias(self) -> &'static str {
        match self {
            Self::Med => "MED",
            Self::Avg => "AVG",
            Self::Weighted => "WEIGHTED",
            Self::Max => "MAX",
            Self::Min => "MIN",
            Self::Wgt => "WGT",
            Self::Wgt2 => "WGT2",
            Self::Dst => "DST",
            Self::Prm => "PRM",
            Self::DstZscore => "DST_ZSCORE",
            Self::PrmZscore => "PRM_ZSCORE",
            Self::GapAvg => "GAP_AVG",
            Self::GapMmm => "GAP_MMM",
        }
    }

    fn idx(self) -> usize {
        match self {
            Self::Med => 0,
            Self::Avg => 1,
            Self::Weighted => 2,
            Self::Max => 3,
            Self::Min => 4,
            Self::Wgt => 5,
            Self::Wgt2 => 6,
            Self::Dst => 7,
            Self::Prm => 8,
            Self::DstZscore => 9,
            Self::PrmZscore => 10,
            Self::GapAvg => 11,
            Self::GapMmm => 12,
        }
    }

    fn source_component(self) -> Self {
        match self {
            Self::DstZscore => Self::Dst,
            Self::PrmZscore => Self::Prm,
            other => other,
        }
    }

    fn is_time_zscore(self) -> bool {
        matches!(self, Self::DstZscore | Self::PrmZscore)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct HazqComparableValueOutput {
    pub base: HazqComparableBase,
    pub component: HazqComparableComponent,
}

impl HazqComparableValueOutput {
    pub const fn new(base: HazqComparableBase, component: HazqComparableComponent) -> Self {
        Self { base, component }
    }

    pub fn id(self) -> String {
        format!("comp_{}_{}", self.base.id(), self.component.id())
    }
}

pub const BASES: [HazqComparableBase; 9] = [
    HazqComparableBase::Bp,
    HazqComparableBase::Dp,
    HazqComparableBase::Ebit2Ev,
    HazqComparableBase::Ep,
    HazqComparableBase::EpQ,
    HazqComparableBase::Ocfp,
    HazqComparableBase::Sales2Ev,
    HazqComparableBase::EpPercentile,
    HazqComparableBase::EpFttm,
];

pub const COMPONENTS: [HazqComparableComponent; 13] = [
    HazqComparableComponent::Med,
    HazqComparableComponent::Avg,
    HazqComparableComponent::Weighted,
    HazqComparableComponent::Max,
    HazqComparableComponent::Min,
    HazqComparableComponent::Wgt,
    HazqComparableComponent::Wgt2,
    HazqComparableComponent::Dst,
    HazqComparableComponent::Prm,
    HazqComparableComponent::DstZscore,
    HazqComparableComponent::PrmZscore,
    HazqComparableComponent::GapAvg,
    HazqComparableComponent::GapMmm,
];

#[derive(Default)]
pub struct HazqComparableValueComputeState {
    snapshot_cache: InstrumentAlignedSnapshotCache<HazqComparableSnapshot>,
    peer_state: ComparablePeerState,
}

#[derive(Clone, Copy, Debug, Default)]
struct HazqComparableSnapshot {
    lifecycle: Option<LifecycleStage>,
    slow_features: [Option<f64>; SLOW_CONTINUOUS_FEATURE_COUNT],
    equity: Option<f64>,
    total_liab: Option<f64>,
    money_cap: Option<f64>,
    revenue_ttm: Option<f64>,
    ebit_ttm: Option<f64>,
    profit_ttm_yoy_growth: Option<f64>,
    profit_q: Option<f64>,
    cfo_ttm: Option<f64>,
}

#[derive(Clone, Debug, Default)]
struct ComparablePeerState {
    peers: Vec<PeerProfile>,
    last_processed_trade_date: Option<i32>,
}

impl ComparablePeerState {
    fn mark_processed(&mut self, trade_date: i32) {
        self.last_processed_trade_date = Some(trade_date);
    }
}

#[derive(Clone, Debug, Default)]
struct PeerProfile {
    all: Vec<PeerLink>,
    top: Vec<PeerLink>,
}

#[derive(Clone, Debug)]
struct PeerLink {
    peer_idx: usize,
    similarity: f64,
}

struct HazqComparableInputs<'a> {
    panel: &'a DailyPanel,
    total_mv: PanelColumn,
    pe_ttm: Option<PanelColumn>,
    income: FinancialPitReader<'a>,
    balance: FinancialPitReader<'a>,
    cashflow: FinancialPitReader<'a>,
    consensus_growth: Option<PanelColumn>,
    consensus_pe_roll: Option<PanelColumn>,
}

#[derive(Clone, Copy, Debug)]
struct ComparablePoint {
    instrument_idx: usize,
    values: [f64; FEATURE_DIM],
}

#[derive(Clone, Copy, Debug)]
struct ComponentStats {
    values: [Option<f64>; COMPONENT_COUNT],
}

#[derive(Clone, Debug)]
struct ComparableRequestPlan {
    outputs: Vec<HazqComparableValueOutput>,
    source_needs: Vec<bool>,
}

impl ComparableRequestPlan {
    fn from_requested_ids(requested_ids: &[String]) -> Self {
        let outputs = requested_outputs(requested_ids);
        let source_needs = source_component_needs(&outputs);
        Self {
            outputs,
            source_needs,
        }
    }

    fn is_empty(&self) -> bool {
        self.outputs.is_empty()
    }

    fn needs_consensus_growth(&self) -> bool {
        self.outputs.iter().any(|output| {
            matches!(
                output.component,
                HazqComparableComponent::GapAvg | HazqComparableComponent::GapMmm
            )
        })
    }

    fn needs_consensus_pe_roll(&self) -> bool {
        self.requested_bases()
            .into_iter()
            .any(|base| base == HazqComparableBase::EpFttm)
    }

    fn requested_bases(&self) -> Vec<HazqComparableBase> {
        BASES
            .into_iter()
            .filter(|base| {
                COMPONENTS
                    .into_iter()
                    .any(|component| self.source_needs[source_key(*base, component)])
            })
            .collect()
    }

    fn needs_ep_intermediate(&self) -> bool {
        self.requested_bases().into_iter().any(|base| {
            matches!(
                base,
                HazqComparableBase::Ep | HazqComparableBase::EpPercentile
            )
        })
    }

    fn needs_source_component(
        &self,
        base: HazqComparableBase,
        component: HazqComparableComponent,
    ) -> bool {
        self.source_needs[source_key(base, component)]
    }
}

#[derive(Clone, Copy, Debug)]
enum LifecycleStage {
    Introduction,
    Growth,
    Mature,
    ShakeOut,
    Decline,
}

impl LifecycleStage {
    fn idx(self) -> usize {
        match self {
            Self::Introduction => 0,
            Self::Growth => 1,
            Self::Mature => 2,
            Self::ShakeOut => 3,
            Self::Decline => 4,
        }
    }
}

pub fn all_outputs() -> Vec<HazqComparableValueOutput> {
    BASES
        .into_iter()
        .flat_map(|base| {
            COMPONENTS
                .into_iter()
                .map(move |component| HazqComparableValueOutput { base, component })
        })
        .collect()
}

pub fn spec(output: HazqComparableValueOutput) -> FactorSpec {
    let id = output.id();
    FactorSpec {
        id: id.clone(),
        aliases: vec![
            format!(
                "HAZQ Comparable {} {}",
                output.base.alias(),
                output.component.alias()
            ),
            format!("{}_{}", output.base.alias(), output.component.alias()),
        ],
        name: id,
        asset_class: AssetClass::Stock,
        frequency: Frequency::Daily,
        version: VERSION.to_string(),
        tags: tags(),
        description: format!(
            "HAZQ comparable-company value factor {} {}. It builds a PIT financial cosine-similarity network, uses peers with similarity above 0.9, excludes BJ stocks, and neutralizes the output by SW level-1 industry and Barra SIZE.",
            output.base.alias(),
            output.component.alias()
        ),
        dependencies: dependencies_for_output(output),
        intraday_raw_dependencies: Vec::new(),
        lookback: Lookback {
            trading_days: LOOKBACK,
        },
    }
}

pub fn compute_requested(
    requested_ids: &[String],
    context: &FactorContext,
    data: &DataPool,
) -> Result<Vec<FactorSeries>> {
    let mut state = HazqComparableValueComputeState::default();
    compute_requested_stateful(requested_ids, context, data, &mut state)
}

pub fn compute_requested_stateful(
    requested_ids: &[String],
    _context: &FactorContext,
    data: &DataPool,
    state: &mut HazqComparableValueComputeState,
) -> Result<Vec<FactorSeries>> {
    let request_plan = ComparableRequestPlan::from_requested_ids(requested_ids);
    if request_plan.is_empty() {
        return Ok(Vec::new());
    }

    let panel = non_bj_panel(data.stock_universe_panel()?)?;
    let inputs = hazq_inputs(
        data,
        &panel,
        request_plan.needs_ep_intermediate(),
        request_plan.needs_consensus_growth(),
        request_plan.needs_consensus_pe_roll(),
    )?;
    let requested_bases = request_plan.requested_bases();
    let mut base_columns = vec![None; BASE_COUNT];
    let mut ep_base_column = None;
    for base in requested_bases.iter().copied() {
        let column = compute_base_column(
            base,
            &inputs,
            data,
            &mut state.snapshot_cache,
            &mut ep_base_column,
        )?;
        base_columns[base.idx()] = Some(column);
    }
    let growth_column = if request_plan.needs_consensus_growth() {
        Some(compute_growth_column(&inputs, &mut state.snapshot_cache)?)
    } else {
        None
    };
    let schedule = hazq_event_schedule(&inputs);
    let first_panel_date = panel.dates().first().copied();
    let mut peer_state = match (state.peer_state.last_processed_trade_date, first_panel_date) {
        (Some(last_processed), Some(first_date)) if first_date > last_processed => {
            state.peer_state.clone()
        }
        _ => ComparablePeerState::default(),
    };
    let event_dates = financial_event_trade_dates(
        peer_state.last_processed_trade_date,
        &schedule,
        panel.dates(),
    )
    .into_iter()
    .collect::<BTreeSet<_>>();
    let mut source_values = source_storage(&request_plan.source_needs, &panel);

    for trade_date in panel.dates().iter().copied() {
        if event_dates.contains(&trade_date) {
            let points =
                comparable_points_for_trade_date(&inputs, &mut state.snapshot_cache, trade_date)?;
            peer_state.peers = peer_profiles_from_points(&points, panel.instruments().len());
        }
        for base in requested_bases.iter().copied() {
            let base_column = base_columns[base.idx()]
                .as_ref()
                .ok_or_else(|| err(format!("missing HAZQ comparable base column {}", base.id())))?;
            write_source_components_for_base_date(
                &panel,
                trade_date,
                &peer_state,
                base,
                base_column,
                growth_column.as_ref(),
                &request_plan,
                &mut source_values,
            )?;
        }
        peer_state.mark_processed(trade_date);
    }
    state.peer_state = peer_state;

    let mut result = Vec::with_capacity(request_plan.outputs.len());
    for output in request_plan.outputs {
        let source = output.component.source_component();
        let key = source_key(output.base, source);
        let values = source_values[key]
            .clone()
            .unwrap_or_else(|| vec![None; panel.shape_len()]);
        let raw = panel.column_from_values(values)?;
        let raw = if output.component.is_time_zscore() {
            raw.ts(|series| ts_zscore(series, LOOKBACK, ZSCORE_MIN_PERIODS))?
        } else {
            raw
        };
        let neutralized = neutralize_size_sector(&raw, &panel, data)?;
        result.push(neutralized.to_factor_series(spec(output)));
    }
    Ok(result)
}

fn hazq_inputs<'a>(
    data: &'a DataPool,
    panel: &'a DailyPanel,
    needs_pe_ttm: bool,
    needs_consensus_growth: bool,
    needs_consensus_pe_roll: bool,
) -> Result<HazqComparableInputs<'a>> {
    let daily_basic = data.daily(DatasetId::StockDailyBasic)?;
    let consensus = if needs_consensus_growth || needs_consensus_pe_roll {
        Some(data.daily(DatasetId::StockConsensus)?)
    } else {
        None
    };
    Ok(HazqComparableInputs {
        panel,
        total_mv: panel.column_from_table(daily_basic, TOTAL_MV_COLUMN)?,
        pe_ttm: needs_pe_ttm
            .then(|| panel.column_from_table(daily_basic, PE_TTM_COLUMN))
            .transpose()?,
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
            ReportTypePreference::consolidated(),
        )?,
        consensus_growth: consensus
            .filter(|_| needs_consensus_growth)
            .map(|table| panel.column_from_table(table, CONSENSUS_GROWTH_COLUMN))
            .transpose()?,
        consensus_pe_roll: consensus
            .filter(|_| needs_consensus_pe_roll)
            .map(|table| panel.column_from_table(table, CONSENSUS_PE_ROLL_COLUMN))
            .transpose()?,
    })
}

fn hazq_event_schedule(inputs: &HazqComparableInputs<'_>) -> FinancialEventSchedule {
    FinancialEventSchedule::from_pit_readers(&[
        inputs.income.clone(),
        inputs.balance.clone(),
        inputs.cashflow.clone(),
    ])
}

fn non_bj_panel(panel: &DailyPanel) -> Result<DailyPanel> {
    let keep_indices = panel
        .instruments()
        .iter()
        .enumerate()
        .filter_map(|(idx, ts_code)| (!is_bj_stock(ts_code)).then_some(idx))
        .collect::<Vec<_>>();
    let instruments = keep_indices
        .iter()
        .map(|idx| panel.instruments()[*idx].clone())
        .collect::<Vec<_>>();
    let source_count = panel.instruments().len();
    let mut present = Vec::with_capacity(panel.dates().len() * instruments.len());
    for date_idx in 0..panel.dates().len() {
        let source_offset = date_idx * source_count;
        for source_idx in &keep_indices {
            present.push(panel.is_present_offset(source_offset + *source_idx));
        }
    }
    let target_dates = panel
        .dates()
        .iter()
        .copied()
        .filter(|date| panel.is_target_date(*date))
        .collect::<Vec<_>>();
    DailyPanel::from_index(panel.dates().to_vec(), instruments, &target_dates, present)
}

fn compute_base_column(
    base: HazqComparableBase,
    inputs: &HazqComparableInputs<'_>,
    data: &DataPool,
    cache: &mut InstrumentAlignedSnapshotCache<HazqComparableSnapshot>,
    ep_base_column: &mut Option<PanelColumn>,
) -> Result<PanelColumn> {
    match base {
        HazqComparableBase::Ep => cached_ep_base_column(inputs, ep_base_column),
        HazqComparableBase::EpPercentile => {
            let ep = cached_ep_base_column(inputs, ep_base_column)?;
            ep.ts(|series| ts_zscore(series, LOOKBACK, ZSCORE_MIN_PERIODS))
        }
        HazqComparableBase::Dp => {
            let dividends = data.dividend_reader()?;
            compute_dp_base_column(inputs.panel, &inputs.total_mv, &dividends)
        }
        HazqComparableBase::EpFttm => compute_ep_fttm_base_column(inputs),
        other => compute_snapshot_base_column(other, inputs, cache),
    }
}

fn cached_ep_base_column(
    inputs: &HazqComparableInputs<'_>,
    ep_base_column: &mut Option<PanelColumn>,
) -> Result<PanelColumn> {
    if let Some(column) = ep_base_column.as_ref() {
        return Ok(column.clone());
    }
    let column = compute_ep_base_column(inputs)?;
    *ep_base_column = Some(column.clone());
    Ok(column)
}

fn compute_ep_base_column(inputs: &HazqComparableInputs<'_>) -> Result<PanelColumn> {
    let pe_ttm = inputs
        .pe_ttm
        .as_ref()
        .ok_or_else(|| err("HAZQ comparable EP requires daily_basic pe_ttm"))?;
    Ok(pe_ttm.map_values(reciprocal_valuation))
}

fn compute_snapshot_base_column(
    base: HazqComparableBase,
    inputs: &HazqComparableInputs<'_>,
    cache: &mut InstrumentAlignedSnapshotCache<HazqComparableSnapshot>,
) -> Result<PanelColumn> {
    if matches!(
        base,
        HazqComparableBase::Dp
            | HazqComparableBase::Ep
            | HazqComparableBase::EpPercentile
            | HazqComparableBase::EpFttm
    ) {
        return Err(err(format!(
            "base {} requires a dedicated HAZQ comparable base helper",
            base.id()
        )));
    }
    let panel = inputs.panel;
    let instrument_count = panel.instruments().len();
    let mut values = vec![None; panel.shape_len()];

    for (date_idx, trade_date) in panel.dates().iter().copied().enumerate() {
        let snapshots = hazq_snapshots_for_date(inputs, cache, trade_date);
        let date_offset = date_idx * instrument_count;
        for (instrument_idx, snapshot) in snapshots.into_iter().enumerate() {
            let offset = date_offset + instrument_idx;
            if !panel.is_present_offset(offset) {
                continue;
            }
            let Some(snapshot) = snapshot else {
                continue;
            };
            let market_cap = clean(inputs.total_mv.values()[offset]).filter(|value| *value > EPS);
            values[offset] = base_value_from_snapshot(base, &snapshot, market_cap);
        }
    }

    panel.column_from_values(values)
}

fn compute_ep_fttm_base_column(inputs: &HazqComparableInputs<'_>) -> Result<PanelColumn> {
    let con_pe_roll = inputs
        .consensus_pe_roll
        .as_ref()
        .ok_or_else(|| err("HAZQ comparable EP_FTTM requires consensus con_pe_roll"))?;
    Ok(con_pe_roll.map_values(reciprocal_valuation))
}

fn reciprocal_valuation(value: Option<f64>) -> Option<f64> {
    let value = clean(value)?;
    (value.abs() > EPS)
        .then_some(1.0 / value)
        .filter(|value| value.is_finite())
}

fn compute_growth_column(
    inputs: &HazqComparableInputs<'_>,
    cache: &mut InstrumentAlignedSnapshotCache<HazqComparableSnapshot>,
) -> Result<PanelColumn> {
    let consensus_growth = inputs
        .consensus_growth
        .as_ref()
        .ok_or_else(|| err("HAZQ comparable GAP requires consensus growth column"))?;
    let panel = inputs.panel;
    let instrument_count = panel.instruments().len();
    let mut values = vec![None; panel.shape_len()];
    for (date_idx, trade_date) in panel.dates().iter().copied().enumerate() {
        let snapshots = hazq_snapshots_for_date(inputs, cache, trade_date);
        let date_offset = date_idx * instrument_count;
        for (instrument_idx, snapshot) in snapshots.into_iter().enumerate() {
            let offset = date_offset + instrument_idx;
            if !panel.is_present_offset(offset) {
                continue;
            }
            values[offset] = clean(consensus_growth.values()[offset])
                .or_else(|| snapshot.and_then(|snapshot| snapshot.profit_ttm_yoy_growth));
        }
    }
    panel.column_from_values(values)
}

fn compute_dp_base_column(
    panel: &DailyPanel,
    total_mv: &PanelColumn,
    dividends: &DividendReader<'_>,
) -> Result<PanelColumn> {
    let instrument_count = panel.instruments().len();
    let mut values = vec![None; panel.shape_len()];
    let dividend_sums_by_date = dividend_sums_by_date(panel, dividends);

    for (date_idx, trade_date) in panel.dates().iter().copied().enumerate() {
        let dividend_sums = dividend_sums_by_date.get(&trade_date);
        let date_offset = date_idx * instrument_count;
        for (instrument_idx, ts_code) in panel.instruments().iter().enumerate() {
            let offset = date_offset + instrument_idx;
            if !panel.is_present_offset(offset) {
                continue;
            }
            let market_cap = clean(total_mv.values()[offset]).filter(|value| *value > EPS);
            let cash_dividend_ltm = dividend_sums
                .and_then(|sums| sums.get(ts_code.as_str()).copied())
                .unwrap_or(0.0);
            values[offset] = safe_div_opt(Some(cash_dividend_ltm), market_cap);
        }
    }

    panel.column_from_values(values)
}

fn comparable_points_for_trade_date(
    inputs: &HazqComparableInputs<'_>,
    cache: &mut InstrumentAlignedSnapshotCache<HazqComparableSnapshot>,
    trade_date: i32,
) -> Result<Vec<ComparablePoint>> {
    let panel = inputs.panel;
    let Some(date_idx) = panel.dates().iter().position(|date| *date == trade_date) else {
        return Ok(Vec::new());
    };
    let instrument_count = panel.instruments().len();
    let date_offset = date_idx * instrument_count;
    let snapshots = hazq_snapshots_for_date(inputs, cache, trade_date);

    let mut continuous_raw = vec![[None; CONTINUOUS_FEATURE_COUNT]; instrument_count];
    let mut lifecycle = vec![None; instrument_count];
    for (instrument_idx, snapshot) in snapshots.into_iter().enumerate() {
        let panel_idx = date_offset + instrument_idx;
        if !panel.is_present_offset(panel_idx) {
            continue;
        }
        let Some(snapshot) = snapshot else {
            continue;
        };
        let total_mv = clean(inputs.total_mv.values()[panel_idx]).filter(|value| *value > EPS);
        continuous_raw[instrument_idx] = continuous_features_from_snapshot(&snapshot, total_mv);
        lifecycle[instrument_idx] = snapshot.lifecycle;
    }

    let mut continuous_z = vec![vec![None; instrument_count]; CONTINUOUS_FEATURE_COUNT];
    for dim in 0..CONTINUOUS_FEATURE_COUNT {
        let raw = continuous_raw
            .iter()
            .map(|features| features[dim])
            .collect::<Vec<_>>();
        let scored = cs_zscore(&raw);
        for instrument_idx in 0..instrument_count {
            let panel_idx = date_offset + instrument_idx;
            if panel.is_present_offset(panel_idx) {
                continuous_z[dim][instrument_idx] = scored[instrument_idx].or(Some(0.0));
            }
        }
    }

    let mut points = Vec::new();
    for instrument_idx in 0..instrument_count {
        let panel_idx = date_offset + instrument_idx;
        if !panel.is_present_offset(panel_idx) {
            continue;
        }
        let Some(values) =
            feature_unit_vector(lifecycle[instrument_idx], &continuous_z, instrument_idx)
        else {
            continue;
        };
        points.push(ComparablePoint {
            instrument_idx,
            values,
        });
    }
    Ok(points)
}

fn hazq_snapshots_for_date(
    inputs: &HazqComparableInputs<'_>,
    cache: &mut InstrumentAlignedSnapshotCache<HazqComparableSnapshot>,
    trade_date: i32,
) -> Vec<Option<HazqComparableSnapshot>> {
    cached_financial_stock_snapshots_for_date(
        inputs.panel,
        trade_date,
        cache,
        |_, _, offset| !inputs.panel.is_present_offset(offset),
        |trade_date, ts_code, _| {
            hazq_snapshot_marker(
                ts_code,
                trade_date,
                &inputs.income,
                &inputs.balance,
                &inputs.cashflow,
            )
        },
        |trade_date, ts_code, _| {
            hazq_snapshot_for_stock(
                ts_code,
                trade_date,
                &inputs.income,
                &inputs.balance,
                &inputs.cashflow,
            )
        },
    )
}

fn hazq_snapshot_marker(
    ts_code: &str,
    trade_date: i32,
    income: &FinancialPitReader<'_>,
    balance: &FinancialPitReader<'_>,
    cashflow: &FinancialPitReader<'_>,
) -> Option<FinancialEventMarker> {
    let latest_end = income.latest_quarter_end_date(ts_code, trade_date)?;
    let previous_end = previous_quarter_end_date(latest_end);
    let yoy_end = same_quarter_previous_year(latest_end);
    let mut builder = FinancialEventMarkerBuilder::new();
    builder.include_reader_ttm_for_end_date(
        FinancialStatementDataset::Income,
        income,
        ts_code,
        trade_date,
        latest_end,
    );
    builder.include_reader_ttm_for_end_date(
        FinancialStatementDataset::CashFlow,
        cashflow,
        ts_code,
        trade_date,
        latest_end,
    );
    builder.include_reader_ttm_for_end_date(
        FinancialStatementDataset::Income,
        income,
        ts_code,
        trade_date,
        yoy_end,
    );
    builder.include_reader_record_for_end_date(
        FinancialStatementDataset::BalanceSheet,
        balance,
        ts_code,
        trade_date,
        latest_end,
    );
    if let Some(end_date) = previous_end {
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

fn hazq_snapshot_for_stock(
    ts_code: &str,
    trade_date: i32,
    income: &FinancialPitReader<'_>,
    balance: &FinancialPitReader<'_>,
    cashflow: &FinancialPitReader<'_>,
) -> Option<HazqComparableSnapshot> {
    let latest_end = income.latest_quarter_end_date(ts_code, trade_date)?;
    let previous_end = previous_quarter_end_date(latest_end);
    let yoy_end = same_quarter_previous_year(latest_end);
    let income_record = income.record_for_end_date(ts_code, trade_date, latest_end)?;
    let balance_record = balance.record_for_end_date(ts_code, trade_date, latest_end)?;
    let previous_balance = previous_end
        .and_then(|end_date| balance.record_for_end_date(ts_code, trade_date, end_date));

    let revenue_ttm =
        clean(income.ttm_sum_for_end_date(ts_code, trade_date, latest_end, REVENUE_COLUMN));
    let profit_ttm = clean(income.ttm_sum_for_end_date(
        ts_code,
        trade_date,
        latest_end,
        NET_PROFIT_ATTR_P_COLUMN,
    ));
    let profit_ttm_yoy =
        clean(income.ttm_sum_for_end_date(ts_code, trade_date, yoy_end, NET_PROFIT_ATTR_P_COLUMN));
    let profit_ttm_yoy_growth = yoy_pct(profit_ttm, profit_ttm_yoy);
    let net_income_ttm =
        clean(income.ttm_sum_for_end_date(ts_code, trade_date, latest_end, NET_PROFIT_COLUMN));
    let income_tax_ttm =
        clean(income.ttm_sum_for_end_date(ts_code, trade_date, latest_end, INCOME_TAX_COLUMN));
    let total_profit_ttm =
        clean(income.ttm_sum_for_end_date(ts_code, trade_date, latest_end, TOTAL_PROFIT_COLUMN));
    let interest_expense_ttm =
        clean(income.ttm_sum_for_end_date(ts_code, trade_date, latest_end, INT_EXP_COLUMN));
    let rd_exp_ttm =
        clean(income.ttm_sum_for_end_date(ts_code, trade_date, latest_end, RD_EXP_COLUMN));
    let cfo_ttm = clean(cashflow.ttm_sum_for_end_date(ts_code, trade_date, latest_end, CFO_COLUMN));
    let cfi_ttm = clean(cashflow.ttm_sum_for_end_date(ts_code, trade_date, latest_end, CFI_COLUMN));
    let cff_ttm = clean(cashflow.ttm_sum_for_end_date(ts_code, trade_date, latest_end, CFF_COLUMN));

    let total_assets = clean(balance_record.column(TOTAL_ASSETS_COLUMN));
    let total_liab = clean(balance_record.column(TOTAL_LIAB_COLUMN));
    let money_cap = clean(balance_record.column(MONEY_CAP_COLUMN));
    let equity = clean(balance_record.column(EQUITY_COLUMN));
    let fix_assets = clean(balance_record.column(FIX_ASSETS_COLUMN));
    let accounts_receiv = clean(balance_record.column(ACCOUNTS_RECEIV_COLUMN));
    let inventories = clean(balance_record.column(INVENTORIES_COLUMN));
    let avg_assets = average_record_value(total_assets, previous_balance, TOTAL_ASSETS_COLUMN);
    let avg_equity = average_record_value(equity, previous_balance, EQUITY_COLUMN);
    let avg_receivables =
        average_record_value(accounts_receiv, previous_balance, ACCOUNTS_RECEIV_COLUMN);
    let avg_inventories = average_record_value(inventories, previous_balance, INVENTORIES_COLUMN);
    let tax = tax_rate(income_tax_ttm, total_profit_ttm);
    let ebit_ttm = derived_ebit_ttm(net_income_ttm, income_tax_ttm, interest_expense_ttm);
    let ic = invested_capital(balance_record);

    let mut slow_features = [None; SLOW_CONTINUOUS_FEATURE_COUNT];
    slow_features[0] = total_assets;
    slow_features[1] = revenue_ttm;
    slow_features[2] = safe_div_opt(profit_ttm, avg_equity);
    slow_features[3] = safe_div_opt(profit_ttm, avg_assets);
    slow_features[4] = ebit_ttm
        .zip(tax)
        .and_then(|(ebit, tax)| safe_div_opt(Some(ebit * (1.0 - tax)), ic));
    slow_features[5] = safe_div_opt(revenue_ttm, avg_assets);
    slow_features[6] = safe_div_opt(revenue_ttm, avg_receivables);
    slow_features[7] = safe_div_opt(revenue_ttm, avg_inventories);
    slow_features[8] = safe_div_opt(total_liab, total_assets);
    slow_features[9] = safe_div_opt(fix_assets, total_assets);
    slow_features[10] = safe_div_opt(rd_exp_ttm, revenue_ttm);

    Some(HazqComparableSnapshot {
        lifecycle: lifecycle_stage(cfo_ttm, cfi_ttm, cff_ttm),
        slow_features,
        equity,
        total_liab,
        money_cap,
        revenue_ttm,
        ebit_ttm,
        profit_ttm_yoy_growth,
        profit_q: clean(income_record.column(NET_PROFIT_ATTR_P_COLUMN)),
        cfo_ttm,
    })
}

fn base_value_from_snapshot(
    base: HazqComparableBase,
    snapshot: &HazqComparableSnapshot,
    market_cap: Option<f64>,
) -> Option<f64> {
    let market_cap = market_cap.filter(|value| *value > EPS);
    let ev = market_cap
        .zip(snapshot.total_liab)
        .zip(snapshot.money_cap)
        .and_then(|((market_cap, total_liab), money_cap)| {
            finite_value(market_cap + total_liab - money_cap).filter(|value| *value > EPS)
        });
    match base {
        HazqComparableBase::Bp => safe_div_opt(snapshot.equity, market_cap),
        HazqComparableBase::Ebit2Ev => safe_div_opt(snapshot.ebit_ttm, ev),
        HazqComparableBase::Ep => None,
        HazqComparableBase::EpQ => safe_div_opt(snapshot.profit_q, market_cap),
        HazqComparableBase::Ocfp => safe_div_opt(snapshot.cfo_ttm, market_cap),
        HazqComparableBase::Sales2Ev => safe_div_opt(snapshot.revenue_ttm, ev),
        HazqComparableBase::Dp | HazqComparableBase::EpPercentile | HazqComparableBase::EpFttm => {
            None
        }
    }
}

fn continuous_features_from_snapshot(
    snapshot: &HazqComparableSnapshot,
    total_mv: Option<f64>,
) -> [Option<f64>; CONTINUOUS_FEATURE_COUNT] {
    let mut values = [None; CONTINUOUS_FEATURE_COUNT];
    values[0] = total_mv;
    for (idx, value) in snapshot.slow_features.iter().enumerate() {
        values[idx + 1] = *value;
    }
    values
}

fn feature_unit_vector(
    lifecycle: Option<LifecycleStage>,
    continuous_z: &[Vec<Option<f64>>],
    instrument_idx: usize,
) -> Option<[f64; FEATURE_DIM]> {
    let mut values = [0.0; FEATURE_DIM];
    if let Some(stage) = lifecycle {
        values[stage.idx()] = 0.5;
    }
    for dim in 0..CONTINUOUS_FEATURE_COUNT {
        values[LIFECYCLE_STAGE_COUNT + dim] =
            clean(*continuous_z.get(dim)?.get(instrument_idx)?).unwrap_or(0.0);
    }
    normalize(values)
}

fn peer_profiles_from_points(
    points: &[ComparablePoint],
    instrument_count: usize,
) -> Vec<PeerProfile> {
    let mut all_peers = vec![Vec::new(); instrument_count];
    let mut top_heaps = vec![BinaryHeap::new(); instrument_count];
    if points.len() >= 2 {
        for left_idx in 0..points.len() - 1 {
            for right_idx in left_idx + 1..points.len() {
                let similarity = cosine_dot(&points[left_idx].values, &points[right_idx].values);
                if similarity <= SIMILARITY_THRESHOLD || !similarity.is_finite() {
                    continue;
                }
                let left = points[left_idx].instrument_idx;
                let right = points[right_idx].instrument_idx;
                all_peers[left].push(PeerLink {
                    peer_idx: right,
                    similarity,
                });
                all_peers[right].push(PeerLink {
                    peer_idx: left,
                    similarity,
                });
                push_top_peer(
                    &mut top_heaps[left],
                    PeerCandidate {
                        similarity,
                        order: right,
                    },
                );
                push_top_peer(
                    &mut top_heaps[right],
                    PeerCandidate {
                        similarity,
                        order: left,
                    },
                );
            }
        }
    }

    all_peers
        .into_iter()
        .zip(top_heaps)
        .map(|(all, heap)| PeerProfile {
            all,
            top: heap
                .into_iter()
                .map(|Reverse(peer)| PeerLink {
                    peer_idx: peer.order,
                    similarity: peer.similarity,
                })
                .collect(),
        })
        .collect()
}

fn write_source_components_for_base_date(
    panel: &DailyPanel,
    trade_date: i32,
    peer_state: &ComparablePeerState,
    base: HazqComparableBase,
    base_column: &PanelColumn,
    growth_column: Option<&PanelColumn>,
    request_plan: &ComparableRequestPlan,
    source_values: &mut [Option<Vec<Option<f64>>>],
) -> Result<()> {
    let Some(date_idx) = panel.dates().iter().position(|date| *date == trade_date) else {
        return Ok(());
    };
    let instrument_count = panel.instruments().len();
    let date_offset = date_idx * instrument_count;
    for instrument_idx in 0..instrument_count {
        let offset = date_offset + instrument_idx;
        if !panel.is_present_offset(offset) {
            continue;
        }
        let profile = peer_state
            .peers
            .get(instrument_idx)
            .cloned()
            .unwrap_or_default();
        let stats = component_stats_for_stock(
            base_column,
            growth_column,
            panel,
            date_offset,
            instrument_idx,
            &profile,
        );
        for component in COMPONENTS {
            if component.is_time_zscore() {
                continue;
            }
            let key = source_key(base, component);
            if !request_plan.needs_source_component(base, component) {
                continue;
            }
            if let Some(values) = source_values[key].as_mut() {
                values[offset] = stats.values[component.idx()];
            }
        }
    }
    Ok(())
}

fn component_stats_for_stock(
    base_column: &PanelColumn,
    growth_column: Option<&PanelColumn>,
    panel: &DailyPanel,
    date_offset: usize,
    instrument_idx: usize,
    profile: &PeerProfile,
) -> ComponentStats {
    let own = clean(base_column.values()[date_offset + instrument_idx]);
    let all = peer_values(base_column, panel, date_offset, &profile.all);
    let top = peer_values(base_column, panel, date_offset, &profile.top);
    let top_values = top.iter().map(|peer| peer.value).collect::<Vec<_>>();
    let all_values = all.iter().map(|peer| peer.value).collect::<Vec<_>>();
    let all_count = all.len();
    let sim_sum = all.iter().map(|peer| peer.similarity).sum::<f64>();
    let sim_avg = if all_count > 0 {
        sim_sum / all_count as f64
    } else {
        0.0
    };
    let weighted = weighted_peer_mean(&all);
    let med = median(&top_values);
    let mut values = [None; COMPONENT_COUNT];
    values[HazqComparableComponent::Med.idx()] = med;
    values[HazqComparableComponent::Avg.idx()] = mean(&all_values);
    values[HazqComparableComponent::Weighted.idx()] = weighted;
    values[HazqComparableComponent::Max.idx()] = max_value(&top_values);
    values[HazqComparableComponent::Min.idx()] = min_value(&top_values);
    values[HazqComparableComponent::Wgt.idx()] =
        comparable_weighted_value(own, weighted, all_count, sim_avg, &all_values, false);
    values[HazqComparableComponent::Wgt2.idx()] =
        comparable_weighted_value(own, weighted, all_count, sim_avg, &all_values, true);
    values[HazqComparableComponent::Dst.idx()] =
        own.zip(med).and_then(|(own, med)| finite_value(own - med));
    values[HazqComparableComponent::Prm.idx()] = own
        .zip(med)
        .and_then(|(own, med)| safe_div(own, med).and_then(|value| finite_value(value - 1.0)));
    let gap = gap_components(
        base_column,
        growth_column,
        panel,
        date_offset,
        instrument_idx,
        &profile.all,
    );
    values[HazqComparableComponent::GapAvg.idx()] = gap.gap_avg;
    values[HazqComparableComponent::GapMmm.idx()] = gap.gap_mmm;
    ComponentStats { values }
}

#[derive(Clone, Copy, Debug, Default)]
struct GapStats {
    gap_avg: Option<f64>,
    gap_mmm: Option<f64>,
}

fn gap_components(
    base_column: &PanelColumn,
    growth_column: Option<&PanelColumn>,
    panel: &DailyPanel,
    date_offset: usize,
    instrument_idx: usize,
    peers: &[PeerLink],
) -> GapStats {
    let Some(growth_column) = growth_column else {
        return GapStats::default();
    };
    let own_growth = clean(growth_column.values()[date_offset + instrument_idx]);
    let Some(own_growth) = own_growth else {
        return GapStats::default();
    };
    let mut high = Vec::new();
    let mut low = Vec::new();
    for peer in peers {
        let offset = date_offset + peer.peer_idx;
        if !panel.is_present_offset(offset) {
            continue;
        }
        let Some(value) = clean(base_column.values()[offset]) else {
            continue;
        };
        let Some(peer_growth) = clean(growth_column.values()[offset]) else {
            continue;
        };
        if peer_growth > own_growth {
            high.push(value);
        } else if peer_growth < own_growth {
            low.push(value);
        }
    }
    GapStats {
        gap_avg: mean(&low)
            .zip(mean(&high))
            .and_then(|(low, high)| finite_value(low - high)),
        gap_mmm: max_value(&low)
            .zip(min_value(&high))
            .and_then(|(low, high)| finite_value(low - high)),
    }
}

#[derive(Clone, Copy, Debug)]
struct PeerValue {
    value: f64,
    similarity: f64,
}

fn peer_values(
    base_column: &PanelColumn,
    panel: &DailyPanel,
    date_offset: usize,
    peers: &[PeerLink],
) -> Vec<PeerValue> {
    peers
        .iter()
        .filter_map(|peer| {
            let offset = date_offset + peer.peer_idx;
            if !panel.is_present_offset(offset) {
                return None;
            }
            let value = clean(base_column.values()[offset])?;
            Some(PeerValue {
                value,
                similarity: peer.similarity,
            })
        })
        .collect()
}

fn comparable_weighted_value(
    own: Option<f64>,
    comparable: Option<f64>,
    peer_count: usize,
    sim_avg: f64,
    peer_values: &[f64],
    count_sigmoid: bool,
) -> Option<f64> {
    let own = own?;
    if peer_count == 0 {
        return finite_value(own);
    }
    let comparable = comparable?;
    let weight = if count_sigmoid {
        sigmoid(peer_count as f64 / TOP_PEER_COUNT as f64) * sim_avg
    } else {
        let min_value = min_value(peer_values)?;
        let max_value = max_value(peer_values)?;
        safe_div(min_value, max_value)? * sim_avg
    };
    if !(0.0..=1.0).contains(&weight) || !weight.is_finite() {
        return None;
    }
    finite_value(own * (1.0 - weight) + comparable * weight)
}

fn source_component_needs(outputs: &[HazqComparableValueOutput]) -> Vec<bool> {
    let mut needs = vec![false; BASE_COUNT * COMPONENT_COUNT];
    for output in outputs {
        needs[source_key(output.base, output.component.source_component())] = true;
    }
    needs
}

fn source_storage(source_needs: &[bool], panel: &DailyPanel) -> Vec<Option<Vec<Option<f64>>>> {
    source_needs
        .iter()
        .map(|needed| needed.then(|| vec![None; panel.shape_len()]))
        .collect()
}

fn source_key(base: HazqComparableBase, component: HazqComparableComponent) -> usize {
    base.idx() * COMPONENT_COUNT + component.idx()
}

fn requested_outputs(requested_ids: &[String]) -> Vec<HazqComparableValueOutput> {
    let requested = requested_ids.iter().cloned().collect::<BTreeSet<_>>();
    all_outputs()
        .into_iter()
        .filter(|output| requested.contains(&output.id()))
        .collect()
}

fn dependencies_for_output(output: HazqComparableValueOutput) -> Vec<DataRequest> {
    let mut dependencies = common_dependencies(output);
    if output.base == HazqComparableBase::Dp {
        dependencies.push(dividend_dependency());
    }
    let mut consensus_columns = Vec::new();
    if matches!(
        output.component,
        HazqComparableComponent::GapAvg | HazqComparableComponent::GapMmm
    ) {
        consensus_columns.push(CONSENSUS_GROWTH_COLUMN);
    }
    if output.base == HazqComparableBase::EpFttm {
        consensus_columns.push(CONSENSUS_PE_ROLL_COLUMN);
    }
    if !consensus_columns.is_empty() {
        dependencies.push(DataRequest::new(
            DatasetId::StockConsensus,
            &consensus_columns,
        ));
    }
    dependencies
}

fn common_dependencies(output: HazqComparableValueOutput) -> Vec<DataRequest> {
    let mut daily_basic_columns = vec![TOTAL_MV_COLUMN];
    if matches!(
        output.base,
        HazqComparableBase::Ep | HazqComparableBase::EpPercentile
    ) {
        daily_basic_columns.push(PE_TTM_COLUMN);
    }
    vec![
        DataRequest::new(DatasetId::StockDailyBasic, &daily_basic_columns),
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
        DataRequest::new(DatasetId::StockBarraDaily, &["SIZE"]),
        DataRequest::new(DatasetId::StockSwClassification, &["l1_code"]),
    ]
}

fn dividend_dependency() -> DataRequest {
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
    )
}

fn tags() -> Vec<String> {
    [
        "HAZQ",
        "cs_network",
        "fundamental",
        "financial",
        "pit",
        "valuation",
        "comparable_company",
        "size_neutralize",
        "sector_neutralize",
        "daily",
    ]
    .iter()
    .map(|tag| (*tag).to_string())
    .collect()
}

fn dividend_sums_by_date<'a>(
    panel: &DailyPanel,
    dividends: &DividendReader<'a>,
) -> BTreeMap<i32, HashMap<&'a str, f64>> {
    panel
        .dates()
        .iter()
        .copied()
        .map(|trade_date| {
            (
                trade_date,
                dividends.implemented_ltm_sum_by_stock(add_months(trade_date, -12), trade_date),
            )
        })
        .collect()
}

fn lifecycle_stage(cfo: Option<f64>, cfi: Option<f64>, cff: Option<f64>) -> Option<LifecycleStage> {
    let cfo_positive = clean(cfo)? >= 0.0;
    let cfi_positive = clean(cfi)? >= 0.0;
    let cff_positive = clean(cff)? >= 0.0;
    match (cfo_positive, cfi_positive, cff_positive) {
        (false, false, true) => Some(LifecycleStage::Introduction),
        (true, false, true) => Some(LifecycleStage::Growth),
        (true, false, false) => Some(LifecycleStage::Mature),
        (false, true, _) => Some(LifecycleStage::Decline),
        (false, false, false) | (true, true, true) | (true, true, false) => {
            Some(LifecycleStage::ShakeOut)
        }
    }
}

fn average_record_value(
    current: Option<f64>,
    previous_balance: Option<PitFinancialRecordView<'_>>,
    column: &str,
) -> Option<f64> {
    let current = current?;
    let previous = previous_balance.and_then(|record| clean(record.column(column)))?;
    finite_value((current + previous) * 0.5)
}

fn same_quarter_previous_year(end_date: i32) -> i32 {
    end_date - 10_000
}

fn yoy_pct(current: Option<f64>, previous: Option<f64>) -> Option<f64> {
    let current = clean(current)?;
    let previous = clean(previous)?;
    (previous.abs() > EPS)
        .then_some(100.0 * (current - previous) / previous.abs())
        .filter(|value| value.is_finite())
}

fn invested_capital(balance: PitFinancialRecordView<'_>) -> Option<f64> {
    let equity = clean(balance.column(EQUITY_COLUMN))?;
    let cash = clean(balance.column(MONEY_CAP_COLUMN))?;
    let debt = clean_or_zero(balance.column(SHORT_BORROW_COLUMN))
        + clean_or_zero(balance.column(NON_CURRENT_LIAB_DUE_1Y_COLUMN))
        + clean_or_zero(balance.column(LONG_BORROW_COLUMN))
        + clean_or_zero(balance.column(BOND_PAYABLE_COLUMN));
    finite_value(equity + debt - cash)
}

fn derived_ebit_ttm(
    net_income_ttm: Option<f64>,
    income_tax_ttm: Option<f64>,
    interest_expense_ttm: Option<f64>,
) -> Option<f64> {
    let value = clean_or_zero(net_income_ttm)
        + clean_or_zero(income_tax_ttm)
        + clean_or_zero(interest_expense_ttm);
    finite_value(value)
}

fn tax_rate(income_tax: Option<f64>, total_profit: Option<f64>) -> Option<f64> {
    let value = safe_div_opt(income_tax, total_profit)?.clamp(0.0, 0.25);
    finite_value(value)
}

fn normalize(mut values: [f64; FEATURE_DIM]) -> Option<[f64; FEATURE_DIM]> {
    let norm_sq = values.iter().map(|value| value * value).sum::<f64>();
    if norm_sq <= EPS || !norm_sq.is_finite() {
        return None;
    }
    let norm = norm_sq.sqrt();
    for value in &mut values {
        *value /= norm;
    }
    Some(values)
}

fn cosine_dot(left: &[f64; FEATURE_DIM], right: &[f64; FEATURE_DIM]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}

#[derive(Clone, Copy, Debug)]
struct PeerCandidate {
    similarity: f64,
    order: usize,
}

impl PartialEq for PeerCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.similarity.total_cmp(&other.similarity) == std::cmp::Ordering::Equal
            && self.order == other.order
    }
}

impl Eq for PeerCandidate {}

impl PartialOrd for PeerCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PeerCandidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.similarity
            .total_cmp(&other.similarity)
            .then_with(|| self.order.cmp(&other.order))
    }
}

fn push_top_peer(heap: &mut BinaryHeap<Reverse<PeerCandidate>>, candidate: PeerCandidate) {
    if !candidate.similarity.is_finite() {
        return;
    }
    if heap.len() < TOP_PEER_COUNT {
        heap.push(Reverse(candidate));
    } else if heap
        .peek()
        .is_some_and(|Reverse(current)| candidate > *current)
    {
        heap.pop();
        heap.push(Reverse(candidate));
    }
}

fn weighted_peer_mean(peers: &[PeerValue]) -> Option<f64> {
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    for peer in peers {
        numerator += peer.similarity * peer.value;
        denominator += peer.similarity;
    }
    (denominator > EPS)
        .then_some(numerator / denominator)
        .filter(|value| value.is_finite())
}

fn mean(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    finite_value(values.iter().sum::<f64>() / values.len() as f64)
}

fn median(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        finite_value((sorted[mid - 1] + sorted[mid]) * 0.5)
    } else {
        finite_value(sorted[mid])
    }
}

fn min_value(values: &[f64]) -> Option<f64> {
    values
        .iter()
        .copied()
        .reduce(f64::min)
        .and_then(finite_value)
}

fn max_value(values: &[f64]) -> Option<f64> {
    values
        .iter()
        .copied()
        .reduce(f64::max)
        .and_then(finite_value)
}

fn sigmoid(value: f64) -> f64 {
    1.0 / (1.0 + (-value).exp())
}

fn safe_div(numerator: f64, denominator: f64) -> Option<f64> {
    (denominator.abs() > EPS)
        .then_some(numerator / denominator)
        .filter(|value| value.is_finite())
}

fn safe_div_opt(numerator: Option<f64>, denominator: Option<f64>) -> Option<f64> {
    safe_div(clean(numerator)?, clean(denominator)?)
}

fn clean_or_zero(value: Option<f64>) -> f64 {
    clean(value).unwrap_or(0.0)
}

fn finite_value(value: f64) -> Option<f64> {
    value.is_finite().then_some(value)
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

#[macro_export]
macro_rules! define_hazq_comparable_value_factor {
    ($struct_name:ident, $base:ident, $component:ident) => {
        use std::any::Any;

        use crate::core::{FactorContext, FactorSeries, FactorSpec};
        use crate::data::DataPool;
        use crate::error::{err, Result};
        use crate::factor::common::hazq_comparable_value::{
            compute_requested, compute_requested_stateful, spec, HazqComparableBase,
            HazqComparableComponent, HazqComparableValueComputeState, HazqComparableValueOutput,
            PROVIDER_KEY,
        };
        use crate::factor::{Factor, FactorUpdatePolicy};

        pub struct $struct_name;

        pub fn create() -> Box<dyn Factor> {
            Box::new($struct_name)
        }

        impl $struct_name {
            fn output() -> HazqComparableValueOutput {
                HazqComparableValueOutput::new(
                    HazqComparableBase::$base,
                    HazqComparableComponent::$component,
                )
            }
        }

        impl Factor for $struct_name {
            fn spec(&self) -> FactorSpec {
                spec(Self::output())
            }

            fn compute_provider_key(&self) -> String {
                PROVIDER_KEY.to_string()
            }

            fn update_policy(&self) -> FactorUpdatePolicy {
                FactorUpdatePolicy::FinancialEventStateDailyFast
            }

            fn initial_compute_state(&self, _requested_ids: &[String]) -> Box<dyn Any + Send> {
                Box::new(HazqComparableValueComputeState::default())
            }

            fn compute(&self, context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
                let id = Self::output().id();
                compute_requested(&[id.clone()], context, data)?
                    .into_iter()
                    .find(|series| series.spec.id == id)
                    .ok_or_else(|| {
                        err(format!(
                            "HAZQ comparable value provider did not return {}",
                            id
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

            fn compute_many_stateful(
                &self,
                requested_ids: &[String],
                context: &FactorContext,
                data: &DataPool,
                state: &mut (dyn Any + Send),
            ) -> Result<Vec<FactorSeries>> {
                let state = state
                    .downcast_mut::<HazqComparableValueComputeState>()
                    .ok_or_else(|| {
                        err("HAZQ comparable value provider received incompatible state")
                    })?;
                compute_requested_stateful(requested_ids, context, data, state)
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use crate::core::{AssetClass, FactorContext, Frequency};

    use super::*;

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1e-10,
            "expected {expected}, got {actual}"
        );
    }

    fn point(instrument_idx: usize, first_dim: f64) -> ComparablePoint {
        let mut values = [0.0; FEATURE_DIM];
        values[0] = first_dim;
        values[1] = (1.0 - first_dim * first_dim).max(0.0).sqrt();
        ComparablePoint {
            instrument_idx,
            values,
        }
    }

    #[test]
    fn hazq_comparable_registers_117_outputs_with_required_tags() {
        let outputs = all_outputs();
        assert_eq!(outputs.len(), 117);
        for id in [
            "comp_ep_med",
            "comp_bp_prm_zscore",
            "comp_ep_gap_avg",
            "comp_bp_gap_mmm",
            "comp_ep_fttm_med",
            "comp_ep_fttm_gap_mmm",
        ] {
            assert!(
                outputs.iter().any(|output| output.id() == id),
                "missing {id}"
            );
        }

        let spec = spec(HazqComparableValueOutput::new(
            HazqComparableBase::Ep,
            HazqComparableComponent::Med,
        ));
        for tag in ["HAZQ", "cs_network", "fundamental"] {
            assert!(spec.tags.contains(&tag.to_string()));
        }
        assert!(!spec
            .dependencies
            .iter()
            .any(|request| request.dataset == DatasetId::StockDailyPv));
        assert!(!spec
            .dependencies
            .iter()
            .any(|request| request.dataset == DatasetId::StockConsensus));
        assert!(spec
            .dependencies
            .iter()
            .any(|request| request.dataset == DatasetId::StockDailyBasic));
    }

    #[test]
    fn hazq_comparable_lifecycle_stage_uses_dickinson_signs() {
        assert!(matches!(
            lifecycle_stage(Some(-1.0), Some(-1.0), Some(1.0)),
            Some(LifecycleStage::Introduction)
        ));
        assert!(matches!(
            lifecycle_stage(Some(1.0), Some(-1.0), Some(1.0)),
            Some(LifecycleStage::Growth)
        ));
        assert!(matches!(
            lifecycle_stage(Some(1.0), Some(-1.0), Some(-1.0)),
            Some(LifecycleStage::Mature)
        ));
        assert!(matches!(
            lifecycle_stage(Some(-1.0), Some(1.0), Some(-1.0)),
            Some(LifecycleStage::Decline)
        ));
        assert!(matches!(
            lifecycle_stage(Some(1.0), Some(1.0), Some(-1.0)),
            Some(LifecycleStage::ShakeOut)
        ));
        assert!(lifecycle_stage(None, Some(1.0), Some(1.0)).is_none());
    }

    #[test]
    fn hazq_comparable_base_values_use_market_cap_and_ev() {
        let snapshot = HazqComparableSnapshot {
            equity: Some(50.0),
            total_liab: Some(40.0),
            money_cap: Some(10.0),
            revenue_ttm: Some(60.0),
            ebit_ttm: Some(12.0),
            profit_q: Some(2.0),
            cfo_ttm: Some(6.0),
            ..Default::default()
        };
        assert_close(
            base_value_from_snapshot(HazqComparableBase::Bp, &snapshot, Some(100.0)).unwrap(),
            0.5,
        );
        assert_close(
            base_value_from_snapshot(HazqComparableBase::Ebit2Ev, &snapshot, Some(100.0)).unwrap(),
            12.0 / 130.0,
        );
        assert_eq!(
            base_value_from_snapshot(HazqComparableBase::Ep, &snapshot, Some(100.0)),
            None
        );
        assert_close(
            base_value_from_snapshot(HazqComparableBase::EpQ, &snapshot, Some(100.0)).unwrap(),
            0.02,
        );
        assert_close(
            base_value_from_snapshot(HazqComparableBase::Ocfp, &snapshot, Some(100.0)).unwrap(),
            0.06,
        );
        assert_close(
            base_value_from_snapshot(HazqComparableBase::Sales2Ev, &snapshot, Some(100.0)).unwrap(),
            60.0 / 130.0,
        );
        assert_eq!(
            base_value_from_snapshot(HazqComparableBase::Dp, &snapshot, Some(100.0)),
            None
        );
        assert_eq!(
            base_value_from_snapshot(HazqComparableBase::EpPercentile, &snapshot, Some(100.0)),
            None
        );
        assert_eq!(
            base_value_from_snapshot(HazqComparableBase::EpFttm, &snapshot, Some(100.0)),
            None
        );
        assert_eq!(
            base_value_from_snapshot(HazqComparableBase::Bp, &snapshot, Some(0.0)),
            None
        );
    }

    #[test]
    fn hazq_comparable_ep_uses_reciprocal_pe_ttm_value() {
        assert_close(reciprocal_valuation(Some(20.0)).unwrap(), 0.05);
        assert_close(reciprocal_valuation(Some(-10.0)).unwrap(), -0.1);
        assert_eq!(reciprocal_valuation(Some(0.0)), None);
        assert_eq!(reciprocal_valuation(None), None);
    }

    #[test]
    fn hazq_comparable_derived_ebit_ttm_uses_net_income_tax_and_interest() {
        assert_close(
            derived_ebit_ttm(Some(100.0), Some(20.0), Some(5.0)).unwrap(),
            125.0,
        );
        assert_close(
            derived_ebit_ttm(Some(100.0), None, Some(5.0)).unwrap(),
            105.0,
        );
    }

    #[test]
    fn hazq_comparable_peer_profiles_threshold_and_top6() {
        let points = vec![point(0, 1.0), point(1, 0.95), point(2, 0.91), point(3, 0.5)];
        let profiles = peer_profiles_from_points(&points, 4);

        assert_eq!(profiles[0].all.len(), 2);
        assert!(profiles[0].all.iter().any(|peer| peer.peer_idx == 1));
        assert!(profiles[0].all.iter().any(|peer| peer.peer_idx == 2));
        assert!(!profiles[0].all.iter().any(|peer| peer.peer_idx == 3));
    }

    #[test]
    fn hazq_comparable_components_use_top_and_all_peers() {
        let context = FactorContext {
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            start_date: 20260105,
            end_date: 20260105,
            load_start_date: 20260105,
            load_dates: vec![20260105],
            target_dates: vec![20260105],
        };
        let panel = DailyPanel::from_index(
            vec![20260105],
            vec![
                "000001.SZ".to_string(),
                "000002.SZ".to_string(),
                "000003.SZ".to_string(),
            ],
            &context.target_dates,
            vec![true, true, true],
        )
        .unwrap();
        let base = panel
            .column_from_values(vec![Some(10.0), Some(12.0), Some(20.0)])
            .unwrap();
        let profile = PeerProfile {
            all: vec![
                PeerLink {
                    peer_idx: 1,
                    similarity: 0.9,
                },
                PeerLink {
                    peer_idx: 2,
                    similarity: 0.95,
                },
            ],
            top: vec![PeerLink {
                peer_idx: 1,
                similarity: 0.9,
            }],
        };

        let stats = component_stats_for_stock(&base, None, &panel, 0, 0, &profile);

        assert_close(
            stats.values[HazqComparableComponent::Med.idx()].unwrap(),
            12.0,
        );
        assert_close(
            stats.values[HazqComparableComponent::Avg.idx()].unwrap(),
            16.0,
        );
        assert_close(
            stats.values[HazqComparableComponent::Weighted.idx()].unwrap(),
            (12.0 * 0.9 + 20.0 * 0.95) / 1.85,
        );
        assert_close(
            stats.values[HazqComparableComponent::Dst.idx()].unwrap(),
            -2.0,
        );
        assert_close(
            stats.values[HazqComparableComponent::Prm.idx()].unwrap(),
            10.0 / 12.0 - 1.0,
        );
        assert_eq!(stats.values[HazqComparableComponent::GapAvg.idx()], None);
        assert_eq!(stats.values[HazqComparableComponent::GapMmm.idx()], None);
    }

    #[test]
    fn hazq_comparable_gap_components_group_by_growth_against_all_peers() {
        let panel = DailyPanel::from_index(
            vec![20260105],
            vec![
                "000001.SZ".to_string(),
                "000002.SZ".to_string(),
                "000003.SZ".to_string(),
                "000004.SZ".to_string(),
                "000005.SZ".to_string(),
            ],
            &[20260105],
            vec![true, true, true, true, true],
        )
        .unwrap();
        let base = panel
            .column_from_values(vec![
                Some(10.0),
                Some(8.0),
                Some(10.0),
                Some(2.0),
                Some(4.0),
            ])
            .unwrap();
        let growth = panel
            .column_from_values(vec![
                Some(10.0),
                Some(5.0),
                Some(7.0),
                Some(12.0),
                Some(15.0),
            ])
            .unwrap();
        let profile = PeerProfile {
            all: vec![
                PeerLink {
                    peer_idx: 1,
                    similarity: 0.91,
                },
                PeerLink {
                    peer_idx: 2,
                    similarity: 0.92,
                },
                PeerLink {
                    peer_idx: 3,
                    similarity: 0.93,
                },
                PeerLink {
                    peer_idx: 4,
                    similarity: 0.94,
                },
            ],
            top: vec![PeerLink {
                peer_idx: 3,
                similarity: 0.93,
            }],
        };

        let stats = component_stats_for_stock(&base, Some(&growth), &panel, 0, 0, &profile);

        assert_close(
            stats.values[HazqComparableComponent::GapAvg.idx()].unwrap(),
            6.0,
        );
        assert_close(
            stats.values[HazqComparableComponent::GapMmm.idx()].unwrap(),
            8.0,
        );
    }

    #[test]
    fn hazq_comparable_gap_components_require_both_growth_groups() {
        let panel = DailyPanel::from_index(
            vec![20260105],
            vec!["000001.SZ".to_string(), "000002.SZ".to_string()],
            &[20260105],
            vec![true, true],
        )
        .unwrap();
        let base = panel
            .column_from_values(vec![Some(10.0), Some(8.0)])
            .unwrap();
        let growth = panel
            .column_from_values(vec![Some(10.0), Some(5.0)])
            .unwrap();
        let profile = PeerProfile {
            all: vec![PeerLink {
                peer_idx: 1,
                similarity: 0.91,
            }],
            top: Vec::new(),
        };

        let stats = component_stats_for_stock(&base, Some(&growth), &panel, 0, 0, &profile);

        assert_eq!(stats.values[HazqComparableComponent::GapAvg.idx()], None);
        assert_eq!(stats.values[HazqComparableComponent::GapMmm.idx()], None);
    }

    #[test]
    fn hazq_comparable_yoy_growth_uses_abs_previous_denominator() {
        assert_close(yoy_pct(Some(120.0), Some(100.0)).unwrap(), 20.0);
        assert_close(yoy_pct(Some(-80.0), Some(-100.0)).unwrap(), 20.0);
        assert_eq!(yoy_pct(Some(120.0), Some(0.0)), None);
    }

    #[test]
    fn hazq_comparable_non_bj_panel_removes_bj_rows() {
        let panel = DailyPanel::from_index(
            vec![20260105],
            vec![
                "000001.SZ".to_string(),
                "920001.BJ".to_string(),
                "600000.SH".to_string(),
            ],
            &[20260105],
            vec![true, true, true],
        )
        .unwrap();

        let filtered = non_bj_panel(&panel).unwrap();

        assert_eq!(
            filtered.instruments(),
            &["000001.SZ".to_string(), "600000.SH".to_string()]
        );
        assert_eq!(filtered.shape_len(), 2);
        assert!(filtered.is_present_offset(0));
        assert!(filtered.is_present_offset(1));
    }

    #[test]
    fn hazq_comparable_requested_outputs_preserve_known_ids() {
        let requested = vec![
            "comp_ep_med".to_string(),
            "comp_bp_prm_zscore".to_string(),
            "unknown".to_string(),
        ];
        let outputs = requested_outputs(&requested);
        let ids = outputs
            .iter()
            .map(|output| output.id())
            .collect::<BTreeSet<_>>();

        assert_eq!(ids.len(), 2);
        assert!(ids.contains("comp_ep_med"));
        assert!(ids.contains("comp_bp_prm_zscore"));
    }

    #[test]
    fn hazq_comparable_request_plan_maps_outputs_to_needed_bases_and_sources() {
        let plan = ComparableRequestPlan::from_requested_ids(&["comp_ep_med".to_string()]);
        assert_eq!(plan.requested_bases(), vec![HazqComparableBase::Ep]);
        assert!(plan.needs_source_component(HazqComparableBase::Ep, HazqComparableComponent::Med));
        assert!(!plan.needs_source_component(HazqComparableBase::Ep, HazqComparableComponent::Avg));
        assert!(!plan.needs_source_component(HazqComparableBase::Bp, HazqComparableComponent::Med));

        let plan = ComparableRequestPlan::from_requested_ids(&["comp_ep_prm_zscore".to_string()]);
        assert_eq!(plan.requested_bases(), vec![HazqComparableBase::Ep]);
        assert!(plan.needs_source_component(HazqComparableBase::Ep, HazqComparableComponent::Prm));
        assert!(!plan
            .needs_source_component(HazqComparableBase::Ep, HazqComparableComponent::PrmZscore));

        let plan =
            ComparableRequestPlan::from_requested_ids(&["comp_ep_percentile_med".to_string()]);
        assert_eq!(
            plan.requested_bases(),
            vec![HazqComparableBase::EpPercentile]
        );
        assert!(plan.needs_ep_intermediate());
        assert!(plan.needs_source_component(
            HazqComparableBase::EpPercentile,
            HazqComparableComponent::Med
        ));

        let plan = ComparableRequestPlan::from_requested_ids(&["comp_ep_gap_avg".to_string()]);
        assert_eq!(plan.requested_bases(), vec![HazqComparableBase::Ep]);
        assert!(plan.needs_consensus_growth());
        assert!(!plan.needs_consensus_pe_roll());
        assert!(
            plan.needs_source_component(HazqComparableBase::Ep, HazqComparableComponent::GapAvg)
        );

        let plan = ComparableRequestPlan::from_requested_ids(&["comp_ep_fttm_med".to_string()]);
        assert_eq!(plan.requested_bases(), vec![HazqComparableBase::EpFttm]);
        assert!(!plan.needs_consensus_growth());
        assert!(plan.needs_consensus_pe_roll());

        let plan = ComparableRequestPlan::from_requested_ids(&[
            "comp_ep_med".to_string(),
            "comp_ep_percentile_avg".to_string(),
        ]);
        assert_eq!(
            plan.requested_bases(),
            vec![HazqComparableBase::Ep, HazqComparableBase::EpPercentile]
        );
        assert!(plan.needs_ep_intermediate());
    }

    #[test]
    fn hazq_comparable_dependencies_include_all_financial_lines_and_requested_consensus() {
        let ep_output =
            HazqComparableValueOutput::new(HazqComparableBase::Ep, HazqComparableComponent::Med);
        let dp_output =
            HazqComparableValueOutput::new(HazqComparableBase::Dp, HazqComparableComponent::Med);
        let gap_output =
            HazqComparableValueOutput::new(HazqComparableBase::Ep, HazqComparableComponent::GapAvg);
        let fttm_output = HazqComparableValueOutput::new(
            HazqComparableBase::EpFttm,
            HazqComparableComponent::Med,
        );
        let fttm_gap_output = HazqComparableValueOutput::new(
            HazqComparableBase::EpFttm,
            HazqComparableComponent::GapMmm,
        );
        let ep_dependencies = dependencies_for_output(ep_output);
        let dp_dependencies = dependencies_for_output(dp_output);
        let gap_dependencies = dependencies_for_output(gap_output);
        let fttm_dependencies = dependencies_for_output(fttm_output);
        let fttm_gap_dependencies = dependencies_for_output(fttm_gap_output);

        assert!(!ep_dependencies
            .iter()
            .any(|request| request.dataset == DatasetId::StockDividend));
        assert!(!ep_dependencies
            .iter()
            .any(|request| request.dataset == DatasetId::StockConsensus));
        assert!(ep_dependencies.iter().any(|request| {
            request.dataset == DatasetId::StockDailyBasic
                && request.columns.contains(&TOTAL_MV_COLUMN.to_string())
                && request.columns.contains(&PE_TTM_COLUMN.to_string())
        }));
        assert!(dp_dependencies.iter().any(|request| {
            request.dataset == DatasetId::StockDailyBasic
                && request.columns.contains(&TOTAL_MV_COLUMN.to_string())
                && !request.columns.contains(&PE_TTM_COLUMN.to_string())
        }));
        assert!(dp_dependencies
            .iter()
            .any(|request| request.dataset == DatasetId::StockDividend));
        assert!(gap_dependencies.iter().any(|request| {
            request.dataset == DatasetId::StockConsensus
                && request
                    .columns
                    .contains(&CONSENSUS_GROWTH_COLUMN.to_string())
        }));
        assert!(fttm_dependencies.iter().any(|request| {
            request.dataset == DatasetId::StockConsensus
                && request
                    .columns
                    .contains(&CONSENSUS_PE_ROLL_COLUMN.to_string())
        }));
        assert!(fttm_gap_dependencies.iter().any(|request| {
            request.dataset == DatasetId::StockConsensus
                && request
                    .columns
                    .contains(&CONSENSUS_GROWTH_COLUMN.to_string())
                && request
                    .columns
                    .contains(&CONSENSUS_PE_ROLL_COLUMN.to_string())
        }));

        let deps = ep_dependencies
            .into_iter()
            .map(|request| (request.dataset, request.financial_quarters))
            .collect::<BTreeMap<_, _>>();

        assert_eq!(
            deps.get(&DatasetId::StockIncome),
            Some(&Some(FINANCIAL_QUARTERS))
        );
        assert_eq!(
            deps.get(&DatasetId::StockBalanceSheet),
            Some(&Some(FINANCIAL_QUARTERS))
        );
        assert_eq!(
            deps.get(&DatasetId::StockCashFlow),
            Some(&Some(FINANCIAL_QUARTERS))
        );
    }
}

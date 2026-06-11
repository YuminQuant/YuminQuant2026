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
const BASE_COUNT: usize = 8;
const COMPONENT_COUNT: usize = 11;
const LIFECYCLE_STAGE_COUNT: usize = 5;
const CONTINUOUS_FEATURE_COUNT: usize = 12;
const SLOW_CONTINUOUS_FEATURE_COUNT: usize = CONTINUOUS_FEATURE_COUNT - 1;
const FEATURE_DIM: usize = LIFECYCLE_STAGE_COUNT + CONTINUOUS_FEATURE_COUNT;

const TOTAL_MV_COLUMN: &str = "total_mv";

const REVENUE_COLUMN: &str = "revenue";
const NET_PROFIT_ATTR_P_COLUMN: &str = "n_income_attr_p";
const EBIT_COLUMN: &str = "ebit";
const INCOME_TAX_COLUMN: &str = "income_tax";
const TOTAL_PROFIT_COLUMN: &str = "total_profit";
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

const INCOME_COLUMNS: [&str; 6] = [
    REVENUE_COLUMN,
    NET_PROFIT_ATTR_P_COLUMN,
    EBIT_COLUMN,
    INCOME_TAX_COLUMN,
    TOTAL_PROFIT_COLUMN,
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
        format!("hazq_comp_{}_{}", self.base.id(), self.component.id())
    }
}

pub const BASES: [HazqComparableBase; 8] = [
    HazqComparableBase::Bp,
    HazqComparableBase::Dp,
    HazqComparableBase::Ebit2Ev,
    HazqComparableBase::Ep,
    HazqComparableBase::EpQ,
    HazqComparableBase::Ocfp,
    HazqComparableBase::Sales2Ev,
    HazqComparableBase::EpPercentile,
];

pub const COMPONENTS: [HazqComparableComponent; 11] = [
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
];

// TODO: Add GAP_AVG and GAP_MMM when analyst expected-growth data is wired into this provider.

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
    profit_ttm: Option<f64>,
    profit_q: Option<f64>,
    cfo_ttm: Option<f64>,
    cash_dividend_ltm: f64,
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
    income: FinancialPitReader<'a>,
    balance: FinancialPitReader<'a>,
    cashflow: FinancialPitReader<'a>,
    dividends: DividendReader<'a>,
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
            "HAZQ comparable-company value factor {} {}. It builds a PIT financial cosine-similarity network, uses peers with similarity above 0.9, skips analyst GAP components, excludes BJ stocks, and neutralizes the output by SW level-1 industry and Barra SIZE.",
            output.base.alias(),
            output.component.alias()
        ),
        dependencies: dependencies(),
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
    let outputs = requested_outputs(requested_ids);
    if outputs.is_empty() {
        return Ok(Vec::new());
    }

    let panel = non_bj_panel(data.stock_universe_panel()?)?;
    let inputs = hazq_inputs(data, &panel)?;
    let base_columns = base_value_columns(&inputs, &mut state.snapshot_cache)?;
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
    let source_needs = source_component_needs(&outputs);
    let mut source_values = source_storage(&source_needs, &panel);

    for trade_date in panel.dates().iter().copied() {
        if event_dates.contains(&trade_date) {
            let points =
                comparable_points_for_trade_date(&inputs, &mut state.snapshot_cache, trade_date)?;
            peer_state.peers = peer_profiles_from_points(&points, panel.instruments().len());
        }
        write_source_components_for_date(
            &panel,
            trade_date,
            &peer_state,
            &base_columns,
            &source_needs,
            &mut source_values,
        )?;
        peer_state.mark_processed(trade_date);
    }
    state.peer_state = peer_state;

    let mut result = Vec::with_capacity(outputs.len());
    for output in outputs {
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

fn hazq_inputs<'a>(data: &'a DataPool, panel: &'a DailyPanel) -> Result<HazqComparableInputs<'a>> {
    Ok(HazqComparableInputs {
        panel,
        total_mv: panel
            .column_from_table(data.daily(DatasetId::StockDailyBasic)?, TOTAL_MV_COLUMN)?,
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
        dividends: data.dividend_reader()?,
    })
}

fn hazq_event_schedule(inputs: &HazqComparableInputs<'_>) -> FinancialEventSchedule {
    let mut schedule = FinancialEventSchedule::from_pit_readers(&[
        inputs.income.clone(),
        inputs.balance.clone(),
        inputs.cashflow.clone(),
    ]);
    schedule.merge(FinancialEventSchedule::from_dividend_reader(
        &inputs.dividends,
    ));
    schedule
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

fn base_value_columns(
    inputs: &HazqComparableInputs<'_>,
    cache: &mut InstrumentAlignedSnapshotCache<HazqComparableSnapshot>,
) -> Result<Vec<PanelColumn>> {
    let panel = inputs.panel;
    let instrument_count = panel.instruments().len();
    let mut values = vec![vec![None; panel.shape_len()]; BASE_COUNT];
    let dividend_sums_by_date = dividend_sums_by_date(panel, &inputs.dividends);

    for (date_idx, trade_date) in panel.dates().iter().copied().enumerate() {
        let dividend_sums = dividend_sums_by_date.get(&trade_date);
        let snapshots = hazq_snapshots_for_date(inputs, cache, trade_date, dividend_sums);
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
            let base = base_values_from_snapshot(&snapshot, market_cap);
            for base_kind in BASES {
                values[base_kind.idx()][offset] = base[base_kind.idx()];
            }
        }
    }

    let ep_raw = panel.column_from_values(values[HazqComparableBase::Ep.idx()].clone())?;
    let ep_percentile = ep_raw.ts(|series| ts_zscore(series, LOOKBACK, ZSCORE_MIN_PERIODS))?;
    values[HazqComparableBase::EpPercentile.idx()] = ep_percentile.values().to_vec();

    values
        .into_iter()
        .map(|values| panel.column_from_values(values))
        .collect()
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
    let dividend_sums = inputs
        .dividends
        .implemented_ltm_sum_by_stock(add_months(trade_date, -12), trade_date);
    let snapshots = hazq_snapshots_for_date(inputs, cache, trade_date, Some(&dividend_sums));

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
    dividend_sums: Option<&HashMap<&str, f64>>,
) -> Vec<Option<HazqComparableSnapshot>> {
    cached_financial_stock_snapshots_for_date(
        inputs.panel,
        trade_date,
        cache,
        |_, _, offset| !inputs.panel.is_present_offset(offset),
        |trade_date, ts_code, _| {
            let cash_dividend = dividend_sums
                .and_then(|sums| sums.get(ts_code).copied())
                .unwrap_or(0.0);
            hazq_snapshot_marker(
                ts_code,
                trade_date,
                &inputs.income,
                &inputs.balance,
                &inputs.cashflow,
                cash_dividend,
            )
        },
        |trade_date, ts_code, _| {
            let cash_dividend = dividend_sums
                .and_then(|sums| sums.get(ts_code).copied())
                .unwrap_or(0.0);
            hazq_snapshot_for_stock(
                ts_code,
                trade_date,
                &inputs.income,
                &inputs.balance,
                &inputs.cashflow,
                cash_dividend,
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
    cash_dividend_ltm: f64,
) -> Option<FinancialEventMarker> {
    let latest_end = income.latest_quarter_end_date(ts_code, trade_date)?;
    let previous_end = previous_quarter_end_date(latest_end);
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
    builder.include_synthetic("cash_dividend_ltm", f64_marker_value(cash_dividend_ltm));
    builder.build()
}

fn hazq_snapshot_for_stock(
    ts_code: &str,
    trade_date: i32,
    income: &FinancialPitReader<'_>,
    balance: &FinancialPitReader<'_>,
    cashflow: &FinancialPitReader<'_>,
    cash_dividend_ltm: f64,
) -> Option<HazqComparableSnapshot> {
    let latest_end = income.latest_quarter_end_date(ts_code, trade_date)?;
    let previous_end = previous_quarter_end_date(latest_end);
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
    let ebit_ttm = clean(income.ttm_sum_for_end_date(ts_code, trade_date, latest_end, EBIT_COLUMN));
    let income_tax_ttm =
        clean(income.ttm_sum_for_end_date(ts_code, trade_date, latest_end, INCOME_TAX_COLUMN));
    let total_profit_ttm =
        clean(income.ttm_sum_for_end_date(ts_code, trade_date, latest_end, TOTAL_PROFIT_COLUMN));
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
        profit_ttm,
        profit_q: clean(income_record.column(NET_PROFIT_ATTR_P_COLUMN)),
        cfo_ttm,
        cash_dividend_ltm,
    })
}

fn base_values_from_snapshot(
    snapshot: &HazqComparableSnapshot,
    market_cap: Option<f64>,
) -> [Option<f64>; BASE_COUNT] {
    let mut values = [None; BASE_COUNT];
    let market_cap = market_cap.filter(|value| *value > EPS);
    let ev = market_cap
        .zip(snapshot.total_liab)
        .zip(snapshot.money_cap)
        .and_then(|((market_cap, total_liab), money_cap)| {
            finite_value(market_cap + total_liab - money_cap).filter(|value| *value > EPS)
        });
    values[HazqComparableBase::Bp.idx()] = safe_div_opt(snapshot.equity, market_cap);
    values[HazqComparableBase::Dp.idx()] =
        safe_div_opt(Some(snapshot.cash_dividend_ltm), market_cap);
    values[HazqComparableBase::Ebit2Ev.idx()] = safe_div_opt(snapshot.ebit_ttm, ev);
    values[HazqComparableBase::Ep.idx()] = safe_div_opt(snapshot.profit_ttm, market_cap);
    values[HazqComparableBase::EpQ.idx()] = safe_div_opt(snapshot.profit_q, market_cap);
    values[HazqComparableBase::Ocfp.idx()] = safe_div_opt(snapshot.cfo_ttm, market_cap);
    values[HazqComparableBase::Sales2Ev.idx()] = safe_div_opt(snapshot.revenue_ttm, ev);
    values
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

fn write_source_components_for_date(
    panel: &DailyPanel,
    trade_date: i32,
    peer_state: &ComparablePeerState,
    base_columns: &[PanelColumn],
    source_needs: &[bool],
    source_values: &mut [Option<Vec<Option<f64>>>],
) -> Result<()> {
    let Some(date_idx) = panel.dates().iter().position(|date| *date == trade_date) else {
        return Ok(());
    };
    let instrument_count = panel.instruments().len();
    let date_offset = date_idx * instrument_count;
    for base in BASES {
        let base_idx = base.idx();
        let has_base_need = COMPONENTS
            .into_iter()
            .any(|component| source_needs[source_key(base, component.source_component())]);
        if !has_base_need {
            continue;
        }
        let base_column = base_columns
            .get(base_idx)
            .ok_or_else(|| err("missing HAZQ comparable base column"))?;
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
                if !source_needs[key] {
                    continue;
                }
                if let Some(values) = source_values[key].as_mut() {
                    values[offset] = stats.values[component.idx()];
                }
            }
        }
    }
    Ok(())
}

fn component_stats_for_stock(
    base_column: &PanelColumn,
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
    ComponentStats { values }
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

fn dependencies() -> Vec<DataRequest> {
    vec![
        DataRequest::new(DatasetId::StockDailyBasic, &[TOTAL_MV_COLUMN]),
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
    ]
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

fn invested_capital(balance: PitFinancialRecordView<'_>) -> Option<f64> {
    let equity = clean(balance.column(EQUITY_COLUMN))?;
    let cash = clean(balance.column(MONEY_CAP_COLUMN))?;
    let debt = clean_or_zero(balance.column(SHORT_BORROW_COLUMN))
        + clean_or_zero(balance.column(NON_CURRENT_LIAB_DUE_1Y_COLUMN))
        + clean_or_zero(balance.column(LONG_BORROW_COLUMN))
        + clean_or_zero(balance.column(BOND_PAYABLE_COLUMN));
    finite_value(equity + debt - cash)
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

fn f64_marker_value(value: f64) -> i64 {
    i64::from_ne_bytes(value.to_bits().to_ne_bytes())
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
    fn hazq_comparable_registers_88_outputs_with_required_tags() {
        let outputs = all_outputs();
        assert_eq!(outputs.len(), 88);
        assert!(outputs
            .iter()
            .any(|output| output.id() == "hazq_comp_ep_med"));
        assert!(outputs
            .iter()
            .any(|output| output.id() == "hazq_comp_bp_prm_zscore"));

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
            profit_ttm: Some(8.0),
            profit_q: Some(2.0),
            cfo_ttm: Some(6.0),
            cash_dividend_ltm: 3.0,
            ..Default::default()
        };
        let values = base_values_from_snapshot(&snapshot, Some(100.0));
        assert_close(values[HazqComparableBase::Bp.idx()].unwrap(), 0.5);
        assert_close(values[HazqComparableBase::Dp.idx()].unwrap(), 0.03);
        assert_close(
            values[HazqComparableBase::Ebit2Ev.idx()].unwrap(),
            12.0 / 130.0,
        );
        assert_close(values[HazqComparableBase::Ep.idx()].unwrap(), 0.08);
        assert_close(values[HazqComparableBase::EpQ.idx()].unwrap(), 0.02);
        assert_close(values[HazqComparableBase::Ocfp.idx()].unwrap(), 0.06);
        assert_close(
            values[HazqComparableBase::Sales2Ev.idx()].unwrap(),
            60.0 / 130.0,
        );
        assert_eq!(base_values_from_snapshot(&snapshot, Some(0.0))[0], None);
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

        let stats = component_stats_for_stock(&base, &panel, 0, 0, &profile);

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
            "hazq_comp_ep_med".to_string(),
            "hazq_comp_bp_prm_zscore".to_string(),
            "unknown".to_string(),
        ];
        let outputs = requested_outputs(&requested);
        let ids = outputs
            .iter()
            .map(|output| output.id())
            .collect::<BTreeSet<_>>();

        assert_eq!(ids.len(), 2);
        assert!(ids.contains("hazq_comp_ep_med"));
        assert!(ids.contains("hazq_comp_bp_prm_zscore"));
    }

    #[test]
    fn hazq_comparable_dependencies_include_all_financial_lines() {
        let deps = dependencies()
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

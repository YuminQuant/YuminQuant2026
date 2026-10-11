use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::stock_daily_ops::neutralize_size_sector;
use crate::factor::common::{
    routed_financial_stock_snapshots_for_date, FinancialStockEventRoute, DailyPanel, FinancialEventMarker,
    FinancialEventMarkerBuilder, FinancialPitReader, FinancialStatementDataset,
    InstrumentAlignedSnapshotCache, PanelColumn, PitFinancialRecordView, ReportTypePreference,
};

pub const PROVIDER_KEY: &str = "stock|daily|hazq_equity_composition";
pub const REP_ID: &str = "rep";
pub const CCP_ID: &str = "ccp";

const VERSION: &str = "0.1.0";
const BALANCE_QUARTERS: usize = 4;
const EPS: f64 = 1e-12;

const UNDISTR_PROFIT_COLUMN: &str = "undistr_porfit";
const SURPLUS_RESE_COLUMN: &str = "surplus_rese";
const TOTAL_SHARE_COLUMN: &str = "total_share";
const CAP_RESE_COLUMN: &str = "cap_rese";
const TREASURY_SHARE_COLUMN: &str = "treasury_share";
const TOTAL_MV_COLUMN: &str = "total_mv";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HazqEquityCompositionOutput {
    Rep,
    Ccp,
}

#[derive(Default)]
pub struct HazqEquityCompositionComputeState {
    snapshot_cache: InstrumentAlignedSnapshotCache<EquityCompositionSnapshot>,
}

#[derive(Clone, Copy, Debug, Default)]
struct EquityCompositionNeeds {
    rep: bool,
    ccp: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct EquityCompositionSnapshot {
    rep_numerator: Option<f64>,
    ccp_numerator: Option<f64>,
}

struct EquityCompositionRawColumns {
    rep: Option<PanelColumn>,
    ccp: Option<PanelColumn>,
}

pub fn spec(output: HazqEquityCompositionOutput) -> FactorSpec {
    let (id, aliases, description) = match output {
        HazqEquityCompositionOutput::Rep => (
            REP_ID,
            vec![
                "REP".to_string(),
                "Retained Earnings to Market Cap".to_string(),
            ],
            "HAZQ retained earnings to market cap factor. It uses the latest PIT consolidated balance sheet retained earnings plus surplus reserve divided by raw StockDailyBasic total_mv, then neutralizes the raw ratio by SW level-1 industry and Barra SIZE.",
        ),
        HazqEquityCompositionOutput::Ccp => (
            CCP_ID,
            vec![
                "CCP".to_string(),
                "Contributed Capital to Market Cap".to_string(),
            ],
            "HAZQ contributed capital to market cap factor. It uses the latest PIT consolidated balance sheet total share plus capital reserve minus treasury shares divided by raw StockDailyBasic total_mv, then neutralizes the raw ratio by SW level-1 industry and Barra SIZE.",
        ),
    };
    FactorSpec {
        id: id.to_string(),
        aliases,
        name: id.to_string(),
        asset_class: AssetClass::Stock,
        frequency: Frequency::Daily,
        version: VERSION.to_string(),
        tags: tags(),
        description: description.to_string(),
        dependencies: dependencies(output),
        intraday_raw_dependencies: Vec::new(),
        lookback: Lookback { trading_days: 0 },
    }
}

pub fn compute_requested(
    requested_ids: &[String],
    context: &FactorContext,
    data: &DataPool,
) -> Result<Vec<FactorSeries>> {
    let mut state = HazqEquityCompositionComputeState::default();
    compute_requested_stateful(requested_ids, context, data, &mut state)
}

pub fn compute_requested_stateful(
    requested_ids: &[String],
    _context: &FactorContext,
    data: &DataPool,
    state: &mut HazqEquityCompositionComputeState,
) -> Result<Vec<FactorSeries>> {
    let needs = needs_from_requested(requested_ids);
    if !needs.rep && !needs.ccp {
        return Ok(Vec::new());
    }

    let panel = data.stock_universe_panel()?;
    let balance = data.financial_reader(
        DatasetId::StockBalanceSheet,
        ReportTypePreference::balance_sheet_consolidated(),
    )?;
    let total_mv =
        panel.column_from_table(data.daily(DatasetId::StockDailyBasic)?, TOTAL_MV_COLUMN)?;
    let raw = raw_columns(panel, &balance, &total_mv, &mut state.snapshot_cache, needs)?;

    let mut output = Vec::new();
    if needs.rep {
        if let Some(rep) = raw.rep {
            output.push(
                neutralize_size_sector(&rep, panel, data)?
                    .to_factor_series(spec(HazqEquityCompositionOutput::Rep)),
            );
        }
    }
    if needs.ccp {
        if let Some(ccp) = raw.ccp {
            output.push(
                neutralize_size_sector(&ccp, panel, data)?
                    .to_factor_series(spec(HazqEquityCompositionOutput::Ccp)),
            );
        }
    }
    Ok(output)
}

fn raw_columns(
    panel: &DailyPanel,
    balance: &FinancialPitReader<'_>,
    total_mv: &PanelColumn,
    cache: &mut InstrumentAlignedSnapshotCache<EquityCompositionSnapshot>,
    needs: EquityCompositionNeeds,
) -> Result<EquityCompositionRawColumns> {
    let instrument_count = panel.instruments().len();
    let mut rep_values = needs.rep.then(|| vec![None; panel.shape_len()]);
    let mut ccp_values = needs.ccp.then(|| vec![None; panel.shape_len()]);

    for (date_idx, trade_date) in panel.dates().iter().copied().enumerate() {
        if !panel.is_target_date(trade_date) {
            continue;
        }
        let snapshots = routed_financial_stock_snapshots_for_date(
            panel,
            trade_date,
            cache,
            &FinancialStockEventRoute::new(&[balance]),
            |_, _, offset| !panel.is_present_offset(offset),
            |trade_date, ts_code, _| equity_composition_marker(ts_code, trade_date, balance),
            |trade_date, ts_code, _| {
                equity_composition_snapshot_for_stock(ts_code, trade_date, balance, needs)
            },
        );
        let date_offset = date_idx * instrument_count;
        for (instrument_idx, _) in panel.instruments().iter().enumerate() {
            let offset = date_offset + instrument_idx;
            if !panel.is_present_offset(offset) {
                continue;
            }
            let Some(snapshot) = snapshots[instrument_idx] else {
                continue;
            };
            let market_cap = total_mv.values()[offset];
            if let Some(values) = &mut rep_values {
                values[offset] = ratio_from_numerator(snapshot.rep_numerator, market_cap);
            }
            if let Some(values) = &mut ccp_values {
                values[offset] = ratio_from_numerator(snapshot.ccp_numerator, market_cap);
            }
        }
    }

    Ok(EquityCompositionRawColumns {
        rep: rep_values
            .map(|values| panel.column_from_values(values))
            .transpose()?,
        ccp: ccp_values
            .map(|values| panel.column_from_values(values))
            .transpose()?,
    })
}

fn equity_composition_marker(
    ts_code: &str,
    trade_date: i32,
    balance: &FinancialPitReader<'_>,
) -> Option<FinancialEventMarker> {
    let end_date = balance.latest_quarter_end_date(ts_code, trade_date)?;
    let mut builder = FinancialEventMarkerBuilder::new();
    builder.include_reader_record_for_end_date(
        FinancialStatementDataset::BalanceSheet,
        balance,
        ts_code,
        trade_date,
        end_date,
    );
    builder.build()
}

fn equity_composition_snapshot_for_stock(
    ts_code: &str,
    trade_date: i32,
    balance: &FinancialPitReader<'_>,
    needs: EquityCompositionNeeds,
) -> Option<EquityCompositionSnapshot> {
    let end_date = balance.latest_quarter_end_date(ts_code, trade_date)?;
    let record = balance.record_for_end_date(ts_code, trade_date, end_date)?;
    Some(EquityCompositionSnapshot {
        rep_numerator: needs
            .rep
            .then(|| rep_numerator_from_record(record))
            .flatten(),
        ccp_numerator: needs
            .ccp
            .then(|| ccp_numerator_from_record(record))
            .flatten(),
    })
}

fn rep_numerator_from_record(record: PitFinancialRecordView<'_>) -> Option<f64> {
    rep_numerator_from_values(
        record.column(UNDISTR_PROFIT_COLUMN),
        record.column(SURPLUS_RESE_COLUMN),
    )
}

fn ccp_numerator_from_record(record: PitFinancialRecordView<'_>) -> Option<f64> {
    ccp_numerator_from_values(
        record.column(TOTAL_SHARE_COLUMN),
        record.column(CAP_RESE_COLUMN),
        record.column(TREASURY_SHARE_COLUMN),
    )
}

fn rep_numerator_from_values(
    undistr_profit: Option<f64>,
    surplus_rese: Option<f64>,
) -> Option<f64> {
    let value = clean(undistr_profit)? + clean(surplus_rese)?;
    value.is_finite().then_some(value)
}

fn ccp_numerator_from_values(
    total_share: Option<f64>,
    cap_rese: Option<f64>,
    treasury_share: Option<f64>,
) -> Option<f64> {
    let value = clean(total_share)? + clean(cap_rese)? - clean(treasury_share).unwrap_or(0.0);
    value.is_finite().then_some(value)
}

fn ratio_from_numerator(numerator: Option<f64>, total_mv: Option<f64>) -> Option<f64> {
    let numerator = clean(numerator)?;
    let total_mv = clean(total_mv).filter(|value| *value > EPS)?;
    let value = numerator / total_mv;
    value.is_finite().then_some(value)
}

fn clean(value: Option<f64>) -> Option<f64> {
    value.filter(|value| value.is_finite())
}

fn needs_from_requested(requested_ids: &[String]) -> EquityCompositionNeeds {
    EquityCompositionNeeds {
        rep: requested_ids.iter().any(|id| id == REP_ID),
        ccp: requested_ids.iter().any(|id| id == CCP_ID),
    }
}

fn dependencies(output: HazqEquityCompositionOutput) -> Vec<DataRequest> {
    let balance_columns: &[&str] = match output {
        HazqEquityCompositionOutput::Rep => &[UNDISTR_PROFIT_COLUMN, SURPLUS_RESE_COLUMN],
        HazqEquityCompositionOutput::Ccp => {
            &[TOTAL_SHARE_COLUMN, CAP_RESE_COLUMN, TREASURY_SHARE_COLUMN]
        }
    };
    vec![
        DataRequest::financial_quarters(
            DatasetId::StockBalanceSheet,
            balance_columns,
            BALANCE_QUARTERS,
        ),
        DataRequest::new(DatasetId::StockDailyBasic, &[TOTAL_MV_COLUMN]),
        DataRequest::new(DatasetId::StockBarraDaily, &["SIZE"]),
        DataRequest::new(DatasetId::StockSwClassification, &["l1_code"]),
    ]
}

fn tags() -> Vec<String> {
    [
        "HAZQ",
        "financial",
        "fundamental",
        "pit",
        "balance_sheet",
        "valuation",
        "size_neutralize",
        "sector_neutralize",
        "daily",
    ]
    .iter()
    .map(|tag| (*tag).to_string())
    .collect()
}

use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::stock_daily_ops::neutralize_size_sector;
use crate::factor::common::{
    cached_financial_stock_snapshots_for_date, DailyPanel, FinancialEventMarker,
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
        let snapshots = cached_financial_stock_snapshots_for_date(
            panel,
            trade_date,
            cache,
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

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, HashMap};

    use crate::core::FactorRowKey;
    use crate::data::{ColumnData, Table};

    use super::*;

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1e-10,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn rep_uses_retained_earnings_and_keeps_negative_values() {
        assert_close(
            rep_numerator_from_values(Some(10.0), Some(5.0)).unwrap(),
            15.0,
        );
        assert_close(
            rep_numerator_from_values(Some(-30.0), Some(5.0)).unwrap(),
            -25.0,
        );
        assert_eq!(rep_numerator_from_values(None, Some(5.0)), None);
    }

    #[test]
    fn ccp_uses_contributed_capital_and_defaults_missing_treasury_share_to_zero() {
        assert_close(
            ccp_numerator_from_values(Some(40.0), Some(20.0), Some(5.0)).unwrap(),
            55.0,
        );
        assert_close(
            ccp_numerator_from_values(Some(40.0), Some(20.0), None).unwrap(),
            60.0,
        );
        assert_eq!(ccp_numerator_from_values(None, Some(20.0), None), None);
    }

    #[test]
    fn ratio_requires_positive_finite_market_cap_and_core_fields() {
        assert_close(ratio_from_numerator(Some(10.0), Some(100.0)).unwrap(), 0.1);
        assert_eq!(ratio_from_numerator(Some(10.0), Some(0.0)), None);
        assert_eq!(ratio_from_numerator(Some(10.0), Some(-1.0)), None);
        assert_eq!(ratio_from_numerator(Some(10.0), Some(f64::NAN)), None);
        assert_eq!(ratio_from_numerator(None, Some(100.0)), None);
        assert_eq!(ratio_from_numerator(Some(f64::NAN), Some(100.0)), None);
    }

    #[test]
    fn specs_have_hazq_tags_and_no_pv_dependency() {
        for output in [
            HazqEquityCompositionOutput::Rep,
            HazqEquityCompositionOutput::Ccp,
        ] {
            let spec = spec(output);
            assert!(spec.tags.contains(&"HAZQ".to_string()));
            assert!(!spec
                .dependencies
                .iter()
                .any(|request| request.dataset == DatasetId::StockDailyPv));
            assert!(spec.dependencies.iter().any(|request| {
                request.dataset == DatasetId::StockDailyBasic
                    && request.columns == vec![TOTAL_MV_COLUMN.to_string()]
            }));
            assert!(spec.dependencies.iter().any(|request| {
                request.dataset == DatasetId::StockBalanceSheet
                    && request.financial_quarters == Some(BALANCE_QUARTERS)
            }));
        }
        assert!(spec(HazqEquityCompositionOutput::Rep)
            .aliases
            .contains(&"REP".to_string()));
        assert!(spec(HazqEquityCompositionOutput::Ccp)
            .aliases
            .contains(&"CCP".to_string()));
    }

    #[test]
    fn hazq_equity_composition_uses_stock_universe_panel_without_pv_anchor() {
        let context = factor_context();
        let data = DataPool::from_daily_tables_for_test(
            HashMap::from([
                (DatasetId::StockBasic, stock_basic_table()),
                (DatasetId::StockBalanceSheet, balance_table()),
                (DatasetId::StockDailyBasic, daily_basic_table()),
                (DatasetId::StockBarraDaily, barra_table()),
                (DatasetId::StockSwClassification, sw_table()),
            ]),
            &context,
        )
        .expect("data pool");
        let requested = vec![REP_ID.to_string(), CCP_ID.to_string()];

        let output = compute_requested(&requested, &context, &data).expect("hazq factors");

        assert_eq!(output.len(), 2);
        let by_id = output
            .iter()
            .map(|series| (series.spec.id.as_str(), series))
            .collect::<BTreeMap<_, _>>();
        let rep = by_id.get(REP_ID).expect("rep series");
        let ccp = by_id.get(CCP_ID).expect("ccp series");
        assert_eq!(rep.values.len(), 4);
        assert_eq!(ccp.values.len(), 4);

        let rep_values = values_by_code(rep);
        let ccp_values = values_by_code(ccp);
        for code in ["000001.SZ", "000002.SZ", "000003.SZ"] {
            assert!(rep_values.get(code).and_then(|value| *value).is_some());
            assert!(ccp_values.get(code).and_then(|value| *value).is_some());
        }
        assert_eq!(rep_values.get("000004.SZ"), Some(&None));
        assert_eq!(ccp_values.get("000004.SZ"), Some(&None));
    }

    fn factor_context() -> FactorContext {
        FactorContext {
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            start_date: 20260105,
            end_date: 20260105,
            load_start_date: 20260105,
            load_dates: vec![20260105],
            target_dates: vec![20260105],
        }
    }

    fn stock_basic_table() -> Table {
        Table::new(BTreeMap::from([
            (
                "ts_code".to_string(),
                ColumnData::Utf8(vec![
                    Some("000001.SZ".to_string()),
                    Some("000002.SZ".to_string()),
                    Some("000003.SZ".to_string()),
                    Some("000004.SZ".to_string()),
                    Some("AAPL.US".to_string()),
                ]),
            ),
            (
                "list_date".to_string(),
                ColumnData::I32(vec![
                    Some(20200101),
                    Some(20200101),
                    Some(20200101),
                    Some(20200101),
                    Some(20200101),
                ]),
            ),
            (
                "delist_date".to_string(),
                ColumnData::I32(vec![None, None, None, None, None]),
            ),
        ]))
        .expect("stock basic")
    }

    fn balance_table() -> Table {
        Table::new(BTreeMap::from([
            (
                "ts_code".to_string(),
                ColumnData::Utf8(vec![
                    Some("000001.SZ".to_string()),
                    Some("000002.SZ".to_string()),
                    Some("000003.SZ".to_string()),
                    Some("000004.SZ".to_string()),
                    Some("999999.XX".to_string()),
                ]),
            ),
            (
                "ann_date".to_string(),
                ColumnData::I32(vec![Some(20250430); 5]),
            ),
            (
                "f_ann_date".to_string(),
                ColumnData::I32(vec![Some(20250430); 5]),
            ),
            (
                "end_date".to_string(),
                ColumnData::I32(vec![Some(20250331); 5]),
            ),
            ("report_type".to_string(), ColumnData::I64(vec![Some(1); 5])),
            ("update_flag".to_string(), ColumnData::I64(vec![Some(0); 5])),
            (
                UNDISTR_PROFIT_COLUMN.to_string(),
                ColumnData::F64(vec![
                    Some(10.0),
                    Some(-30.0),
                    Some(30.0),
                    Some(5.0),
                    Some(999.0),
                ]),
            ),
            (
                SURPLUS_RESE_COLUMN.to_string(),
                ColumnData::F64(vec![
                    Some(5.0),
                    Some(5.0),
                    Some(0.0),
                    Some(5.0),
                    Some(999.0),
                ]),
            ),
            (
                TOTAL_SHARE_COLUMN.to_string(),
                ColumnData::F64(vec![
                    Some(20.0),
                    Some(40.0),
                    Some(30.0),
                    Some(10.0),
                    Some(999.0),
                ]),
            ),
            (
                CAP_RESE_COLUMN.to_string(),
                ColumnData::F64(vec![
                    Some(30.0),
                    Some(20.0),
                    Some(30.0),
                    Some(10.0),
                    Some(999.0),
                ]),
            ),
            (
                TREASURY_SHARE_COLUMN.to_string(),
                ColumnData::F64(vec![Some(5.0), None, Some(10.0), Some(0.0), Some(999.0)]),
            ),
        ]))
        .expect("balance")
    }

    fn daily_basic_table() -> Table {
        Table::new(BTreeMap::from([
            (
                "trade_date".to_string(),
                ColumnData::I32(vec![Some(20260105), Some(20260105), Some(20260105)]),
            ),
            (
                "ts_code".to_string(),
                ColumnData::Utf8(vec![
                    Some("000001.SZ".to_string()),
                    Some("000002.SZ".to_string()),
                    Some("000003.SZ".to_string()),
                ]),
            ),
            (
                TOTAL_MV_COLUMN.to_string(),
                ColumnData::F64(vec![Some(100.0), Some(200.0), Some(300.0)]),
            ),
        ]))
        .expect("daily basic")
    }

    fn barra_table() -> Table {
        Table::new(BTreeMap::from([
            (
                "trade_date".to_string(),
                ColumnData::I32(vec![
                    Some(20260105),
                    Some(20260105),
                    Some(20260105),
                    Some(20260105),
                ]),
            ),
            (
                "ts_code".to_string(),
                ColumnData::Utf8(vec![
                    Some("000001.SZ".to_string()),
                    Some("000002.SZ".to_string()),
                    Some("000003.SZ".to_string()),
                    Some("000004.SZ".to_string()),
                ]),
            ),
            (
                "SIZE".to_string(),
                ColumnData::F64(vec![Some(1.0), Some(2.0), Some(4.0), Some(5.0)]),
            ),
        ]))
        .expect("barra")
    }

    fn sw_table() -> Table {
        Table::new(BTreeMap::from([
            (
                "ts_code".to_string(),
                ColumnData::Utf8(vec![
                    Some("000001.SZ".to_string()),
                    Some("000002.SZ".to_string()),
                    Some("000003.SZ".to_string()),
                    Some("000004.SZ".to_string()),
                ]),
            ),
            (
                "in_date".to_string(),
                ColumnData::I32(vec![Some(20200101); 4]),
            ),
            (
                "out_date".to_string(),
                ColumnData::I32(vec![None, None, None, None]),
            ),
            (
                "l1_code".to_string(),
                ColumnData::Utf8(vec![
                    Some("10".to_string()),
                    Some("10".to_string()),
                    Some("10".to_string()),
                    Some("10".to_string()),
                ]),
            ),
        ]))
        .expect("sw")
    }

    fn values_by_code(series: &FactorSeries) -> BTreeMap<&str, Option<f64>> {
        series
            .values
            .iter()
            .map(|item| match &item.key {
                FactorRowKey::Daily { ts_code, .. } => (ts_code.as_str(), item.value),
                _ => unreachable!("daily factor"),
            })
            .collect()
    }
}

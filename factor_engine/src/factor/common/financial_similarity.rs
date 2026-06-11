use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap};

use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, FactorValue,
    Frequency, Lookback,
};
use crate::data::DataPool;
use crate::error::{err, Result};
use crate::factor::common::financial::previous_quarter_end_date;
use crate::factor::common::stock_daily_ops::{
    adjusted_20d_return, is_bj_stock, mask_bj, neutralize_size_sector,
};
use crate::factor::common::vector::clean;
use crate::factor::common::{
    cached_financial_stock_snapshots_for_date, factor_series_to_panel_column,
    financial_event_trade_dates, DailyPanel, DividendReader, EventDrivenCrossSectionCache,
    FinancialEventMarker, FinancialEventMarkerBuilder, FinancialEventSchedule, FinancialPitReader,
    FinancialStatementDataset, InstrumentAlignedSnapshotCache, PanelColumn, ReportTypePreference,
};
use crate::operators::{cs_pctrank, cs_regression_residual};

pub const F_MOMENTUM_80PEC_ID: &str = "f_momentum_80pec";
pub const LINK_NEW_ID: &str = "link_new";
pub const PROVIDER_KEY: &str = "stock|daily|financial_similarity";
const LINK_NEW_RAW_ID: &str = "__link_new_raw";

const VERSION: &str = "0.1.0";
const LOOKBACK: usize = 252;
const METRIC_DIM: usize = 10;
const FINANCIAL_QUARTERS: usize = 8;
const TOP_PEER_RETAIN_RATIO: f64 = 0.20;

const INCOME_COLUMNS: [&str; 2] = ["revenue", "n_income_attr_p"];
const BALANCE_COLUMNS: [&str; 6] = [
    "total_cur_assets",
    "total_cur_liab",
    "total_ncl",
    "total_hldr_eqy_exc_min_int",
    "inventories",
    "accounts_receiv",
];

#[derive(Clone, Copy, Debug)]
struct FinancialMetricSlowSnapshot {
    metrics: [Option<f64>; METRIC_DIM],
    cash_dividend_ltm: f64,
    total_mv_snapshot: Option<f64>,
}

#[derive(Clone, Debug)]
struct FinancialPeerLink {
    peer_idx: usize,
    similarity: f64,
}

#[derive(Clone, Debug, Default)]
struct FinancialSimilarityPeerState {
    top_peers: Vec<Vec<FinancialPeerLink>>,
    last_processed_trade_date: Option<i32>,
}

impl FinancialSimilarityPeerState {
    fn mark_processed(&mut self, trade_date: i32) {
        self.last_processed_trade_date = Some(trade_date);
    }
}

#[derive(Clone, Debug, Default)]
pub struct FinancialSimilarityComputeState {
    link_raw_cache: EventDrivenCrossSectionCache,
    peer_state: FinancialSimilarityPeerState,
    snapshot_cache: InstrumentAlignedSnapshotCache<FinancialMetricSlowSnapshot>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FinancialSimilarityOutput {
    FMomentum80Pec,
    LinkNew,
}

impl FinancialSimilarityOutput {
    pub fn id(self) -> &'static str {
        match self {
            Self::FMomentum80Pec => F_MOMENTUM_80PEC_ID,
            Self::LinkNew => LINK_NEW_ID,
        }
    }
}

pub fn spec(kind: FinancialSimilarityOutput) -> FactorSpec {
    let (id, aliases, description) = match kind {
        FinancialSimilarityOutput::FMomentum80Pec => (
            F_MOMENTUM_80PEC_ID,
            vec!["F-Momentum-80Pec".to_string(), "F Momentum 80Pec".to_string()],
            "Financial similarity momentum factor. It builds a 10-metric PIT financial vector, keeps the top 20% most similar peers by F-Link cosine similarity, computes peer Ret20 weighted by similarity, residualizes by own Ret20, and neutralizes by Barra SIZE and SW sector.",
        ),
        FinancialSimilarityOutput::LinkNew => (
            LINK_NEW_ID,
            vec!["Link_New".to_string(), "Financial Link New".to_string()],
            "Financial similarity signal factor. It builds a 10-metric PIT financial vector, averages F-Link cosine similarity to other stocks, and neutralizes by Barra SIZE and SW sector.",
        ),
    };
    let mut dependencies = vec![
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
    ];
    if kind == FinancialSimilarityOutput::FMomentum80Pec {
        dependencies.insert(0, DataRequest::new(DatasetId::StockDailyPv, &["close"]));
        dependencies.insert(
            1,
            DataRequest::new(DatasetId::StockAdjFactor, &["adj_factor"]),
        );
    }

    FactorSpec {
        id: id.to_string(),
        aliases,
        name: id.to_string(),
        asset_class: AssetClass::Stock,
        frequency: Frequency::Daily,
        version: VERSION.to_string(),
        tags: tags(),
        description: description.to_string(),
        dependencies,
        intraday_raw_dependencies: Vec::new(),
        lookback: Lookback {
            trading_days: LOOKBACK,
        },
    }
}

fn link_new_raw_spec() -> FactorSpec {
    FactorSpec {
        id: LINK_NEW_RAW_ID.to_string(),
        aliases: Vec::new(),
        name: LINK_NEW_RAW_ID.to_string(),
        asset_class: AssetClass::Stock,
        frequency: Frequency::Daily,
        version: VERSION.to_string(),
        tags: vec!["internal".to_string(), "financial_raw".to_string()],
        description: "Internal link_new raw financial similarity series.".to_string(),
        dependencies: Vec::new(),
        intraday_raw_dependencies: Vec::new(),
        lookback: Lookback { trading_days: 0 },
    }
}

pub fn compute_requested(
    requested_ids: &[String],
    _context: &FactorContext,
    data: &DataPool,
) -> Result<Vec<FactorSeries>> {
    let want_f_momentum = requested_ids.iter().any(|id| id == F_MOMENTUM_80PEC_ID);
    let want_link_new = requested_ids.iter().any(|id| id == LINK_NEW_ID);
    if !want_f_momentum && !want_link_new {
        return Ok(Vec::new());
    }

    let panel = data.stock_universe_panel()?;
    let total_mv = panel.column_from_table(data.daily(DatasetId::StockDailyBasic)?, "total_mv")?;
    let income = data.financial_reader(
        DatasetId::StockIncome,
        ReportTypePreference::income_single_quarter(),
    )?;
    let balance = data.financial_reader(
        DatasetId::StockBalanceSheet,
        ReportTypePreference::balance_sheet_consolidated(),
    )?;
    let dividends = data.dividend_reader()?;
    let ret20 = if want_f_momentum {
        Some(adjusted_20d_return(data, &panel)?)
    } else {
        None
    };

    let mut snapshot_cache = InstrumentAlignedSnapshotCache::default();
    let metric_columns = financial_metric_columns(
        &panel,
        &income,
        &balance,
        &total_mv,
        &dividends,
        &mut snapshot_cache,
    )?;
    let standardized_metrics = metric_columns
        .into_iter()
        .map(|column| {
            let ranked = column.cs(|values| cs_pctrank(values, true))?;
            fill_present_non_bj_missing_ranks_with_zero(&ranked, &panel)
        })
        .collect::<Result<Vec<_>>>()?;

    let (f_momentum_raw, link_raw) = financial_similarity_raw_outputs(
        &standardized_metrics,
        ret20.as_ref(),
        &panel,
        want_f_momentum,
        want_link_new,
    )?;

    let mut output = Vec::new();
    if want_f_momentum {
        let raw = panel.column_from_values(f_momentum_raw)?;
        let ret20 = ret20
            .as_ref()
            .expect("f_momentum_80pec requires ret20 when requested");
        let residual = raw.cs_binary(ret20, cs_regression_residual)?;
        let masked = mask_bj(&residual, &panel)?;
        let neutralized = neutralize_size_sector(&masked, &panel, data)?;
        output.push(
            mask_bj(&neutralized, &panel)?
                .to_factor_series(spec(FinancialSimilarityOutput::FMomentum80Pec)),
        );
    }
    if want_link_new {
        let raw = panel.column_from_values(link_raw)?;
        let masked = mask_bj(&raw, &panel)?;
        let neutralized = neutralize_size_sector(&masked, &panel, data)?;
        output.push(
            mask_bj(&neutralized, &panel)?
                .to_factor_series(spec(FinancialSimilarityOutput::LinkNew)),
        );
    }
    Ok(output)
}

pub fn compute_requested_stateful(
    requested_ids: &[String],
    context: &FactorContext,
    data: &DataPool,
    state: &mut FinancialSimilarityComputeState,
) -> Result<Vec<FactorSeries>> {
    let want_f_momentum = requested_ids.iter().any(|id| id == F_MOMENTUM_80PEC_ID);
    let want_link_new = requested_ids.iter().any(|id| id == LINK_NEW_ID);
    if !want_f_momentum && !want_link_new {
        return Ok(Vec::new());
    }

    let panel = data.stock_universe_panel()?;
    let income_reader = data.financial_reader(
        DatasetId::StockIncome,
        ReportTypePreference::income_single_quarter(),
    )?;
    let balance_reader = data.financial_reader(
        DatasetId::StockBalanceSheet,
        ReportTypePreference::balance_sheet_consolidated(),
    )?;
    let mut schedule =
        FinancialEventSchedule::from_pit_readers(&[income_reader.clone(), balance_reader.clone()]);
    let dividend_reader = data.dividend_reader()?;
    schedule.merge(FinancialEventSchedule::from_dividend_reader(
        &dividend_reader,
    ));
    let ret20 = if want_f_momentum {
        Some(adjusted_20d_return(data, &panel)?)
    } else {
        None
    };

    let event_trade_dates = financial_event_trade_dates(
        state.peer_state.last_processed_trade_date,
        &schedule,
        &context.target_dates,
    );
    let event_trade_date_set = event_trade_dates.iter().copied().collect::<BTreeSet<_>>();
    let event_inputs = if event_trade_dates.is_empty() {
        None
    } else {
        Some(financial_similarity_inputs(data)?)
    };

    let mut f_momentum_raw = want_f_momentum.then(|| vec![None; panel.shape_len()]);
    let mut link_values = Vec::new();
    for trade_date in context.target_dates.iter().copied() {
        if event_trade_date_set.contains(&trade_date) {
            if let Some(inputs) = event_inputs.as_ref() {
                let points = financial_similarity_points_for_trade_date(
                    inputs.panel,
                    &inputs.income,
                    &inputs.balance,
                    &inputs.total_mv,
                    &inputs.dividends,
                    &mut state.snapshot_cache,
                    trade_date,
                )?;
                update_financial_similarity_event_state(
                    &mut state.peer_state,
                    &mut state.link_raw_cache,
                    &points,
                    &panel,
                    trade_date,
                    want_f_momentum,
                    want_link_new,
                )?;
            }
        }
        if let (Some(output), Some(ret20)) = (f_momentum_raw.as_mut(), ret20.as_ref()) {
            write_f_momentum_raw_for_date(
                &state.peer_state.top_peers,
                ret20,
                &panel,
                trade_date,
                output,
            )?;
        }
        if want_link_new {
            let mut replay =
                state
                    .link_raw_cache
                    .replay_series(link_new_raw_spec(), &panel, trade_date);
            link_values.append(&mut replay.values);
        }
        state.peer_state.mark_processed(trade_date);
        state.link_raw_cache.mark_processed(trade_date);
    }

    let mut output = Vec::new();
    if let Some(raw_values) = f_momentum_raw {
        let raw = panel.column_from_values(raw_values)?;
        let ret20 = ret20
            .as_ref()
            .expect("f_momentum_80pec requires ret20 when requested");
        let residual = raw.cs_binary(ret20, cs_regression_residual)?;
        let masked = mask_bj(&residual, &panel)?;
        let neutralized = neutralize_size_sector(&masked, &panel, data)?;
        output.push(
            mask_bj(&neutralized, &panel)?
                .to_factor_series(spec(FinancialSimilarityOutput::FMomentum80Pec)),
        );
    }
    if want_link_new {
        let raw_series = FactorSeries {
            spec: link_new_raw_spec(),
            values: link_values,
        };
        let raw = factor_series_to_panel_column(&panel, &raw_series)?;
        let masked = mask_bj(&raw, &panel)?;
        let neutralized = neutralize_size_sector(&masked, &panel, data)?;
        output.push(
            mask_bj(&neutralized, &panel)?
                .to_factor_series(spec(FinancialSimilarityOutput::LinkNew)),
        );
    }
    Ok(output)
}

struct FinancialSimilarityInputs<'a> {
    panel: &'a DailyPanel,
    total_mv: PanelColumn,
    income: FinancialPitReader<'a>,
    balance: FinancialPitReader<'a>,
    dividends: DividendReader<'a>,
}

fn financial_similarity_inputs(data: &DataPool) -> Result<FinancialSimilarityInputs<'_>> {
    let panel = data.stock_universe_panel()?;
    let total_mv = panel.column_from_table(data.daily(DatasetId::StockDailyBasic)?, "total_mv")?;
    let income = data.financial_reader(
        DatasetId::StockIncome,
        ReportTypePreference::income_single_quarter(),
    )?;
    let balance = data.financial_reader(
        DatasetId::StockBalanceSheet,
        ReportTypePreference::balance_sheet_consolidated(),
    )?;
    let dividends = data.dividend_reader()?;
    Ok(FinancialSimilarityInputs {
        panel,
        total_mv,
        income,
        balance,
        dividends,
    })
}

fn update_financial_similarity_event_state(
    peer_state: &mut FinancialSimilarityPeerState,
    link_raw_cache: &mut EventDrivenCrossSectionCache,
    points: &[FinancialPoint],
    panel: &DailyPanel,
    trade_date: i32,
    want_f_momentum: bool,
    want_link_new: bool,
) -> Result<()> {
    let instrument_count = panel.instruments().len();
    let Some(date_idx) = panel.dates().iter().position(|date| *date == trade_date) else {
        return Ok(());
    };
    let offset = date_idx * instrument_count;
    if want_f_momentum {
        peer_state.top_peers = financial_top_peer_links(points, instrument_count);
    }
    if want_link_new {
        let day_link = link_new_from_vector_sum(points, instrument_count);
        let mut raw_values = vec![None; panel.shape_len()];
        for instrument_idx in 0..instrument_count {
            raw_values[offset + instrument_idx] = day_link[instrument_idx];
        }
        let raw = panel.column_from_values(raw_values)?;
        let series = factor_series_for_trade_date(link_new_raw_spec(), panel, trade_date, &raw);
        link_raw_cache.update_series(&series, panel);
    }
    Ok(())
}

fn financial_top_peer_links(
    points: &[FinancialPoint],
    instrument_count: usize,
) -> Vec<Vec<FinancialPeerLink>> {
    let keep_count = points
        .len()
        .saturating_sub(1)
        .checked_sub(0)
        .map(|count| ((count as f64) * TOP_PEER_RETAIN_RATIO).ceil() as usize)
        .unwrap_or(0)
        .max(1);
    let mut heaps = vec![BinaryHeap::new(); instrument_count];
    if points.len() >= 2 {
        for left_idx in 0..points.len() - 1 {
            for right_idx in left_idx + 1..points.len() {
                let similarity = cosine_dot(&points[left_idx].values, &points[right_idx].values);
                let left = points[left_idx].instrument_idx;
                let right = points[right_idx].instrument_idx;
                push_top_peer(
                    &mut heaps[left],
                    keep_count,
                    PeerCandidate {
                        similarity,
                        order: right,
                        ret20: None,
                    },
                );
                push_top_peer(
                    &mut heaps[right],
                    keep_count,
                    PeerCandidate {
                        similarity,
                        order: left,
                        ret20: None,
                    },
                );
            }
        }
    }

    heaps
        .into_iter()
        .enumerate()
        .map(|(_, heap)| {
            heap.into_iter()
                .map(|Reverse(peer)| FinancialPeerLink {
                    peer_idx: peer.order,
                    similarity: peer.similarity,
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

fn write_f_momentum_raw_for_date(
    top_peers: &[Vec<FinancialPeerLink>],
    ret20: &PanelColumn,
    panel: &DailyPanel,
    trade_date: i32,
    output: &mut [Option<f64>],
) -> Result<()> {
    let Some(date_idx) = panel.dates().iter().position(|date| *date == trade_date) else {
        return Ok(());
    };
    let instrument_count = panel.instruments().len();
    let offset = date_idx * instrument_count;
    for (instrument_idx, ts_code) in panel.instruments().iter().enumerate() {
        let panel_idx = offset + instrument_idx;
        if is_bj_stock(ts_code) || !panel.is_present_offset(panel_idx) {
            continue;
        }
        let Some(peers) = top_peers.get(instrument_idx) else {
            continue;
        };
        let mut numerator = 0.0;
        let mut denominator = 0.0;
        for peer in peers {
            let peer_idx = peer.peer_idx;
            let peer_panel_idx = offset + peer_idx;
            if !panel.is_present_offset(peer_panel_idx) {
                continue;
            }
            if let Some(peer_ret20) = clean(ret20.values()[peer_panel_idx]) {
                numerator += peer.similarity * peer_ret20;
                denominator += peer.similarity;
            }
        }
        if denominator > f64::EPSILON {
            output[panel_idx] = finite_value(numerator / denominator);
        }
    }
    Ok(())
}

fn factor_series_for_trade_date(
    spec: FactorSpec,
    panel: &DailyPanel,
    trade_date: i32,
    column: &PanelColumn,
) -> FactorSeries {
    let Some(date_idx) = panel.dates().iter().position(|date| *date == trade_date) else {
        return FactorSeries {
            spec,
            values: Vec::new(),
        };
    };
    let instrument_count = panel.instruments().len();
    let offset = date_idx * instrument_count;
    let mut values = Vec::new();
    for (instrument_idx, ts_code) in panel.instruments().iter().enumerate() {
        let panel_idx = offset + instrument_idx;
        if !panel.is_present_offset(panel_idx) {
            continue;
        }
        values.push(FactorValue {
            key: crate::core::FactorRowKey::Daily {
                trade_date,
                ts_code: ts_code.clone(),
            },
            value: column.values()[panel_idx],
        });
    }
    FactorSeries { spec, values }
}

fn tags() -> Vec<String> {
    [
        "XYZQ",
        "financial",
        "fundamental",
        "pit",
        "f_momentum",
        "cs_network",
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

fn financial_metric_columns(
    panel: &DailyPanel,
    income: &FinancialPitReader<'_>,
    balance: &FinancialPitReader<'_>,
    total_mv: &PanelColumn,
    dividends: &DividendReader<'_>,
    cache: &mut InstrumentAlignedSnapshotCache<FinancialMetricSlowSnapshot>,
) -> Result<Vec<PanelColumn>> {
    let mut metric_values = vec![vec![None; panel.shape_len()]; METRIC_DIM];
    let dividend_sums_by_date = panel
        .dates()
        .iter()
        .copied()
        .filter(|trade_date| panel.is_target_date(*trade_date))
        .map(|trade_date| {
            (
                trade_date,
                dividends.implemented_ltm_sum_by_stock(add_months(trade_date, -12), trade_date),
            )
        })
        .collect::<BTreeMap<_, _>>();
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
                financial_metric_marker(ts_code, trade_date, income, balance, cash_dividend)
            },
            |trade_date, ts_code, offset| {
                let cash_dividend = dividend_sums_by_date
                    .get(&trade_date)
                    .and_then(|sum| sum.get(ts_code).copied())
                    .unwrap_or(0.0);
                let total_mv_value = clean(total_mv.values()[offset]).filter(|value| *value > 0.0);
                financial_metrics_slow_for_stock(
                    ts_code,
                    trade_date,
                    income,
                    balance,
                    cash_dividend,
                    total_mv_value,
                )
            },
        );
        let date_offset = date_idx * instrument_count;
        for (instrument_idx, snapshot) in snapshots.into_iter().enumerate() {
            let Some(snapshot) = snapshot else {
                continue;
            };
            let offset = date_offset + instrument_idx;
            let mut metrics = snapshot.metrics;
            metrics[6] = safe_div_opt(Some(snapshot.cash_dividend_ltm), snapshot.total_mv_snapshot);
            for metric_idx in 0..METRIC_DIM {
                metric_values[metric_idx][offset] = metrics[metric_idx];
            }
        }
    }

    metric_values
        .into_iter()
        .map(|values| panel.column_from_values(values))
        .collect()
}

fn financial_similarity_points_for_trade_date(
    panel: &DailyPanel,
    income: &FinancialPitReader<'_>,
    balance: &FinancialPitReader<'_>,
    total_mv: &PanelColumn,
    dividends: &DividendReader<'_>,
    cache: &mut InstrumentAlignedSnapshotCache<FinancialMetricSlowSnapshot>,
    trade_date: i32,
) -> Result<Vec<FinancialPoint>> {
    let Some(date_idx) = panel.dates().iter().position(|date| *date == trade_date) else {
        return Ok(Vec::new());
    };
    let instrument_count = panel.instruments().len();
    let date_offset = date_idx * instrument_count;
    let dividend_sums =
        dividends.implemented_ltm_sum_by_stock(add_months(trade_date, -12), trade_date);
    let snapshots = cached_financial_stock_snapshots_for_date(
        panel,
        trade_date,
        cache,
        |_, ts_code, offset| is_bj_stock(ts_code) || !panel.is_present_offset(offset),
        |trade_date, ts_code, _| {
            let cash_dividend = dividend_sums.get(ts_code).copied().unwrap_or(0.0);
            financial_metric_marker(ts_code, trade_date, income, balance, cash_dividend)
        },
        |trade_date, ts_code, offset| {
            let cash_dividend = dividend_sums.get(ts_code).copied().unwrap_or(0.0);
            let total_mv_value = clean(total_mv.values()[offset]).filter(|value| *value > 0.0);
            financial_metrics_slow_for_stock(
                ts_code,
                trade_date,
                income,
                balance,
                cash_dividend,
                total_mv_value,
            )
        },
    );
    let mut metric_values = vec![vec![None; instrument_count]; METRIC_DIM];
    for (instrument_idx, snapshot) in snapshots.into_iter().enumerate() {
        let Some(snapshot) = snapshot else {
            continue;
        };
        let mut metrics = snapshot.metrics;
        metrics[6] = safe_div_opt(Some(snapshot.cash_dividend_ltm), snapshot.total_mv_snapshot);
        for metric_idx in 0..METRIC_DIM {
            metric_values[metric_idx][instrument_idx] = metrics[metric_idx];
        }
    }

    let mut ranked_metrics = Vec::with_capacity(METRIC_DIM);
    for values in metric_values {
        let mut ranked = cs_pctrank(&values, true);
        for (instrument_idx, ts_code) in panel.instruments().iter().enumerate() {
            let panel_idx = date_offset + instrument_idx;
            if panel.is_present_offset(panel_idx)
                && !is_bj_stock(ts_code)
                && ranked[instrument_idx].is_none()
            {
                ranked[instrument_idx] = Some(0.0);
            }
        }
        ranked_metrics.push(ranked);
    }

    let mut points = Vec::new();
    for (instrument_idx, ts_code) in panel.instruments().iter().enumerate() {
        let panel_idx = date_offset + instrument_idx;
        if is_bj_stock(ts_code) || !panel.is_present_offset(panel_idx) {
            continue;
        }
        let Some(values) =
            financial_unit_vector_from_cross_section(&ranked_metrics, instrument_idx)
        else {
            continue;
        };
        points.push(FinancialPoint {
            instrument_idx,
            values,
            ret20: None,
        });
    }
    Ok(points)
}

fn financial_metric_marker(
    ts_code: &str,
    trade_date: i32,
    income: &FinancialPitReader<'_>,
    balance: &FinancialPitReader<'_>,
    cash_dividend_ltm: f64,
) -> Option<FinancialEventMarker> {
    let latest_end = income.latest_quarter_end_date(ts_code, trade_date)?;
    let yoy_end = same_quarter_previous_year(latest_end);
    let previous_end = previous_quarter_end_date(latest_end);
    let previous_yoy_end = previous_end.map(same_quarter_previous_year);
    let mut builder = FinancialEventMarkerBuilder::new();
    builder.include_reader_record_for_end_date(
        FinancialStatementDataset::Income,
        income,
        ts_code,
        trade_date,
        latest_end,
    );
    builder.include_reader_record_for_end_date(
        FinancialStatementDataset::Income,
        income,
        ts_code,
        trade_date,
        yoy_end,
    );
    if let Some(end_date) = previous_end {
        builder.include_reader_record_for_end_date(
            FinancialStatementDataset::Income,
            income,
            ts_code,
            trade_date,
            end_date,
        );
    }
    if let Some(end_date) = previous_yoy_end {
        builder.include_reader_record_for_end_date(
            FinancialStatementDataset::Income,
            income,
            ts_code,
            trade_date,
            end_date,
        );
    }
    builder.include_reader_ttm_for_end_date(
        FinancialStatementDataset::Income,
        income,
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
    builder.include_reader_record_for_end_date(
        FinancialStatementDataset::BalanceSheet,
        balance,
        ts_code,
        trade_date,
        yoy_end,
    );
    builder.include_synthetic("cash_dividend_ltm", f64_marker_value(cash_dividend_ltm));
    builder.build()
}

fn financial_metrics_slow_for_stock(
    ts_code: &str,
    trade_date: i32,
    income: &FinancialPitReader<'_>,
    balance: &FinancialPitReader<'_>,
    cash_dividend_ltm: f64,
    total_mv_snapshot: Option<f64>,
) -> Option<FinancialMetricSlowSnapshot> {
    let mut metrics = [None; METRIC_DIM];
    let latest_end = income.latest_quarter_end_date(ts_code, trade_date)?;
    let yoy_end = same_quarter_previous_year(latest_end);
    let previous_end = previous_quarter_end_date(latest_end);
    let previous_yoy_end = previous_end.map(same_quarter_previous_year);

    let current_assets =
        balance_value(balance, ts_code, trade_date, latest_end, "total_cur_assets");
    let current_liab = balance_value(balance, ts_code, trade_date, latest_end, "total_cur_liab");
    let non_current_liab = balance_value(balance, ts_code, trade_date, latest_end, "total_ncl");
    let equity = balance_value(
        balance,
        ts_code,
        trade_date,
        latest_end,
        "total_hldr_eqy_exc_min_int",
    );
    let current_liab_yoy = balance_value(balance, ts_code, trade_date, yoy_end, "total_cur_liab");

    let revenue = income_value(income, ts_code, trade_date, latest_end, "revenue");
    let revenue_yoy = income_value(income, ts_code, trade_date, yoy_end, "revenue");
    let profit = income_value(income, ts_code, trade_date, latest_end, "n_income_attr_p");
    let profit_yoy = income_value(income, ts_code, trade_date, yoy_end, "n_income_attr_p");
    let previous_profit = previous_end.and_then(|end_date| {
        income_value(income, ts_code, trade_date, end_date, "n_income_attr_p")
    });
    let previous_profit_yoy = previous_yoy_end.and_then(|end_date| {
        income_value(income, ts_code, trade_date, end_date, "n_income_attr_p")
    });

    let revenue_ttm = income.ttm_sum_for_end_date(ts_code, trade_date, latest_end, "revenue");
    let profit_ttm =
        income.ttm_sum_for_end_date(ts_code, trade_date, latest_end, "n_income_attr_p");
    let profit_ttm_yoy =
        income.ttm_sum_for_end_date(ts_code, trade_date, yoy_end, "n_income_attr_p");
    let equity_yoy = balance_value(
        balance,
        ts_code,
        trade_date,
        yoy_end,
        "total_hldr_eqy_exc_min_int",
    );
    let inventories = balance_value(balance, ts_code, trade_date, latest_end, "inventories");
    let inventories_yoy = balance_value(balance, ts_code, trade_date, yoy_end, "inventories");
    let receivables = balance_value(balance, ts_code, trade_date, latest_end, "accounts_receiv");
    let receivables_yoy = balance_value(balance, ts_code, trade_date, yoy_end, "accounts_receiv");

    let profit_yoy_growth = growth_rate_opt(profit, profit_yoy);
    let previous_profit_yoy_growth = growth_rate_opt(previous_profit, previous_profit_yoy);
    let roe_ttm = safe_div_opt(profit_ttm, equity);
    let roe_ttm_yoy = safe_div_opt(profit_ttm_yoy, equity_yoy);
    let inventory_base = sum_pair(inventories, inventories_yoy);
    let receivable_base = sum_pair(receivables, receivables_yoy);

    metrics[0] = safe_div_opt(current_assets, current_liab);
    metrics[1] = safe_div_opt(non_current_liab, equity);
    metrics[2] = growth_rate_opt(current_liab, current_liab_yoy);
    metrics[3] = growth_rate_opt(revenue, revenue_yoy);
    metrics[4] = profit_yoy_growth;
    metrics[5] = profit_yoy_growth
        .zip(previous_profit_yoy_growth)
        .and_then(|(latest_growth, previous_growth)| finite_value(latest_growth - previous_growth));
    metrics[7] = growth_rate_opt(roe_ttm, roe_ttm_yoy);
    metrics[8] = safe_div_opt(revenue_ttm.map(|value| 2.0 * value), inventory_base);
    metrics[9] = safe_div_opt(revenue_ttm.map(|value| 2.0 * value), receivable_base);
    Some(FinancialMetricSlowSnapshot {
        metrics,
        cash_dividend_ltm,
        total_mv_snapshot,
    })
}

fn income_value(
    data: &FinancialPitReader<'_>,
    ts_code: &str,
    trade_date: i32,
    end_date: i32,
    column: &str,
) -> Option<f64> {
    data.record_for_end_date(ts_code, trade_date, end_date)?
        .column(column)
}

fn balance_value(
    data: &FinancialPitReader<'_>,
    ts_code: &str,
    trade_date: i32,
    end_date: i32,
    column: &str,
) -> Option<f64> {
    data.record_for_end_date(ts_code, trade_date, end_date)?
        .column(column)
}

fn growth_rate(current: f64, previous: f64) -> Option<f64> {
    (previous.abs() > f64::EPSILON).then_some((current - previous) / previous.abs())
}

fn growth_rate_opt(current: Option<f64>, previous: Option<f64>) -> Option<f64> {
    growth_rate(current?, previous?)
}

fn safe_div(numerator: f64, denominator: f64) -> Option<f64> {
    (denominator.abs() > f64::EPSILON)
        .then_some(numerator / denominator)
        .filter(|value| value.is_finite())
}

fn safe_div_opt(numerator: Option<f64>, denominator: Option<f64>) -> Option<f64> {
    safe_div(numerator?, denominator?)
}

fn sum_pair(left: Option<f64>, right: Option<f64>) -> Option<f64> {
    finite_value(left? + right?)
}

fn finite_value(value: f64) -> Option<f64> {
    value.is_finite().then_some(value)
}

fn f64_marker_value(value: f64) -> i64 {
    i64::from_ne_bytes(value.to_bits().to_ne_bytes())
}

fn same_quarter_previous_year(end_date: i32) -> i32 {
    (end_date / 10_000 - 1) * 10_000 + end_date % 10_000
}

fn fill_present_non_bj_missing_ranks_with_zero(
    column: &PanelColumn,
    panel: &DailyPanel,
) -> Result<PanelColumn> {
    let instrument_count = panel.instruments().len();
    let mut values = column.values().to_vec();
    for date_idx in 0..panel.dates().len() {
        let offset = date_idx * instrument_count;
        for (instrument_idx, ts_code) in panel.instruments().iter().enumerate() {
            let panel_idx = offset + instrument_idx;
            if panel.is_present_offset(panel_idx)
                && !is_bj_stock(ts_code)
                && values[panel_idx].is_none()
            {
                values[panel_idx] = Some(0.0);
            }
        }
    }
    panel.column_from_values(values)
}

fn financial_similarity_raw_outputs(
    metric_columns: &[PanelColumn],
    ret20: Option<&PanelColumn>,
    panel: &DailyPanel,
    want_f_momentum: bool,
    want_link_new: bool,
) -> Result<(Vec<Option<f64>>, Vec<Option<f64>>)> {
    if metric_columns.len() != METRIC_DIM {
        return Err(err(format!(
            "financial similarity expected {} standardized metrics, got {}",
            METRIC_DIM,
            metric_columns.len()
        )));
    }
    let code_count = panel.instruments().len();
    let mut f_momentum = vec![None; panel.shape_len()];
    let mut link_new = vec![None; panel.shape_len()];

    for date_idx in 0..panel.dates().len() {
        let offset = date_idx * code_count;
        let points = financial_points_for_date(metric_columns, ret20, panel, offset, code_count);
        let (day_f_momentum, day_link) =
            financial_peer_outputs(&points, code_count, want_f_momentum, want_link_new);
        for code_idx in 0..code_count {
            f_momentum[offset + code_idx] = day_f_momentum[code_idx];
            link_new[offset + code_idx] = day_link[code_idx];
        }
    }

    Ok((f_momentum, link_new))
}

fn financial_points_for_date(
    metric_columns: &[PanelColumn],
    ret20: Option<&PanelColumn>,
    panel: &DailyPanel,
    offset: usize,
    code_count: usize,
) -> Vec<FinancialPoint> {
    let mut points = Vec::new();
    for code_idx in 0..code_count {
        let ts_code = &panel.instruments()[code_idx];
        if is_bj_stock(ts_code) {
            continue;
        }
        let panel_idx = offset + code_idx;
        if !panel.is_present_offset(panel_idx) {
            continue;
        }
        let Some(values) = financial_unit_vector_at(metric_columns, panel_idx) else {
            continue;
        };
        points.push(FinancialPoint {
            instrument_idx: code_idx,
            values,
            ret20: ret20.and_then(|ret20| clean(ret20.values()[panel_idx])),
        });
    }
    points
}

fn financial_unit_vector_at(
    metric_columns: &[PanelColumn],
    panel_idx: usize,
) -> Option<[f64; METRIC_DIM]> {
    let mut values = [0.0; METRIC_DIM];
    let mut norm_sq = 0.0;
    for dim in 0..METRIC_DIM {
        let value = clean(metric_columns[dim].values()[panel_idx])?;
        values[dim] = value;
        norm_sq += value * value;
    }
    if norm_sq <= f64::EPSILON {
        return None;
    }
    let norm = norm_sq.sqrt();
    for value in &mut values {
        *value /= norm;
    }
    Some(values)
}

fn financial_unit_vector_from_cross_section(
    metric_values: &[Vec<Option<f64>>],
    instrument_idx: usize,
) -> Option<[f64; METRIC_DIM]> {
    if metric_values.len() != METRIC_DIM {
        return None;
    }
    let mut values = [0.0; METRIC_DIM];
    let mut norm_sq = 0.0;
    for dim in 0..METRIC_DIM {
        let value = clean(*metric_values.get(dim)?.get(instrument_idx)?)?;
        values[dim] = value;
        norm_sq += value * value;
    }
    if norm_sq <= f64::EPSILON {
        return None;
    }
    let norm = norm_sq.sqrt();
    for value in &mut values {
        *value /= norm;
    }
    Some(values)
}

#[derive(Clone, Copy, Debug)]
struct FinancialPoint {
    instrument_idx: usize,
    values: [f64; METRIC_DIM],
    ret20: Option<f64>,
}

fn financial_peer_outputs(
    points: &[FinancialPoint],
    instrument_count: usize,
    want_f_momentum: bool,
    want_link_new: bool,
) -> (Vec<Option<f64>>, Vec<Option<f64>>) {
    let keep_count = points
        .len()
        .saturating_sub(1)
        .checked_sub(0)
        .map(|count| ((count as f64) * TOP_PEER_RETAIN_RATIO).ceil() as usize)
        .unwrap_or(0)
        .max(1);
    let mut top_peers = want_f_momentum.then(|| vec![BinaryHeap::new(); instrument_count]);
    let link = if want_link_new {
        link_new_from_vector_sum(points, instrument_count)
    } else {
        vec![None; instrument_count]
    };

    if want_f_momentum && points.len() >= 2 {
        for left_idx in 0..points.len() - 1 {
            for right_idx in left_idx + 1..points.len() {
                let similarity = cosine_dot(&points[left_idx].values, &points[right_idx].values);
                let left = points[left_idx].instrument_idx;
                let right = points[right_idx].instrument_idx;
                if let Some(heaps) = top_peers.as_mut() {
                    push_top_peer(
                        &mut heaps[left],
                        keep_count,
                        PeerCandidate {
                            similarity,
                            order: right,
                            ret20: points[right_idx].ret20,
                        },
                    );
                    push_top_peer(
                        &mut heaps[right],
                        keep_count,
                        PeerCandidate {
                            similarity,
                            order: left,
                            ret20: points[left_idx].ret20,
                        },
                    );
                }
            }
        }
    }

    let f_momentum = top_peers
        .map(weighted_top_peer_returns)
        .unwrap_or_else(|| vec![None; instrument_count]);
    (f_momentum, link)
}

fn link_new_from_vector_sum(
    points: &[FinancialPoint],
    instrument_count: usize,
) -> Vec<Option<f64>> {
    let mut output = vec![None; instrument_count];
    if points.len() < 2 {
        return output;
    }
    let mut vector_sum = [0.0; METRIC_DIM];
    for point in points {
        for (dim, value) in point.values.iter().enumerate() {
            vector_sum[dim] += value;
        }
    }
    let denominator = points.len() as f64 - 1.0;
    for point in points {
        let self_dot_sum = cosine_dot(&point.values, &vector_sum);
        let value = (self_dot_sum - 1.0) / denominator;
        output[point.instrument_idx] = value.is_finite().then_some(value);
    }
    output
}

fn cosine_dot(left: &[f64; METRIC_DIM], right: &[f64; METRIC_DIM]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}

#[derive(Clone, Copy, Debug)]
struct PeerCandidate {
    similarity: f64,
    order: usize,
    ret20: Option<f64>,
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

fn push_top_peer(
    heap: &mut BinaryHeap<Reverse<PeerCandidate>>,
    keep_count: usize,
    candidate: PeerCandidate,
) {
    if keep_count == 0 || !candidate.similarity.is_finite() {
        return;
    }
    if heap.len() < keep_count {
        heap.push(Reverse(candidate));
    } else if heap
        .peek()
        .is_some_and(|Reverse(current)| candidate > *current)
    {
        heap.pop();
        heap.push(Reverse(candidate));
    }
}

fn weighted_top_peer_returns(heaps: Vec<BinaryHeap<Reverse<PeerCandidate>>>) -> Vec<Option<f64>> {
    heaps
        .into_iter()
        .map(|heap| {
            let mut numerator = 0.0;
            let mut denominator = 0.0;
            for Reverse(peer) in heap {
                if let Some(ret20) = clean(peer.ret20) {
                    numerator += peer.similarity * ret20;
                    denominator += peer.similarity;
                }
            }
            if denominator > f64::EPSILON {
                let value = numerator / denominator;
                value.is_finite().then_some(value)
            } else {
                None
            }
        })
        .collect()
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

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::Arc;

    use crate::core::{AssetClass, FactorContext, Frequency};
    use crate::data::{ColumnData, Table};
    use crate::factor::common::{DividendIndex, FinancialPitIndex};

    use super::*;

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1e-12,
            "actual={actual}, expected={expected}"
        );
    }

    fn test_context(target_dates: Vec<i32>) -> FactorContext {
        FactorContext {
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            start_date: *target_dates.first().unwrap(),
            end_date: *target_dates.last().unwrap(),
            load_start_date: *target_dates.first().unwrap(),
            load_dates: target_dates.clone(),
            target_dates,
        }
    }

    fn test_panel(rows: &[(i32, &str)]) -> DailyPanel {
        let table = Table::new(BTreeMap::from([
            (
                "trade_date".to_string(),
                ColumnData::I32(rows.iter().map(|(date, _)| Some(*date)).collect()),
            ),
            (
                "ts_code".to_string(),
                ColumnData::Utf8(
                    rows.iter()
                        .map(|(_, ts_code)| Some((*ts_code).to_string()))
                        .collect(),
                ),
            ),
            (
                "close".to_string(),
                ColumnData::F64(rows.iter().map(|_| Some(1.0)).collect()),
            ),
        ]))
        .expect("valid table");
        let dates = rows.iter().map(|(date, _)| *date).collect::<Vec<_>>();
        DailyPanel::from_table(&table, &test_context(dates)).expect("panel")
    }

    fn dividend_table(rows: &[(&str, i32, &str, f64, i32, f64)]) -> Table {
        Table::new(BTreeMap::from([
            (
                "ts_code".to_string(),
                ColumnData::Utf8(
                    rows.iter()
                        .map(|row| Some(row.0.to_string()))
                        .collect::<Vec<_>>(),
                ),
            ),
            (
                "ann_date".to_string(),
                ColumnData::I32(rows.iter().map(|row| Some(row.1)).collect()),
            ),
            (
                "div_proc".to_string(),
                ColumnData::Utf8(
                    rows.iter()
                        .map(|row| Some(row.2.to_string()))
                        .collect::<Vec<_>>(),
                ),
            ),
            (
                "cash_div_tax".to_string(),
                ColumnData::F64(rows.iter().map(|row| Some(row.3)).collect()),
            ),
            (
                "ex_date".to_string(),
                ColumnData::I32(rows.iter().map(|row| Some(row.4)).collect()),
            ),
            (
                "base_share".to_string(),
                ColumnData::F64(rows.iter().map(|row| Some(row.5)).collect()),
            ),
        ]))
        .expect("valid dividend table")
    }

    fn financial_similarity_income_table(ts_codes: &[&str]) -> Table {
        let mut columns = BTreeMap::from([
            (
                "ts_code".to_string(),
                ColumnData::Utf8(
                    ts_codes
                        .iter()
                        .map(|ts_code| Some((*ts_code).to_string()))
                        .collect(),
                ),
            ),
            (
                "ann_date".to_string(),
                ColumnData::I32(ts_codes.iter().map(|_| Some(20260101)).collect()),
            ),
            (
                "f_ann_date".to_string(),
                ColumnData::I32(ts_codes.iter().map(|_| Some(20260101)).collect()),
            ),
            (
                "end_date".to_string(),
                ColumnData::I32(ts_codes.iter().map(|_| Some(20251231)).collect()),
            ),
            (
                "report_type".to_string(),
                ColumnData::I64(ts_codes.iter().map(|_| Some(3)).collect()),
            ),
            (
                "update_flag".to_string(),
                ColumnData::I64(ts_codes.iter().map(|_| Some(0)).collect()),
            ),
        ]);
        for column in INCOME_COLUMNS {
            columns.insert(
                column.to_string(),
                ColumnData::F64(ts_codes.iter().map(|_| Some(1.0)).collect()),
            );
        }
        Table::new(columns).expect("valid income table")
    }

    fn financial_similarity_balance_table(ts_codes: &[&str]) -> Table {
        let mut columns = BTreeMap::from([
            (
                "ts_code".to_string(),
                ColumnData::Utf8(
                    ts_codes
                        .iter()
                        .map(|ts_code| Some((*ts_code).to_string()))
                        .collect(),
                ),
            ),
            (
                "ann_date".to_string(),
                ColumnData::I32(ts_codes.iter().map(|_| Some(20260101)).collect()),
            ),
            (
                "f_ann_date".to_string(),
                ColumnData::I32(ts_codes.iter().map(|_| Some(20260101)).collect()),
            ),
            (
                "end_date".to_string(),
                ColumnData::I32(ts_codes.iter().map(|_| Some(20251231)).collect()),
            ),
            (
                "report_type".to_string(),
                ColumnData::I64(ts_codes.iter().map(|_| Some(1)).collect()),
            ),
            (
                "update_flag".to_string(),
                ColumnData::I64(ts_codes.iter().map(|_| Some(0)).collect()),
            ),
        ]);
        for column in BALANCE_COLUMNS {
            let value = if column == "total_cur_assets" {
                2.0
            } else {
                1.0
            };
            columns.insert(
                column.to_string(),
                ColumnData::F64(ts_codes.iter().map(|_| Some(value)).collect()),
            );
        }
        Table::new(columns).expect("valid balance table")
    }

    fn point(instrument_idx: usize, first_dim: f64, ret20: Option<f64>) -> FinancialPoint {
        let mut values = [0.0; METRIC_DIM];
        values[0] = first_dim;
        values[1] = (1.0 - first_dim * first_dim).max(0.0).sqrt();
        FinancialPoint {
            instrument_idx,
            values,
            ret20,
        }
    }

    #[test]
    fn financial_similarity_same_quarter_previous_year_preserves_quarter() {
        assert_eq!(same_quarter_previous_year(20250331), 20240331);
        assert_eq!(same_quarter_previous_year(20251231), 20241231);
    }

    #[test]
    fn financial_similarity_growth_rate_uses_abs_base() {
        assert_close(growth_rate(3.0, 2.0).unwrap(), 0.5);
        assert_close(growth_rate(-1.0, -2.0).unwrap(), 0.5);
        assert_eq!(growth_rate(1.0, 0.0), None);
    }

    #[test]
    fn financial_similarity_metric_raw_skips_not_present_slots_even_with_financial_records() {
        let panel = test_panel(&[
            (20260101, "000001.SZ"),
            (20260101, "000002.SZ"),
            (20260102, "000001.SZ"),
        ]);
        let income_table = financial_similarity_income_table(&["000001.SZ", "000002.SZ"]);
        let balance_table = financial_similarity_balance_table(&["000001.SZ", "000002.SZ"]);
        let income_index = FinancialPitIndex::from_table(Arc::new(income_table)).unwrap();
        let balance_index = FinancialPitIndex::from_table(Arc::new(balance_table)).unwrap();
        let income = income_index.reader(ReportTypePreference::income_single_quarter());
        let balance = balance_index.reader(ReportTypePreference::balance_sheet_consolidated());
        let total_mv = panel
            .column_from_values(vec![Some(100.0); panel.shape_len()])
            .unwrap();

        let mut cache = InstrumentAlignedSnapshotCache::default();
        let dividend_index = DividendIndex::from_table(Arc::new(dividend_table(&[]))).unwrap();
        let dividends = dividend_index.reader();
        let metric_columns =
            financial_metric_columns(&panel, &income, &balance, &total_mv, &dividends, &mut cache)
                .unwrap();

        assert_eq!(metric_columns[0].values()[2], Some(2.0));
        assert_eq!(metric_columns[0].values()[3], None);
    }

    #[test]
    fn financial_similarity_fills_missing_rank_only_for_present_non_bj() {
        let panel = test_panel(&[
            (20260101, "000001.SZ"),
            (20260101, "000002.SZ"),
            (20260101, "920001.BJ"),
            (20260102, "000001.SZ"),
            (20260102, "920001.BJ"),
        ]);
        let raw = panel
            .column_from_values(vec![Some(1.0), Some(2.0), None, None, None, None])
            .unwrap();
        let ranked = raw.cs(|values| cs_pctrank(values, true)).unwrap();
        let filled = fill_present_non_bj_missing_ranks_with_zero(&ranked, &panel).unwrap();

        assert_eq!(
            filled.values(),
            &[Some(0.0), Some(1.0), None, Some(0.0), None, None]
        );
    }

    #[test]
    fn financial_similarity_partial_zero_filled_vector_still_enters_points() {
        let panel = test_panel(&[
            (20260101, "000001.SZ"),
            (20260101, "000002.SZ"),
            (20260101, "920001.BJ"),
        ]);
        let mut metric_columns = Vec::new();
        metric_columns.push(
            panel
                .column_from_values(vec![Some(0.5), Some(0.0), None])
                .unwrap(),
        );
        for _ in 1..METRIC_DIM {
            metric_columns.push(
                panel
                    .column_from_values(vec![Some(0.0), Some(0.0), None])
                    .unwrap(),
            );
        }

        let points =
            financial_points_for_date(&metric_columns, None, &panel, 0, panel.instruments().len());

        assert_eq!(points.len(), 1);
        assert_eq!(points[0].instrument_idx, 0);
    }

    #[test]
    fn financial_similarity_points_skip_not_present_slots_even_if_values_exist() {
        let panel = test_panel(&[
            (20260101, "000001.SZ"),
            (20260101, "000002.SZ"),
            (20260102, "000001.SZ"),
        ]);
        let metric_columns = (0..METRIC_DIM)
            .map(|_| {
                panel
                    .column_from_values(vec![Some(0.1), Some(0.1), Some(0.1), Some(0.1)])
                    .unwrap()
            })
            .collect::<Vec<_>>();

        let points =
            financial_points_for_date(&metric_columns, None, &panel, 2, panel.instruments().len());

        assert_eq!(points.len(), 1);
        assert_eq!(points[0].instrument_idx, 0);
    }

    #[test]
    fn financial_similarity_keeps_top_peer_set_before_return_filter() {
        let points = vec![
            point(0, 1.0, Some(0.1)),
            point(1, 0.9, None),
            point(2, 0.7, Some(0.3)),
            point(3, 0.1, Some(0.9)),
            point(4, 0.0, Some(1.0)),
            point(5, -0.1, Some(1.1)),
        ];
        let (f_momentum, link) = financial_peer_outputs(&points, 6, true, true);

        assert_eq!(f_momentum[0], None);
        assert!(link[0].is_some());
    }

    #[test]
    fn financial_similarity_link_new_averages_row_similarity() {
        let points = vec![
            point(0, 1.0, Some(0.1)),
            point(1, 1.0, Some(0.2)),
            point(2, 0.0, Some(0.3)),
        ];
        let (_, link) = financial_peer_outputs(&points, 3, false, true);

        assert_close(link[0].unwrap(), 0.5);
        assert_close(link[1].unwrap(), 0.5);
        assert_close(link[2].unwrap(), 0.0);
    }

    #[test]
    fn financial_similarity_dtop_uses_only_implemented_visible_records() {
        let index = DividendIndex::from_table(Arc::new(dividend_table(&[
            (
                "000001.SZ",
                20260101,
                "\u{5b9e}\u{65bd}",
                0.2,
                20260301,
                100.0,
            ),
            (
                "000001.SZ",
                20260101,
                "\u{9884}\u{6848}",
                0.3,
                20260302,
                100.0,
            ),
            (
                "000001.SZ",
                20270101,
                "\u{5b9e}\u{65bd}",
                0.4,
                20260301,
                100.0,
            ),
        ])))
        .unwrap();
        let reader = index.reader();
        let sums = reader.implemented_ltm_sum_by_stock(20250424, 20260424);

        assert_close(*sums.get("000001.SZ").unwrap(), 20.0);
    }
}

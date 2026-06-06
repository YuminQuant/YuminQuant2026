use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::{DataPool, Table};
use crate::error::{err, Result};
use crate::factor::common::financial::previous_quarter_end_date;
use crate::factor::common::stock_daily_ops::{
    adjusted_20d_return, is_bj_stock, mask_bj, neutralize_size_sector,
};
use crate::factor::common::vector::clean;
use crate::factor::common::{DailyPanel, PanelColumn, PitFinancialData, ReportTypePreference};
use crate::operators::{cs_pctrank, cs_regression_residual};

pub const F_MOMENTUM_80PEC_ID: &str = "f_momentum_80pec";
pub const LINK_NEW_ID: &str = "link_new";
pub const PROVIDER_KEY: &str = "stock|daily|financial_similarity";

const VERSION: &str = "0.1.0";
const LOOKBACK: usize = 252;
const METRIC_DIM: usize = 10;
const FINANCIAL_QUARTERS: usize = 8;
const TOP_PEER_RETAIN_RATIO: f64 = 0.20;
const IMPLEMENTED_DIV_PROC: &str = "\u{5b9e}\u{65bd}";

const INCOME_COLUMNS: [&str; 2] = ["revenue", "n_income_attr_p"];
const BALANCE_COLUMNS: [&str; 6] = [
    "total_cur_assets",
    "total_cur_liab",
    "total_ncl",
    "total_hldr_eqy_exc_min_int",
    "inventories",
    "accounts_receiv",
];

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

    let panel = data.daily_panel(DatasetId::StockDailyPv)?;
    let total_mv = panel.column_from_table(data.daily(DatasetId::StockDailyBasic)?, "total_mv")?;
    let income = PitFinancialData::from_table(
        data.daily(DatasetId::StockIncome)?,
        &INCOME_COLUMNS,
        ReportTypePreference::income_single_quarter(),
    )?;
    let balance = PitFinancialData::from_table(
        data.daily(DatasetId::StockBalanceSheet)?,
        &BALANCE_COLUMNS,
        ReportTypePreference::balance_sheet_consolidated(),
    )?;
    let dividends = parse_dividend_records(data.daily(DatasetId::StockDividend)?)?;
    let ret20 = if want_f_momentum {
        Some(adjusted_20d_return(data, &panel)?)
    } else {
        None
    };

    let metric_columns =
        financial_metric_columns(&panel, &income, &balance, &total_mv, &dividends)?;
    let standardized_metrics = metric_columns
        .into_iter()
        .map(|column| column.cs(|values| cs_pctrank(values, true)))
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
    income: &PitFinancialData,
    balance: &PitFinancialData,
    total_mv: &PanelColumn,
    dividends: &[DividendRecord],
) -> Result<Vec<PanelColumn>> {
    let mut metric_values = vec![vec![None; panel.shape_len()]; METRIC_DIM];
    let instrument_count = panel.instruments().len();

    for (date_idx, trade_date) in panel.dates().iter().copied().enumerate() {
        if !panel.is_target_date(trade_date) {
            continue;
        }
        let dividend_sum =
            dividend_sum_by_stock(dividends, add_months(trade_date, -12), trade_date);
        for (instrument_idx, ts_code) in panel.instruments().iter().enumerate() {
            if is_bj_stock(ts_code) {
                continue;
            }
            let offset = date_idx * instrument_count + instrument_idx;
            let total_mv_value = clean(total_mv.values()[offset]).filter(|value| *value > 0.0);
            let cash_dividend = dividend_sum.get(ts_code.as_str()).copied().unwrap_or(0.0);
            let Some(metrics) = financial_metrics_for_stock(
                ts_code,
                trade_date,
                income,
                balance,
                total_mv_value,
                cash_dividend,
            ) else {
                continue;
            };
            for metric_idx in 0..METRIC_DIM {
                metric_values[metric_idx][offset] = Some(metrics[metric_idx]);
            }
        }
    }

    metric_values
        .into_iter()
        .map(|values| panel.column_from_values(values))
        .collect()
}

fn financial_metrics_for_stock(
    ts_code: &str,
    trade_date: i32,
    income: &PitFinancialData,
    balance: &PitFinancialData,
    total_mv: Option<f64>,
    cash_dividend_ltm: f64,
) -> Option<[f64; METRIC_DIM]> {
    let latest_end = income.latest_quarter_end_date(ts_code, trade_date)?;
    let yoy_end = same_quarter_previous_year(latest_end);
    let previous_end = previous_quarter_end_date(latest_end)?;
    let previous_yoy_end = same_quarter_previous_year(previous_end);

    let current_assets =
        balance_value(balance, ts_code, trade_date, latest_end, "total_cur_assets")?;
    let current_liab = balance_value(balance, ts_code, trade_date, latest_end, "total_cur_liab")?;
    let non_current_liab = balance_value(balance, ts_code, trade_date, latest_end, "total_ncl")?;
    let equity = balance_value(
        balance,
        ts_code,
        trade_date,
        latest_end,
        "total_hldr_eqy_exc_min_int",
    )?;
    let current_liab_yoy = balance_value(balance, ts_code, trade_date, yoy_end, "total_cur_liab")?;

    let revenue = income_value(income, ts_code, trade_date, latest_end, "revenue")?;
    let revenue_yoy = income_value(income, ts_code, trade_date, yoy_end, "revenue")?;
    let profit = income_value(income, ts_code, trade_date, latest_end, "n_income_attr_p")?;
    let profit_yoy = income_value(income, ts_code, trade_date, yoy_end, "n_income_attr_p")?;
    let previous_profit =
        income_value(income, ts_code, trade_date, previous_end, "n_income_attr_p")?;
    let previous_profit_yoy = income_value(
        income,
        ts_code,
        trade_date,
        previous_yoy_end,
        "n_income_attr_p",
    )?;

    let revenue_ttm = income.ttm_sum_for_end_date(ts_code, trade_date, latest_end, "revenue")?;
    let profit_ttm =
        income.ttm_sum_for_end_date(ts_code, trade_date, latest_end, "n_income_attr_p")?;
    let profit_ttm_yoy =
        income.ttm_sum_for_end_date(ts_code, trade_date, yoy_end, "n_income_attr_p")?;
    let equity_yoy = balance_value(
        balance,
        ts_code,
        trade_date,
        yoy_end,
        "total_hldr_eqy_exc_min_int",
    )?;
    let inventories = balance_value(balance, ts_code, trade_date, latest_end, "inventories")?;
    let inventories_yoy = balance_value(balance, ts_code, trade_date, yoy_end, "inventories")?;
    let receivables = balance_value(balance, ts_code, trade_date, latest_end, "accounts_receiv")?;
    let receivables_yoy = balance_value(balance, ts_code, trade_date, yoy_end, "accounts_receiv")?;

    let profit_yoy_growth = growth_rate(profit, profit_yoy)?;
    let previous_profit_yoy_growth = growth_rate(previous_profit, previous_profit_yoy)?;
    let roe_ttm = safe_div(profit_ttm, equity)?;
    let roe_ttm_yoy = safe_div(profit_ttm_yoy, equity_yoy)?;

    finite_array([
        safe_div(current_assets, current_liab)?,
        safe_div(non_current_liab, equity)?,
        growth_rate(current_liab, current_liab_yoy)?,
        growth_rate(revenue, revenue_yoy)?,
        profit_yoy_growth,
        profit_yoy_growth - previous_profit_yoy_growth,
        safe_div(cash_dividend_ltm, total_mv?)?,
        growth_rate(roe_ttm, roe_ttm_yoy)?,
        safe_div(2.0 * revenue_ttm, inventories + inventories_yoy)?,
        safe_div(2.0 * revenue_ttm, receivables + receivables_yoy)?,
    ])
}

fn income_value(
    data: &PitFinancialData,
    ts_code: &str,
    trade_date: i32,
    end_date: i32,
    column: &str,
) -> Option<f64> {
    data.record_for_end_date(ts_code, trade_date, end_date)?
        .column(column)
}

fn balance_value(
    data: &PitFinancialData,
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

fn safe_div(numerator: f64, denominator: f64) -> Option<f64> {
    (denominator.abs() > f64::EPSILON)
        .then_some(numerator / denominator)
        .filter(|value| value.is_finite())
}

fn finite_array(values: [f64; METRIC_DIM]) -> Option<[f64; METRIC_DIM]> {
    values
        .iter()
        .all(|value| value.is_finite())
        .then_some(values)
}

fn same_quarter_previous_year(end_date: i32) -> i32 {
    (end_date / 10_000 - 1) * 10_000 + end_date % 10_000
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

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1e-12,
            "actual={actual}, expected={expected}"
        );
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
}

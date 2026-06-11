use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::financial::previous_quarter_end_date;
use crate::factor::common::stock_daily_ops::{is_bj_stock, neutralize_size_sector};
use crate::factor::common::{
    cached_financial_stock_snapshots_for_date, DailyPanel, FinancialEventMarker,
    FinancialEventMarkerBuilder, FinancialEventSchedule, FinancialPitReader,
    FinancialStatementDataset, InstrumentAlignedSnapshotCache, PanelColumn, ReportTypePreference,
};
use crate::operators::ts_skew;

pub const PROVIDER_KEY: &str = "stock|daily|dbzq_profit_skew";
pub const OP_MARGIN_TTM_SKEW_ID: &str = "op_margin_ttm_skew";
pub const POP_SKEW_ID: &str = "pop_skew";

const VERSION: &str = "0.1.0";
const ROLLING_POP_WINDOW: usize = 240;
const ROLLING_POP_MIN_PERIODS: usize = 1;
const DAILY_LOOKBACK: usize = ROLLING_POP_WINDOW - 1;
const OP_MARGIN_TTM_SKEW_QUARTERS: usize = 8;
const OP_MARGIN_REQUIRED_QUARTERS: usize = 11;
const POP_REQUIRED_QUARTERS: usize = 4;
const EPS: f64 = 1e-12;

const OPERATE_PROFIT_COLUMN: &str = "operate_profit";
const REVENUE_COLUMN: &str = "revenue";
const TOTAL_MV_COLUMN: &str = "total_mv";

const INCOME_COLUMNS: [&str; 2] = [OPERATE_PROFIT_COLUMN, REVENUE_COLUMN];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProfitSkewOutput {
    OpMarginTtmSkew,
    PopSkew,
}

#[derive(Default)]
pub struct ProfitSkewComputeState {
    snapshot_cache: InstrumentAlignedSnapshotCache<ProfitSkewSnapshot>,
}

#[derive(Clone, Copy, Debug, Default)]
struct ProfitSkewNeeds {
    op_margin: bool,
    pop: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct ProfitSkewSnapshot {
    op_margin_ttm_skew: Option<f64>,
    operate_profit_ttm: Option<f64>,
}

pub fn spec(output: ProfitSkewOutput) -> FactorSpec {
    let (id, aliases, description, dependencies, lookback) = match output {
        ProfitSkewOutput::OpMarginTtmSkew => (
            OP_MARGIN_TTM_SKEW_ID,
            vec![
                "Operating Profit Margin TTM Skew".to_string(),
                "OP Margin TTM Skew".to_string(),
            ],
            "DBZQ operating-profit-margin TTM skew factor. It computes PIT single-quarter rolling TTM operating profit margin, takes strict eight-quarter skewness, replays slow raw values between financial events, and neutralizes by Barra SIZE and SW sector.",
            dependencies(false),
            0,
        ),
        ProfitSkewOutput::PopSkew => (
            POP_SKEW_ID,
            vec!["POP Skew".to_string(), "Operating Profit Valuation Skew".to_string()],
            "DBZQ POP skew factor. It uses PIT operating profit TTM as a slow financial state, daily total_mv as a fast variable, computes daily POP=total_mv/operate_profit_ttm, applies 240-day skewness with min_periods=1, and neutralizes by Barra SIZE and SW sector.",
            dependencies(true),
            DAILY_LOOKBACK,
        ),
    };
    FactorSpec {
        id: id.to_string(),
        aliases,
        name: id.to_string(),
        asset_class: AssetClass::Stock,
        frequency: Frequency::Daily,
        version: VERSION.to_string(),
        tags: tags(output),
        description: description.to_string(),
        dependencies,
        intraday_raw_dependencies: Vec::new(),
        lookback: Lookback {
            trading_days: lookback,
        },
    }
}

pub fn compute_requested(
    requested_ids: &[String],
    context: &FactorContext,
    data: &DataPool,
) -> Result<Vec<FactorSeries>> {
    let mut state = ProfitSkewComputeState::default();
    compute_requested_stateful(requested_ids, context, data, &mut state)
}

pub fn compute_requested_stateful(
    requested_ids: &[String],
    _context: &FactorContext,
    data: &DataPool,
    state: &mut ProfitSkewComputeState,
) -> Result<Vec<FactorSeries>> {
    let needs = needs_from_requested(requested_ids);
    if !needs.op_margin && !needs.pop {
        return Ok(Vec::new());
    }

    // The engine passes overlapping warmup dates between date batches. Resetting the
    // provider cache per batch prevents a later batch's warmup rows from replaying
    // the previous batch's future financial state.
    state.snapshot_cache = InstrumentAlignedSnapshotCache::default();

    let panel = data.stock_universe_panel()?;
    let income = data.financial_reader(
        DatasetId::StockIncome,
        ReportTypePreference::income_single_quarter(),
    )?;
    let (op_margin_raw, operate_profit_ttm) =
        slow_profit_columns(&panel, &income, &mut state.snapshot_cache, needs)?;

    let mut output = Vec::new();
    if needs.op_margin {
        let factor = neutralize_size_sector(
            &op_margin_raw.expect("op margin raw computed when requested"),
            &panel,
            data,
        )?;
        output.push(factor.to_factor_series(spec(ProfitSkewOutput::OpMarginTtmSkew)));
    }
    if needs.pop {
        let basic = data.daily(DatasetId::StockDailyBasic)?;
        let total_mv = panel.column_from_table(basic, TOTAL_MV_COLUMN)?;
        let pop = operate_profit_ttm
            .expect("operating profit TTM computed when POP is requested")
            .zip_binary(&total_mv, pop_from_mv_and_operating_profit)?;
        let pop_skew =
            pop.ts(|series| ts_skew(series, ROLLING_POP_WINDOW, ROLLING_POP_MIN_PERIODS))?;
        let factor = neutralize_size_sector(&pop_skew, &panel, data)?;
        output.push(factor.to_factor_series(spec(ProfitSkewOutput::PopSkew)));
    }
    Ok(output)
}

fn slow_profit_columns(
    panel: &DailyPanel,
    income: &FinancialPitReader<'_>,
    cache: &mut InstrumentAlignedSnapshotCache<ProfitSkewSnapshot>,
    needs: ProfitSkewNeeds,
) -> Result<(Option<PanelColumn>, Option<PanelColumn>)> {
    let instrument_count = panel.instruments().len();
    let mut op_margin_values = needs.op_margin.then(|| vec![None; panel.shape_len()]);
    let mut operate_profit_values = needs.pop.then(|| vec![None; panel.shape_len()]);
    let mut current_snapshots = vec![None; instrument_count];
    let schedule = FinancialEventSchedule::from_pit_readers(&[income.clone()]);
    let mut last_processed_trade_date = None;

    for (date_idx, trade_date) in panel.dates().iter().copied().enumerate() {
        let should_recompute = last_processed_trade_date.is_none()
            || schedule.has_event_after_until(last_processed_trade_date, trade_date);
        if should_recompute {
            current_snapshots = cached_financial_stock_snapshots_for_date(
                panel,
                trade_date,
                cache,
                |_, ts_code, offset| is_bj_stock(ts_code) || !panel.is_present_offset(offset),
                |trade_date, ts_code, _| profit_skew_marker(ts_code, trade_date, income, needs),
                |trade_date, ts_code, _| profit_skew_snapshot(ts_code, trade_date, income, needs),
            );
        }

        let date_offset = date_idx * instrument_count;
        for (instrument_idx, snapshot) in current_snapshots.iter().enumerate() {
            let offset = date_offset + instrument_idx;
            if !panel.is_present_offset(offset) {
                continue;
            }
            if let Some(values) = &mut op_margin_values {
                values[offset] = snapshot.and_then(|item| item.op_margin_ttm_skew);
            }
            if let Some(values) = &mut operate_profit_values {
                values[offset] = snapshot.and_then(|item| item.operate_profit_ttm);
            }
        }
        last_processed_trade_date = Some(trade_date);
    }

    Ok((
        op_margin_values
            .map(|values| panel.column_from_values(values))
            .transpose()?,
        operate_profit_values
            .map(|values| panel.column_from_values(values))
            .transpose()?,
    ))
}

fn profit_skew_marker(
    ts_code: &str,
    trade_date: i32,
    income: &FinancialPitReader<'_>,
    needs: ProfitSkewNeeds,
) -> Option<FinancialEventMarker> {
    let anchor = income.latest_quarter_end_date(ts_code, trade_date)?;
    let quarter_count = if needs.op_margin {
        OP_MARGIN_REQUIRED_QUARTERS
    } else {
        POP_REQUIRED_QUARTERS
    };
    let ends = quarter_chain(anchor, quarter_count)?;
    let mut builder = FinancialEventMarkerBuilder::new();
    for end_date in ends {
        builder.include_reader_record_for_end_date(
            FinancialStatementDataset::Income,
            income,
            ts_code,
            trade_date,
            end_date,
        );
    }
    builder.build()
}

fn profit_skew_snapshot(
    ts_code: &str,
    trade_date: i32,
    income: &FinancialPitReader<'_>,
    needs: ProfitSkewNeeds,
) -> Option<ProfitSkewSnapshot> {
    let operate_profit_ttm = needs
        .pop
        .then(|| operating_profit_ttm(ts_code, trade_date, income))
        .flatten();
    let op_margin_ttm_skew = needs
        .op_margin
        .then(|| op_margin_ttm_skew(ts_code, trade_date, income))
        .flatten();
    Some(ProfitSkewSnapshot {
        op_margin_ttm_skew,
        operate_profit_ttm,
    })
}

fn operating_profit_ttm(
    ts_code: &str,
    trade_date: i32,
    income: &FinancialPitReader<'_>,
) -> Option<f64> {
    let anchor = income.latest_quarter_end_date(ts_code, trade_date)?;
    clean(income.ttm_sum_for_end_date(ts_code, trade_date, anchor, OPERATE_PROFIT_COLUMN))
}

fn op_margin_ttm_skew(
    ts_code: &str,
    trade_date: i32,
    income: &FinancialPitReader<'_>,
) -> Option<f64> {
    let anchor = income.latest_quarter_end_date(ts_code, trade_date)?;
    let ends = quarter_chain(anchor, OP_MARGIN_TTM_SKEW_QUARTERS)?;
    let mut margins = Vec::with_capacity(OP_MARGIN_TTM_SKEW_QUARTERS);
    for end_date in ends {
        let operate_profit = clean(income.ttm_sum_for_end_date(
            ts_code,
            trade_date,
            end_date,
            OPERATE_PROFIT_COLUMN,
        ))?;
        let revenue =
            clean(income.ttm_sum_for_end_date(ts_code, trade_date, end_date, REVENUE_COLUMN))
                .filter(|value| *value > EPS)?;
        let margin = operate_profit / revenue;
        if !margin.is_finite() {
            return None;
        }
        margins.push(margin);
    }
    skewness(&margins)
}

fn pop_from_mv_and_operating_profit(
    operate_profit_ttm: Option<f64>,
    total_mv: Option<f64>,
) -> Option<f64> {
    let denominator = clean(operate_profit_ttm).filter(|value| value.abs() > EPS)?;
    let numerator = clean(total_mv).filter(|value| *value > 0.0)?;
    let value = numerator / denominator;
    value.is_finite().then_some(value)
}

fn skewness(values: &[f64]) -> Option<f64> {
    if values.len() != OP_MARGIN_TTM_SKEW_QUARTERS {
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
        / values.len() as f64;
    let std_dev = variance.sqrt();
    if std_dev <= f64::EPSILON {
        return None;
    }
    let third_moment = values
        .iter()
        .map(|value| (value - mean).powi(3))
        .sum::<f64>()
        / values.len() as f64;
    let skew = third_moment / std_dev.powi(3);
    skew.is_finite().then_some(skew)
}

fn quarter_chain(anchor: i32, len: usize) -> Option<Vec<i32>> {
    let mut output = Vec::with_capacity(len);
    let mut current = anchor;
    for _ in 0..len {
        output.push(current);
        current = previous_quarter_end_date(current)?;
    }
    Some(output)
}

fn needs_from_requested(requested_ids: &[String]) -> ProfitSkewNeeds {
    ProfitSkewNeeds {
        op_margin: requested_ids.iter().any(|id| id == OP_MARGIN_TTM_SKEW_ID),
        pop: requested_ids.iter().any(|id| id == POP_SKEW_ID),
    }
}

fn dependencies(include_daily_basic: bool) -> Vec<DataRequest> {
    let mut requests = vec![
        DataRequest::financial_quarters(
            DatasetId::StockIncome,
            &INCOME_COLUMNS,
            OP_MARGIN_REQUIRED_QUARTERS,
        ),
        DataRequest::new(DatasetId::StockBarraDaily, &["SIZE"]),
        DataRequest::new(DatasetId::StockSwClassification, &["l1_code"]),
    ];
    if include_daily_basic {
        requests.push(DataRequest::new(
            DatasetId::StockDailyBasic,
            &[TOTAL_MV_COLUMN],
        ));
    }
    requests
}

fn tags(_output: ProfitSkewOutput) -> Vec<String> {
    [
        "DBZQ",
        "financial",
        "fundamental",
        "pit",
        "skewness",
        "profitability",
        "valuation",
        "neutralize",
        "barra",
        "size",
        "sector",
        "daily",
    ]
    .iter()
    .map(|tag| (*tag).to_string())
    .collect::<Vec<_>>()
}

fn clean(value: Option<f64>) -> Option<f64> {
    value.filter(|value| value.is_finite())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pop_keeps_negative_operating_profit_and_rejects_zero_denominator() {
        assert_eq!(
            pop_from_mv_and_operating_profit(Some(-10.0), Some(100.0)),
            Some(-10.0)
        );
        assert_eq!(
            pop_from_mv_and_operating_profit(Some(0.0), Some(100.0)),
            None
        );
        assert_eq!(
            pop_from_mv_and_operating_profit(Some(10.0), Some(0.0)),
            None
        );
    }

    #[test]
    fn skewness_requires_strict_eight_values_and_rejects_constant_series() {
        assert_eq!(skewness(&[1.0, 2.0]), None);
        assert_eq!(skewness(&[1.0; OP_MARGIN_TTM_SKEW_QUARTERS]), None);
        assert!(skewness(&[1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]).is_some());
    }

    #[test]
    fn requested_outputs_are_requested_aware() {
        let requested = vec![OP_MARGIN_TTM_SKEW_ID.to_string()];
        let needs = needs_from_requested(&requested);
        assert!(needs.op_margin);
        assert!(!needs.pop);
    }

    #[test]
    fn specs_have_dbzq_tags_and_pop_uses_daily_basic() {
        let op = spec(ProfitSkewOutput::OpMarginTtmSkew);
        let pop = spec(ProfitSkewOutput::PopSkew);
        for spec in [&op, &pop] {
            assert!(spec.tags.contains(&"DBZQ".to_string()));
            assert!(spec.tags.contains(&"financial".to_string()));
        }
        assert_eq!(op.lookback.trading_days, 0);
        assert_eq!(pop.lookback.trading_days, DAILY_LOOKBACK);
        assert!(!op
            .dependencies
            .iter()
            .any(|request| request.dataset == DatasetId::StockDailyBasic));
        assert!(pop.dependencies.iter().any(|request| {
            request.dataset == DatasetId::StockDailyBasic
                && request
                    .columns
                    .iter()
                    .any(|column| column == TOTAL_MV_COLUMN)
        }));
    }
}

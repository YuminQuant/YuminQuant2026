use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::financial::previous_quarter_end_date;
use crate::factor::common::stock_daily_ops::{
    is_bj_stock, mask_bj, neutralize_size_sector_with_inputs,
};
use crate::factor::common::vector::clean;
use crate::factor::common::{
    ClassificationLevel, ClassificationMap, DailyPanel, PanelColumn, ReportTypePreference,
};
use crate::factor::{Factor, FactorUpdatePolicy};
use crate::operators::cs_pctrank;

pub const FACTOR_ID: &str = "aog_rank_demax_20d";

const VERSION: &str = "0.1.0";
const LOOKBACK: usize = 252;
const HISTORY_WINDOW: usize = 20;
const FINANCIAL_QUARTERS: usize = 8;
const EPS: f64 = 1e-12;
const INCOME_COLUMNS: [&str; 1] = ["n_income_attr_p"];

pub struct StockDailyAogRankDemax20d;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyAogRankDemax20d)
}

impl Factor for StockDailyAogRankDemax20d {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: FACTOR_ID.to_string(),
            aliases: vec!["AOG_RANK_DEMAX_20d".to_string()],
            name: FACTOR_ID.to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: tags(),
            description: "DFZQ earnings announcement opening-gap rank factor. It ranks the first trading day's opening gap after each PIT earnings disclosure, subtracts the maximum opening-gap rank from the 20 trading days ending at the announcement anchor date, holds the raw event value until the next disclosure reaction day, and neutralizes by SW level-1 industry and Barra SIZE.".to_string(),
            dependencies: vec![
                DataRequest::new(DatasetId::StockDailyPv, &["open", "pre_close"]),
                DataRequest::financial_quarters(
                    DatasetId::StockIncome,
                    &INCOME_COLUMNS,
                    FINANCIAL_QUARTERS,
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

    fn update_policy(&self) -> FactorUpdatePolicy {
        FactorUpdatePolicy::FinancialEventStateDailyFast
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let panel = data.stock_universe_panel()?;
        let pv = data.daily(DatasetId::StockDailyPv)?;
        let open = panel.column_from_table(pv, "open")?;
        let pre_close = panel.column_from_table(pv, "pre_close")?;
        let income = data.financial_reader(
            DatasetId::StockIncome,
            ReportTypePreference::income_single_quarter(),
        )?;
        let size = panel.column_from_table(data.daily(DatasetId::StockBarraDaily)?, "SIZE")?;
        let sector_map = ClassificationMap::from_table(
            data.daily(DatasetId::StockSwClassification)?,
            ClassificationLevel::Sector,
        )?;

        let gap_rank = ranked_open_gap_column(panel, &open, &pre_close)?;
        let raw = aog_raw_column(panel, &gap_rank, &income)?;
        let masked = mask_bj(&raw, panel)?;
        let factor = neutralize_size_sector_with_inputs(&masked, panel, &size, &sector_map)?;
        Ok(factor.to_factor_series(self.spec()))
    }
}

fn tags() -> Vec<String> {
    [
        "DFZQ",
        "financial",
        "fundamental",
        "earnings_announcement",
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

fn ranked_open_gap_column(
    panel: &DailyPanel,
    open: &PanelColumn,
    pre_close: &PanelColumn,
) -> Result<PanelColumn> {
    let instrument_count = panel.instruments().len();
    let mut gaps = vec![None; panel.shape_len()];
    for (date_idx, _) in panel.dates().iter().enumerate() {
        let date_offset = date_idx * instrument_count;
        for (instrument_idx, ts_code) in panel.instruments().iter().enumerate() {
            if is_bj_stock(ts_code) {
                continue;
            }
            let offset = date_offset + instrument_idx;
            if !panel.is_present_offset(offset) {
                continue;
            }
            gaps[offset] = open_gap(open.values()[offset], pre_close.values()[offset]);
        }
    }
    panel
        .column_from_values(gaps)?
        .cs(|values| cs_pctrank(values, true))
}

fn aog_raw_column(
    panel: &DailyPanel,
    gap_rank: &PanelColumn,
    income: &crate::factor::common::FinancialPitReader<'_>,
) -> Result<PanelColumn> {
    let instrument_count = panel.instruments().len();
    let mut values = vec![None; panel.shape_len()];

    for (date_idx, trade_date) in panel.dates().iter().copied().enumerate() {
        let date_offset = date_idx * instrument_count;
        for (instrument_idx, ts_code) in panel.instruments().iter().enumerate() {
            let offset = date_offset + instrument_idx;
            if !panel.is_present_offset(offset) || is_bj_stock(ts_code) {
                continue;
            }
            if let Some((anchor_idx, reaction_idx)) =
                announcement_indices_for_stock(panel, income, ts_code, trade_date)
            {
                values[offset] = aog_event_value(
                    gap_rank,
                    instrument_count,
                    instrument_idx,
                    anchor_idx,
                    reaction_idx,
                );
            }
        }
    }

    panel.column_from_values(values)
}

fn announcement_indices_for_stock(
    panel: &DailyPanel,
    income: &crate::factor::common::FinancialPitReader<'_>,
    ts_code: &str,
    trade_date: i32,
) -> Option<(usize, usize)> {
    let mut end_date = income.latest_quarter_end_date(ts_code, trade_date)?;
    for _ in 0..FINANCIAL_QUARTERS {
        let record = income.record_for_end_date(ts_code, trade_date, end_date)?;
        if let Some((anchor_idx, reaction_idx)) =
            announcement_anchor_reaction_indices(panel.dates(), record.disclosure_date())
        {
            if panel.dates()[reaction_idx] <= trade_date {
                return Some((anchor_idx, reaction_idx));
            }
        }
        end_date = previous_quarter_end_date(end_date)?;
    }
    None
}

fn announcement_anchor_reaction_indices(dates: &[i32], ann_date: i32) -> Option<(usize, usize)> {
    let anchor_idx = last_date_on_or_before(dates, ann_date)?;
    let reaction_idx = first_date_after(dates, ann_date)?;
    (anchor_idx < reaction_idx).then_some((anchor_idx, reaction_idx))
}

fn last_date_on_or_before(dates: &[i32], target: i32) -> Option<usize> {
    match dates.binary_search(&target) {
        Ok(idx) => Some(idx),
        Err(0) => None,
        Err(idx) => Some(idx - 1),
    }
}

fn first_date_after(dates: &[i32], target: i32) -> Option<usize> {
    match dates.binary_search(&target) {
        Ok(idx) => (idx + 1 < dates.len()).then_some(idx + 1),
        Err(idx) => (idx < dates.len()).then_some(idx),
    }
}

fn aog_event_value(
    gap_rank: &PanelColumn,
    instrument_count: usize,
    instrument_idx: usize,
    anchor_idx: usize,
    reaction_idx: usize,
) -> Option<f64> {
    let start_idx = anchor_idx.checked_add(1)?.checked_sub(HISTORY_WINDOW)?;
    let reaction_offset = reaction_idx * instrument_count + instrument_idx;
    let reaction_rank = clean(gap_rank.values()[reaction_offset])?;
    let mut max_history = f64::NEG_INFINITY;
    for date_idx in start_idx..=anchor_idx {
        let offset = date_idx * instrument_count + instrument_idx;
        let value = clean(gap_rank.values()[offset])?;
        max_history = max_history.max(value);
    }
    let value = reaction_rank - max_history;
    value.is_finite().then_some(value)
}

fn open_gap(open: Option<f64>, pre_close: Option<f64>) -> Option<f64> {
    match (clean(open), clean(pre_close)) {
        (Some(open), Some(pre_close)) if pre_close.abs() > EPS => Some(open / pre_close - 1.0),
        _ => None,
    }
}

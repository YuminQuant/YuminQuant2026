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
    ClassificationLevel, ClassificationMap, DailyPanel, PanelColumn, ReportTypePreference,
};
use crate::factor::{Factor, FactorUpdatePolicy};

pub const PROVIDER_KEY: &str = "stock|daily|earnings_reaction_gaussian";
pub const MARKET_INDEX: &str = "000985.CSI";

const VERSION: &str = "0.1.0";
const LOOKBACK: usize = 252;
const FINANCIAL_QUARTERS: usize = 8;
const EPS: f64 = 1e-12;
const INCOME_COLUMNS: [&str; 1] = ["n_income_attr_p"];

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum EarningsReactionOutput {
    GapIndustryExcess,
    TurnoverMarketExcess,
}

impl EarningsReactionOutput {
    pub fn id(self) -> &'static str {
        match self {
            Self::GapIndustryExcess => "earnings_gap_ind_excess_gauss_resid",
            Self::TurnoverMarketExcess => "earnings_turnover_mkt_excess_gauss_resid",
        }
    }

    fn alias(self) -> &'static str {
        match self {
            Self::GapIndustryExcess => "Earnings Next-Day Gap Industry Excess Gaussian Residual",
            Self::TurnoverMarketExcess => {
                "Earnings Next-Day Turnover Market Excess Gaussian Residual"
            }
        }
    }

    fn from_id(id: &str) -> Option<Self> {
        Some(match id {
            "earnings_gap_ind_excess_gauss_resid" => Self::GapIndustryExcess,
            "earnings_turnover_mkt_excess_gauss_resid" => Self::TurnoverMarketExcess,
            _ => return None,
        })
    }
}

pub struct EarningsReactionGaussianFactor {
    kind: EarningsReactionOutput,
}

impl EarningsReactionGaussianFactor {
    pub fn new(kind: EarningsReactionOutput) -> Self {
        Self { kind }
    }
}

impl Factor for EarningsReactionGaussianFactor {
    fn spec(&self) -> FactorSpec {
        spec(self.kind)
    }

    fn compute_provider_key(&self) -> String {
        PROVIDER_KEY.to_string()
    }

    fn update_policy(&self) -> FactorUpdatePolicy {
        FactorUpdatePolicy::FinancialEventStateDailyFast
    }

    fn compute(&self, context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let requested = [self.kind.id().to_string()];
        compute_requested(&requested, context, data)?
            .into_iter()
            .find(|series| series.spec.id == self.kind.id())
            .ok_or_else(|| {
                err(format!(
                    "earnings reaction provider did not return {}",
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
}

pub fn spec(kind: EarningsReactionOutput) -> FactorSpec {
    FactorSpec {
        id: kind.id().to_string(),
        aliases: vec![kind.alias().to_string()],
        name: kind.id().to_string(),
        asset_class: AssetClass::Stock,
        frequency: Frequency::Daily,
        version: VERSION.to_string(),
        tags: tags(),
        description: format!(
            "Earnings-announcement reaction Gaussian-rank residual factor {}. It uses latest PIT earnings announcement dates, announcement-next-day reactions, cumulative excess return controls, then SIZE and SW-sector neutralization.",
            kind.id()
        ),
        dependencies: vec![
            DataRequest::new(DatasetId::StockDailyPv, &["open", "close", "pre_close"]),
            DataRequest::new(DatasetId::StockDailyBasic, &["turnover_rate_f"]),
            DataRequest::financial_quarters(DatasetId::StockIncome, &INCOME_COLUMNS, FINANCIAL_QUARTERS),
            DataRequest::new(DatasetId::StockBarraDaily, &["SIZE"]),
            DataRequest::new(DatasetId::StockSwClassification, &["l1_code"]),
            DataRequest::index_daily(MARKET_INDEX, &["close", "pre_close"]),
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
        "earnings_announcement",
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

pub fn compute_requested(
    requested_ids: &[String],
    _context: &FactorContext,
    data: &DataPool,
) -> Result<Vec<FactorSeries>> {
    let mut requested = requested_ids
        .iter()
        .filter_map(|id| EarningsReactionOutput::from_id(id))
        .collect::<Vec<_>>();
    requested.sort();
    requested.dedup();
    if requested.is_empty() {
        return Ok(Vec::new());
    }

    let panel = data.stock_universe_panel()?;
    let pv = data.daily(DatasetId::StockDailyPv)?;
    let open = panel.column_from_table(pv, "open")?;
    let close = panel.column_from_table(pv, "close")?;
    let pre_close = panel.column_from_table(pv, "pre_close")?;
    let stock_return = close.zip_binary(&pre_close, ret)?;
    let turnover =
        panel.column_from_table(data.daily(DatasetId::StockDailyBasic)?, "turnover_rate_f")?;
    let sector_map = ClassificationMap::from_table(
        data.daily(DatasetId::StockSwClassification)?,
        ClassificationLevel::Sector,
    )?;
    let size = panel.column_from_table(data.daily(DatasetId::StockBarraDaily)?, "SIZE")?;
    let income = data.financial_reader(
        DatasetId::StockIncome,
        ReportTypePreference::income_single_quarter(),
    )?;

    let groups_by_date = panel
        .dates()
        .iter()
        .map(|date| sector_map.groups_for(*date, panel.instruments()))
        .collect::<Vec<_>>();
    let market_gap_mean = market_open_gap_mean(panel, &open, &close);
    let industry_returns = industry_equal_weight_returns(panel, &stock_return, &groups_by_date);
    let index_close_by_date = index_close_map(data.index_daily_panel(MARKET_INDEX)?)?;

    let mut gap_y = vec![None; panel.shape_len()];
    let mut ind_x = vec![None; panel.shape_len()];
    let mut turnover_y = vec![None; panel.shape_len()];
    let mut mkt_x = vec![None; panel.shape_len()];
    let instrument_count = panel.instruments().len();

    for (date_idx, trade_date) in panel.dates().iter().copied().enumerate() {
        if !panel.is_target_date(trade_date) {
            continue;
        }
        let current_offset = date_idx * instrument_count;
        for (instrument_idx, ts_code) in panel.instruments().iter().enumerate() {
            let offset = current_offset + instrument_idx;
            if !panel.is_present_offset(offset) || is_bj_stock(ts_code) {
                continue;
            }
            let Some((ann_trade_idx, ann_next_idx)) =
                announcement_trade_indices(panel, &income, ts_code, trade_date)
            else {
                continue;
            };
            if date_idx < ann_next_idx {
                continue;
            }
            let ann_trade_offset = ann_trade_idx * instrument_count + instrument_idx;
            let ann_next_offset = ann_next_idx * instrument_count + instrument_idx;
            if !panel.is_present_offset(ann_trade_offset)
                || !panel.is_present_offset(ann_next_offset)
            {
                continue;
            }

            let stock_gap = open_gap(
                open.values()[ann_next_offset],
                close.values()[ann_trade_offset],
            );
            gap_y[offset] = subtract(stock_gap, market_gap_mean[ann_next_idx]);
            turnover_y[offset] = clean(turnover.values()[ann_next_offset]);

            let stock_interval =
                price_return(close.values()[offset], close.values()[ann_trade_offset]);
            let sector = groups_by_date[date_idx][instrument_idx].as_deref();
            let industry_interval = sector.and_then(|sector| {
                cumulative_group_return(&industry_returns, sector, ann_trade_idx + 1, date_idx)
            });
            ind_x[offset] = subtract(stock_interval, industry_interval);

            let ann_trade_date = panel.dates()[ann_trade_idx];
            let current_date = panel.dates()[date_idx];
            let market_interval = match (
                index_close_by_date.get(&current_date).copied(),
                index_close_by_date.get(&ann_trade_date).copied(),
            ) {
                (Some(current), Some(start)) => price_return(Some(current), Some(start)),
                _ => None,
            };
            mkt_x[offset] = subtract(stock_interval, market_interval);
        }
    }

    let gap_y = panel.column_from_values(gap_y)?;
    let ind_x = panel.column_from_values(ind_x)?;
    let turnover_y = panel.column_from_values(turnover_y)?;
    let mkt_x = panel.column_from_values(mkt_x)?;

    let gap_raw = if requested.contains(&EarningsReactionOutput::GapIndustryExcess) {
        Some(gaussian_residual(&gap_y, &[&ind_x])?)
    } else {
        None
    };
    let turnover_raw = if requested.contains(&EarningsReactionOutput::TurnoverMarketExcess) {
        Some(gaussian_residual(&turnover_y, &[&mkt_x])?)
    } else {
        None
    };

    let mut output = Vec::new();
    for kind in requested {
        let raw = match kind {
            EarningsReactionOutput::GapIndustryExcess => gap_raw.as_ref().unwrap(),
            EarningsReactionOutput::TurnoverMarketExcess => turnover_raw.as_ref().unwrap(),
        };
        let masked = mask_bj(raw, panel)?;
        let factor = neutralize_size_sector_with_inputs(&masked, panel, &size, &sector_map)?;
        output.push(factor.to_factor_series(spec(kind)));
    }
    Ok(output)
}

fn announcement_trade_indices(
    panel: &DailyPanel,
    income: &crate::factor::common::FinancialPitReader<'_>,
    ts_code: &str,
    trade_date: i32,
) -> Option<(usize, usize)> {
    let end_date = income.latest_quarter_end_date(ts_code, trade_date)?;
    let record = income.record_for_end_date(ts_code, trade_date, end_date)?;
    let ann_date = record.disclosure_date();
    let ann_trade_idx = first_date_on_or_after(panel.dates(), ann_date)?;
    let ann_next_idx = ann_trade_idx + 1;
    (ann_next_idx < panel.dates().len()).then_some((ann_trade_idx, ann_next_idx))
}

fn first_date_on_or_after(dates: &[i32], target: i32) -> Option<usize> {
    dates
        .binary_search(&target)
        .map(Some)
        .unwrap_or_else(|idx| (idx < dates.len()).then_some(idx))
}

fn market_open_gap_mean(
    panel: &DailyPanel,
    open: &PanelColumn,
    close: &PanelColumn,
) -> Vec<Option<f64>> {
    let instrument_count = panel.instruments().len();
    let mut output = vec![None; panel.dates().len()];
    for (date_idx, _) in panel.dates().iter().enumerate().skip(1) {
        let mut sum = 0.0;
        let mut count = 0usize;
        let date_offset = date_idx * instrument_count;
        let prev_offset = (date_idx - 1) * instrument_count;
        for (instrument_idx, ts_code) in panel.instruments().iter().enumerate() {
            if is_bj_stock(ts_code) {
                continue;
            }
            let offset = date_offset + instrument_idx;
            let prev = prev_offset + instrument_idx;
            if !panel.is_present_offset(offset) || !panel.is_present_offset(prev) {
                continue;
            }
            if let Some(value) = open_gap(open.values()[offset], close.values()[prev]) {
                sum += value;
                count += 1;
            }
        }
        if count > 0 {
            output[date_idx] = Some(sum / count as f64);
        }
    }
    output
}

fn industry_equal_weight_returns(
    panel: &DailyPanel,
    stock_return: &PanelColumn,
    groups_by_date: &[Vec<Option<String>>],
) -> Vec<HashMap<String, f64>> {
    let instrument_count = panel.instruments().len();
    let mut output = Vec::with_capacity(panel.dates().len());
    for date_idx in 0..panel.dates().len() {
        let mut sums = HashMap::<String, (f64, usize)>::new();
        let date_offset = date_idx * instrument_count;
        for (instrument_idx, ts_code) in panel.instruments().iter().enumerate() {
            if is_bj_stock(ts_code) {
                continue;
            }
            let offset = date_offset + instrument_idx;
            if !panel.is_present_offset(offset) {
                continue;
            }
            let (Some(group), Some(ret)) = (
                groups_by_date[date_idx][instrument_idx].clone(),
                clean(stock_return.values()[offset]),
            ) else {
                continue;
            };
            let entry = sums.entry(group).or_insert((0.0, 0));
            entry.0 += ret;
            entry.1 += 1;
        }
        output.push(
            sums.into_iter()
                .filter_map(|(group, (sum, count))| {
                    (count > 0).then_some((group, sum / count as f64))
                })
                .collect(),
        );
    }
    output
}

fn cumulative_group_return(
    returns_by_date: &[HashMap<String, f64>],
    group: &str,
    start_idx: usize,
    end_idx: usize,
) -> Option<f64> {
    if start_idx > end_idx || end_idx >= returns_by_date.len() {
        return Some(0.0);
    }
    let mut product = 1.0;
    for date_idx in start_idx..=end_idx {
        let ret = *returns_by_date[date_idx].get(group)?;
        if ret <= -1.0 || !ret.is_finite() {
            return None;
        }
        product *= 1.0 + ret;
    }
    Some(product - 1.0)
}

fn index_close_map(index_panel: &DailyPanel) -> Result<HashMap<i32, f64>> {
    let close = index_panel.column("close")?;
    let instrument_count = index_panel.instruments().len();
    let mut output = HashMap::new();
    if instrument_count == 0 {
        return Ok(output);
    }
    for (date_idx, trade_date) in index_panel.dates().iter().copied().enumerate() {
        let offset = date_idx * instrument_count;
        if let Some(value) = clean(close.values()[offset]) {
            output.insert(trade_date, value);
        }
    }
    Ok(output)
}

fn open_gap(open: Option<f64>, previous_close: Option<f64>) -> Option<f64> {
    match (clean(open), clean(previous_close)) {
        (Some(open), Some(previous_close)) if previous_close.abs() > EPS => {
            Some(open / previous_close - 1.0)
        }
        _ => None,
    }
}

fn price_return(current: Option<f64>, start: Option<f64>) -> Option<f64> {
    match (clean(current), clean(start)) {
        (Some(current), Some(start)) if start.abs() > EPS => Some(current / start - 1.0),
        _ => None,
    }
}

fn ret(numerator: Option<f64>, denominator: Option<f64>) -> Option<f64> {
    price_return(numerator, denominator)
}

fn subtract(left: Option<f64>, right: Option<f64>) -> Option<f64> {
    let value = left? - right?;
    value.is_finite().then_some(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: Option<f64>, expected: f64) {
        let actual = actual.expect("value");
        assert!(
            (actual - expected).abs() < 1e-10,
            "actual={actual}, expected={expected}"
        );
    }

    #[test]
    fn first_date_on_or_after_finds_announcement_trade_date() {
        let dates = [20260102, 20260105, 20260106];
        assert_eq!(first_date_on_or_after(&dates, 20260101), Some(0));
        assert_eq!(first_date_on_or_after(&dates, 20260105), Some(1));
        assert_eq!(first_date_on_or_after(&dates, 20260104), Some(1));
        assert_eq!(first_date_on_or_after(&dates, 20260107), None);
    }

    #[test]
    fn open_gap_and_excess_formula_use_market_mean() {
        let stock_gap = open_gap(Some(12.0), Some(10.0));
        assert_close(stock_gap, 0.2);
        assert_close(subtract(stock_gap, Some(0.05)), 0.15);
    }

    #[test]
    fn cumulative_group_return_compounds_daily_returns() {
        let mut d0 = HashMap::new();
        d0.insert("801010".to_string(), 0.10);
        let mut d1 = HashMap::new();
        d1.insert("801010".to_string(), -0.05);
        let returns = vec![d0, d1];
        assert_close(
            cumulative_group_return(&returns, "801010", 0, 1),
            (1.10 * 0.95) - 1.0,
        );
    }

    #[test]
    fn earnings_reaction_specs_have_fundamental_gaussian_tags() {
        let factor_spec = spec(EarningsReactionOutput::GapIndustryExcess);
        assert_eq!(factor_spec.id, "earnings_gap_ind_excess_gauss_resid");
        assert!(factor_spec.tags.contains(&"fundamental".to_string()));
        assert!(factor_spec.tags.contains(&"gaussian_rank".to_string()));
    }
}

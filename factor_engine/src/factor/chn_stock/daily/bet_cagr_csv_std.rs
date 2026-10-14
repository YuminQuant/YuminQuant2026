use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::financial::previous_quarter_end_date;
use crate::factor::common::stock_daily_ops::{is_bj_stock, neutralize_and_fill_missing};
use crate::factor::common::{
    ClassificationLevel, ClassificationMap,
    routed_financial_stock_snapshots_for_date, FinancialStockEventRoute, DailyPanel, FinancialEventMarker,
    FinancialEventMarkerBuilder, FinancialEventSchedule, FinancialPitReader,
    FinancialStatementDataset, InstrumentAlignedSnapshotCache, PanelColumn, ReportTypePreference,
};
use crate::factor::{Factor, FactorUpdatePolicy};
use crate::operators::ts_zscore;

const WINDOW: usize = 252;
const MIN_PERIODS: usize = 60;
const QUARTERS: usize = 8;
const PROFIT: &str = "n_income_attr_p";
const CAGR: &str = "con_npcgrate_2y_roll";
const EPS: f64 = 1e-12;

pub struct StockDailyBetCagrCsvStd;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyBetCagrCsvStd)
}

impl Factor for StockDailyBetCagrCsvStd {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "bet_cagr_csv_std".into(),
            aliases: vec!["BET_CAGR_CSV_STD".into()],
            name: "Conservative Growth Equilibrium Valuation Z-score".into(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: "0.1.0".into(),
            tags: ["HAZQ", "fundamental", "analyst", "financial", "consensus", "valuation", "pit", "size_neutralize", "sector_neutralize", "daily"]
                .into_iter().map(str::to_string).collect(),
            description: "Daily BET uses g=min(consensus two-year growth / 100, PIT parent net-profit TTM YoY with absolute base). BET=ln1p(g*pe_ttm)/ln1p(g), with BET=pe_ttm at zero growth; positive PE and valid logarithms required. Apply 252-trading-day population z-score including today, min_periods=60, then SW level-1 and Barra SIZE neutralization. Lower is cheaper; missing final raw gets zero residual only with valid SIZE and a fitted industry; no sign flip or winsorization; excludes BJ.".into(),
            dependencies: vec![
                DataRequest::financial_quarters(DatasetId::StockIncome, &[PROFIT], QUARTERS),
                DataRequest::new(DatasetId::StockDailyBasic, &["pe_ttm"]),
                DataRequest::new(DatasetId::StockConsensus, &[CAGR]),
                DataRequest::new(DatasetId::StockBarraDaily, &["SIZE"]),
                DataRequest::new(DatasetId::StockSwClassification, &["l1_code"]),
            ],
            intraday_raw_dependencies: Vec::new(),
            lookback: Lookback { trading_days: WINDOW - 1 },
        }
    }

    fn update_policy(&self) -> FactorUpdatePolicy {
        FactorUpdatePolicy::FinancialEventStateDailyFast
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let panel = data.stock_universe_panel()?;
        let income = data.financial_reader(
            DatasetId::StockIncome,
            ReportTypePreference::income_single_quarter(),
        )?;
        let pe = panel.column_from_table(data.daily(DatasetId::StockDailyBasic)?, "pe_ttm")?;
        let cagr = panel.column_from_table(data.daily(DatasetId::StockConsensus)?, CAGR)?;
        let bet = raw_bet(&panel, &income, &pe, &cagr)?;
        let standardized = bet.ts(|values| ts_zscore(values, WINDOW, MIN_PERIODS))?;
        Ok(finish_analyst(&standardized, &panel, data)?.to_factor_series(self.spec()))
    }
}

fn raw_bet(
    panel: &DailyPanel,
    income: &FinancialPitReader<'_>,
    pe: &PanelColumn,
    cagr: &PanelColumn,
) -> Result<PanelColumn> {
    // Rebuild at each batch's warmup start; never replay future financial state backwards.
    let mut cache = InstrumentAlignedSnapshotCache::<f64>::default();
    let schedule = FinancialEventSchedule::from_pit_readers(&[income.clone()]);
    let n = panel.instruments().len();
    let mut snapshots = vec![None; n];
    let mut values = vec![None; panel.shape_len()];
    let mut last_date = None;
    for (date_idx, date) in panel.dates().iter().copied().enumerate() {
        let start = date_idx * n;
        let presence_changed = date_idx > 0
            && (0..n).any(|i| {
                panel.is_present_offset(start + i) != panel.is_present_offset(start + i - n)
            });
        if last_date.is_none()
            || presence_changed
            || schedule.has_event_after_until(last_date, date)
        {
            snapshots = routed_financial_stock_snapshots_for_date(
                panel,
                date,
                &mut cache,
                &FinancialStockEventRoute::new(&[income]),
                |_, code, offset| is_bj_stock(code) || !panel.is_present_offset(offset),
                |date, code, _| marker(income, code, date),
                |date, code, _| snapshot(income, code, date),
            );
        }
        for (i, growth) in snapshots.iter().enumerate() {
            let offset = start + i;
            if panel.is_present_offset(offset) {
                values[offset] = bet_value(pe.values()[offset], cagr.values()[offset], *growth);
            }
        }
        last_date = Some(date);
    }
    panel.column_from_values(values)
}

fn quarter_ends(anchor: i32) -> Option<[i32; QUARTERS]> {
    let mut ends = [anchor; QUARTERS];
    for i in 1..QUARTERS {
        ends[i] = previous_quarter_end_date(ends[i - 1])?;
    }
    Some(ends)
}

fn marker(income: &FinancialPitReader<'_>, code: &str, date: i32) -> Option<FinancialEventMarker> {
    let ends = quarter_ends(income.latest_quarter_end_date(code, date)?)?;
    let mut builder = FinancialEventMarkerBuilder::new();
    for end in ends {
        builder.include_reader_record_for_end_date(
            FinancialStatementDataset::Income,
            income,
            code,
            date,
            end,
        );
    }
    builder.build()
}

fn snapshot(income: &FinancialPitReader<'_>, code: &str, date: i32) -> Option<f64> {
    let ends = quarter_ends(income.latest_quarter_end_date(code, date)?)?;
    let profits = ends.map(|end| {
        income
            .record_for_end_date(code, date, end)
            .and_then(|record| record.column(PROFIT))
    });
    ttm_yoy(&profits)
}

fn ttm_yoy(profits: &[Option<f64>; QUARTERS]) -> Option<f64> {
    let sum = |values: &[Option<f64>]| -> Option<f64> {
        values
            .iter()
            .try_fold(0.0, |sum, value| {
                Some(sum + value.filter(|v| v.is_finite())?)
            })
            .filter(|v| v.is_finite())
    };
    let current = sum(&profits[..4])?;
    let previous = sum(&profits[4..])?;
    if previous.abs() <= EPS {
        return None;
    }
    let value = (current - previous) / previous.abs();
    value.is_finite().then_some(value)
}

fn bet_value(pe: Option<f64>, cagr_pct: Option<f64>, actual_yoy: Option<f64>) -> Option<f64> {
    let pe = pe.filter(|v| v.is_finite() && *v > 0.0)?;
    let cagr = cagr_pct.filter(|v| v.is_finite())? / 100.0;
    let actual_yoy = actual_yoy.filter(|v| v.is_finite())?;
    let g = cagr.min(actual_yoy);
    let product = g * pe;
    if g <= -1.0 || product <= -1.0 || !product.is_finite() {
        return None;
    }
    let value = if g == 0.0 {
        pe
    } else {
        product.ln_1p() / g.ln_1p()
    };
    value.is_finite().then_some(value)
}

// Fill only the final signal; financial and time-series inputs retain their missing values.
fn finish_analyst(raw: &PanelColumn, panel: &DailyPanel, data: &DataPool) -> Result<PanelColumn> {
    let size = panel.column_from_panel(data.daily_panel(DatasetId::StockBarraDaily)?, "SIZE")?;
    let sector = ClassificationMap::from_table(
        data.daily(DatasetId::StockSwClassification)?, ClassificationLevel::Sector,
    )?;
    let n = panel.instruments().len();
    let mut values = vec![None; panel.shape_len()];
    for (day, &date) in panel.dates().iter().enumerate() {
        if !panel.is_target_date(date) {
            continue;
        }
        let start = day * n;
        let mut groups = sector.groups_for(date, panel.instruments());
        for (i, code) in panel.instruments().iter().enumerate() {
            if is_bj_stock(code) || !panel.is_present_offset(start + i) {
                groups[i] = None;
            }
        }
        let residual = neutralize_and_fill_missing(
            &raw.values()[start..start + n],
            &size.values()[start..start + n],
            &groups,
        );
        values[start..start + n].copy_from_slice(&residual);
    }
    panel.column_from_values(values)
}

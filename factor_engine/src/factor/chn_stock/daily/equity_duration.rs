use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::stock_daily_ops::{is_bj_stock, neutralize_and_fill_missing};
use crate::factor::common::{
    ClassificationLevel, ClassificationMap,
    routed_financial_stock_snapshots_for_date, FinancialStockEventRoute, DailyPanel, DividendReader, FinancialEventMarker,
    FinancialEventMarkerBuilder, FinancialEventSchedule, FinancialPitReader,
    FinancialStatementDataset, InstrumentAlignedSnapshotCache, PanelColumn, ReportTypePreference,
};
use crate::factor::{Factor, FactorUpdatePolicy};

const EQUITY: &str = "total_hldr_eqy_exc_min_int";
const PROFIT: &str = "n_income_attr_p";
const FORECASTS: [&str; 3] = ["con_np_fy0", "con_np_fy1", "con_np_fy2"];
const YUAN_PER_WAN: f64 = 10_000.0;
const RATE_STEP: f64 = 0.0001;

pub struct StockDailyEquityDuration;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyEquityDuration)
}

impl Factor for StockDailyEquityDuration {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "equity_duration".into(),
            aliases: vec!["Equity Duration".into(), "EQUITY_DURATION".into()],
            name: "Negative Implied Equity Duration".into(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: "0.2.1".into(),
            tags: ["XYZQ", "fundamental", "analyst", "financial", "consensus", "valuation", "pit", "size_neutralize", "sector_neutralize", "daily"]
                .into_iter().map(str::to_string).collect(),
            description: "Negative residual-income equity duration using strictly three consensus periods FY0/FY1/FY2 with a PIT annual-disclosure anchor and May 1 fallback; all three must be finite, with no two-period fallback. All monetary totals in wan yuan. Payout=implemented LTM cash dividends / latest PIT parent profit TTM; positive current/recursive equity and TTM profit required. Terminal residual income stays constant. Require a unique positive implied discount rate above 0.0001; central price sensitivity at +/-0.0001. SW level-1 and Barra SIZE neutralization; missing raw gets zero residual only with valid SIZE and a fitted industry; no winsorization or zscore; excludes BJ.".into(),
            dependencies: vec![
                DataRequest::financial_quarters(DatasetId::StockIncome, &[PROFIT], 4),
                DataRequest::financial_quarters(DatasetId::StockBalanceSheet, &[EQUITY], 4),
                DataRequest::new(DatasetId::StockConsensus, &FORECASTS),
                DataRequest::new(DatasetId::StockDailyBasic, &["total_mv"]),
                DataRequest::new(DatasetId::StockDividend, &["ts_code", "ann_date", "div_proc", "cash_div_tax", "ex_date", "base_share"]),
                DataRequest::new(DatasetId::StockBarraDaily, &["SIZE"]),
                DataRequest::new(DatasetId::StockSwClassification, &["l1_code"]),
            ],
            intraday_raw_dependencies: Vec::new(),
            lookback: Lookback { trading_days: 0 },
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
        let balance = data.financial_reader(
            DatasetId::StockBalanceSheet,
            ReportTypePreference::balance_sheet_consolidated(),
        )?;
        let consensus = data.daily(DatasetId::StockConsensus)?;
        let forecasts = [
            panel.column_from_table(consensus, FORECASTS[0])?,
            panel.column_from_table(consensus, FORECASTS[1])?,
            panel.column_from_table(consensus, FORECASTS[2])?,
        ];
        let market_cap =
            panel.column_from_table(data.daily(DatasetId::StockDailyBasic)?, "total_mv")?;
        let raw = raw_duration(
            &panel,
            &income,
            &balance,
            &data.dividend_reader()?,
            &market_cap,
            &forecasts,
        )?;
        Ok(finish_analyst(&raw, &panel, data)?.to_factor_series(self.spec()))
    }
}

#[derive(Clone, Debug)]
struct Snapshot {
    equity: f64,
    profit_ttm: f64,
}

fn positive(value: f64) -> Option<f64> {
    (value.is_finite() && value > 0.0).then_some(value)
}

fn snapshot(
    income: &FinancialPitReader<'_>,
    balance: &FinancialPitReader<'_>,
    code: &str,
    date: i32,
) -> Option<Snapshot> {
    let end = income.latest_quarter_end_date(code, date)?;
    let profit_ttm =
        positive(income.ttm_sum_for_end_date(code, date, end, PROFIT)?)? / YUAN_PER_WAN;
    let end = balance.latest_quarter_end_date(code, date)?;
    let equity = positive(
        balance
            .record_for_end_date(code, date, end)?
            .column(EQUITY)?,
    )? / YUAN_PER_WAN;
    Some(Snapshot { equity, profit_ttm })
}

fn marker(
    income: &FinancialPitReader<'_>,
    balance: &FinancialPitReader<'_>,
    code: &str,
    date: i32,
) -> Option<FinancialEventMarker> {
    let mut marker = FinancialEventMarkerBuilder::new();
    marker.include_reader_latest_ttm(FinancialStatementDataset::Income, income, code, date);
    marker.include_reader_latest_quarter(
        FinancialStatementDataset::BalanceSheet,
        balance,
        code,
        date,
    );
    marker.build()
}

fn ltm_start(date: i32) -> i32 {
    // Only February 29 needs clamping when subtracting exactly one year.
    let month_day = if date % 10000 == 229 {
        228
    } else {
        date % 10000
    };
    (date / 10000 - 1) * 10000 + month_day
}

fn raw_duration(
    panel: &DailyPanel,
    income: &FinancialPitReader<'_>,
    balance: &FinancialPitReader<'_>,
    dividends: &DividendReader<'_>,
    market_cap: &PanelColumn,
    forecasts: &[PanelColumn; 3],
) -> Result<PanelColumn> {
    // Cache only PIT slow inputs; forecasts, dividend expiry and market cap vary daily.
    let mut cache = InstrumentAlignedSnapshotCache::<Snapshot>::default();
    let schedule = FinancialEventSchedule::from_pit_readers(&[income.clone(), balance.clone()]);
    let n = panel.instruments().len();
    let mut snapshots = vec![None; n];
    let mut values = vec![None; panel.shape_len()];
    let mut last_date = None;
    for (day, date) in panel.dates().iter().copied().enumerate() {
        let start = day * n;
        let presence_changed = day > 0
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
                &FinancialStockEventRoute::new(&[income, balance]),
                |_, code, offset| is_bj_stock(code) || !panel.is_present_offset(offset),
                |date, code, _| marker(income, balance, code, date),
                |date, code, _| snapshot(income, balance, code, date),
            );
        }
        // Only one date's dividend map is retained, not a dates-by-stocks history.
        let cash = dividends.implemented_ltm_sum_by_stock(ltm_start(date), date);
        for (i, code) in panel.instruments().iter().enumerate() {
            let offset = start + i;
            if !panel.is_present_offset(offset) || is_bj_stock(code) {
                continue;
            }
            let Some(slow) = snapshots[i].as_ref() else {
                continue;
            };
            let Some(cap) = market_cap.values()[offset] else {
                continue;
            };
            // DividendReader returns cash_div_tax (yuan/share) * base_share (wan shares).
            let payout = cash.get(code.as_str()).copied().unwrap_or(0.0) / slow.profit_ttm;
            values[offset] = duration_alpha(
                slow.equity,
                cap,
                payout,
                std::array::from_fn(|j| forecasts[j].values()[offset]),
            );
        }
        last_date = Some(date);
    }
    panel.column_from_values(values)
}

struct Valuation {
    earnings: [f64; 3],
    payout: f64,
}

impl Valuation {
    fn new(equity: f64, cap: f64, payout: f64, forecasts: [Option<f64>; 3]) -> Option<Self> {
        positive(equity)?;
        positive(cap)?;
        if !payout.is_finite() || payout < 0.0 {
            return None;
        }
        let forecasts = forecasts.map(|value| value.filter(|v| v.is_finite()));
        let earnings = [forecasts[0]?, forecasts[1]?, forecasts[2]?];
        let mut book = equity;
        for earning in &earnings[..2] {
            book = positive(book + earning * (1.0 - payout))?;
        }
        let earnings = earnings.map(|earning| earning / cap);
        earnings
            .iter()
            .all(|e| e.is_finite())
            .then_some(Self { earnings, payout })
    }

    fn price_ratio(&self, rate: f64) -> Option<f64> {
        positive(rate)?;
        // Clean-surplus telescoping of the residual-income formula avoids subtracting
        // large book values: early dividends + terminal earnings perpetuity.
        let q = 1.0 + rate;
        let value = self.payout * self.earnings[0] / q
            + (self.payout * self.earnings[1] + self.earnings[2] / rate) / (q * q);
        value.is_finite().then_some(value)
    }

    fn implied_rate(&self) -> Option<f64> {
        // V(r)=market cap is a monic cubic. Partition at derivative
        // roots so each interval is monotone, detecting multiple positive roots.
        let (b, c, d) = (
            2.0 - self.payout * self.earnings[0],
            1.0 - self.payout * (self.earnings[0] + self.earnings[1]),
            -self.earnings[2],
        );
        if ![b, c, d].iter().all(|v| v.is_finite()) {
            return None;
        }
        let poly = |r: f64| ((r + b) * r + c) * r + d;
        let bound = 1.0 + b.abs().max(c.abs()).max(d.abs());
        let mut points = [0.0; 4];
        let mut count = 1;
        let disc = b * b - 3.0 * c;
        if !disc.is_finite() {
            return None;
        }
        let critical = if disc >= 0.0 {
            [(-b - disc.sqrt()) / 3.0, (-b + disc.sqrt()) / 3.0]
        } else {
            [0.0; 2]
        };
        for r in critical {
            if r > 0.0 && r < bound && (count == 1 || r > points[count - 1]) {
                points[count] = r;
                count += 1;
            }
        }
        points[count] = bound;
        count += 1;
        let mut root = None;
        for idx in 1..count {
            let (mut lo, mut hi) = (points[idx - 1], points[idx]);
            let (mut flo, mut fhi) = (poly(lo), poly(hi));
            if !flo.is_finite() || !fhi.is_finite() {
                return None;
            }
            // Repeated roots touch zero without a sign change. Check price space
            // to avoid a polynomial-scale-dependent zero tolerance.
            if lo > 0.0 && (self.price_ratio(lo)? - 1.0).abs() <= 1e-12 {
                flo = 0.0;
            }
            if (self.price_ratio(hi)? - 1.0).abs() <= 1e-12 {
                fhi = 0.0;
            }
            let candidate = if fhi == 0.0 {
                Some(hi)
            } else if flo != 0.0 && flo.signum() != fhi.signum() {
                for _ in 0..100 {
                    let mid = lo + (hi - lo) * 0.5;
                    if hi - lo <= 1e-13 * mid.abs().max(RATE_STEP) {
                        break;
                    }
                    let fm = poly(mid);
                    if !fm.is_finite() {
                        return None;
                    }
                    if fm == 0.0 {
                        lo = mid;
                        hi = mid;
                        break;
                    }
                    if fm.signum() == flo.signum() {
                        lo = mid;
                        flo = fm;
                    } else {
                        hi = mid;
                    }
                }
                Some(lo + (hi - lo) * 0.5)
            } else {
                None
            };
            if let Some(candidate) = candidate {
                if root.is_some() {
                    return None;
                }
                root = Some(candidate);
            }
        }
        let root = root?;
        (root > RATE_STEP && (self.price_ratio(root)? - 1.0).abs() <= 1e-8).then_some(root)
    }
}

fn duration_alpha(equity: f64, cap: f64, payout: f64, forecasts: [Option<f64>; 3]) -> Option<f64> {
    let model = Valuation::new(equity, cap, payout, forecasts)?;
    let rate = model.implied_rate()?;
    // Alpha=-Duration, so the central price derivative is NOT negated.
    let alpha = (model.price_ratio(rate + RATE_STEP)? - model.price_ratio(rate - RATE_STEP)?)
        / (2.0 * RATE_STEP);
    alpha.is_finite().then_some(alpha)
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

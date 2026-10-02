use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::stock_daily_ops::{is_bj_stock, neutralize_size_sector};
use crate::factor::common::{
    cached_financial_stock_snapshots_for_date, DailyPanel, DividendReader, FinancialEventMarker,
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
            description: "Negative residual-income equity duration using strictly three consensus periods FY0/FY1/FY2 with a PIT annual-disclosure anchor and May 1 fallback; all three must be finite, with no two-period fallback. All monetary totals in wan yuan. Payout=implemented LTM cash dividends / latest PIT parent profit TTM; positive current/recursive equity and TTM profit required. Terminal residual income stays constant. Require a unique positive implied discount rate above 0.0001; central price sensitivity at +/-0.0001. SW level-1 and Barra SIZE neutralization; no winsorization or zscore; excludes BJ.".into(),
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
        Ok(neutralize_size_sector(&raw, &panel, data)?.to_factor_series(self.spec()))
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
            snapshots = cached_financial_stock_snapshots_for_date(
                panel,
                date,
                &mut cache,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{ColumnData, Table};
    use crate::factor::common::{DividendIndex, FinancialPitIndex};
    use std::collections::BTreeMap;
    use std::sync::Arc;

    fn statement(
        rows: &[(&str, i32, i32, f64)],
        column: &str,
        report_type: i64,
    ) -> FinancialPitIndex {
        FinancialPitIndex::from_table(Arc::new(
            Table::new(BTreeMap::from([
                (
                    "ts_code".into(),
                    ColumnData::Utf8(rows.iter().map(|r| Some(r.0.into())).collect()),
                ),
                (
                    "end_date".into(),
                    ColumnData::I32(rows.iter().map(|r| Some(r.1)).collect()),
                ),
                (
                    "ann_date".into(),
                    ColumnData::I32(rows.iter().map(|r| Some(r.2)).collect()),
                ),
                (
                    "f_ann_date".into(),
                    ColumnData::I32(rows.iter().map(|r| Some(r.2)).collect()),
                ),
                (
                    "report_type".into(),
                    ColumnData::I64(vec![Some(report_type); rows.len()]),
                ),
                (
                    "update_flag".into(),
                    ColumnData::I64(vec![Some(0); rows.len()]),
                ),
                (
                    column.into(),
                    ColumnData::F64(rows.iter().map(|r| Some(r.3)).collect()),
                ),
            ]))
            .unwrap(),
        ))
        .unwrap()
    }

    #[test]
    fn equity_duration_pit_units_revisions_dividends_and_batch_alignment() {
        let codes = ["000001.SZ", "600000.SH", "430001.BJ"];
        let dates = [20250530, 20250602, 20250603, 20250604];
        let mut rows = Vec::new();
        for code in codes {
            for end in [20250331, 20241231, 20240930, 20240630] {
                rows.push((code, end, 20250501, 25_000.0));
            }
        }
        rows.push((codes[0], 20240630, 20250602, 50_000.0));
        let income_index = statement(&rows, PROFIT, 2);
        let income = income_index.reader(ReportTypePreference::income_single_quarter());
        let mut rows: Vec<_> = codes
            .iter()
            .map(|code| (*code, 20250331, 20250501, 1_000_000.0))
            .collect();
        rows.push((codes[1], 20250331, 20250603, -100.0));
        let balance_index = statement(&rows, EQUITY, 1);
        let balance = balance_index.reader(ReportTypePreference::balance_sheet_consolidated());
        let before = snapshot(&income, &balance, codes[0], 20250530).unwrap();
        assert_eq!((before.equity, before.profit_ttm), (100.0, 10.0));
        assert_eq!(
            snapshot(&income, &balance, codes[0], 20250602)
                .unwrap()
                .profit_ttm,
            12.5
        );
        assert!(snapshot(&income, &balance, codes[0], 20250430).is_none());
        assert!(snapshot(&income, &balance, codes[1], 20250603).is_none());
        assert_ne!(
            marker(&income, &balance, codes[0], 20250530),
            marker(&income, &balance, codes[0], 20250602)
        );
        let div_index = DividendIndex::from_table(Arc::new(
            Table::new(BTreeMap::from([
                (
                    "ts_code".into(),
                    ColumnData::Utf8(vec![Some(codes[0].into()); 2]),
                ),
                (
                    "ann_date".into(),
                    ColumnData::I32(vec![Some(20240601), Some(20250602)]),
                ),
                (
                    "ex_date".into(),
                    ColumnData::I32(vec![Some(20240603), Some(20250603)]),
                ),
                (
                    "div_proc".into(),
                    ColumnData::Utf8(vec![Some("\u{5b9e}\u{65bd}".into()); 2]),
                ),
                (
                    "cash_div_tax".into(),
                    ColumnData::F64(vec![Some(0.2), Some(0.4)]),
                ),
                ("base_share".into(), ColumnData::F64(vec![Some(10.0); 2])),
            ]))
            .unwrap(),
        ))
        .unwrap();
        let dividends = div_index.reader();
        assert_eq!(
            dividends.implemented_ltm_sum(codes[0], ltm_start(20250602), 20250602),
            2.0
        );
        assert_eq!(
            dividends.implemented_ltm_sum(codes[0], ltm_start(20250603), 20250603),
            6.0
        );
        assert_eq!(
            dividends.implemented_ltm_sum(codes[0], ltm_start(20250604), 20250604),
            4.0
        );

        let make_panel = |dates: &[i32], codes: &[&str]| {
            DailyPanel::from_index(
                dates.to_vec(),
                codes.iter().map(|c| c.to_string()).collect(),
                &[20250604],
                vec![true; dates.len() * codes.len()],
            )
            .unwrap()
        };
        let compute = |panel: &DailyPanel| {
            // Reversed source rows exercise date/code mapping for daily inputs.
            let keys: Vec<_> = panel
                .dates()
                .iter()
                .flat_map(|date| codes.iter().map(move |code| (*date, *code)))
                .rev()
                .collect();
            let source = Table::new(BTreeMap::from([
                (
                    "trade_date".into(),
                    ColumnData::I32(keys.iter().map(|r| Some(r.0)).collect()),
                ),
                (
                    "ts_code".into(),
                    ColumnData::Utf8(keys.iter().map(|r| Some(r.1.into())).collect()),
                ),
                (
                    "total_mv".into(),
                    ColumnData::F64(
                        keys.iter()
                            .map(|r| Some(if r.1 == codes[0] { 200.0 } else { 250.0 }))
                            .collect(),
                    ),
                ),
            ]))
            .unwrap();
            let cap = panel.column_from_table(&source, "total_mv").unwrap();
            let forecasts = [10.0, 12.0, 14.0].map(|v| {
                panel
                    .column_from_values(vec![Some(v); panel.shape_len()])
                    .unwrap()
            });
            raw_duration(panel, &income, &balance, &dividends, &cap, &forecasts).unwrap()
        };
        let full = compute(&make_panel(&dates, &codes));
        for (day, payout) in [0.2, 2.0 / 12.5, 6.0 / 12.5, 4.0 / 12.5]
            .into_iter()
            .enumerate()
        {
            assert_eq!(
                full.values()[day * 3],
                duration_alpha(100.0, 200.0, payout, [Some(10.0), Some(12.0), Some(14.0)])
            );
            assert!(full.values()[day * 3 + 2].is_none());
        }
        assert!(full.values()[1].is_some());
        assert!(full.values()[7].is_none());
        let reordered = compute(&make_panel(&dates[1..], &[codes[2], codes[0], codes[1]]));
        for day in 0..3 {
            for (i, original) in [2, 0, 1].into_iter().enumerate() {
                assert_eq!(
                    reordered.values()[day * 3 + i],
                    full.values()[(day + 1) * 3 + original]
                );
            }
        }
        let output = full.to_factor_series(StockDailyEquityDuration.spec());
        assert_eq!(output.values.len(), 3);
        assert!(output
            .values
            .iter()
            .all(|value| value.key.trade_date() == 20250604));
    }

    #[test]
    fn equity_duration_requires_all_three_forecasts_and_valid_inputs() {
        assert!(Valuation::new(100.0, 200.0, 0.4, [Some(10.0), Some(12.0), Some(14.0)]).is_some());
        for idx in 0..3 {
            for invalid in [None, Some(f64::NAN), Some(f64::INFINITY)] {
                let mut forecasts = [Some(10.0), Some(12.0), Some(14.0)];
                forecasts[idx] = invalid;
                assert!(duration_alpha(100.0, 200.0, 0.4, forecasts).is_none());
            }
        }
        assert!(duration_alpha(0.0, 200.0, 0.4, [Some(10.0); 3]).is_none());
        assert!(duration_alpha(100.0, 0.0, 0.4, [Some(10.0); 3]).is_none());
        assert!(duration_alpha(100.0, 200.0, -0.1, [Some(10.0); 3]).is_none());
        assert!(duration_alpha(100.0, 200.0, 20.0, [Some(10.0); 3]).is_none());
        assert!(duration_alpha(100.0, 200.0, 0.0, [Some(-10.0); 3]).is_none());
    }

    fn residual_price(book: f64, earnings: &[f64], payout: f64, rate: f64) -> f64 {
        let mut book = book;
        let mut value = book;
        for (idx, earning) in earnings.iter().enumerate() {
            let residual = earning - rate * book;
            value += if idx + 1 == earnings.len() {
                residual / (rate * (1.0 + rate).powi(idx as i32))
            } else {
                residual / (1.0 + rate).powi(idx as i32 + 1)
            };
            book += earning * (1.0 - payout);
        }
        value
    }

    #[test]
    fn equity_duration_matches_residual_income_and_recovers_rate() {
        for payout in [0.0, 0.4, 1.0, 1.2] {
            let earnings = [10.0, 12.0, 14.0];
            let cap = residual_price(100.0, &earnings, payout, 0.08);
            let forecasts = earnings.map(Some);
            let model = Valuation::new(100.0, cap, payout, forecasts).unwrap();
            assert!((model.price_ratio(0.08).unwrap() - 1.0).abs() < 1e-12);
            assert!((model.implied_rate().unwrap() - 0.08).abs() < 1e-12);
            let expected = (residual_price(100.0, &earnings, payout, 0.0801)
                - residual_price(100.0, &earnings, payout, 0.0799))
                / (cap * 0.0002);
            let actual = duration_alpha(100.0, cap, payout, forecasts).unwrap();
            assert!((actual - expected).abs() < 1e-8);
            assert!(actual < 0.0);
            let scaled = duration_alpha(
                1_000_000.0,
                cap * 10000.0,
                payout,
                forecasts.map(|v| v.map(|v| v * 10000.0)),
            )
            .unwrap();
            assert!((actual - scaled).abs() < 1e-8);
        }
    }

    #[test]
    fn equity_duration_rejects_multiple_roots_and_near_zero_rate() {
        // Three positive roots, and a repeated root plus another positive root.
        for (second, terminal) in [(-1.71, 0.006), (-1.69, 0.004)] {
            assert!(
                duration_alpha(100.0, 1.0, 1.0, [Some(2.6), Some(second), Some(terminal)])
                    .is_none()
            );
        }
        let cap = residual_price(100.0, &[10.0, 12.0, 14.0], 0.4, RATE_STEP / 2.0);
        assert!(duration_alpha(100.0, cap, 0.4, [Some(10.0), Some(12.0), Some(14.0)]).is_none());
    }

    #[test]
    fn equity_duration_metadata_and_calendar() {
        let spec = StockDailyEquityDuration.spec();
        assert_eq!(spec.id, "equity_duration");
        for tag in ["XYZQ", "fundamental", "analyst"] {
            assert!(spec.tags.contains(&tag.into()));
        }
        assert!(!spec.tags.contains(&"deprecated".into()));
        assert!(!spec
            .dependencies
            .iter()
            .any(|d| d.dataset == DatasetId::StockDailyPv));
        let columns = &spec
            .dependencies
            .iter()
            .find(|d| d.dataset == DatasetId::StockConsensus)
            .unwrap()
            .columns;
        assert_eq!(
            columns,
            &["con_np_fy0", "con_np_fy1", "con_np_fy2"]
                .map(str::to_string)
                .to_vec()
        );
        assert_eq!(ltm_start(20240229), 20230228);
    }
}

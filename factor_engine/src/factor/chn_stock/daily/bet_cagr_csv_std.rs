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
            description: "Daily BET uses g=min(consensus two-year growth / 100, PIT parent net-profit TTM YoY with absolute base). BET=ln1p(g*pe_ttm)/ln1p(g), with BET=pe_ttm at zero growth; positive PE and valid logarithms required. Apply 252-trading-day population z-score including today, min_periods=60, then SW level-1 and Barra SIZE neutralization. Lower is cheaper; no sign flip, fills or winsorization; excludes BJ.".into(),
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
        Ok(neutralize_size_sector(&standardized, &panel, data)?.to_factor_series(self.spec()))
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
            snapshots = cached_financial_stock_snapshots_for_date(
                panel,
                date,
                &mut cache,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{ColumnData, Table};
    use crate::factor::common::FinancialPitIndex;
    use std::collections::BTreeMap;
    use std::sync::Arc;

    fn financial_fixture() -> FinancialPitIndex {
        let mut rows = Vec::new();
        for (code, current) in [
            ("000001.SZ", 30.0),
            ("600000.SH", 35.0),
            ("430001.BJ", 40.0),
        ] {
            for (i, end) in quarter_ends(20241231).unwrap().into_iter().enumerate() {
                rows.push((code, end, 20250301, if i < 4 { current } else { 25.0 }));
            }
        }
        // A later disclosure revises the oldest quarter used by the YoY denominator.
        rows.push(("000001.SZ", 20230331, 20250601, 45.0));
        let table = Table::new(BTreeMap::from([
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
                ColumnData::I64(vec![Some(2); rows.len()]),
            ),
            (
                "update_flag".into(),
                ColumnData::I64(vec![Some(0); rows.len()]),
            ),
            (
                PROFIT.into(),
                ColumnData::F64(rows.iter().map(|r| Some(r.3)).collect()),
            ),
        ]))
        .unwrap();
        FinancialPitIndex::from_table(Arc::new(table)).unwrap()
    }

    fn panel(dates: &[i32], codes: &[&str], targets: &[i32]) -> DailyPanel {
        DailyPanel::from_index(
            dates.to_vec(),
            codes.iter().map(|v| v.to_string()).collect(),
            targets,
            vec![true; dates.len() * codes.len()],
        )
        .unwrap()
    }

    #[test]
    fn bet_cagr_csv_std_pit_revisions_order_and_warmup_output() {
        let index = financial_fixture();
        let income = index.reader(ReportTypePreference::income_single_quarter());
        assert_eq!(snapshot(&income, "000001.SZ", 20250228), None);
        assert_eq!(snapshot(&income, "000001.SZ", 20250530), Some(0.2));
        assert_eq!(snapshot(&income, "000001.SZ", 20250602), Some(0.0));
        assert_ne!(
            marker(&income, "000001.SZ", 20250530),
            marker(&income, "000001.SZ", 20250602)
        );
        let dates = [20250529, 20250530, 20250602, 20250603];
        let codes = ["000001.SZ", "600000.SH", "430001.BJ"];
        let full_panel = panel(&dates, &codes, &[20250603]);
        let compute = |panel: &DailyPanel| {
            raw_bet(
                panel,
                &income,
                &panel
                    .column_from_values(vec![Some(20.0); panel.shape_len()])
                    .unwrap(),
                &panel
                    .column_from_values(vec![Some(50.0); panel.shape_len()])
                    .unwrap(),
            )
            .unwrap()
        };
        let full = compute(&full_panel);
        assert_eq!(
            full.values()[0],
            bet_value(Some(20.0), Some(50.0), Some(0.2))
        );
        assert_eq!(full.values()[6], Some(20.0));
        assert!((0..dates.len()).all(|i| full.values()[i * 3 + 2].is_none()));
        let reversed = panel(&dates[1..], &[codes[2], codes[1], codes[0]], &[20250603]);
        let batch = compute(&reversed);
        for day in 0..3 {
            for stock in 0..3 {
                assert_eq!(
                    batch.values()[day * 3 + stock],
                    full.values()[(day + 1) * 3 + 2 - stock]
                );
            }
        }
        // Computing the earlier batch after the later one must still use its own PIT state.
        let earlier = compute(&panel(&dates[..2], &codes, &[20250530]));
        assert_eq!(earlier.values(), &full.values()[..6]);
        let output = full.to_factor_series(StockDailyBetCagrCsvStd.spec());
        assert_eq!(output.values.len(), 3);
        assert!(output.values.iter().all(|v| matches!(
            v.key,
            crate::core::FactorRowKey::Daily {
                trade_date: 20250603,
                ..
            }
        )));
    }

    #[test]
    fn bet_cagr_csv_std_presence_changes_refresh_slow_snapshot() {
        let index = financial_fixture();
        let income = index.reader(ReportTypePreference::income_single_quarter());
        let panel = DailyPanel::from_index(
            vec![20250528, 20250529],
            vec!["000001.SZ".into()],
            &[20250528, 20250529],
            vec![false, true],
        )
        .unwrap();
        let pe = panel.column_from_values(vec![Some(20.0); 2]).unwrap();
        let cagr = panel.column_from_values(vec![Some(50.0); 2]).unwrap();
        let raw = raw_bet(&panel, &income, &pe, &cagr).unwrap();
        assert_eq!(raw.values()[0], None);
        assert!(raw.values()[1].is_some());
    }

    #[test]
    fn bet_cagr_csv_std_formula_units_and_conservative_growth() {
        let expected = (1.0 + 0.1 * 20.0_f64).ln() / 1.1_f64.ln();
        assert!((bet_value(Some(20.0), Some(20.0), Some(0.1)).unwrap() - expected).abs() < 1e-12);
        assert!((bet_value(Some(20.0), Some(10.0), Some(0.2)).unwrap() - expected).abs() < 1e-12);
        assert_eq!(bet_value(Some(20.0), Some(0.0), Some(0.2)), Some(20.0));
        assert!((bet_value(Some(20.0), Some(1e-12), Some(0.2)).unwrap() - 20.0).abs() < 1e-10);
        assert!(bet_value(Some(20.0), Some(-1.0), Some(0.2)).unwrap() > 20.0);
    }

    #[test]
    fn bet_cagr_csv_std_invalid_inputs_and_logarithm_boundaries() {
        for pe in [
            None,
            Some(0.0),
            Some(-1.0),
            Some(f64::NAN),
            Some(f64::INFINITY),
        ] {
            assert_eq!(bet_value(pe, Some(20.0), Some(0.1)), None);
        }
        for growth in [None, Some(f64::NAN), Some(f64::INFINITY)] {
            assert_eq!(bet_value(Some(20.0), growth, Some(0.1)), None);
            assert_eq!(bet_value(Some(20.0), Some(20.0), growth), None);
        }
        assert_eq!(bet_value(Some(20.0), Some(-5.0), Some(0.1)), None);
        assert_eq!(bet_value(Some(0.5), Some(-100.0), Some(0.1)), None);
        assert_eq!(bet_value(Some(20.0), Some(20.0), Some(-0.1)), None);
    }

    #[test]
    fn bet_cagr_csv_std_ttm_uses_eight_quarters_and_absolute_base() {
        let mut profits = [Some(10.0); 8];
        profits[..4].fill(Some(15.0));
        assert_eq!(ttm_yoy(&profits), Some(0.5));
        profits[4..].fill(Some(-10.0));
        assert_eq!(ttm_yoy(&profits), Some(2.5));
        for i in 0..8 {
            let mut missing = profits;
            missing[i] = None;
            assert_eq!(ttm_yoy(&missing), None);
        }
        profits[4..].fill(Some(0.0));
        assert_eq!(ttm_yoy(&profits), None);
    }

    #[test]
    fn bet_cagr_csv_std_population_window_minimum_and_batch_equivalence() {
        let mut values = (0..600)
            .map(|i| Some(10.0 + (i % 73) as f64))
            .collect::<Vec<_>>();
        values[10] = None;
        values[300] = None;
        let full = ts_zscore(&values, WINDOW, MIN_PERIODS);
        assert!(full[..60].iter().all(Option::is_none));
        assert!(full[60].is_some());
        assert_eq!(full[300], None);
        for start in [100usize, 320, 500] {
            let load_start = start.saturating_sub(WINDOW - 1);
            let batch = ts_zscore(&values[load_start..], WINDOW, MIN_PERIODS);
            for i in start..600 {
                assert_eq!(full[i], batch[i - load_start]);
            }
        }
        let sample = (1..=60).map(|v| Some(v as f64)).collect::<Vec<_>>();
        let expected = (60.0 - 30.5) / ((60.0_f64.powi(2) - 1.0) / 12.0).sqrt();
        assert!((ts_zscore(&sample, WINDOW, MIN_PERIODS)[59].unwrap() - expected).abs() < 1e-12);
        assert!(ts_zscore(&[Some(10.0); 300], WINDOW, MIN_PERIODS)
            .iter()
            .all(Option::is_none));
    }

    #[test]
    fn bet_cagr_csv_std_spec_has_no_pv_anchor() {
        let spec = StockDailyBetCagrCsvStd.spec();
        assert_eq!(spec.id, "bet_cagr_csv_std");
        assert!(!spec.tags.iter().any(|tag| tag == "deprecated"));
        assert!(spec.aliases.contains(&"BET_CAGR_CSV_STD".into()));
        assert_eq!(spec.lookback.trading_days, 251);
        for tag in ["HAZQ", "fundamental"] {
            assert!(spec.tags.contains(&tag.into()));
        }
        assert!(!spec
            .dependencies
            .iter()
            .any(|r| r.dataset == DatasetId::StockDailyPv));
        assert_eq!(
            StockDailyBetCagrCsvStd.update_policy(),
            FactorUpdatePolicy::FinancialEventStateDailyFast
        );
    }
}

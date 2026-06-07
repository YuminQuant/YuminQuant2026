use std::any::Any;
use std::collections::{BTreeSet, HashMap};

use crate::barra::common::{
    sqrt_circ_mv_weights, standardize_panel_industry_filled_weighted,
    zscore_panel_weighted_filled_zero,
};
use crate::barra::{BarraExposure, BarraSharedCache};
use crate::core::{
    AssetClass, BarraSeries, BarraSpec, DataRequest, DatasetId, FactorContext, Frequency, Lookback,
};
use crate::data::{DataPool, Table};
use crate::error::Result;
use crate::factor::common::{
    cached_financial_stock_snapshots_for_date, DailyPanel, DividendReader,
    FinancialEventMarkerBuilder, InstrumentAlignedSnapshotCache, PanelColumn,
};

pub struct StockDailyBarraCne6DividendYield;

const MODEL: &str = "CNE6";
const VERSION: &str = "0.4.0";
const LOOKBACK: usize = 252;

pub fn create() -> Box<dyn BarraExposure> {
    Box::new(StockDailyBarraCne6DividendYield)
}

#[derive(Default)]
struct DividendYieldComputeState {
    dtop_cache: InstrumentAlignedSnapshotCache<DtopSlowSnapshot>,
}

impl BarraExposure for StockDailyBarraCne6DividendYield {
    fn family_id(&self) -> &'static str {
        "DIVIDEND_YIELD"
    }

    fn specs(&self) -> Vec<BarraSpec> {
        vec![
            dividend_yield_spec(
                "DTOP",
                &[],
                "CNE6 dividend-to-price",
                "Past 12 natural months implemented cash dividend amount divided by event-snapshot total market value.",
            ),
            dividend_yield_spec(
                "DTOPF",
                &[],
                "CNE6 forecast dividend-to-price",
                "Mean FY1 analyst forecast dividend yield from reports in the past three natural months.",
            ),
            dividend_yield_spec(
                "DIVIDEND_YIELD",
                &[],
                "CNE6 DIVIDEND_YIELD style exposure",
                "Composite of DTOP and DTOPF, using the available side when the other is missing.",
            ),
        ]
    }

    fn initial_compute_state(&self, _selected_ids: &BTreeSet<String>) -> Box<dyn Any + Send> {
        Box::new(DividendYieldComputeState::default())
    }

    fn compute_stateful(
        &self,
        context: &FactorContext,
        data: &DataPool,
        state: &mut (dyn Any + Send),
        _shared_cache: &BarraSharedCache,
    ) -> Result<Vec<BarraSeries>> {
        let state = state
            .downcast_mut::<DividendYieldComputeState>()
            .expect("DIVIDEND_YIELD compute state type");
        self.compute_with_cache(context, data, &mut state.dtop_cache)
    }

    fn compute(&self, context: &FactorContext, data: &DataPool) -> Result<Vec<BarraSeries>> {
        let mut cache = InstrumentAlignedSnapshotCache::default();
        self.compute_with_cache(context, data, &mut cache)
    }
}

impl StockDailyBarraCne6DividendYield {
    fn compute_with_cache(
        &self,
        _context: &FactorContext,
        data: &DataPool,
        dtop_cache: &mut InstrumentAlignedSnapshotCache<DtopSlowSnapshot>,
    ) -> Result<Vec<BarraSeries>> {
        let panel = data.daily_panel(DatasetId::StockDailyPv)?;
        let total_mv =
            panel.column_from_table(data.daily(DatasetId::StockDailyBasic)?, "total_mv")?;
        let dividends = data.dividend_reader()?;
        let analyst_reports = parse_analyst_records(data.daily(DatasetId::StockAnalystReport)?)?;
        let weights = sqrt_circ_mv_weights(panel, data)?;

        let dtop_raw = dtop_column(panel, &total_mv, &dividends, dtop_cache)?;
        let dtop = standardize_panel_industry_filled_weighted(&dtop_raw, &weights, data)?;
        let dtopf_raw = dtopf_column(panel, &analyst_reports)?;
        let dtopf = standardize_panel_industry_filled_weighted(&dtopf_raw, &weights, data)?;
        let composite_raw = dtop.zip_binary(&dtopf, composite_available)?;
        let dividend_yield = zscore_panel_weighted_filled_zero(&composite_raw, &weights)?;

        let specs = self.specs();
        Ok(vec![
            dtop.to_barra_series(specs[0].clone()),
            dtopf.to_barra_series(specs[1].clone()),
            dividend_yield.to_barra_series(specs[2].clone()),
        ])
    }
}

fn dividend_yield_spec(id: &str, aliases: &[&str], name: &str, description: &str) -> BarraSpec {
    BarraSpec {
        id: id.to_string(),
        aliases: aliases.iter().map(|value| value.to_string()).collect(),
        name: name.to_string(),
        model: MODEL.to_string(),
        asset_class: AssetClass::Stock,
        frequency: Frequency::Daily,
        version: VERSION.to_string(),
        tags: ["barra", "cne6", "style", "dividend_yield", "daily", "stock"]
            .iter()
            .map(|value| value.to_string())
            .collect(),
        description: description.to_string(),
        dependencies: vec![
            DataRequest::new(DatasetId::StockDailyPv, &[]),
            DataRequest::new(DatasetId::StockDailyBasic, &["total_mv", "circ_mv"]),
            DataRequest::new(DatasetId::StockSwClassification, &["l1_code"]),
            DataRequest::new(
                DatasetId::StockDividend,
                &[
                    "ts_code",
                    "end_date",
                    "ann_date",
                    "div_proc",
                    "cash_div_tax",
                    "ex_date",
                    "base_date",
                    "base_share",
                ],
            ),
            DataRequest::new(
                DatasetId::StockAnalystReport,
                &["ts_code", "report_date", "quarter", "rd"],
            ),
        ],
        lookback: Lookback {
            trading_days: LOOKBACK,
        },
    }
}

#[derive(Clone, Debug)]
struct AnalystRecord {
    ts_code: String,
    report_date: i32,
    quarter: String,
    rd: f64,
}

#[derive(Clone, Copy, Debug)]
struct DtopSlowSnapshot {
    cash: f64,
    total_mv: f64,
}

fn parse_analyst_records(table: &Table) -> Result<Vec<AnalystRecord>> {
    let ts_codes = table.required_utf8("ts_code")?;
    let report_dates = table.required_i32_date_cast("report_date")?;
    let quarters = table.required_utf8("quarter")?;
    let rd_values = table.required_f64_cast("rd")?;

    let mut records = Vec::new();
    for idx in 0..table.len {
        let (Some(ts_code), Some(report_date), Some(quarter), Some(rd)) = (
            ts_codes[idx].clone(),
            report_dates[idx],
            quarters[idx].clone(),
            clean(rd_values[idx]),
        ) else {
            continue;
        };
        records.push(AnalystRecord {
            ts_code,
            report_date,
            quarter,
            rd,
        });
    }
    Ok(records)
}

fn dtop_column(
    panel: &DailyPanel,
    total_mv: &PanelColumn,
    dividends: &DividendReader<'_>,
    cache: &mut InstrumentAlignedSnapshotCache<DtopSlowSnapshot>,
) -> Result<PanelColumn> {
    let mut values = vec![None; panel.shape_len()];
    let instrument_count = panel.instruments().len();

    for (date_idx, trade_date) in panel.dates().iter().copied().enumerate() {
        if !panel.is_target_date(trade_date) {
            continue;
        }
        let dividend_sums =
            dividends.implemented_ltm_sum_by_stock(add_months(trade_date, -12), trade_date);
        let snapshots = cached_financial_stock_snapshots_for_date(
            panel,
            trade_date,
            cache,
            |_, _, offset| !panel.is_present_offset(offset),
            |_, ts_code, _| {
                let cash = dividend_sums.get(ts_code).copied().unwrap_or(0.0);
                let mut builder = FinancialEventMarkerBuilder::new();
                builder.include_synthetic("dtop_cash_ltm", f64_marker_value(cash));
                builder.build()
            },
            |_, ts_code, offset| {
                let cash = dividend_sums.get(ts_code).copied().unwrap_or(0.0);
                let total_mv = clean(total_mv.values()[offset]).filter(|value| *value > 0.0)?;
                Some(DtopSlowSnapshot { cash, total_mv })
            },
        );
        for (instrument_idx, snapshot) in snapshots.into_iter().enumerate() {
            let offset = date_idx * instrument_count + instrument_idx;
            values[offset] = snapshot.map(|snapshot| snapshot.cash / snapshot.total_mv);
        }
    }

    panel.column_from_values(values)
}

fn f64_marker_value(value: f64) -> i64 {
    i64::from_ne_bytes(value.to_bits().to_ne_bytes())
}

fn dtopf_column(panel: &DailyPanel, records: &[AnalystRecord]) -> Result<PanelColumn> {
    let instrument_count = panel.instruments().len();
    let mut values = vec![None; panel.shape_len()];

    for (date_idx, trade_date) in panel.dates().iter().copied().enumerate() {
        if !panel.is_target_date(trade_date) {
            continue;
        }
        let start_date = add_months(trade_date, -3);
        let fy1 = fy1_quarter(trade_date);
        let forecast = forecast_mean_by_stock(records, start_date, trade_date, &fy1);

        for (instrument_idx, ts_code) in panel.instruments().iter().enumerate() {
            values[date_idx * instrument_count + instrument_idx] =
                forecast.get(ts_code.as_str()).copied().unwrap_or(None);
        }
    }

    panel.column_from_values(values)
}

fn forecast_mean_by_stock<'a>(
    records: &'a [AnalystRecord],
    start_date: i32,
    trade_date: i32,
    fy1: &str,
) -> HashMap<&'a str, Option<f64>> {
    let mut sums = HashMap::<&str, (f64, usize)>::new();
    for record in records {
        if record.report_date < start_date
            || record.report_date > trade_date
            || record.quarter.trim() != fy1
        {
            continue;
        }
        let entry = sums.entry(record.ts_code.as_str()).or_default();
        entry.0 += record.rd;
        entry.1 += 1;
    }
    sums.into_iter()
        .map(|(ts_code, (sum, count))| (ts_code, (count > 0).then_some(sum / count as f64)))
        .collect()
}

fn fy1_quarter(trade_date: i32) -> String {
    let (year, _month, _day) = ymd(trade_date);
    let fy1_year = if trade_date <= year * 10_000 + 430 {
        year
    } else {
        year + 1
    };
    format!("{fy1_year}Q4")
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

fn composite_available(dtop: Option<f64>, dtopf: Option<f64>) -> Option<f64> {
    match (clean(dtop), clean(dtopf)) {
        (Some(dtop), Some(dtopf)) => Some((dtop + dtopf) / 2.0),
        (Some(dtop), None) => Some(dtop),
        (None, Some(dtopf)) => Some(dtopf),
        _ => None,
    }
}

fn clean(value: Option<f64>) -> Option<f64> {
    value.filter(|value| !value.is_nan())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::Arc;

    use crate::data::{ColumnData, Table};
    use crate::factor::common::{DailyPanel, DividendIndex, InstrumentAlignedSnapshotCache};

    use super::{
        add_months, composite_available, dtop_column, forecast_mean_by_stock, fy1_quarter,
        AnalystRecord,
    };

    fn assert_close(actual: f64, expected: f64) {
        assert!((actual - expected).abs() < 1e-10, "{actual} != {expected}");
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

    #[test]
    fn fy1_rule_rolls_after_april_thirtieth() {
        assert_eq!(fy1_quarter(20260430), "2026Q4");
        assert_eq!(fy1_quarter(20260501), "2027Q4");
    }

    #[test]
    fn month_arithmetic_clamps_to_valid_month_days() {
        assert_eq!(add_months(20260331, -1), 20260228);
        assert_eq!(add_months(20240331, -1), 20240229);
        assert_eq!(add_months(20260115, -12), 20250115);
    }

    #[test]
    fn dtop_uses_only_implemented_announced_ex_date_records_in_past_year() {
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
                20260101,
                "\u{5b9e}\u{65bd}",
                0.4,
                20270301,
                100.0,
            ),
            (
                "000002.SZ",
                20260101,
                "\u{5b9e}\u{65bd}",
                0.5,
                20260301,
                200.0,
            ),
        ])))
        .unwrap();
        let reader = index.reader();
        let sums = reader.implemented_ltm_sum_by_stock(20250424, 20260424);

        assert_close(*sums.get("000001.SZ").unwrap(), 20.0);
        assert_close(*sums.get("000002.SZ").unwrap(), 100.0);
    }

    #[test]
    fn dtop_uses_slow_total_market_value_until_dividend_marker_changes() {
        let panel = DailyPanel::from_index(
            vec![20260424, 20260425],
            vec!["000001.SZ".to_string(), "000002.SZ".to_string()],
            &[20260424, 20260425],
            vec![true, true, true, true],
        )
        .unwrap();
        let total_mv = panel
            .column_from_values(vec![Some(1000.0), Some(0.0), Some(2000.0), Some(2000.0)])
            .unwrap();
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
                "000002.SZ",
                20260101,
                "\u{5b9e}\u{65bd}",
                0.5,
                20260301,
                200.0,
            ),
        ])))
        .unwrap();
        let reader = index.reader();

        let mut cache = InstrumentAlignedSnapshotCache::default();
        let dtop = dtop_column(&panel, &total_mv, &reader, &mut cache).unwrap();

        assert_close(dtop.values()[0].unwrap(), 0.02);
        assert_eq!(dtop.values()[1], None);
        assert_close(dtop.values()[2].unwrap(), 0.02);
        assert_eq!(dtop.values()[3], None);
    }

    #[test]
    fn dtopf_uses_three_month_window_fy1_and_mean_rd() {
        let records = vec![
            AnalystRecord {
                ts_code: "000001.SZ".to_string(),
                report_date: 20260201,
                quarter: "2026Q4".to_string(),
                rd: 0.02,
            },
            AnalystRecord {
                ts_code: "000001.SZ".to_string(),
                report_date: 20260401,
                quarter: "2026Q4".to_string(),
                rd: 0.04,
            },
            AnalystRecord {
                ts_code: "000001.SZ".to_string(),
                report_date: 20260401,
                quarter: "2027Q4".to_string(),
                rd: 0.10,
            },
        ];
        let means = forecast_mean_by_stock(&records, 20260130, 20260430, "2026Q4");

        assert_close(means.get("000001.SZ").unwrap().unwrap(), 0.03);
    }

    #[test]
    fn dividend_yield_composite_uses_available_side() {
        assert_close(composite_available(Some(1.0), Some(3.0)).unwrap(), 2.0);
        assert_eq!(composite_available(Some(1.0), None), Some(1.0));
        assert_eq!(composite_available(None, Some(3.0)), Some(3.0));
        assert_eq!(composite_available(None, None), None);
    }
}

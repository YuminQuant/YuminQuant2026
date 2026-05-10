use std::collections::HashMap;

use crate::barra::common::{
    sqrt_circ_mv_weights, standardize_panel_industry_filled_weighted,
    zscore_panel_weighted_filled_zero,
};
use crate::barra::BarraExposure;
use crate::core::{
    AssetClass, BarraSeries, BarraSpec, DataRequest, DatasetId, FactorContext, Frequency, Lookback,
};
use crate::data::{DataPool, Table};
use crate::error::Result;
use crate::factor::common::{DailyPanel, PanelColumn};

pub struct StockDailyBarraCne6DividendYield;

const MODEL: &str = "CNE6";
const VERSION: &str = "0.4.0";
const LOOKBACK: usize = 252;
const IMPLEMENTED_DIV_PROC: &str = "\u{5b9e}\u{65bd}";

pub fn create() -> Box<dyn BarraExposure> {
    Box::new(StockDailyBarraCne6DividendYield)
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
                "Past 12 natural months implemented cash dividend amount divided by target-date total market value.",
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

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<Vec<BarraSeries>> {
        let panel = data.daily_panel(DatasetId::StockDailyPv)?;
        let total_mv =
            panel.column_from_table(data.daily(DatasetId::StockDailyBasic)?, "total_mv")?;
        let dividends = parse_dividend_records(data.daily(DatasetId::StockDividend)?)?;
        let analyst_reports = parse_analyst_records(data.daily(DatasetId::StockAnalystReport)?)?;
        let weights = sqrt_circ_mv_weights(panel, data)?;

        let dtop_raw = dtop_column(panel, &total_mv, &dividends)?;
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
struct DividendRecord {
    ts_code: String,
    ann_date: i32,
    ex_date: i32,
    cash_div_tax: f64,
    base_share: f64,
    implemented: bool,
}

#[derive(Clone, Debug)]
struct AnalystRecord {
    ts_code: String,
    report_date: i32,
    quarter: String,
    rd: f64,
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
    records: &[DividendRecord],
) -> Result<PanelColumn> {
    let instrument_count = panel.instruments().len();
    let mut values = vec![None; panel.shape_len()];

    for (date_idx, trade_date) in panel.dates().iter().copied().enumerate() {
        if !panel.is_target_date(trade_date) {
            continue;
        }
        let start_date = add_months(trade_date, -12);
        let dividend_sum = dividend_sum_by_stock(records, start_date, trade_date);

        for (instrument_idx, ts_code) in panel.instruments().iter().enumerate() {
            let offset = date_idx * instrument_count + instrument_idx;
            let Some(market_value) = clean(total_mv.values()[offset]).filter(|value| *value > 0.0)
            else {
                continue;
            };
            let cash = dividend_sum.get(ts_code.as_str()).copied().unwrap_or(0.0);
            values[offset] = Some(cash / market_value);
        }
    }

    panel.column_from_values(values)
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
    use crate::factor::common::DailyPanel;

    use super::{
        add_months, composite_available, dividend_sum_by_stock, dtop_column,
        forecast_mean_by_stock, fy1_quarter, AnalystRecord, DividendRecord, IMPLEMENTED_DIV_PROC,
    };

    fn assert_close(actual: f64, expected: f64) {
        assert!((actual - expected).abs() < 1e-10, "{actual} != {expected}");
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
                ann_date: 20260101,
                ex_date: 20270301,
                cash_div_tax: 0.4,
                base_share: 100.0,
                implemented: true,
            },
            DividendRecord {
                ts_code: "000002.SZ".to_string(),
                ann_date: 20260101,
                ex_date: 20260301,
                cash_div_tax: 0.5,
                base_share: 200.0,
                implemented: IMPLEMENTED_DIV_PROC == "\u{5b9e}\u{65bd}",
            },
        ];
        let sums = dividend_sum_by_stock(&records, 20250424, 20260424);

        assert_close(*sums.get("000001.SZ").unwrap(), 20.0);
        assert_close(*sums.get("000002.SZ").unwrap(), 100.0);
    }

    #[test]
    fn dtop_uses_target_date_total_market_value() {
        let panel = DailyPanel::from_index(
            vec![20260424],
            vec!["000001.SZ".to_string(), "000002.SZ".to_string()],
            &[20260424],
            vec![true, true],
        )
        .unwrap();
        let total_mv = panel
            .column_from_values(vec![Some(1000.0), Some(0.0)])
            .unwrap();
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
                ts_code: "000002.SZ".to_string(),
                ann_date: 20260101,
                ex_date: 20260301,
                cash_div_tax: 0.5,
                base_share: 200.0,
                implemented: true,
            },
        ];

        let dtop = dtop_column(&panel, &total_mv, &records).unwrap();

        assert_close(dtop.values()[0].unwrap(), 0.02);
        assert_eq!(dtop.values()[1], None);
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

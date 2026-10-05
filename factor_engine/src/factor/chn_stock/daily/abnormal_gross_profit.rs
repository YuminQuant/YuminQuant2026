use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::{err, Result};
use crate::factor::common::stock_daily_ops::{is_bj_stock, neutralize_size_sector_with_inputs};
use crate::factor::common::{
    cached_financial_stock_snapshots_for_date, ClassificationLevel, ClassificationMap,
    FinancialEventMarker, FinancialEventMarkerBuilder, FinancialPitReader,
    FinancialStatementDataset, InstrumentAlignedSnapshotCache, ReportTypePreference,
};
use crate::factor::{Factor, FactorUpdatePolicy};
use std::any::Any;

pub struct StockDailyAbnormalGrossProfit;
pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyAbnormalGrossProfit)
}

impl Factor for StockDailyAbnormalGrossProfit {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "abnormal_gross_profit".into(), aliases: vec!["ABNORMAL_GROSS_PROFIT".into(), "AbnormalGP".into()],
            name: "Abnormal Gross Profit".into(), asset_class: AssetClass::Stock, frequency: Frequency::Daily,
            version: "0.1.0".into(),
            tags: ["SWHYZQ", "fundamental", "financial", "pit", "profitability", "size_neutralize", "sector_neutralize", "daily"].into_iter().map(str::to_string).collect(),
            description: "(GP_q-GP_q-4*CashSales_q/CashSales_q-4)/total_assets_q; GP=revenue-oper_cost, CashSales=c_fr_sale_sg. Single-quarter flows. Raw updates May 1/Sep 1/Nov 1 using Q1/Q2/Q3 and only records visible by that calendar cutoff; frozen until next cutoff, no Q4 or fallback quarter. Partial missing additive operands use zero, all missing remains null; invalid cash growth or nonpositive assets stays null. Positive direction, excludes BJ and SW financials; daily SW L1 and Barra SIZE neutralization.".into(),
            dependencies: vec![
                DataRequest::financial_quarters(DatasetId::StockIncome, &["revenue", "oper_cost"], 8),
                DataRequest::financial_quarters(DatasetId::StockCashFlow, &["c_fr_sale_sg"], 8),
                DataRequest::financial_quarters(DatasetId::StockBalanceSheet, &["total_assets"], 8),
                DataRequest::new(DatasetId::StockBarraDaily, &["SIZE"]),
                DataRequest::new(DatasetId::StockSwClassification, &["l1_code"]),
            ], intraday_raw_dependencies: vec![], lookback: Lookback { trading_days: 0 },
        }
    }
    fn update_policy(&self) -> FactorUpdatePolicy {
        FactorUpdatePolicy::FinancialEventStateDailyFast
    }
    fn initial_compute_state(&self, _: &[String]) -> Box<dyn Any + Send> {
        Box::new(InstrumentAlignedSnapshotCache::<f64>::default())
    }
    fn compute(&self, _: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        compute(data, &mut InstrumentAlignedSnapshotCache::default())
    }
    fn compute_many_stateful(
        &self,
        ids: &[String],
        _: &FactorContext,
        data: &DataPool,
        state: &mut (dyn Any + Send),
    ) -> Result<Vec<FactorSeries>> {
        if !ids.iter().any(|id| id == "abnormal_gross_profit") {
            return Ok(vec![]);
        }
        Ok(vec![compute(
            data,
            state
                .downcast_mut::<InstrumentAlignedSnapshotCache<f64>>()
                .ok_or_else(|| err("abnormal gross profit state mismatch"))?,
        )?])
    }
}

// Calendar cutoffs, not first trading days: later weekend disclosures cannot leak in.
fn reporting_period(date: i32) -> (i32, i32) {
    let year = date / 10000;
    match date % 10000 {
        1101.. => (year * 10000 + 1101, year * 10000 + 930),
        901.. => (year * 10000 + 901, year * 10000 + 630),
        501.. => (year * 10000 + 501, year * 10000 + 331),
        _ => ((year - 1) * 10000 + 1101, (year - 1) * 10000 + 930),
    }
}

fn is_financial_sector(code: &str) -> bool {
    matches!(
        code.split('.').next().unwrap_or(code),
        "801780" | "801790" | "801190"
    )
}

struct Readers<'a> {
    income: FinancialPitReader<'a>,
    cash: FinancialPitReader<'a>,
    balance: FinancialPitReader<'a>,
}
impl Readers<'_> {
    fn marker(&self, code: &str, cutoff: i32, end: i32) -> Option<FinancialEventMarker> {
        let mut marker = FinancialEventMarkerBuilder::new();
        for end in [end, end - 10000] {
            marker.include_reader_record_for_end_date(
                FinancialStatementDataset::Income,
                &self.income,
                code,
                cutoff,
                end,
            );
            marker.include_reader_record_for_end_date(
                FinancialStatementDataset::CashFlow,
                &self.cash,
                code,
                cutoff,
                end,
            );
        }
        marker.include_reader_record_for_end_date(
            FinancialStatementDataset::BalanceSheet,
            &self.balance,
            code,
            cutoff,
            end,
        );
        marker.build()
    }
    fn snapshot(&self, code: &str, cutoff: i32, end: i32) -> Option<f64> {
        let gp = |end| {
            let record = self.income.record_for_end_date(code, cutoff, end)?;
            difference(record.column("revenue"), record.column("oper_cost"))
        };
        let cash = |end| {
            self.cash
                .record_for_end_date(code, cutoff, end)
                .and_then(|r| r.column("c_fr_sale_sg"))
        };
        abnormal(
            gp(end),
            gp(end - 10000),
            cash(end),
            cash(end - 10000),
            self.balance
                .record_for_end_date(code, cutoff, end)
                .and_then(|r| r.column("total_assets")),
        )
    }
}

fn compute(
    data: &DataPool,
    cache: &mut InstrumentAlignedSnapshotCache<f64>,
) -> Result<FactorSeries> {
    let panel = data.stock_universe_panel()?;
    let readers = Readers {
        income: data.financial_reader(
            DatasetId::StockIncome,
            ReportTypePreference::income_single_quarter(),
        )?,
        cash: data.financial_reader(
            DatasetId::StockCashFlow,
            ReportTypePreference::income_single_quarter(),
        )?,
        balance: data.financial_reader(
            DatasetId::StockBalanceSheet,
            ReportTypePreference::balance_sheet_consolidated(),
        )?,
    };
    let sector = ClassificationMap::from_table(
        data.daily(DatasetId::StockSwClassification)?,
        ClassificationLevel::Sector,
    )?;
    let n = panel.instruments().len();
    let mut values = vec![None; panel.shape_len()];
    let mut snapshots = vec![None; n];
    let mut previous: Option<(usize, (i32, i32))> = None;
    for (day, date) in panel.dates().iter().copied().enumerate() {
        if !panel.is_target_date(date) {
            continue;
        }
        let period = reporting_period(date);
        let refresh = previous.is_none_or(|(prev, old_period)| {
            period != old_period
                || (0..n).any(|i| {
                    panel.is_present_offset(day * n + i) != panel.is_present_offset(prev * n + i)
                })
        });
        if refresh {
            snapshots = cached_financial_stock_snapshots_for_date(
                &panel,
                date,
                cache,
                |_, code, offset| is_bj_stock(code) || !panel.is_present_offset(offset),
                |_, code, _| readers.marker(code, period.0, period.1),
                |_, code, _| readers.snapshot(code, period.0, period.1),
            );
        }
        for (i, code) in panel.instruments().iter().enumerate() {
            if panel.is_present_offset(day * n + i)
                && sector
                    .group_for(date, code)
                    .is_some_and(|s| !is_financial_sector(s))
            {
                values[day * n + i] = snapshots[i];
            }
        }
        previous = Some((day, period));
    }
    let raw = panel.column_from_values(values)?;
    let size = panel.column_from_table(data.daily(DatasetId::StockBarraDaily)?, "SIZE")?;
    Ok(
        neutralize_size_sector_with_inputs(&raw, &panel, &size, &sector)?
            .to_factor_series(StockDailyAbnormalGrossProfit.spec()),
    )
}

fn clean(v: Option<f64>) -> Option<f64> {
    v.filter(|v| v.is_finite())
}
fn difference(a: Option<f64>, b: Option<f64>) -> Option<f64> {
    let (a, b) = (clean(a), clean(b));
    if a.is_none() && b.is_none() {
        None
    } else {
        clean(Some(a.unwrap_or(0.0) - b.unwrap_or(0.0)))
    }
}
fn abnormal(
    gp: Option<f64>,
    prior_gp: Option<f64>,
    cash: Option<f64>,
    prior_cash: Option<f64>,
    assets: Option<f64>,
) -> Option<f64> {
    let growth = clean(Some(
        clean(cash)? / clean(prior_cash).filter(|v| *v > 1e-12)?,
    ))?;
    let assets = clean(assets).filter(|v| *v > 1e-12)?;
    let expected = match clean(prior_gp) {
        Some(gp) => Some(clean(Some(gp * growth))?),
        None => None,
    };
    clean(Some(difference(gp, expected)? / assets))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{ColumnData, Table};
    use crate::factor::common::FinancialPitIndex;
    use std::{collections::BTreeMap, sync::Arc};
    #[test]
    fn abnormal_gross_profit_formula_and_missing_rules() {
        assert!(
            (abnormal(
                Some(125.0),
                Some(100.0),
                Some(110.0),
                Some(100.0),
                Some(500.0)
            )
            .unwrap()
                - 0.03)
                .abs()
                < 1e-12
        );
        assert_eq!(difference(None, Some(30.0)), Some(-30.0));
        assert_eq!(difference(Some(30.0), None), Some(30.0));
        assert_eq!(difference(None, Some(f64::NAN)), None);
        assert_eq!(
            abnormal(None, Some(10.0), Some(20.0), Some(10.0), Some(100.0)),
            Some(-0.2)
        );
        assert_eq!(
            abnormal(Some(10.0), None, Some(20.0), Some(10.0), Some(100.0)),
            Some(0.1)
        );
        assert_eq!(
            abnormal(None, None, Some(20.0), Some(10.0), Some(100.0)),
            None
        );
        for bad in [
            None,
            Some(0.0),
            Some(-1.0),
            Some(f64::NAN),
            Some(f64::INFINITY),
        ] {
            assert!(abnormal(Some(10.0), Some(5.0), Some(20.0), bad, Some(100.0)).is_none());
            assert!(abnormal(Some(10.0), Some(5.0), Some(20.0), Some(10.0), bad).is_none());
        }
    }
    #[test]
    fn abnormal_gross_profit_three_calendar_updates_and_no_annual_anchor() {
        for (date, expected) in [
            (20250101, (20241101, 20240930)),
            (20250430, (20241101, 20240930)),
            (20250501, (20250501, 20250331)),
            (20250831, (20250501, 20250331)),
            (20250901, (20250901, 20250630)),
            (20251031, (20250901, 20250630)),
            (20251101, (20251101, 20250930)),
            (20251103, (20251101, 20250930)),
            (20251231, (20251101, 20250930)),
        ] {
            assert_eq!(reporting_period(date), expected);
        }
        for code in ["801780.SI", "801790.SI", "801190", "801780"] {
            assert!(is_financial_sector(code));
        }
        assert!(!is_financial_sector("801180.SI"));
        let spec = StockDailyAbnormalGrossProfit.spec();
        assert!(spec.tags.contains(&"SWHYZQ".into()) && spec.tags.contains(&"fundamental".into()));
        assert!(!spec
            .dependencies
            .iter()
            .any(|d| d.dataset == DatasetId::StockDailyPv));
    }
    fn index(
        fields: &[&str],
        values: Vec<Vec<Option<f64>>>,
        report_type: i64,
    ) -> FinancialPitIndex {
        let mut cols = BTreeMap::from([
            (
                "ts_code".into(),
                ColumnData::Utf8(vec![Some("000001.SZ".into()); 4]),
            ),
            (
                "end_date".into(),
                ColumnData::I32(vec![
                    Some(20250331),
                    Some(20240331),
                    Some(20250331),
                    Some(20240331),
                ]),
            ),
            (
                "ann_date".into(),
                ColumnData::I32(vec![
                    Some(20250429),
                    Some(20240429),
                    Some(20250502),
                    Some(20250502),
                ]),
            ),
            ("f_ann_date".into(), ColumnData::I32(vec![None; 4])),
            (
                "report_type".into(),
                ColumnData::I64(vec![Some(report_type); 4]),
            ),
            ("update_flag".into(), ColumnData::I64(vec![Some(0); 4])),
        ]);
        for (name, values) in fields.iter().zip(values) {
            cols.insert((*name).into(), ColumnData::F64(values));
        }
        FinancialPitIndex::from_table(Arc::new(Table::new(cols).unwrap())).unwrap()
    }
    #[test]
    fn abnormal_gross_profit_freezes_disclosure_versions_at_update_cutoff() {
        let income = index(
            &["revenue", "oper_cost"],
            vec![
                vec![Some(225.0), Some(200.0), Some(999.0), Some(777.0)],
                vec![Some(100.0); 4],
            ],
            2,
        );
        let cash = index(
            &["c_fr_sale_sg"],
            vec![vec![Some(110.0), Some(100.0), Some(300.0), Some(250.0)]],
            2,
        );
        let balance = index(&["total_assets"], vec![vec![Some(500.0); 4]], 1);
        let readers = Readers {
            income: income.reader(ReportTypePreference::income_single_quarter()),
            cash: cash.reader(ReportTypePreference::income_single_quarter()),
            balance: balance.reader(ReportTypePreference::balance_sheet_consolidated()),
        };
        let before = readers.snapshot("000001.SZ", 20250501, 20250331).unwrap();
        assert!((before - 0.03).abs() < 1e-12);
        let mut cache = InstrumentAlignedSnapshotCache::default();
        for (date, codes) in [
            (20250501, ["000001.SZ", "000002.SZ", "430001.BJ"]),
            (20250831, ["430001.BJ", "000001.SZ", "000002.SZ"]),
            (20250901, ["000002.SZ", "430001.BJ", "000001.SZ"]),
            (20250501, ["000001.SZ", "430001.BJ", "000002.SZ"]),
        ] {
            let panel = crate::factor::common::DailyPanel::from_index(
                vec![date],
                codes.iter().map(|s| s.to_string()).collect(),
                &[date],
                vec![true; 3],
            )
            .unwrap();
            let (cutoff, end) = reporting_period(date);
            let values = cached_financial_stock_snapshots_for_date(
                &panel,
                date,
                &mut cache,
                |_, code, _| is_bj_stock(code),
                |_, code, _| readers.marker(code, cutoff, end),
                |_, code, _| readers.snapshot(code, cutoff, end),
            );
            for (i, code) in codes.iter().enumerate() {
                assert_eq!(
                    values[i],
                    if *code == "000001.SZ" && date < 20250901 {
                        Some(before)
                    } else {
                        None
                    }
                );
            }
        }
        for date in [20250501, 20250502, 20250601, 20250831] {
            let (cutoff, end) = reporting_period(date);
            assert_eq!(readers.snapshot("000001.SZ", cutoff, end), Some(before));
        }
        assert_ne!(
            readers.marker("000001.SZ", 20250501, 20250331),
            readers.marker("000001.SZ", 20250502, 20250331)
        );
        assert!(readers.snapshot("000001.SZ", 20250901, 20250630).is_none());
        assert!(readers.snapshot("000001.SZ", 20240401, 20250331).is_none());
    }
}

use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::{err, Result};
use crate::factor::common::stock_daily_ops::{is_bj_stock, neutralize_size_sector_with_inputs};
use crate::factor::common::{
    routed_financial_stock_snapshots_for_date, FinancialStockEventRoute, ClassificationLevel, ClassificationMap,
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
            snapshots = routed_financial_stock_snapshots_for_date(
                &panel,
                date,
                cache,
                &FinancialStockEventRoute::new(&[&readers.income, &readers.balance, &readers.cash]).at_date(period.0).with_key(period.1 as i64),
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

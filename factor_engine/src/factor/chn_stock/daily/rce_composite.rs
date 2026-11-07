use std::any::Any;
use std::collections::HashMap;

use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::{err, Result};
use crate::factor::common::financial::previous_quarter_end_date;
use crate::factor::common::vector::clean;
use crate::factor::common::{
    ClassificationLevel, ClassificationMap, FinancialPitReader, PitFinancialRecordView,
    ReportTypePreference,
};
use crate::factor::{Factor, FactorUpdatePolicy};
use crate::operators::{cs_neutralize_regression, cs_pctrank, cs_winsorize_quantile, cs_zscore};

const ID: &str = "rce_composite";
const HISTORY: usize = 8;
const GCE_QUARTERS: usize = HISTORY + 5;
// Covers all available income history (from 2009) across the supported research dates.
const RD_HISTORY_QUARTERS: usize = 160;
const QUARTER_COST: f64 = 0.025;
const MIN_GOA: f64 = 10_000_000.0;
const MIN_SURPRISE_STD: f64 = 1_000_000.0;
const DEPRECIATION: &str = "depr_fa_coga_dpba";
const INCOME: &[&str] = &[
    "total_revenue",
    "oper_cost",
    "biz_tax_surchg",
    "sell_exp",
    "admin_exp",
    "income_tax",
    "rd_exp",
];
const BALANCE: &[&str] = &[
    "defer_tax_liab",
    "defer_tax_assets",
    "total_liab",
    "money_cap",
    "fix_assets",
    "cip_total",
    "cip",
    "const_materials",
    "goodwill",
    "intan_assets",
    "r_and_d",
    "accounts_receiv_bill",
    "notes_receiv",
    "accounts_receiv",
    "prepayment",
    "inventories",
    "contract_assets",
    "accounts_pay",
    "notes_payable",
    "acct_payable",
    "adv_receipts",
    "contract_liab",
    "payroll_payable",
    "taxes_payable",
];

pub struct StockDailyRceComposite;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyRceComposite)
}

#[derive(Default)]
struct State {
    anchor: Option<i32>,
    values: HashMap<String, Option<f64>>,
}

impl Factor for StockDailyRceComposite {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: ID.into(),
            aliases: vec!["RCE_COMPOSITE".into()],
            name: "Residual Cash Earnings Composite".into(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: "0.1.0".into(),
            tags: ["TFZQ", "fundamental", "financial", "pit", "monthly_update", "neutralize", "size", "sector", "daily"]
                .into_iter().map(str::to_string).collect(),
            description: "Month-end PIT residual cash earnings composite. Combine quarterly GCE/EV, percentile rank of GCE/AvgGOA minus 2.5% less percentile rank of PGOA, and SUGCE (at least six of eight preceding YoY differences, sample std above RMB 1m). GOA approximates gross fixed assets with fix_assets (net); adds CIP, operating NWC, R&D capital, goodwill, intangibles and development expenditure. R&D capital starts at the first visible complete annual statement, annual rd_exp divided by 25%, then decays at 20% annually on a quarterly basis. Depreciation uses single-quarter or cumulative difference, then a PIT cumulative run-rate (Q4 prefers the second half). Partial missing additive fields are zero, all-missing sums null. Daily basic total_mv converted from RMB 10k to RMB. Clip each leg at 1/99 percentiles, zscore, fill missing with zero and average all three; SW L1 and Barra SIZE neutralize the final composite and hold the month-end residual daily.".into(),
            dependencies: vec![
                DataRequest::financial_quarters(DatasetId::StockIncome, INCOME, RD_HISTORY_QUARTERS),
                DataRequest::financial_quarters(DatasetId::StockBalanceSheet, BALANCE, GCE_QUARTERS + 1),
                DataRequest::financial_quarters(DatasetId::StockCashFlow, &[DEPRECIATION], GCE_QUARTERS + 5),
                DataRequest::new(DatasetId::StockDailyBasic, &["total_mv"]),
                DataRequest::new(DatasetId::StockBarraDaily, &["SIZE"]),
                DataRequest::new(DatasetId::StockSwClassification, &["l1_code"]),
            ],
            intraday_raw_dependencies: Vec::new(),
            lookback: Lookback { trading_days: 32 },
        }
    }

    fn update_policy(&self) -> FactorUpdatePolicy {
        FactorUpdatePolicy::FinancialEventSnapshot
    }

    fn initial_compute_state(&self, _: &[String]) -> Box<dyn Any + Send> {
        Box::<State>::default()
    }

    fn compute(&self, _: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        self.compute_on_panel(data, &mut State::default())
    }

    fn compute_many_stateful(
        &self,
        ids: &[String],
        _: &FactorContext,
        data: &DataPool,
        state: &mut (dyn Any + Send),
    ) -> Result<Vec<FactorSeries>> {
        if !ids.iter().any(|id| id == ID) {
            return Ok(Vec::new());
        }
        let state = state
            .downcast_mut::<State>()
            .ok_or_else(|| err("rce_composite state mismatch"))?;
        Ok(vec![self.compute_on_panel(data, state)?])
    }
}

impl StockDailyRceComposite {
    fn compute_on_panel(&self, data: &DataPool, state: &mut State) -> Result<FactorSeries> {
        let panel = data.stock_universe_panel()?;
        let calendar = data.trading_calendar()?;
        let market_cap =
            panel.column_from_table(data.daily(DatasetId::StockDailyBasic)?, "total_mv")?;
        let size = panel.column_from_table(data.daily(DatasetId::StockBarraDaily)?, "SIZE")?;
        let sectors = ClassificationMap::from_table(
            data.daily(DatasetId::StockSwClassification)?,
            ClassificationLevel::Sector,
        )?;
        let statements = Statements {
            income: data.financial_reader(
                DatasetId::StockIncome,
                ReportTypePreference::income_single_quarter(),
            )?,
            annual_income: data
                .financial_reader(DatasetId::StockIncome, ReportTypePreference::consolidated())?,
            balance: data.financial_reader(
                DatasetId::StockBalanceSheet,
                ReportTypePreference::balance_sheet_consolidated(),
            )?,
            cash_quarter: data.financial_reader(
                DatasetId::StockCashFlow,
                ReportTypePreference::income_single_quarter(),
            )?,
            cash_cumulative: data.financial_reader(
                DatasetId::StockCashFlow,
                ReportTypePreference::consolidated(),
            )?,
        };
        let instruments = panel.instruments();
        let n = instruments.len();
        let mut values = vec![None; panel.shape_len()];
        for (day, &date) in panel.dates().iter().enumerate() {
            if !panel.is_target_date(date) {
                continue;
            }
            if let Some(anchor) = calendar.month_end_on_or_before(date) {
                if state.anchor != Some(anchor) {
                    let i = panel
                        .dates()
                        .binary_search(&anchor)
                        .map_err(|_| err("rce_composite month-end missing from warmup panel"))?;
                    let offset = i * n;
                    let mut groups = sectors.groups_for(anchor, instruments);
                    let mut yield_ev = Vec::with_capacity(n);
                    let mut spread = Vec::with_capacity(n);
                    let mut pgoa = Vec::with_capacity(n);
                    let mut surprise = Vec::with_capacity(n);
                    for (j, code) in instruments.iter().enumerate() {
                        let legs = if panel.is_present_offset(offset + j) {
                            let mv = clean(market_cap.values()[offset + j]).filter(|v| *v > 0.0);
                            statements.legs(code, anchor, mv.map(|v| v * 10_000.0))
                        } else {
                            groups[j] = None;
                            [None; 4]
                        };
                        yield_ev.push(legs[0]);
                        spread.push(legs[1]);
                        pgoa.push(legs[2]);
                        surprise.push(legs[3]);
                    }
                    let rank_gap: Vec<_> = cs_pctrank(&spread, true)
                        .into_iter()
                        .zip(cs_pctrank(&pgoa, true))
                        .map(|(s, p)| finite(s? - p?))
                        .collect();
                    let [ev_z, gap_z, surprise_z] = [&yield_ev, &rank_gap, &surprise]
                        .map(|leg| cs_zscore(&cs_winsorize_quantile(leg, 0.01, 0.99)));
                    let combined: Vec<_> = (0..n)
                        .map(|j| {
                            panel.is_present_offset(offset + j).then(|| {
                                (zero(ev_z[j]) + zero(gap_z[j]) + zero(surprise_z[j])) / 3.0
                            })
                        })
                        .collect();
                    let exposure = &size.values()[offset..offset + n];
                    let residuals =
                        cs_neutralize_regression(&combined, &[exposure], Some(&groups), None);
                    state.values = instruments.iter().cloned().zip(residuals).collect();
                    state.anchor = Some(anchor);
                }
                for (j, code) in instruments.iter().enumerate() {
                    let offset = day * n + j;
                    if panel.is_present_offset(offset) {
                        values[offset] = state.values.get(code).copied().flatten();
                    }
                }
            }
        }
        Ok(panel
            .column_from_values(values)?
            .to_factor_series(self.spec()))
    }
}

struct Statements<'a> {
    income: FinancialPitReader<'a>,
    annual_income: FinancialPitReader<'a>,
    balance: FinancialPitReader<'a>,
    cash_quarter: FinancialPitReader<'a>,
    cash_cumulative: FinancialPitReader<'a>,
}

impl Statements<'_> {
    fn legs(&self, code: &str, date: i32, market_cap: Option<f64>) -> [Option<f64>; 4] {
        let Some(end) = self
            .income
            .latest_quarter_end_date(code, date)
            .max(self.annual_income.latest_quarter_end_date(code, date))
        else {
            return [None; 4];
        };
        let mut current = Some(end);
        let gce = std::array::from_fn(|_| {
            let quarter = current?;
            current = previous_quarter_end_date(quarter);
            self.gce(code, date, quarter)
        });
        let latest_balance = self
            .balance
            .latest_quarter_end_date(code, date)
            .and_then(|q| self.balance.record_for_end_date(code, date, q));
        let ev_yield = (|| {
            let b = latest_balance?;
            let ev =
                finite(market_cap? + zero(b.column("total_liab")) - zero(b.column("money_cap")))?;
            positive_ratio(gce[0]?, ev)
        })();
        let (rd_now, rd_prior) = self.rd_capital(code, date, end);
        let now = self
            .balance
            .record_for_end_date(code, date, end)
            .and_then(|b| goa(b, rd_now));
        let prior = self
            .balance
            .record_for_end_date(code, date, end - 10_000)
            .and_then(|b| goa(b, rd_prior));
        let spread = (|| finite(positive_ratio(gce[0]?, (now? + prior?) * 0.5)? - QUARTER_COST))();
        let pgoa = (|| positive_ratio(market_cap?, now.filter(|v| *v > MIN_GOA)?))();
        [ev_yield, spread, pgoa, sugce(&gce)]
    }

    fn income_quarter(&self, code: &str, date: i32, end: i32, fields: &[&str]) -> Option<f64> {
        let sum = |record: PitFinancialRecordView<'_>| {
            signed_sum(
                fields
                    .iter()
                    .enumerate()
                    .map(|(i, name)| (record.column(name), if i == 0 { 1.0 } else { -1.0 })),
            )
        };
        if let Some(value) = self
            .income
            .record_for_end_date(code, date, end)
            .and_then(sum)
        {
            return Some(value);
        }
        let cumulative = self.annual_income.record_for_end_date(code, date, end)?;
        if end % 10_000 == 331 {
            return sum(cumulative);
        }
        let previous =
            self.annual_income
                .record_for_end_date(code, date, previous_quarter_end_date(end)?)?;
        signed_sum([(sum(cumulative), 1.0), (sum(previous), -1.0)])
    }

    fn gce(&self, code: &str, date: i32, end: i32) -> Option<f64> {
        self.income
            .record_for_end_date(code, date, end)
            .or_else(|| self.annual_income.record_for_end_date(code, date, end))?;
        // The expensed R&D subtracted in OpEBIT is added back in GCE.
        let operating = self.income_quarter(code, date, end, &INCOME[..5]);
        let tax = self.income_quarter(code, date, end, &["income_tax"]);
        let deferred = (|| {
            let b = self.balance.record_for_end_date(code, date, end)?;
            let prev =
                self.balance
                    .record_for_end_date(code, date, previous_quarter_end_date(end)?)?;
            signed_sum([
                (b.column("defer_tax_liab"), 1.0),
                (prev.column("defer_tax_liab"), -1.0),
                (b.column("defer_tax_assets"), -1.0),
                (prev.column("defer_tax_assets"), 1.0),
            ])
        })();
        signed_sum([
            (operating, 1.0),
            (self.depreciation(code, date, end), 1.0),
            (tax, -1.0),
            (deferred, 1.0),
        ])
    }

    fn depreciation(&self, code: &str, date: i32, end: i32) -> Option<f64> {
        if let Some(value) = self
            .cash_quarter
            .record_for_end_date(code, date, end)
            .and_then(|r| clean(r.column(DEPRECIATION)))
        {
            return Some(value);
        }
        let cumulative = |q| {
            self.cash_cumulative
                .record_for_end_date(code, date, q)
                .and_then(|r| clean(r.column(DEPRECIATION)))
        };
        if let Some(now) = cumulative(end) {
            if end % 10_000 == 331 {
                return Some(now);
            }
            if let Some(previous) = cumulative(previous_quarter_end_date(end)?) {
                return finite(now - previous);
            }
            if end % 10_000 == 1231 {
                if let Some(half) = cumulative(end / 10_000 * 10_000 + 630) {
                    return finite((now - half) / 2.0);
                }
            }
        }
        self.cash_cumulative
            .end_dates(code)
            .rev()
            .filter(|q| *q <= end)
            .find_map(|q| finite(cumulative(q)? / quarter_number(q)? as f64))
    }

    fn rd_capital(&self, code: &str, date: i32, end: i32) -> (Option<f64>, Option<f64>) {
        let seed = self
            .annual_income
            .end_dates(code)
            .take_while(|q| *q <= end)
            .filter(|q| q % 10_000 == 1231)
            .find_map(|q| {
                let annual = self.annual_income.record_for_end_date(code, date, q)?;
                (clean(annual.column("total_revenue")).is_some()
                    || clean(annual.column("oper_cost")).is_some())
                .then(|| (q, zero(annual.column("rd_exp")) / 0.25))
            });
        let (Some((seed_end, mut capital)), Some(end_slot)) = (seed, quarter_slot(end)) else {
            return (None, None);
        };
        let seed_slot = 4 * (seed_end / 10_000) + 3;
        let mut prior = None;
        let keep = 0.8_f64.powf(0.25);
        for slot in seed_slot..=end_slot {
            if slot != seed_slot {
                let q = (slot / 4) * 10_000 + [331, 630, 930, 1231][(slot % 4) as usize];
                let rd = zero(self.income_quarter(code, date, q, &["rd_exp"]));
                capital = capital * keep + rd;
            }
            if slot == end_slot - 4 {
                prior = finite(capital);
            }
        }
        (finite(capital), prior)
    }
}

fn goa(balance: PitFinancialRecordView<'_>, rd_capital: Option<f64>) -> Option<f64> {
    let col = |name| clean(balance.column(name));
    let nwc = signed_sum([
        (
            col("accounts_receiv_bill").or_else(|| {
                signed_sum([(col("notes_receiv"), 1.0), (col("accounts_receiv"), 1.0)])
            }),
            1.0,
        ),
        (col("prepayment"), 1.0),
        (col("inventories"), 1.0),
        (col("contract_assets"), 1.0),
        (
            col("accounts_pay")
                .or_else(|| signed_sum([(col("notes_payable"), 1.0), (col("acct_payable"), 1.0)])),
            -1.0,
        ),
        (col("adv_receipts"), -1.0),
        (col("contract_liab"), -1.0),
        (col("payroll_payable"), -1.0),
        (col("taxes_payable"), -1.0),
    ]);
    let cip =
        col("cip_total").or_else(|| signed_sum([(col("cip"), 1.0), (col("const_materials"), 1.0)]));
    signed_sum([
        (col("fix_assets"), 1.0),
        (cip, 1.0),
        (nwc, 1.0),
        (Some(rd_capital?), 1.0),
        (col("goodwill"), 1.0),
        (col("intan_assets"), 1.0),
        (col("r_and_d"), 1.0),
    ])
}

fn sugce(values: &[Option<f64>; GCE_QUARTERS]) -> Option<f64> {
    let yoy = |i: usize| finite(values[i]? - values[i + 4]?);
    let current = yoy(0)?;
    let past: Vec<_> = (1..=HISTORY).filter_map(yoy).collect();
    if past.len() < 6 {
        return None;
    }
    let mean = past.iter().sum::<f64>() / past.len() as f64;
    let std =
        (past.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (past.len() - 1) as f64).sqrt();
    (std > MIN_SURPRISE_STD)
        .then(|| finite((current - mean) / std))
        .flatten()
}

fn quarter_number(end: i32) -> Option<i32> {
    match end % 10_000 {
        331 => Some(1),
        630 => Some(2),
        930 => Some(3),
        1231 => Some(4),
        _ => None,
    }
}

fn quarter_slot(end: i32) -> Option<i32> {
    Some(4 * (end / 10_000) + quarter_number(end)? - 1)
}

fn finite(value: f64) -> Option<f64> {
    clean(Some(value))
}

fn zero(value: Option<f64>) -> f64 {
    clean(value).unwrap_or(0.0)
}

fn signed_sum(values: impl IntoIterator<Item = (Option<f64>, f64)>) -> Option<f64> {
    let mut valid = values
        .into_iter()
        .filter_map(|(v, sign)| clean(v).map(|x| x * sign));
    finite(valid.next()? + valid.sum::<f64>())
}

fn positive_ratio(numerator: f64, denominator: f64) -> Option<f64> {
    if denominator > 0.0 && denominator.is_finite() {
        finite(numerator / denominator)
    } else {
        None
    }
}

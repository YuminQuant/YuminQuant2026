use std::any::Any;

use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::{err, Result};
use crate::factor::common::financial::previous_quarter_end_date;
use crate::factor::common::stock_daily_ops::{is_bj_stock, neutralize_size_sector_with_inputs};
use crate::factor::common::{
    routed_financial_stock_snapshots_for_date, FinancialStockEventRoute, ClassificationLevel, ClassificationMap,
    FinancialEventMarker, FinancialEventMarkerBuilder, FinancialEventSchedule, FinancialPitReader,
    FinancialStatementDataset, InstrumentAlignedSnapshotCache, ReportTypePreference,
};
use crate::factor::{Factor, FactorUpdatePolicy};

const PROVIDER_KEY: &str = "stock|daily|gdzq_financial_ts_residual";
const WINDOW: usize = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Output {
    OperProfitEbit,
    OperProfitGrossProfit,
    OperProfitCashCollection,
    RoeSurplusReserve,
    OperMarginSurplusReserve,
    TotalCostRatioSurplusReserve,
    AdvanceInventoryDays,
    CashSalesInventoryTurnover,
    NoninterestCurLiabInventoryTurnover,
}

const ALL: [Output; 9] = [
    Output::OperProfitEbit,
    Output::OperProfitGrossProfit,
    Output::OperProfitCashCollection,
    Output::RoeSurplusReserve,
    Output::OperMarginSurplusReserve,
    Output::TotalCostRatioSurplusReserve,
    Output::AdvanceInventoryDays,
    Output::CashSalesInventoryTurnover,
    Output::NoninterestCurLiabInventoryTurnover,
];

impl Output {
    pub fn id(self) -> &'static str {
        match self {
            Self::OperProfitEbit => "oper_profit_ebit_ts_resid",
            Self::OperProfitGrossProfit => "oper_profit_gross_profit_ts_resid",
            Self::OperProfitCashCollection => "oper_profit_cash_collection_ts_resid",
            Self::RoeSurplusReserve => "roe_surplus_reserve_ts_resid",
            Self::OperMarginSurplusReserve => "oper_margin_surplus_reserve_ts_resid",
            Self::TotalCostRatioSurplusReserve => "total_cost_ratio_surplus_reserve_ts_resid",
            Self::AdvanceInventoryDays => "advance_inventory_days_ts_resid",
            Self::CashSalesInventoryTurnover => "cash_sales_inventory_turnover_ts_resid",
            Self::NoninterestCurLiabInventoryTurnover => {
                "noninterest_cur_liab_inventory_turnover_ts_resid"
            }
        }
    }

    fn fields(
        self,
    ) -> (
        &'static [&'static str],
        &'static [&'static str],
        &'static [&'static str],
    ) {
        match self {
            Self::OperProfitEbit => (
                &["operate_profit", "n_income", "income_tax", "int_exp"],
                &[],
                &[],
            ),
            Self::OperProfitGrossProfit => (&["operate_profit", "revenue", "oper_cost"], &[], &[]),
            Self::OperProfitCashCollection => {
                (&["operate_profit", "revenue"], &[], &["c_fr_sale_sg"])
            }
            Self::RoeSurplusReserve => (
                &["n_income_attr_p"],
                &["total_hldr_eqy_exc_min_int", "surplus_rese"],
                &[],
            ),
            Self::OperMarginSurplusReserve => {
                (&["operate_profit", "total_revenue"], &["surplus_rese"], &[])
            }
            Self::TotalCostRatioSurplusReserve => {
                (&["total_cogs", "total_revenue"], &["surplus_rese"], &[])
            }
            Self::AdvanceInventoryDays => (&["oper_cost"], &["adv_receipts", "inventories"], &[]),
            Self::CashSalesInventoryTurnover => {
                (&["oper_cost"], &["inventories"], &["c_fr_sale_sg"])
            }
            Self::NoninterestCurLiabInventoryTurnover => (
                &["oper_cost"],
                &[
                    "inventories",
                    "total_cur_liab",
                    "st_borr",
                    "non_cur_liab_due_1y",
                    "st_bonds_payable",
                    "st_fin_payable",
                ],
                &[],
            ),
        }
    }

    fn sign(self) -> f64 {
        if self == Self::TotalCostRatioSurplusReserve {
            -1.0
        } else {
            1.0
        }
    }

    fn prior_balance(self) -> bool {
        matches!(
            self,
            Self::RoeSurplusReserve
                | Self::AdvanceInventoryDays
                | Self::CashSalesInventoryTurnover
                | Self::NoninterestCurLiabInventoryTurnover
        )
    }
}

fn spec(output: Output) -> FactorSpec {
    let (income, balance, cash) = output.fields();
    let mut dependencies = vec![DataRequest::financial_quarters(
        DatasetId::StockIncome,
        income,
        WINDOW,
    )];
    if !balance.is_empty() {
        dependencies.push(DataRequest::financial_quarters(
            DatasetId::StockBalanceSheet,
            balance,
            WINDOW + usize::from(output.prior_balance()),
        ));
    }
    if !cash.is_empty() {
        dependencies.push(DataRequest::financial_quarters(
            DatasetId::StockCashFlow,
            cash,
            WINDOW,
        ));
    }
    dependencies.push(DataRequest::new(DatasetId::StockBarraDaily, &["SIZE"]));
    dependencies.push(DataRequest::new(
        DatasetId::StockSwClassification,
        &["l1_code"],
    ));
    FactorSpec {
        id: output.id().into(), aliases: vec![output.id().to_ascii_uppercase()],
        name: output.id().replace('_', " "), asset_class: AssetClass::Stock, frequency: Frequency::Daily,
        version: "0.1.0".into(),
        tags: ["GDZQ", "fundamental", "financial", "pit", "ts_regression", "neutralize", "size", "sector", "daily"].into_iter().map(str::to_string).collect(),
        description: format!("Eight consecutive PIT single quarters, within-stock population zscore of Y and X, OLS with intercept including the latest quarter; latest residual times {}. Partial missing additive operands use zero, all missing stays null. SW L1 and Barra SIZE neutralized daily, excludes BJ. See financial development README for field definitions.", output.sign()),
        dependencies, intraday_raw_dependencies: vec![], lookback: Lookback { trading_days: 0 },
    }
}

pub struct GdzqFinancialTsResidual {
    output: Output,
}

impl GdzqFinancialTsResidual {
    pub fn new(output: Output) -> Self {
        Self { output }
    }
}

#[derive(Default)]
struct ComputeState {
    requested: Vec<Output>,
    snapshots: InstrumentAlignedSnapshotCache<[Option<f64>; 9]>,
}

impl Factor for GdzqFinancialTsResidual {
    fn spec(&self) -> FactorSpec {
        spec(self.output)
    }
    fn compute_provider_key(&self) -> String {
        PROVIDER_KEY.into()
    }
    fn update_policy(&self) -> FactorUpdatePolicy {
        FactorUpdatePolicy::FinancialEventStateDailyFast
    }
    fn initial_compute_state(&self, _: &[String]) -> Box<dyn Any + Send> {
        Box::new(ComputeState::default())
    }
    fn compute(&self, _: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        compute_requested(
            &[self.output.id().into()],
            data,
            &mut ComputeState::default(),
        )?
        .pop()
        .ok_or_else(|| err("GDZQ provider returned no output"))
    }
    fn compute_many(
        &self,
        ids: &[String],
        _: &FactorContext,
        data: &DataPool,
    ) -> Result<Vec<FactorSeries>> {
        compute_requested(ids, data, &mut ComputeState::default())
    }
    fn compute_many_stateful(
        &self,
        ids: &[String],
        _: &FactorContext,
        data: &DataPool,
        state: &mut (dyn Any + Send),
    ) -> Result<Vec<FactorSeries>> {
        compute_requested(
            ids,
            data,
            state
                .downcast_mut::<ComputeState>()
                .ok_or_else(|| err("GDZQ provider state mismatch"))?,
        )
    }
}

struct Readers<'a> {
    income: FinancialPitReader<'a>,
    balance: Option<FinancialPitReader<'a>>,
    cash: Option<FinancialPitReader<'a>>,
    balance_quarters: usize,
}

fn compute_requested(
    ids: &[String],
    data: &DataPool,
    state: &mut ComputeState,
) -> Result<Vec<FactorSeries>> {
    let requested: Vec<_> = ALL
        .into_iter()
        .filter(|output| ids.iter().any(|id| id == output.id()))
        .collect();
    if requested.is_empty() {
        return Ok(vec![]);
    }
    // Provider jobs may request different subsets in successive factor batches.
    if requested != state.requested {
        state.snapshots = InstrumentAlignedSnapshotCache::default();
        state.requested = requested.clone();
    }
    let readers = Readers {
        income: data.financial_reader(
            DatasetId::StockIncome,
            ReportTypePreference::income_single_quarter(),
        )?,
        balance: if requested.iter().any(|o| !o.fields().1.is_empty()) {
            Some(data.financial_reader(
                DatasetId::StockBalanceSheet,
                ReportTypePreference::balance_sheet_consolidated(),
            )?)
        } else {
            None
        },
        cash: if requested.iter().any(|o| !o.fields().2.is_empty()) {
            Some(data.financial_reader(
                DatasetId::StockCashFlow,
                ReportTypePreference::income_single_quarter(),
            )?)
        } else {
            None
        },
        balance_quarters: WINDOW + usize::from(requested.iter().any(|o| o.prior_balance())),
    };
    let mut event_readers = vec![readers.income.clone()];
    event_readers.extend(readers.balance.iter().cloned());
    event_readers.extend(readers.cash.iter().cloned());
    let schedule = FinancialEventSchedule::from_pit_readers(&event_readers);
    let panel = data.stock_universe_panel()?;
    let n = panel.instruments().len();
    let mut values = vec![vec![None; panel.shape_len()]; requested.len()];
    let mut snapshots = vec![None; n];
    let mut last_date = None;
    let mut previous_day: Option<usize> = None;
    for (day, date) in panel.dates().iter().copied().enumerate() {
        if !panel.is_target_date(date) {
            continue;
        }
        let changed = previous_day.is_some_and(|prev| {
            (0..n).any(|i| {
                panel.is_present_offset(day * n + i) != panel.is_present_offset(prev * n + i)
            })
        });
        if last_date.is_none() || changed || schedule.has_event_after_until(last_date, date) {
            snapshots = routed_financial_stock_snapshots_for_date(
                &panel,
                date,
                &mut state.snapshots,
                &FinancialStockEventRoute::new(&event_readers.iter().collect::<Vec<_>>()),
                |_, code, offset| is_bj_stock(code) || !panel.is_present_offset(offset),
                |date, code, _| readers.marker(code, date),
                |date, code, _| readers.snapshot(code, date, &requested),
            );
        }
        for (i, snapshot) in snapshots.iter().enumerate() {
            if panel.is_present_offset(day * n + i) {
                for (j, output) in requested.iter().enumerate() {
                    values[j][day * n + i] = snapshot.and_then(|s| s[*output as usize]);
                }
            }
        }
        last_date = Some(date);
        previous_day = Some(day);
    }
    let size = panel.column_from_table(data.daily(DatasetId::StockBarraDaily)?, "SIZE")?;
    let sector = ClassificationMap::from_table(
        data.daily(DatasetId::StockSwClassification)?,
        ClassificationLevel::Sector,
    )?;
    requested
        .into_iter()
        .zip(values)
        .map(|(output, values)| {
            let raw = panel.column_from_values(values)?;
            Ok(
                neutralize_size_sector_with_inputs(&raw, &panel, &size, &sector)?
                    .to_factor_series(spec(output)),
            )
        })
        .collect()
}

fn quarter_ends(anchor: i32) -> Option<[i32; 9]> {
    let mut ends = [anchor; 9];
    for i in 1..9 {
        ends[i] = previous_quarter_end_date(ends[i - 1])?;
    }
    Some(ends)
}

impl Readers<'_> {
    fn marker(&self, code: &str, date: i32) -> Option<FinancialEventMarker> {
        let ends = quarter_ends(self.income.latest_quarter_end_date(code, date)?)?;
        let mut marker = FinancialEventMarkerBuilder::new();
        for (i, end) in ends.into_iter().enumerate() {
            if i < WINDOW {
                marker.include_reader_record_for_end_date(
                    FinancialStatementDataset::Income,
                    &self.income,
                    code,
                    date,
                    end,
                );
                if let Some(reader) = &self.cash {
                    marker.include_reader_record_for_end_date(
                        FinancialStatementDataset::CashFlow,
                        reader,
                        code,
                        date,
                        end,
                    );
                }
            }
            if i < self.balance_quarters {
                if let Some(reader) = &self.balance {
                    marker.include_reader_record_for_end_date(
                        FinancialStatementDataset::BalanceSheet,
                        reader,
                        code,
                        date,
                        end,
                    );
                }
            }
        }
        marker.build()
    }

    fn snapshot(&self, code: &str, date: i32, requested: &[Output]) -> Option<[Option<f64>; 9]> {
        let ends = quarter_ends(self.income.latest_quarter_end_date(code, date)?)?;
        let income = ends.map(|end| self.income.record_for_end_date(code, date, end));
        let balance = ends.map(|end| {
            self.balance
                .as_ref()
                .and_then(|r| r.record_for_end_date(code, date, end))
        });
        let cash = ends.map(|end| {
            self.cash
                .as_ref()
                .and_then(|r| r.record_for_end_date(code, date, end))
        });
        let mut result = [None; 9];
        for output in requested {
            let pairs = std::array::from_fn(|q| {
                pair(
                    *output,
                    |name| income[q].and_then(|r| clean(r.column(name))),
                    |name| balance[q].and_then(|r| clean(r.column(name))),
                    |name| balance[q + 1].and_then(|r| clean(r.column(name))),
                    |name| cash[q].and_then(|r| clean(r.column(name))),
                )
            });
            result[*output as usize] = latest_residual(pairs).map(|v| output.sign() * v);
        }
        Some(result)
    }
}

fn clean(value: Option<f64>) -> Option<f64> {
    value.filter(|v| v.is_finite())
}

// Null operands contribute zero, but a wholly missing expression stays null.
fn additive(terms: &[(Option<f64>, f64)]) -> Option<f64> {
    let mut total = 0.0;
    let mut present = false;
    for (value, sign) in terms {
        if let Some(value) = clean(*value) {
            present = true;
            total += value * sign;
        }
    }
    clean(present.then_some(total))
}

fn ratio(numerator: Option<f64>, denominator: Option<f64>) -> Option<f64> {
    clean(Some(
        clean(numerator)? / clean(denominator).filter(|d| *d > 1e-12)?,
    ))
}

fn pair(
    output: Output,
    income: impl Fn(&str) -> Option<f64>,
    balance: impl Fn(&str) -> Option<f64>,
    prior: impl Fn(&str) -> Option<f64>,
    cash: impl Fn(&str) -> Option<f64>,
) -> Option<(f64, f64)> {
    let avg = |name| additive(&[(balance(name), 0.5), (prior(name), 0.5)]);
    let turnover = || ratio(income("oper_cost"), avg("inventories"));
    let (y, x) = match output {
        Output::OperProfitEbit => (
            income("operate_profit"),
            additive(&[
                (income("n_income"), 1.0),
                (income("income_tax"), 1.0),
                (income("int_exp"), 1.0),
            ]),
        ),
        Output::OperProfitGrossProfit => (
            income("operate_profit"),
            additive(&[(income("revenue"), 1.0), (income("oper_cost"), -1.0)]),
        ),
        Output::OperProfitCashCollection => (
            income("operate_profit"),
            ratio(cash("c_fr_sale_sg"), income("revenue")),
        ),
        Output::RoeSurplusReserve => {
            // Missing endpoints follow the additive rule; observed nonpositive equity is invalid.
            let field = "total_hldr_eqy_exc_min_int";
            if [balance(field), prior(field)]
                .into_iter()
                .flatten()
                .any(|v| v <= 0.0)
            {
                return None;
            }
            (
                ratio(income("n_income_attr_p").map(|v| v * 4.0), avg(field)),
                balance("surplus_rese"),
            )
        }
        Output::OperMarginSurplusReserve => (
            ratio(income("operate_profit"), income("total_revenue")),
            balance("surplus_rese"),
        ),
        Output::TotalCostRatioSurplusReserve => (
            ratio(income("total_cogs"), income("total_revenue")),
            balance("surplus_rese"),
        ),
        Output::AdvanceInventoryDays => (balance("adv_receipts"), ratio(Some(90.0), turnover())),
        Output::CashSalesInventoryTurnover => (cash("c_fr_sale_sg"), turnover()),
        Output::NoninterestCurLiabInventoryTurnover => (
            additive(&[
                (balance("total_cur_liab"), 1.0),
                (balance("st_borr"), -1.0),
                (balance("non_cur_liab_due_1y"), -1.0),
                (balance("st_bonds_payable"), -1.0),
                (balance("st_fin_payable"), -1.0),
            ]),
            turnover(),
        ),
    };
    Some((clean(y)?, clean(x)?))
}

fn latest_residual(pairs: [Option<(f64, f64)>; WINDOW]) -> Option<f64> {
    let mut y = [0.0; WINDOW];
    let mut x = [0.0; WINDOW];
    for (i, pair) in pairs.into_iter().enumerate() {
        let (yi, xi) = pair?;
        y[i] = clean(Some(yi))?;
        x[i] = clean(Some(xi))?;
    }
    let standardize = |values: [f64; WINDOW]| -> Option<[f64; WINDOW]> {
        let scale = values.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        if scale == 0.0 {
            return None;
        }
        let values = values.map(|v| v / scale);
        let mean = values.iter().sum::<f64>() / WINDOW as f64;
        let centered = values.map(|v| v - mean);
        let sd = (centered.iter().map(|v| v * v).sum::<f64>() / WINDOW as f64).sqrt();
        if sd <= 1e-12 {
            return None;
        }
        Some(centered.map(|v| v / sd))
    };
    let y = standardize(y)?;
    let x = standardize(x)?;
    let beta =
        x.iter().zip(y).map(|(x, y)| x * y).sum::<f64>() / x.iter().map(|x| x * x).sum::<f64>();
    // Standardized columns have zero means, absorbing the intercept. Newest first.
    clean(Some(y[0] - beta * x[0]))
}

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
    cached_financial_stock_snapshots_for_date, ClassificationLevel, ClassificationMap,
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
            snapshots = cached_financial_stock_snapshots_for_date(
                &panel,
                date,
                &mut state.snapshots,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{ColumnData, Table};
    use crate::factor::common::{DailyPanel, FinancialPitIndex};
    use std::collections::{BTreeMap, HashMap};
    use std::sync::Arc;

    #[test]
    fn additive_missing_policy_and_denominators() {
        assert_eq!(additive(&[(None, 1.0), (Some(f64::NAN), -1.0)]), None);
        assert_eq!(additive(&[(None, 1.0), (Some(5.0), -1.0)]), Some(-5.0));
        assert_eq!(additive(&[(Some(0.0), 1.0), (None, -1.0)]), Some(0.0));
        assert_eq!(ratio(Some(1.0), Some(0.0)), None);
        assert_eq!(ratio(Some(1.0), Some(-2.0)), None);
        assert_eq!(ratio(None, Some(2.0)), None);
    }

    #[test]
    fn all_nine_pair_definitions_and_signs() {
        let income = |name: &str| match name {
            "operate_profit" => Some(12.0),
            "n_income" => Some(8.0),
            "income_tax" => Some(2.0),
            "revenue" => Some(40.0),
            "oper_cost" => Some(20.0),
            "n_income_attr_p" => Some(10.0),
            "total_revenue" => Some(50.0),
            "total_cogs" => Some(30.0),
            _ => None,
        };
        let balance = |name: &str| match name {
            "total_hldr_eqy_exc_min_int" => Some(100.0),
            "surplus_rese" => Some(6.0),
            "inventories" => Some(10.0),
            "adv_receipts" => Some(9.0),
            "total_cur_liab" => Some(40.0),
            "st_borr" => Some(5.0),
            "non_cur_liab_due_1y" => Some(3.0),
            "st_bonds_payable" => Some(2.0),
            _ => None,
        };
        let prior = |name: &str| match name {
            "inventories" => Some(30.0),
            "total_hldr_eqy_exc_min_int" => Some(60.0),
            _ => None,
        };
        let cash = |name: &str| (name == "c_fr_sale_sg").then_some(24.0);
        let expected = [
            (12.0, 10.0),
            (12.0, 20.0),
            (12.0, 0.6),
            (0.5, 6.0),
            (0.24, 6.0),
            (0.6, 6.0),
            (9.0, 90.0),
            (24.0, 1.0),
            (30.0, 1.0),
        ];
        for (output, expected) in ALL.into_iter().zip(expected) {
            let actual = pair(output, income, balance, prior, cash).unwrap();
            assert!((actual.0 - expected.0).abs() < 1e-12, "{output:?}");
            assert!((actual.1 - expected.1).abs() < 1e-12, "{output:?}");
            assert_eq!(
                output.sign(),
                if output == Output::TotalCostRatioSurplusReserve {
                    -1.0
                } else {
                    1.0
                }
            );
        }
        // Even the leading term of a sum can be absent under the user policy.
        assert_eq!(
            pair(
                Output::OperProfitEbit,
                |n| match n {
                    "operate_profit" => Some(5.0),
                    "int_exp" => Some(3.0),
                    _ => None,
                },
                |_| None,
                |_| None,
                |_| None
            ),
            Some((5.0, 3.0))
        );
        assert_eq!(
            pair(
                Output::OperProfitGrossProfit,
                |n| match n {
                    "operate_profit" => Some(5.0),
                    "oper_cost" => Some(3.0),
                    _ => None,
                },
                |_| None,
                |_| None,
                |_| None
            ),
            Some((5.0, -3.0))
        );
        assert_eq!(
            pair(Output::RoeSurplusReserve, income, balance, |_| None, cash),
            Some((0.8, 6.0))
        );
        assert!(pair(
            Output::RoeSurplusReserve,
            income,
            balance,
            |_| Some(-1.0),
            cash
        )
        .is_none());
        for output in ALL {
            assert!(pair(output, |_| None, |_| None, |_| None, |_| None).is_none());
        }
    }

    #[test]
    fn regression_is_latest_in_sample_population_zscore_with_intercept() {
        let y = [18.0, 6.0, 3.0, 4.0, 2.0, 1.0, 7.0, 9.0];
        let x = [2.0, 8.0, 3.0, 1.0, 6.0, 7.0, 4.0, 5.0];
        let ym = y.iter().sum::<f64>() / 8.0;
        let xm = x.iter().sum::<f64>() / 8.0;
        let sy = (y.iter().map(|v| (v - ym).powi(2)).sum::<f64>() / 8.0).sqrt();
        let beta = (0..8).map(|i| (x[i] - xm) * (y[i] - ym)).sum::<f64>()
            / x.iter().map(|v| (v - xm).powi(2)).sum::<f64>();
        let expected = (y[0] - ym - beta * (x[0] - xm)) / sy;
        let pairs = std::array::from_fn(|i| Some((y[i], x[i])));
        assert!((latest_residual(pairs).unwrap() - expected).abs() < 1e-12);
        assert!(
            (latest_residual(pairs.map(|p| p.map(|(y, x)| (y * 1e100, x * 1e-100)))).unwrap()
                - expected)
                .abs()
                < 1e-12
        );
        assert!(latest_residual(std::array::from_fn(|i| Some((y[i], 3.0)))).is_none());
        assert!(latest_residual(std::array::from_fn(|i| Some((3.0, x[i])))).is_none());
        assert!(
            latest_residual(std::array::from_fn(|i| Some((2.0 * x[i] + 3.0, x[i]))))
                .unwrap()
                .abs()
                < 1e-12
        );
        for i in 0..8 {
            let mut missing = pairs;
            missing[i] = None;
            assert!(latest_residual(missing).is_none());
        }
    }

    fn financial_table(dataset: DatasetId) -> Table {
        let ends = quarter_ends(20250331).unwrap();
        // Last row per stock revises the oldest quarter, after the initial test date.
        let rows: Vec<_> = (0..8)
            .flat_map(|stock| (0..10).map(move |q| (stock, q)))
            .collect();
        let mut cols = BTreeMap::from([
            (
                "ts_code".into(),
                ColumnData::Utf8(
                    rows.iter()
                        .map(|(s, _)| Some(format!("{:06}.SZ", s + 1)))
                        .collect(),
                ),
            ),
            (
                "end_date".into(),
                ColumnData::I32(rows.iter().map(|(_, q)| Some(ends[(*q).min(8)])).collect()),
            ),
            (
                "ann_date".into(),
                ColumnData::I32(
                    rows.iter()
                        .map(|(_, q)| Some(if *q == 9 { 20250602 } else { 20250501 }))
                        .collect(),
                ),
            ),
            (
                "f_ann_date".into(),
                ColumnData::I32(
                    rows.iter()
                        .map(|(_, q)| Some(if *q == 9 { 20250602 } else { 20250501 }))
                        .collect(),
                ),
            ),
            (
                "report_type".into(),
                ColumnData::I64(vec![
                    Some(if dataset == DatasetId::StockBalanceSheet {
                        1
                    } else {
                        2
                    });
                    rows.len()
                ]),
            ),
            (
                "update_flag".into(),
                ColumnData::I64(vec![Some(0); rows.len()]),
            ),
        ]);
        let mut fields = vec![];
        for output in ALL {
            let f = output.fields();
            fields.extend_from_slice(match dataset {
                DatasetId::StockIncome => f.0,
                DatasetId::StockBalanceSheet => f.1,
                _ => f.2,
            });
        }
        fields.sort();
        fields.dedup();
        for (j, field) in fields.into_iter().enumerate() {
            cols.insert(
                field.into(),
                ColumnData::F64(
                    rows.iter()
                        .map(|(s, q)| {
                            Some(
                                100.0
                                    + (j + 1) as f64 * (q + 1) as f64
                                    + (((s + 1) * (q + 2) * (j + 3)) % 17) as f64,
                            )
                        })
                        .collect(),
                ),
            );
        }
        Table::new(cols).unwrap()
    }

    #[test]
    fn oldest_balance_revision_and_instrument_aligned_cache() {
        let income =
            FinancialPitIndex::from_table(Arc::new(financial_table(DatasetId::StockIncome)))
                .unwrap();
        let balance =
            FinancialPitIndex::from_table(Arc::new(financial_table(DatasetId::StockBalanceSheet)))
                .unwrap();
        let readers = Readers {
            income: income.reader(ReportTypePreference::income_single_quarter()),
            balance: Some(balance.reader(ReportTypePreference::balance_sheet_consolidated())),
            cash: None,
            balance_quarters: 9,
        };
        let output = Output::RoeSurplusReserve;
        let before = readers.snapshot("000001.SZ", 20250502, &[output]).unwrap();
        let after = readers.snapshot("000001.SZ", 20250602, &[output]).unwrap();
        assert!(before[output as usize].is_some());
        assert_ne!(before, after);
        assert_ne!(
            readers.marker("000001.SZ", 20250502),
            readers.marker("000001.SZ", 20250602)
        );
        let mut cache = InstrumentAlignedSnapshotCache::default();
        for (date, codes) in [
            (20250502, vec!["000001.SZ", "000002.SZ", "430001.BJ"]),
            (20250602, vec!["000002.SZ", "430001.BJ", "000001.SZ"]),
            (20250502, vec!["430001.BJ", "000001.SZ", "000002.SZ"]),
        ] {
            let panel = DailyPanel::from_index(
                vec![date],
                codes.iter().map(|v| v.to_string()).collect(),
                &[date],
                vec![true; 3],
            )
            .unwrap();
            let result = cached_financial_stock_snapshots_for_date(
                &panel,
                date,
                &mut cache,
                |_, code, _| is_bj_stock(code),
                |date, code, _| readers.marker(code, date),
                |date, code, _| readers.snapshot(code, date, &[output]),
            );
            for (i, code) in codes.iter().enumerate() {
                if is_bj_stock(code) {
                    assert!(result[i].is_none());
                } else {
                    assert_eq!(result[i], readers.snapshot(code, date, &[output]));
                }
            }
        }
    }

    fn pool(dates: Vec<i32>) -> DataPool {
        let context = FactorContext {
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            start_date: dates[0],
            end_date: *dates.last().unwrap(),
            load_start_date: dates[0],
            load_dates: dates.clone(),
            target_dates: dates.clone(),
        };
        let codes: Vec<_> = (1..=8)
            .map(|s| Some(format!("{s:06}.SZ")))
            .chain([Some("430001.BJ".into())])
            .collect();
        let basic = Table::new(BTreeMap::from([
            ("ts_code".into(), ColumnData::Utf8(codes.clone())),
            ("list_date".into(), ColumnData::I32(vec![Some(20200101); 9])),
            ("delist_date".into(), ColumnData::I32(vec![None; 9])),
        ]))
        .unwrap();
        let sw = Table::new(BTreeMap::from([
            ("ts_code".into(), ColumnData::Utf8(codes.clone())),
            ("in_date".into(), ColumnData::I32(vec![Some(20200101); 9])),
            ("out_date".into(), ColumnData::I32(vec![None; 9])),
            (
                "l1_code".into(),
                ColumnData::Utf8(vec![Some("10".into()); 9]),
            ),
        ]))
        .unwrap();
        let barra = Table::new(BTreeMap::from([
            (
                "ts_code".into(),
                ColumnData::Utf8(
                    dates
                        .iter()
                        .flat_map(|_| codes.clone().into_iter().rev())
                        .collect(),
                ),
            ),
            (
                "trade_date".into(),
                ColumnData::I32(dates.iter().flat_map(|d| vec![Some(*d); 9]).collect()),
            ),
            (
                "SIZE".into(),
                ColumnData::F64(
                    dates
                        .iter()
                        .flat_map(|_| {
                            (0..9)
                                .rev()
                                .map(|i| if i == 7 { None } else { Some((i + 1) as f64) })
                        })
                        .collect(),
                ),
            ),
        ]))
        .unwrap();
        DataPool::from_daily_tables_for_test(
            HashMap::from([
                (DatasetId::StockBasic, basic),
                (DatasetId::StockSwClassification, sw),
                (DatasetId::StockBarraDaily, barra),
                (
                    DatasetId::StockIncome,
                    financial_table(DatasetId::StockIncome),
                ),
                (
                    DatasetId::StockBalanceSheet,
                    financial_table(DatasetId::StockBalanceSheet),
                ),
                (
                    DatasetId::StockCashFlow,
                    financial_table(DatasetId::StockCashFlow),
                ),
            ]),
            &context,
        )
        .unwrap()
    }

    #[test]
    fn requested_subsets_batches_and_daily_neutralization_without_pv() {
        let dates = vec![20250502, 20250505, 20250602];
        let data = pool(dates.clone());
        let all_ids: Vec<_> = ALL.iter().map(|o| o.id().to_string()).collect();
        let mut state = ComputeState::default();
        let all = compute_requested(&all_ids, &data, &mut state).unwrap();
        assert_eq!(all.len(), 9);
        for output in ALL {
            let id = output.id().to_string();
            let single = compute_requested(&[id.clone()], &data, &mut state).unwrap();
            assert_eq!(single.len(), 1);
            let expected = all.iter().find(|s| s.spec.id == id).unwrap();
            assert_eq!(
                single[0]
                    .values
                    .iter()
                    .map(|v| (&v.key, v.value))
                    .collect::<Vec<_>>(),
                expected
                    .values
                    .iter()
                    .map(|v| (&v.key, v.value))
                    .collect::<Vec<_>>()
            );
            assert!(single[0].values.iter().any(|v| v.value.is_some()), "{id}");
            for value in &single[0].values {
                if let crate::core::FactorRowKey::Daily { ts_code, .. } = &value.key {
                    if ts_code == "000008.SZ" || ts_code.ends_with(".BJ") {
                        assert!(value.value.is_none());
                    }
                }
            }
            for date in &dates {
                let batch =
                    compute_requested(&[id.clone()], &pool(vec![*date]), &mut state).unwrap();
                let expected: Vec<_> = single[0]
                    .values
                    .iter()
                    .filter(|v| v.key.trade_date() == *date)
                    .map(|v| (&v.key, v.value))
                    .collect();
                assert_eq!(
                    batch[0]
                        .values
                        .iter()
                        .map(|v| (&v.key, v.value))
                        .collect::<Vec<_>>(),
                    expected
                );
            }
        }
        for output in ALL {
            let spec = spec(output);
            assert!(
                spec.tags.contains(&"GDZQ".into()) && spec.tags.contains(&"fundamental".into())
            );
            assert!(!spec
                .dependencies
                .iter()
                .any(|d| d.dataset == DatasetId::StockDailyPv));
            assert_eq!(
                spec.dependencies
                    .iter()
                    .any(|d| d.dataset == DatasetId::StockCashFlow),
                !output.fields().2.is_empty()
            );
        }
    }
}

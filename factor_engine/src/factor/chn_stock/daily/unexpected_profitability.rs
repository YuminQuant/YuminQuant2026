use std::any::Any;
use std::collections::{BTreeMap, BTreeSet, HashMap};

use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorRowKey, FactorSeries, FactorSpec,
    FactorValue, Frequency, Lookback,
};
use crate::data::DataPool;
use crate::error::{err, Result};
use crate::factor::common::financial::previous_quarter_end_date;
use crate::factor::common::stock_daily_ops::{is_bj_stock, neutralize_size_sector_with_inputs};
use crate::factor::common::{
    routed_financial_stock_snapshots_for_date, FinancialStockEventRoute, ClassificationLevel, ClassificationMap,
    DividendReader, EventDrivenCrossSectionCache, FinancialEventMarker,
    FinancialEventMarkerBuilder, FinancialEventSchedule, FinancialPitReader,
    FinancialStatementDataset, InstrumentAlignedSnapshotCache, ReportTypePreference,
};
use crate::factor::{Factor, FactorUpdatePolicy};

const ID: &str = "unexpected_profitability";
const EPS: f64 = 1e-12;
const HISTORY: usize = 13;
const N_FEATURES: usize = 9;
const OI_FIELDS: &[&str] = &[
    "n_income",
    "non_oper_income",
    "non_oper_exp",
    "invest_income",
    "fv_value_chg_gain",
    "fin_exp",
    "int_income",
];
const OI_WEIGHTS: [f64; 7] = [1.0, -0.75, 0.75, -0.75, -0.75, 0.75, -0.75];
const BALANCE_FIELDS: &[&str] = &[
    "total_hldr_eqy_inc_min_int",
    "st_borr",
    "trading_fl",
    "notes_payable",
    "non_cur_liab_due_1y",
    "lt_borr",
    "bond_payable",
    "money_cap",
    "trad_asset",
    "fa_avail_for_sale",
    "htm_invest",
    "invest_real_estate",
    "time_deposits",
    "oth_assets",
    "lt_rec",
];

pub struct StockDailyUnexpectedProfitability;
pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyUnexpectedProfitability)
}

impl Factor for StockDailyUnexpectedProfitability {
    fn spec(&self) -> FactorSpec {
        let mut income = OI_FIELDS.to_vec();
        income.push("operate_profit");
        FactorSpec {
            id: ID.into(), aliases: vec!["UP".into(), "Unexpected Profitability".into()],
            name: "Unexpected Profitability".into(), asset_class: AssetClass::Stock,
            frequency: Frequency::Daily, version: "0.1.0".into(),
            tags: ["DFZQ", "fundamental", "financial", "pit", "profitability", "neutralize", "size", "sector", "daily"].into_iter().map(str::to_string).collect(),
            description: "Daily PIT approximation of UP. Quarterly OI=n_income-0.75*(non_oper_income-non_oper_exp+invest_income+fv_value_chg_gain)+0.75*(fin_exp-int_income); missing additive operands are zero unless all missing. RNOA=OI/average(begin,end NOA), positive NOA required. Nine lagged predictors; Q1 uses prior Q3. Cross-sectional OLS trained on previous-year same-quarter outcomes and features visible at the previous-year same calendar date, never current outcomes. Current features use daily latest PIT revisions; market cap uses lag-quarter last trading day. Annual dividends use implemented fiscal-year distributions known at each as-of date, not proposals. SW financials and BJ excluded; daily SW L1/SIZE neutralization. No winsorization or final zscore.".into(),
            dependencies: vec![
                DataRequest::financial_quarters(DatasetId::StockIncome, &income, 20),
                DataRequest::financial_quarters(DatasetId::StockBalanceSheet, BALANCE_FIELDS, 20),
                DataRequest::financial_quarters(DatasetId::StockCashFlow, &["n_cashflow_act"], 20),
                DataRequest::new(DatasetId::StockDailyBasic, &["total_mv"]),
                DataRequest::new(DatasetId::StockDividend, &["end_date", "ann_date", "div_proc", "cash_div_tax", "ex_date", "base_share"]),
                DataRequest::new(DatasetId::StockBarraDaily, &["SIZE"]),
                DataRequest::new(DatasetId::StockSwClassification, &["l1_code"]),
            ], intraday_raw_dependencies: vec![], lookback: Lookback { trading_days: 600 },
        }
    }
    fn update_policy(&self) -> FactorUpdatePolicy {
        FactorUpdatePolicy::FinancialEventStateDailyFast
    }
    fn initial_compute_state(&self, _: &[String]) -> Box<dyn Any + Send> {
        Box::new(State::default())
    }
    fn compute(&self, _: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        compute(data, &mut State::default())
    }
    fn compute_many_stateful(
        &self,
        ids: &[String],
        _: &FactorContext,
        data: &DataPool,
        state: &mut (dyn Any + Send),
    ) -> Result<Vec<FactorSeries>> {
        if !ids.iter().any(|id| id == ID) {
            return Ok(vec![]);
        }
        Ok(vec![compute(
            data,
            state
                .downcast_mut::<State>()
                .ok_or_else(|| err("UP state mismatch"))?,
        )?])
    }
}

#[derive(Default)]
struct State {
    current: InstrumentAlignedSnapshotCache<Snapshot>,
    historical: InstrumentAlignedSnapshotCache<Snapshot>,
    current_dividends: AnnualDividendCache,
    historical_dividends: AnnualDividendCache,
    models: BTreeMap<i32, CachedModel>,
    raw: EventDrivenCrossSectionCache,
}

#[derive(Default)]
struct AnnualDividendCache {
    asof: Option<i32>,
    sums: BTreeMap<i32, HashMap<String, f64>>,
}
impl AnnualDividendCache {
    fn refresh(
        &mut self,
        reader: &DividendReader<'_>,
        schedule: &FinancialEventSchedule,
        asof: i32,
        validate_batch: bool,
    ) -> bool {
        let refresh = validate_batch
            || self
                .asof
                .is_none_or(|last| asof < last || asof / 10000 != last / 10000)
            || schedule.has_event_after_until(self.asof, asof);
        if refresh {
            self.sums = reader
                .implemented_annual_sums_by_stock(asof / 10000 - 4, asof / 10000, asof)
                .into_iter()
                .map(|(year, sums)| {
                    (
                        year,
                        sums.into_iter()
                            .map(|(code, amount)| (code.to_string(), amount))
                            .collect(),
                    )
                })
                .collect();
        }
        self.asof = Some(asof);
        refresh
    }
}

struct CachedModel {
    rows: Vec<(f64, [f64; N_FEATURES])>,
    model: Option<Model>,
}
#[derive(Clone, Copy, Default)]
struct Quarter {
    end: i32,
    equity: Option<f64>,
    noa: Option<f64>,
    oi: Option<f64>,
    acc: Option<f64>,
}
#[derive(Clone)]
struct Snapshot {
    quarters: [Quarter; HISTORY],
}
impl Snapshot {
    fn quarter(&self, end: i32) -> Option<&Quarter> {
        self.quarters.iter().find(|q| q.end == end)
    }
    fn rnoa(&self, end: i32) -> Option<f64> {
        let now = self.quarter(end)?;
        let prev = self.quarter(previous_quarter_end_date(end)?)?;
        let a = positive(now.noa)?;
        let b = positive(prev.noa)?;
        finite(Some(now.oi? / (0.5 * a + 0.5 * b)))
    }
    fn acc_ttm(&self, end: i32) -> Option<f64> {
        let mut sum = 0.0;
        let mut end = end;
        for _ in 0..4 {
            sum += self.quarter(end)?.acc?;
            end = previous_quarter_end_date(end)?;
        }
        finite(Some(sum))
    }
    fn annual_end(&self, lag: i32) -> Option<i32> {
        self.quarters
            .iter()
            .find(|q| q.end <= lag && q.end % 10000 == 1231 && q.noa.is_some())
            .map(|q| q.end)
    }
    fn features(
        &self,
        end: i32,
        market_cap_yuan: f64,
        dividend_yuan: f64,
        annual: i32,
    ) -> Option<[f64; N_FEATURES]> {
        let lag = lag_quarter(end)?;
        let q = self.quarter(lag)?;
        let noa = positive(q.noa)?;
        let equity = positive(q.equity)?;
        let last_noa = positive(self.quarter(lag - 10000)?.noa)?;
        let seasonal = self.rnoa(end - 10000)?;
        let x = [
            (equity / positive(Some(market_cap_yuan))?).ln(),
            market_cap_yuan.ln(),
            (noa - last_noa) / last_noa,
            seasonal,
            seasonal - self.rnoa(end - 20000)?,
            self.rnoa(lag)? - self.rnoa(lag - 10000)?,
            self.acc_ttm(lag)? / noa,
            if dividend_yuan > 0.0 { 0.0 } else { 1.0 },
            dividend_yuan / positive(self.quarter(annual)?.noa)?,
        ];
        x.iter().all(|v| v.is_finite()).then_some(x)
    }
}

struct Readers<'a> {
    income: FinancialPitReader<'a>,
    balance: FinancialPitReader<'a>,
    cash: FinancialPitReader<'a>,
}
impl Readers<'_> {
    fn marker(&self, code: &str, date: i32) -> Option<FinancialEventMarker> {
        let mut end = self.income.latest_quarter_end_date(code, date)?;
        let mut marker = FinancialEventMarkerBuilder::new();
        for _ in 0..HISTORY {
            for (dataset, reader) in [
                (FinancialStatementDataset::Income, &self.income),
                (FinancialStatementDataset::BalanceSheet, &self.balance),
                (FinancialStatementDataset::CashFlow, &self.cash),
            ] {
                marker.include_reader_record_for_end_date(dataset, reader, code, date, end);
            }
            end = previous_quarter_end_date(end)?;
        }
        marker.build()
    }
    fn snapshot(&self, code: &str, date: i32) -> Option<Snapshot> {
        let mut end = self.income.latest_quarter_end_date(code, date)?;
        let mut quarters = [Quarter::default(); HISTORY];
        for q in &mut quarters {
            let income = self.income.record_for_end_date(code, date, end);
            let balance = self.balance.record_for_end_date(code, date, end);
            let cash = self.cash.record_for_end_date(code, date, end);
            let oi = additive(
                OI_FIELDS
                    .iter()
                    .zip(OI_WEIGHTS)
                    .map(|(field, weight)| (income.and_then(|r| r.column(field)), weight)),
            );
            let noa = additive(BALANCE_FIELDS.iter().enumerate().map(|(i, field)| {
                (
                    balance.and_then(|r| r.column(field)),
                    if i < 7 { 1.0 } else { -1.0 },
                )
            }));
            *q = Quarter {
                end,
                oi,
                noa,
                equity: balance.and_then(|r| finite(r.column(BALANCE_FIELDS[0]))),
                acc: additive([
                    (income.and_then(|r| r.column("operate_profit")), 1.0),
                    (cash.and_then(|r| r.column("n_cashflow_act")), -1.0),
                ]),
            };
            end = previous_quarter_end_date(end)?;
        }
        Some(Snapshot { quarters })
    }
}

fn lag_quarter(end: i32) -> Option<i32> {
    if end % 10000 == 331 {
        Some((end / 10000 - 1) * 10000 + 930)
    } else {
        previous_quarter_end_date(end)
    }
}
fn prior_year(date: i32) -> i32 {
    if date % 10000 == 229 {
        date - 10001
    } else {
        date - 10000
    }
}
fn finite(value: Option<f64>) -> Option<f64> {
    value.filter(|v| v.is_finite())
}
fn positive(value: Option<f64>) -> Option<f64> {
    finite(value).filter(|v| *v > EPS)
}
fn additive(items: impl IntoIterator<Item = (Option<f64>, f64)>) -> Option<f64> {
    let mut sum = 0.0;
    let mut found = false;
    for (value, weight) in items {
        if let Some(value) = finite(value) {
            sum += value * weight;
            found = true;
        }
    }
    finite(found.then_some(sum))
}
fn eligible(sector: &ClassificationMap, date: i32, code: &str) -> bool {
    !is_bj_stock(code)
        && sector.group_for(date, code).is_some_and(|s| {
            !matches!(
                s.split('.').next().unwrap_or(s),
                "801780" | "801790" | "801190"
            )
        })
}

// Centered, scaled OLS with re-orthogonalized QR. Constant predictors are removed;
// nonconstant collinearity yields null instead of unstable extrapolation.
#[derive(Clone, Copy)]
struct Model {
    mean: [f64; N_FEATURES],
    scale: [f64; N_FEATURES],
    beta: [f64; N_FEATURES],
    intercept: f64,
}
impl Model {
    fn fit(rows: &[(f64, [f64; N_FEATURES])]) -> Option<Self> {
        if rows.len() <= N_FEATURES + 1
            || rows
                .iter()
                .any(|(y, x)| !y.is_finite() || x.iter().any(|v| !v.is_finite()))
        {
            return None;
        }
        let n = rows.len() as f64;
        let intercept = rows.iter().map(|r| r.0).sum::<f64>() / n;
        let mean = std::array::from_fn(|j| rows.iter().map(|r| r.1[j]).sum::<f64>() / n);
        let scale = std::array::from_fn(|j| {
            (rows.iter().map(|r| (r.1[j] - mean[j]).powi(2)).sum::<f64>() / n).sqrt()
        });
        let active: Vec<_> = (0..N_FEATURES).filter(|&j| scale[j] > EPS).collect();
        let mut basis: Vec<Vec<f64>> = Vec::new();
        let mut r = [[0.0; N_FEATURES]; N_FEATURES];
        let mut rhs = [0.0; N_FEATURES];
        for (k, &j) in active.iter().enumerate() {
            let mut col: Vec<_> = rows
                .iter()
                .map(|row| (row.1[j] - mean[j]) / scale[j])
                .collect();
            for _ in 0..2 {
                for (i, q) in basis.iter().enumerate() {
                    let projection: f64 = col.iter().zip(q).map(|(a, b)| a * b).sum();
                    r[i][k] += projection;
                    for (a, b) in col.iter_mut().zip(q) {
                        *a -= projection * b;
                    }
                }
            }
            r[k][k] = col.iter().map(|v| v * v).sum::<f64>().sqrt();
            if r[k][k] <= 1e-8 * n.sqrt() {
                return None;
            }
            for v in &mut col {
                *v /= r[k][k];
            }
            rhs[k] = col
                .iter()
                .zip(rows)
                .map(|(q, row)| q * (row.0 - intercept))
                .sum();
            basis.push(col);
        }
        let mut coefficients = [0.0; N_FEATURES];
        for k in (0..active.len()).rev() {
            coefficients[k] = (rhs[k]
                - ((k + 1)..active.len())
                    .map(|j| r[k][j] * coefficients[j])
                    .sum::<f64>())
                / r[k][k];
        }
        let mut beta = [0.0; N_FEATURES];
        for (k, &j) in active.iter().enumerate() {
            beta[j] = coefficients[k];
        }
        beta.iter().all(|v| v.is_finite()).then_some(Self {
            mean,
            scale,
            beta,
            intercept,
        })
    }
    fn predict(&self, x: [f64; N_FEATURES]) -> Option<f64> {
        finite(Some(
            self.intercept
                + (0..N_FEATURES)
                    .filter(|&j| self.scale[j] > EPS)
                    .map(|j| self.beta[j] * (x[j] - self.mean[j]) / self.scale[j])
                    .sum::<f64>(),
        ))
    }
}

fn compute(data: &DataPool, state: &mut State) -> Result<FactorSeries> {
    let panel = data.stock_universe_panel()?;
    let readers = Readers {
        income: data.financial_reader(
            DatasetId::StockIncome,
            ReportTypePreference::income_single_quarter(),
        )?,
        balance: data.financial_reader(
            DatasetId::StockBalanceSheet,
            ReportTypePreference::balance_sheet_consolidated(),
        )?,
        cash: data.financial_reader(
            DatasetId::StockCashFlow,
            ReportTypePreference::income_single_quarter(),
        )?,
    };
    let sector = ClassificationMap::from_table(
        data.daily(DatasetId::StockSwClassification)?,
        ClassificationLevel::Sector,
    )?;
    let dividend = data.dividend_reader()?;
    let dividend_schedule = FinancialEventSchedule::from_annual_dividend_reader(&dividend);
    let mv = panel.column_from_table(data.daily(DatasetId::StockDailyBasic)?, "total_mv")?;
    let n = panel.instruments().len();
    let dates = panel.dates();
    let mut values = vec![None; panel.shape_len()];
    let schedule = FinancialEventSchedule::from_pit_readers(&[
        readers.income.clone(),
        readers.balance.clone(),
        readers.cash.clone(),
    ]);
    let mut last_date = None;
    let mut last_historical_date = None;
    let mut current = vec![None; n];
    let mut past = vec![None; n];
    let mut last_eligibility = None;
    let lookup: BTreeMap<_, _> = panel
        .instruments()
        .iter()
        .enumerate()
        .map(|(i, code)| (code.as_str(), i))
        .collect();
    for (day, &date) in dates.iter().enumerate() {
        if !panel.is_target_date(date) {
            continue;
        }
        let historical_date = prior_year(date);
        // Validate the first cross-section of each newly loaded batch, including
        // its lagged market caps and instrument set. Never carry positional data across batches.
        let first_in_batch = last_date.is_none();
        let current_changed = first_in_batch || schedule.has_event_after_until(last_date, date);
        let historical_changed = last_historical_date.is_none()
            || schedule.has_event_after_until(last_historical_date, historical_date);
        if current_changed {
            current = routed_financial_stock_snapshots_for_date(
                &panel,
                date,
                &mut state.current,
                &FinancialStockEventRoute::new(&[&readers.income, &readers.balance, &readers.cash]),
                |_, code, _| is_bj_stock(code),
                |_, code, _| readers.marker(code, date),
                |_, code, _| readers.snapshot(code, date),
            );
        }
        if historical_changed {
            past = routed_financial_stock_snapshots_for_date(
                &panel,
                date,
                &mut state.historical,
                &FinancialStockEventRoute::new(&[&readers.income, &readers.balance, &readers.cash]).at_date(historical_date),
                |_, code, _| is_bj_stock(code),
                |_, code, _| readers.marker(code, historical_date),
                |_, code, _| readers.snapshot(code, historical_date),
            );
        }
        last_date = Some(date);
        last_historical_date = Some(historical_date);
        let current_div_changed =
            state
                .current_dividends
                .refresh(&dividend, &dividend_schedule, date, first_in_batch);
        let historical_div_changed = state.historical_dividends.refresh(
            &dividend,
            &dividend_schedule,
            historical_date,
            first_in_batch,
        );
        let historical_day = dates
            .partition_point(|d| *d <= historical_date)
            .checked_sub(1);
        let eligibility: Vec<_> = panel
            .instruments()
            .iter()
            .enumerate()
            .map(|(i, code)| {
                [
                    panel.is_present_offset(day * n + i) && eligible(&sector, date, code),
                    historical_day.is_some_and(|d| panel.is_present_offset(d * n + i))
                        && eligible(&sector, historical_date, code),
                ]
            })
            .collect();
        let refresh_raw = current_changed
            || historical_changed
            || current_div_changed
            || historical_div_changed
            || last_eligibility.as_ref() != Some(&eligibility);
        last_eligibility = Some(eligibility);
        if !refresh_raw {
            let replay =
                state
                    .raw
                    .replay_series(StockDailyUnexpectedProfitability.spec(), &panel, date);
            for item in replay.values {
                if let FactorRowKey::Daily { ts_code, .. } = item.key {
                    if let Some(&i) = lookup.get(ts_code.as_str()) {
                        values[day * n + i] = item.value;
                    }
                }
            }
            state.raw.mark_processed(date);
            continue;
        }
        let targets: BTreeSet<_> = current
            .iter()
            .enumerate()
            .filter(|(i, _)| panel.is_present_offset(day * n + i))
            .filter_map(|(_, s)| s.as_ref().map(|s| s.quarters[0].end))
            .collect();
        state
            .models
            .retain(|end, _| targets.contains(&(end + 10000)));
        let current_annual = &state.current_dividends.sums;
        let historical_annual = &state.historical_dividends.sums;
        let features =
            |snapshot: &Snapshot, end: i32, asof: i32, i: usize| -> Option<[f64; N_FEATURES]> {
                let lag = lag_quarter(end)?;
                let market_day = dates.partition_point(|d| *d <= lag).checked_sub(1)?;
                // Do not silently substitute the first loaded day for missing historical data.
                if lag - dates[market_day] > 40 {
                    return None;
                }
                let cap = positive(mv.values()[market_day * n + i])? * 10000.0;
                let fiscal_end = snapshot.annual_end(lag)?;
                let annual = if asof == date {
                    current_annual
                } else {
                    historical_annual
                };
                let amount = annual
                    .get(&(fiscal_end / 10000))?
                    .get(panel.instruments()[i].as_str())
                    .copied()
                    .unwrap_or(0.0)
                    * 10000.0;
                snapshot.features(end, cap, amount, fiscal_end)
            };
        for end in targets {
            let train_end = end - 10000;
            let rows: Vec<_> = past
                .iter()
                .enumerate()
                .filter_map(|(i, s)| {
                    let code = &panel.instruments()[i];
                    if !panel.is_present_offset(historical_day? * n + i)
                        || !eligible(&sector, historical_date, code)
                    {
                        return None;
                    }
                    let s = s.as_ref()?;
                    Some((
                        s.rnoa(train_end)?,
                        features(s, train_end, historical_date, i)?,
                    ))
                })
                .collect();
            // Current-period disclosures need not change last year's training
            // cross-section. Refit only when its actual ordered numeric inputs change.
            if state
                .models
                .get(&train_end)
                .is_none_or(|cached| cached.rows != rows)
            {
                let model = Model::fit(&rows);
                state.models.insert(train_end, CachedModel { rows, model });
            }
            let Some(model) = state.models[&train_end].model else {
                continue;
            };
            for (i, s) in current.iter().enumerate() {
                if !panel.is_present_offset(day * n + i)
                    || !eligible(&sector, date, &panel.instruments()[i])
                {
                    continue;
                }
                values[day * n + i] = s
                    .as_ref()
                    .filter(|s| s.quarters[0].end == end)
                    .and_then(|s| {
                        finite(Some(
                            s.rnoa(end)? - model.predict(features(s, end, date, i)?)?,
                        ))
                    })
                    .or(values[day * n + i]);
            }
        }
        let raw_day = FactorSeries {
            spec: StockDailyUnexpectedProfitability.spec(),
            values: panel
                .instruments()
                .iter()
                .enumerate()
                .filter(|(i, _)| panel.is_present_offset(day * n + i))
                .map(|(i, code)| FactorValue {
                    key: FactorRowKey::Daily {
                        trade_date: date,
                        ts_code: code.clone(),
                    },
                    value: values[day * n + i],
                })
                .collect(),
        };
        state.raw.update_series(&raw_day, &panel);
        state.raw.mark_processed(date);
    }
    let size = panel.column_from_table(data.daily(DatasetId::StockBarraDaily)?, "SIZE")?;
    let raw = panel.column_from_values(values)?;
    Ok(
        neutralize_size_sector_with_inputs(&raw, &panel, &size, &sector)?
            .to_factor_series(StockDailyUnexpectedProfitability.spec()),
    )
}

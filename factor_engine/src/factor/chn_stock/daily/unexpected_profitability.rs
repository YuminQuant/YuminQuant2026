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
    cached_financial_stock_snapshots_for_date, ClassificationLevel, ClassificationMap,
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
    #[cfg(test)]
    raw_updates: usize,
    #[cfg(test)]
    model_fits: usize,
    #[cfg(test)]
    disable_caches: bool,
}

#[derive(Default)]
struct AnnualDividendCache {
    asof: Option<i32>,
    sums: BTreeMap<i32, HashMap<String, f64>>,
    #[cfg(test)]
    rebuilds: usize,
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
            #[cfg(test)]
            {
                self.rebuilds += 1;
            }
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
        #[cfg(test)]
        let first_in_batch = first_in_batch || state.disable_caches;
        let current_changed = first_in_batch || schedule.has_event_after_until(last_date, date);
        let historical_changed = last_historical_date.is_none()
            || schedule.has_event_after_until(last_historical_date, historical_date);
        if current_changed {
            current = cached_financial_stock_snapshots_for_date(
                &panel,
                date,
                &mut state.current,
                |_, code, _| is_bj_stock(code),
                |_, code, _| readers.marker(code, date),
                |_, code, _| readers.snapshot(code, date),
            );
        }
        if historical_changed {
            past = cached_financial_stock_snapshots_for_date(
                &panel,
                date,
                &mut state.historical,
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
        #[cfg(test)]
        {
            state.raw_updates += 1;
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
            #[cfg(test)]
            if state.disable_caches {
                state.models.remove(&train_end);
            }
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
                #[cfg(test)]
                {
                    state.model_fits += 1;
                }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{ColumnData, Table};
    use std::collections::HashMap;

    fn fixture(targets: Vec<i32>, future_profit: f64) -> DataPool {
        fixture_custom(targets, future_profit, |_| {})
    }
    fn fixture_custom(
        targets: Vec<i32>,
        future_profit: f64,
        edit: impl FnOnce(&mut HashMap<DatasetId, Table>),
    ) -> DataPool {
        let dates = vec![
            20230929, 20240506, 20240507, 20240602, 20240930, 20250506, 20250507, 20250602,
        ];
        let codes: Vec<_> = (1..=80)
            .map(|s| {
                Some(if s == 79 {
                    "430001.BJ".into()
                } else {
                    format!("{s:06}.SZ")
                })
            })
            .collect();
        let basic = Table::new(BTreeMap::from([
            ("ts_code".into(), ColumnData::Utf8(codes.clone())),
            (
                "list_date".into(),
                ColumnData::I32(vec![Some(20100101); 80]),
            ),
            ("delist_date".into(), ColumnData::I32(vec![None; 80])),
        ]))
        .unwrap();
        let sector = Table::new(BTreeMap::from([
            ("ts_code".into(), ColumnData::Utf8(codes.clone())),
            ("in_date".into(), ColumnData::I32(vec![Some(20100101); 80])),
            ("out_date".into(), ColumnData::I32(vec![None; 80])),
            (
                "l1_code".into(),
                ColumnData::Utf8(
                    (0..80)
                        .map(|i| Some(if i == 77 { "801780" } else { "801010" }.into()))
                        .collect(),
                ),
            ),
        ]))
        .unwrap();
        let daily = |field: &str| {
            Table::new(BTreeMap::from([
                (
                    "ts_code".into(),
                    ColumnData::Utf8(
                        dates
                            .iter()
                            .flat_map(|_| codes.iter().rev().cloned())
                            .collect(),
                    ),
                ),
                (
                    "trade_date".into(),
                    ColumnData::I32(dates.iter().flat_map(|d| vec![Some(*d); 80]).collect()),
                ),
                (
                    field.into(),
                    ColumnData::F64(
                        dates
                            .iter()
                            .flat_map(|_| {
                                (0..80).rev().map(|i| {
                                    if field == "SIZE" {
                                        (i != 79).then_some((i + 1) as f64)
                                    } else {
                                        Some(5000.0 + (i * i + 7 * i) as f64)
                                    }
                                })
                            })
                            .collect(),
                    ),
                ),
            ]))
            .unwrap()
        };
        let mut ends = vec![20250331];
        for _ in 1..24 {
            ends.push(previous_quarter_end_date(*ends.last().unwrap()).unwrap());
        }
        let statement = |dataset| {
            let rows: Vec<_> = (0..80)
                .flat_map(|s| (0..24).map(move |q| (s, q, false)))
                .chain([(0, 0, true)])
                .collect();
            let announcements: Vec<_> = rows
                .iter()
                .map(|(_, q, revision)| {
                    Some(if *revision {
                        20250602
                    } else {
                        let end = ends[*q];
                        let year = end / 10000;
                        match end % 10000 {
                            331 => year * 10000 + 420,
                            630 => year * 10000 + 720,
                            930 => year * 10000 + 1020,
                            _ => (year + 1) * 10000 + 320,
                        }
                    })
                })
                .collect();
            let mut columns = BTreeMap::from([
                (
                    "ts_code".into(),
                    ColumnData::Utf8(rows.iter().map(|(s, _, _)| codes[*s].clone()).collect()),
                ),
                (
                    "end_date".into(),
                    ColumnData::I32(rows.iter().map(|(_, q, _)| Some(ends[*q])).collect()),
                ),
                ("ann_date".into(), ColumnData::I32(announcements.clone())),
                ("f_ann_date".into(), ColumnData::I32(announcements)),
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
                    ColumnData::I64(rows.iter().map(|(_, _, r)| Some(i64::from(*r))).collect()),
                ),
            ]);
            let fields = match dataset {
                DatasetId::StockIncome => {
                    let mut fields = OI_FIELDS.to_vec();
                    fields.push("operate_profit");
                    fields
                }
                DatasetId::StockBalanceSheet => BALANCE_FIELDS.to_vec(),
                _ => vec!["n_cashflow_act"],
            };
            for (j, field) in fields.iter().enumerate() {
                columns.insert(
                    (*field).into(),
                    ColumnData::F64(
                        rows.iter()
                            .map(|(s, q, revision)| {
                                let noise =
                                    ((((s + 1) * 7919 + (q + 1) * 104729 + (j + 1) * 15485863)
                                        as f64)
                                        .sin()
                                        * 43758.5453)
                                        .fract()
                                        .abs();
                                Some(if *revision && *field == "n_income" {
                                    future_profit
                                } else if *field == "total_hldr_eqy_inc_min_int" {
                                    1000.0 + noise * 1000.0
                                } else {
                                    5.0 + noise * 50.0
                                })
                            })
                            .collect(),
                    ),
                );
            }
            Table::new(columns).unwrap()
        };
        let dividend = Table::new(BTreeMap::from([
            ("ts_code".into(), ColumnData::Utf8(vec![])),
            ("end_date".into(), ColumnData::I32(vec![])),
            ("ann_date".into(), ColumnData::I32(vec![])),
            ("div_proc".into(), ColumnData::Utf8(vec![])),
            ("cash_div_tax".into(), ColumnData::F64(vec![])),
            ("ex_date".into(), ColumnData::I32(vec![])),
            ("base_share".into(), ColumnData::F64(vec![])),
        ]))
        .unwrap();
        let context = FactorContext {
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            start_date: targets[0],
            end_date: *targets.last().unwrap(),
            load_start_date: dates[0],
            load_dates: dates.clone(),
            target_dates: targets,
        };
        let mut tables = HashMap::from([
            (DatasetId::StockBasic, basic),
            (DatasetId::StockSwClassification, sector),
            (DatasetId::StockBarraDaily, daily("SIZE")),
            (DatasetId::StockDailyBasic, daily("total_mv")),
            (DatasetId::StockIncome, statement(DatasetId::StockIncome)),
            (
                DatasetId::StockBalanceSheet,
                statement(DatasetId::StockBalanceSheet),
            ),
            (
                DatasetId::StockCashFlow,
                statement(DatasetId::StockCashFlow),
            ),
            (DatasetId::StockDividend, dividend),
        ]);
        edit(&mut tables);
        DataPool::from_daily_tables_for_test(tables, &context).unwrap()
    }

    #[test]
    fn unexpected_profitability_pit_batches_alignment_and_neutralization() {
        let dates = vec![20250506, 20250507, 20250602];
        let all = compute(&fixture(dates.clone(), 1000.0), &mut State::default()).unwrap();
        assert_eq!(all.values.len(), 240);
        assert!(all.values.iter().any(|v| v.value.is_some()));
        for v in &all.values {
            if let crate::core::FactorRowKey::Daily { ts_code, .. } = &v.key {
                if ["000078.SZ", "430001.BJ", "000080.SZ"].contains(&ts_code.as_str()) {
                    assert!(v.value.is_none());
                }
            }
        }
        let mut state = State::default();
        for date in dates {
            let batch = compute(&fixture(vec![date], 1000.0), &mut state).unwrap();
            let expected: Vec<_> = all
                .values
                .iter()
                .filter(|v| v.key.trade_date() == date)
                .map(|v| (&v.key, v.value))
                .collect();
            assert_eq!(
                expected,
                batch
                    .values
                    .iter()
                    .map(|v| (&v.key, v.value))
                    .collect::<Vec<_>>()
            );
        }
        let changed = compute(
            &fixture(vec![20250506, 20250507, 20250602], 100000.0),
            &mut State::default(),
        )
        .unwrap();
        for (a, b) in all
            .values
            .iter()
            .zip(&changed.values)
            .filter(|(a, _)| a.key.trade_date() < 20250602)
        {
            assert_eq!(a.value, b.value);
        }
        assert!(all
            .values
            .iter()
            .zip(&changed.values)
            .any(|(a, b)| a.key.trade_date() == 20250602 && a.value != b.value));
        // No historical date is emitted and no PV anchor is supplied to the fixture.
        assert!(all.values.iter().all(|v| v.key.trade_date() >= 20250506));
    }
    #[test]
    fn unexpected_profitability_event_cache_matches_daily_refits() {
        let data = fixture(vec![20250506, 20250507, 20250602], 1000.0);
        let mut cached = State::default();
        let result = compute(&data, &mut cached).unwrap();
        let mut uncached = State {
            disable_caches: true,
            ..State::default()
        };
        let reference = compute(&data, &mut uncached).unwrap();
        assert_same(&result, &reference);
        assert_eq!(cached.raw_updates, 2);
        assert_eq!(cached.model_fits, 1);
        assert_eq!(cached.current_dividends.rebuilds, 1);
        assert_eq!(cached.historical_dividends.rebuilds, 1);
        assert_eq!(uncached.raw_updates, 3);
        assert_eq!(uncached.model_fits, 3);

        let mut batches = State::default();
        for date in [20250506, 20250507, 20250602] {
            compute(&fixture(vec![date], 1000.0), &mut batches).unwrap();
        }
        assert_eq!(batches.model_fits, 1);
        let changed_panel = fixture_custom(vec![20250602], 1000.0, |tables| {
            let basic = tables.get_mut(&DatasetId::StockBasic).unwrap();
            *basic = basic.take(&(1..80).rev().collect::<Vec<_>>()).unwrap();
            let mv = tables.get_mut(&DatasetId::StockDailyBasic).unwrap();
            if let ColumnData::F64(values) = mv.columns.get_mut("total_mv").unwrap() {
                for value in values {
                    *value = value.map(|v| v * 1.1);
                }
            }
        });
        let changed = compute(&changed_panel, &mut batches).unwrap();
        assert_same(
            &changed,
            &compute(&changed_panel, &mut State::default()).unwrap(),
        );
        assert_eq!(batches.model_fits, 2);
    }

    fn assert_same(a: &FactorSeries, b: &FactorSeries) {
        assert_eq!(a.values.len(), b.values.len());
        for (a, b) in a.values.iter().zip(&b.values) {
            assert_eq!(a.key, b.key);
            assert_eq!(a.value.map(f64::to_bits), b.value.map(f64::to_bits));
        }
    }

    #[test]
    fn unexpected_profitability_daily_size_and_eligibility_are_not_frozen() {
        for change_membership in [false, true] {
            let data = fixture_custom(vec![20250506, 20250507], 1000.0, |tables| {
                let size = tables.get_mut(&DatasetId::StockBarraDaily).unwrap();
                let dates = size.required_i32("trade_date").unwrap().clone();
                if let ColumnData::F64(values) = size.columns.get_mut("SIZE").unwrap() {
                    for (i, value) in values.iter_mut().enumerate() {
                        if dates[i] == Some(20250507) {
                            *value = value.map(|v| v + (i as f64).sin() * 20.0);
                        }
                    }
                }
                if change_membership {
                    tables
                        .get_mut(&DatasetId::StockSwClassification)
                        .unwrap()
                        .columns
                        .insert("out_date".into(), ColumnData::I32(vec![None; 80]));
                    if let ColumnData::I32(dates) = tables
                        .get_mut(&DatasetId::StockSwClassification)
                        .unwrap()
                        .columns
                        .get_mut("out_date")
                        .unwrap()
                    {
                        dates[0] = Some(20250506);
                    }
                    if let ColumnData::I32(dates) = tables
                        .get_mut(&DatasetId::StockBasic)
                        .unwrap()
                        .columns
                        .get_mut("delist_date")
                        .unwrap()
                    {
                        dates[1] = Some(20250506);
                    }
                }
            });
            let mut state = State::default();
            let result = compute(&data, &mut state).unwrap();
            let reference = compute(
                &data,
                &mut State {
                    disable_caches: true,
                    ..State::default()
                },
            )
            .unwrap();
            assert_same(&result, &reference);
            assert_eq!(state.raw_updates, if change_membership { 2 } else { 1 });
            let first: Vec<_> = result
                .values
                .iter()
                .filter(|v| v.key.trade_date() == 20250506)
                .map(|v| v.value)
                .collect();
            let next: Vec<_> = result
                .values
                .iter()
                .filter(|v| v.key.trade_date() == 20250507)
                .map(|v| v.value)
                .collect();
            assert_ne!(first, next);
        }
    }

    fn test_dividends() -> Table {
        Table::new(BTreeMap::from([
            (
                "ts_code".into(),
                ColumnData::Utf8(vec![Some("000001.SZ".into()), Some("000002.SZ".into())]),
            ),
            (
                "end_date".into(),
                ColumnData::I32(vec![Some(20221231), Some(20231231)]),
            ),
            (
                "ann_date".into(),
                ColumnData::I32(vec![Some(20240506), Some(20250506)]),
            ),
            (
                "ex_date".into(),
                ColumnData::I32(vec![Some(20240507), Some(20250507)]),
            ),
            (
                "div_proc".into(),
                ColumnData::Utf8(vec![Some("\u{5b9e}\u{65bd}".into()); 2]),
            ),
            (
                "cash_div_tax".into(),
                ColumnData::F64(vec![Some(0.2), Some(0.3)]),
            ),
            ("base_share".into(), ColumnData::F64(vec![Some(100.0); 2])),
        ]))
        .unwrap()
    }

    #[test]
    fn unexpected_profitability_dividend_events_and_year_rollover() {
        let index =
            crate::factor::common::DividendIndex::from_table(std::sync::Arc::new(test_dividends()))
                .unwrap();
        let reader = index.reader();
        let schedule = FinancialEventSchedule::from_annual_dividend_reader(&reader);
        let mut cache = AnnualDividendCache::default();
        for (date, rebuild) in [
            (20250506, true),
            (20250507, true),
            (20250508, false),
            (20260102, true),
            (20240506, true),
        ] {
            assert_eq!(cache.refresh(&reader, &schedule, date, false), rebuild);
            for (year, sums) in &cache.sums {
                let reference: HashMap<String, f64> = reader
                    .implemented_annual_sum_by_stock(*year, date)
                    .into_iter()
                    .map(|(k, v)| (k.to_string(), v))
                    .collect();
                assert_eq!(sums, &reference);
            }
        }
        let data = fixture_custom(vec![20250506, 20250507], 1000.0, |tables| {
            tables.insert(DatasetId::StockDividend, test_dividends());
        });
        let mut state = State::default();
        let result = compute(&data, &mut state).unwrap();
        assert_same(
            &result,
            &compute(
                &data,
                &mut State {
                    disable_caches: true,
                    ..State::default()
                },
            )
            .unwrap(),
        );
        assert_eq!(state.raw_updates, 2);
        assert_eq!(state.current_dividends.rebuilds, 2);
        assert_eq!(state.historical_dividends.rebuilds, 2);
        assert_eq!(state.model_fits, 2);
    }

    #[test]
    fn unexpected_profitability_additive_missing_policy() {
        assert_eq!(additive([(None, 1.0), (None, -1.0)]), None);
        assert_eq!(additive([(Some(4.0), 1.0), (None, -1.0)]), Some(4.0));
        assert_eq!(
            additive([(Some(f64::NAN), 1.0), (Some(3.0), -1.0)]),
            Some(-3.0)
        );
        let vals = [100.0, 10.0, 2.0, 3.0, 1.0, 8.0, 2.0];
        assert_eq!(
            additive(vals.into_iter().zip(OI_WEIGHTS).map(|(v, w)| (Some(v), w))),
            Some(95.5)
        );
    }
    #[test]
    fn unexpected_profitability_periods_and_metadata() {
        assert_eq!(lag_quarter(20250331), Some(20240930));
        assert_eq!(lag_quarter(20250630), Some(20250331));
        assert_eq!(prior_year(20240229), 20230228);
        let spec = StockDailyUnexpectedProfitability.spec();
        assert!(spec.tags.contains(&"DFZQ".into()));
        assert!(spec.tags.contains(&"fundamental".into()));
        assert!(!spec
            .dependencies
            .iter()
            .any(|r| r.dataset == DatasetId::StockDailyPv));
    }
    #[test]
    fn unexpected_profitability_regression_and_degeneracy() {
        let rows: Vec<_> = (0..80)
            .map(|i| {
                let x = std::array::from_fn(|j| {
                    if j == 7 {
                        1.0
                    } else {
                        ((i as f64 + 1.0) * (j as f64 + 1.0)).sin()
                    }
                });
                (0.3 + 0.2 * x[0] - 0.1 * x[3], x)
            })
            .collect();
        let model = Model::fit(&rows).unwrap();
        for (y, x) in &rows {
            assert!((model.predict(*x).unwrap() - y).abs() < 1e-10);
        }
        let mut reversed = rows.clone();
        reversed.reverse();
        let other = Model::fit(&reversed).unwrap();
        assert!(
            (model.predict(rows[0].1).unwrap() - other.predict(rows[0].1).unwrap()).abs() < 1e-10
        );
        assert!(Model::fit(&rows[..10]).is_none());
        let collinear: Vec<_> = rows
            .iter()
            .map(|(y, x)| {
                let mut x = *x;
                x[1] = x[0];
                (*y, x)
            })
            .collect();
        assert!(Model::fit(&collinear).is_none());
    }
    #[test]
    fn unexpected_profitability_rnoa_average_and_acc() {
        let mut q = [Quarter::default(); HISTORY];
        let mut end = 20250331;
        for item in &mut q {
            *item = Quarter {
                end,
                noa: Some(200.0),
                oi: Some(20.0),
                acc: Some(3.0),
                equity: Some(100.0),
            };
            end = previous_quarter_end_date(end).unwrap();
        }
        q[1].noa = Some(100.0);
        let mut snapshot = Snapshot { quarters: q };
        assert_eq!(snapshot.rnoa(20250331), Some(20.0 / 150.0));
        assert_eq!(snapshot.acc_ttm(20250331), Some(12.0));
        snapshot.quarters[1].noa = Some(-100.0);
        assert_eq!(snapshot.rnoa(20250331), None);
        snapshot.quarters[1].acc = None;
        assert_eq!(snapshot.acc_ttm(20250331), None);
    }
}

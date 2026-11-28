//! Point-in-time fundamental inputs for the LFQ end-to-end factor.
use crate::calendar::TradingCalendar;
use crate::config::EngineConfig;
use crate::core::DatasetId;
use crate::data::loader::DisclosureTableCache;
use crate::data::parquet_io::read_parquet;
use crate::data::read_ahead::{configure_threads, visit_filtered_inputs, write_files};
use crate::data::{ColumnData, DataCatalog, MarketDataLoader, Table};
use crate::derive::storage::{check_derived_manifest, write_derived_parquet};
use crate::error::{err, Result};
use crate::factor::common::financial::{
    previous_quarter_end_date, FinancialPitIndex, FinancialPitReader, PitFinancialRecordView,
    ReportTypePreference,
};
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::path::PathBuf;

mod institutional;
use institutional::{AnalystHistory, FundHistory};

const VERSION: &str = "astgnn_lfq_daily_v1";
const COLUMNS: [&str; 56] = [
    "earnings_yield",
    "earnings_yield_z252",
    "dividend_yield",
    "book_to_market",
    "sales_yield",
    "operating_cash_yield",
    "free_cash_yield",
    "ebit_ev",
    "roe",
    "roa",
    "roic",
    "gross_profit_to_assets",
    "gross_margin",
    "operating_margin",
    "net_margin",
    "deducted_net_margin",
    "deducted_profit_yoy",
    "revenue_yoy",
    "operating_profit_yoy",
    "parent_profit_yoy",
    "operating_cash_yoy",
    "assets_yoy",
    "gross_margin_yoy_delta",
    "roe_yoy_delta",
    "operating_cash_margin",
    "free_cash_margin",
    "operating_cash_to_profit",
    "accruals_to_assets",
    "cash_collection_ratio",
    "capex_to_sales",
    "rd_to_sales",
    "deducted_profit_share",
    "liabilities_to_assets",
    "interest_debt_to_assets",
    "net_debt_to_equity",
    "current_ratio",
    "quick_ratio",
    "interest_coverage",
    "cash_to_assets",
    "short_debt_share",
    "assets_turnover",
    "fixed_assets_turnover",
    "receivable_days",
    "inventory_days",
    "payable_days",
    "cash_conversion_days",
    "receivables_to_sales",
    "inventory_to_assets",
    "sue",
    "fund_holding_ratio",
    "analyst_coverage",
    "analyst_dispersion",
    "analyst_growth",
    "analyst_revision_60d",
    "analyst_upgrade_share_60d",
    "fund_holding_qoq",
];
type Metrics = [Option<f64>; COLUMNS.len()];
type Records<'a, const N: usize> = [Option<PitFinancialRecordView<'a>>; N];

const INCOME: &[&str] = &[
    "revenue",
    "oper_cost",
    "operate_profit",
    "n_income",
    "n_income_attr_p",
    "ebit",
    "income_tax",
    "total_profit",
    "int_exp",
    "fin_exp_int_exp",
    "rd_exp",
];
const BALANCE: &[&str] = &[
    "total_assets",
    "total_liab",
    "total_hldr_eqy_exc_min_int",
    "money_cap",
    "inventories",
    "total_cur_assets",
    "total_cur_liab",
    "fix_assets",
    "accounts_receiv",
    "acct_payable",
    "st_borr",
    "lt_borr",
    "bond_payable",
    "st_bonds_payable",
    "non_cur_liab_due_1y",
];
const CASHFLOW: &[&str] = &["n_cashflow_act", "c_pay_acq_const_fiolta", "c_fr_sale_sg"];

#[derive(Clone, Debug)]
pub struct FundamentalLfqRequest {
    pub start_date: i32,
    pub end_date: i32,
    pub date_batch_size: usize,
    pub threads: Option<usize>,
}

fn columns(names: &[&str]) -> Vec<String> {
    names.iter().map(|name| (*name).to_owned()).collect()
}

fn number(value: Option<f64>) -> Option<f64> {
    value.filter(|v| v.is_finite())
}

fn sum(values: impl IntoIterator<Item = Option<f64>>) -> Option<f64> {
    let mut values = values.into_iter().flatten();
    number(
        values
            .next()
            .map(|first| values.fold(first, |total, value| total + value)),
    )
}

fn difference(left: Option<f64>, right: Option<f64>) -> Option<f64> {
    sum([left, right.map(|value| -value)])
}

fn ratio(numerator: Option<f64>, denominator: Option<f64>) -> Option<f64> {
    let denominator = denominator.filter(|d| d.abs() > 1e-12)?;
    number(numerator.map(|value| value / denominator))
}

fn financial_ratio(numerator: Option<f64>, denominator: Option<f64>) -> Option<f64> {
    ratio(Some(numerator.unwrap_or(0.0)), denominator)
}

fn yoy(now: Option<f64>, prior: Option<f64>) -> Option<f64> {
    let prior = prior?;
    ratio(Some(now? - prior), Some(prior.abs()))
}

fn quarter_ends<const N: usize>(anchor: i32) -> [i32; N] {
    let mut cursor = anchor;
    std::array::from_fn(|_| {
        let end = cursor;
        cursor = previous_quarter_end_date(cursor).unwrap_or(0);
        end
    })
}

fn records<'a, const N: usize>(
    reader: &FinancialPitReader<'a>,
    stock: &str,
    date: i32,
    ends: &[i32; N],
) -> Records<'a, N> {
    ends.map(|end| reader.record_for_end_date(stock, date, end))
}

fn field(record: Option<PitFinancialRecordView<'_>>, column: &str) -> Option<f64> {
    number(record.and_then(|r| r.column(column)))
}

fn ttm(records: &[Option<PitFinancialRecordView<'_>>], column: &str) -> Option<f64> {
    sum(records.iter().map(|record| field(*record, column)))
}

fn debt(record: Option<PitFinancialRecordView<'_>>, short_only: bool) -> Option<f64> {
    let short =
        sum(["st_borr", "st_bonds_payable", "non_cur_liab_due_1y"].map(|c| field(record, c)));
    if short_only {
        return short;
    }
    sum([
        short,
        field(record, "lt_borr"),
        field(record, "bond_payable"),
    ])
}

fn mean_pair(current: Option<f64>, prior: Option<f64>) -> Option<f64> {
    sum([current, prior]).map(|total| total / 2.0)
}

fn days(value: Option<f64>, flow: Option<f64>) -> Option<f64> {
    ratio(value.map(|value| value * 365.0), flow)
}

struct Statements<'a> {
    income: FinancialPitReader<'a>,
    balance: FinancialPitReader<'a>,
    cashflow: FinancialPitReader<'a>,
}

#[derive(Clone)]
struct FinancialSnapshot {
    metrics: Metrics,
    ebit: Option<f64>,
    interest_debt: Option<f64>,
    cash: Option<f64>,
}

impl Statements<'_> {
    fn events(&self, stock: &str, start: i32, end: i32) -> Vec<i32> {
        let mut dates = Vec::new();
        for reader in [&self.income, &self.balance, &self.cashflow] {
            dates.extend(reader.disclosure_dates_between(stock, start, end));
        }
        dates.sort_unstable();
        dates.dedup();
        dates
    }

    fn snapshot(&self, stock: &str, date: i32) -> FinancialSnapshot {
        let mut m = [None; COLUMNS.len()];
        let anchor = self
            .income
            .latest_quarter_end_date(stock, date)
            .or_else(|| self.balance.latest_quarter_end_date(stock, date))
            .or_else(|| self.cashflow.latest_quarter_end_date(stock, date));
        let Some(anchor) = anchor else {
            return FinancialSnapshot {
                metrics: m,
                ebit: None,
                interest_debt: None,
                cash: None,
            };
        };
        let income = records(&self.income, stock, date, &quarter_ends::<13>(anchor));
        let cashflow = records(&self.cashflow, stock, date, &quarter_ends::<8>(anchor));
        let ends = quarter_ends::<9>(anchor);
        let balances = records(&self.balance, stock, date, &[ends[0], ends[4], ends[8]]);
        let [balance, prior_balance, older_balance] = balances;
        let inc = |column| ttm(&income[..4], column);
        let inc_prior = |column| ttm(&income[4..8], column);
        let cf = |column| ttm(&cashflow[..4], column);
        let bal = |column| field(balance, column);
        let bal_prior = |column| field(prior_balance, column);
        let average = |column| mean_pair(bal(column), bal_prior(column));

        let revenue = inc("revenue");
        let prior_revenue = inc_prior("revenue");
        let cost = inc("oper_cost");
        let operating_profit = inc("operate_profit");
        let parent_profit = inc("n_income_attr_p");
        let prior_profit = inc_prior("n_income_attr_p");
        let gross_profit = difference(revenue, cost);
        let assets = bal("total_assets");
        let equity = bal("total_hldr_eqy_exc_min_int");
        let operating_cash = cf("n_cashflow_act");
        let capex = cf("c_pay_acq_const_fiolta");
        let free_cash = difference(operating_cash, capex);
        let cash = bal("money_cap");
        let interest_debt = debt(balance, false);
        let interest = ttm_interest(&income[..4]);
        let ebit = inc("ebit").or_else(|| sum([operating_profit, interest]));
        let tax_rate = ratio(
            inc("income_tax"),
            inc("total_profit").filter(|value| *value > 0.0),
        )
        .unwrap_or(0.0)
        .clamp(0.0, 1.0);
        let invested_capital = difference(sum([equity, interest_debt]), cash);
        let prior_capital = difference(
            sum([
                bal_prior("total_hldr_eqy_exc_min_int"),
                debt(prior_balance, false),
            ]),
            bal_prior("money_cap"),
        );
        let inv = bal("inventories");
        let receivables = bal("accounts_receiv");

        // Daily market denominators are applied to these numerators separately.
        m[0] = parent_profit;
        m[3] = equity;
        m[4] = revenue;
        m[5] = operating_cash;
        m[6] = free_cash;
        m[8] = financial_ratio(parent_profit, average("total_hldr_eqy_exc_min_int"));
        m[9] = financial_ratio(inc("n_income"), average("total_assets"));
        m[10] = financial_ratio(
            ebit.map(|value| value * (1.0 - tax_rate)),
            mean_pair(invested_capital, prior_capital),
        );
        m[11] = financial_ratio(gross_profit, average("total_assets"));
        m[12] = financial_ratio(gross_profit, revenue);
        m[13] = financial_ratio(operating_profit, revenue);
        m[14] = financial_ratio(inc("n_income"), revenue);
        m[15] = financial_ratio(parent_profit, revenue);
        m[16] = yoy(parent_profit, prior_profit);
        m[17] = yoy(revenue, prior_revenue);
        m[18] = yoy(operating_profit, inc_prior("operate_profit"));
        m[19] = yoy(parent_profit, prior_profit);
        m[20] = yoy(operating_cash, ttm(&cashflow[4..8], "n_cashflow_act"));
        m[21] = yoy(assets, bal_prior("total_assets"));
        m[22] = m[12]
            .zip(ratio(
                difference(prior_revenue, inc_prior("oper_cost")),
                prior_revenue,
            ))
            .map(|(current, previous)| current - previous);
        m[23] = m[8]
            .zip(ratio(
                prior_profit,
                mean_pair(
                    bal_prior("total_hldr_eqy_exc_min_int"),
                    field(older_balance, "total_hldr_eqy_exc_min_int"),
                ),
            ))
            .map(|(current, previous)| current - previous);
        m[24] = financial_ratio(operating_cash, revenue);
        m[25] = financial_ratio(free_cash, revenue);
        m[26] = financial_ratio(operating_cash, parent_profit);
        m[27] = financial_ratio(
            difference(inc("n_income"), operating_cash),
            average("total_assets"),
        );
        m[28] = financial_ratio(cf("c_fr_sale_sg"), revenue);
        m[29] = financial_ratio(capex, revenue);
        m[30] = ratio(inc("rd_exp"), revenue);
        m[31] = ratio(parent_profit, parent_profit);
        m[32] = financial_ratio(bal("total_liab"), assets);
        m[33] = financial_ratio(interest_debt, assets);
        m[34] = financial_ratio(difference(interest_debt, cash), equity);
        m[35] = financial_ratio(bal("total_cur_assets"), bal("total_cur_liab"));
        m[36] = financial_ratio(
            difference(bal("total_cur_assets"), inv),
            bal("total_cur_liab"),
        );
        m[37] = ratio(ebit, interest);
        m[38] = financial_ratio(cash, assets);
        m[39] = financial_ratio(debt(balance, true), interest_debt);
        m[40] = financial_ratio(revenue, average("total_assets"));
        m[41] = financial_ratio(revenue, average("fix_assets"));
        m[42] = days(average("accounts_receiv"), revenue);
        m[43] = days(average("inventories"), cost);
        m[44] = days(average("acct_payable"), cost);
        m[45] = sum([m[42], m[43], m[44].map(|value| -value)]);
        m[46] = financial_ratio(receivables, revenue);
        m[47] = financial_ratio(inv, assets);
        m[48] = crate::factor::common::dfzq_seasonal_surprise::surprise(
            &income.map(|record| field(record, "n_income")),
        );
        FinancialSnapshot {
            metrics: m,
            ebit,
            interest_debt,
            cash,
        }
    }
}

fn ttm_interest(records: &[Option<PitFinancialRecordView<'_>>]) -> Option<f64> {
    sum(records
        .iter()
        .map(|record| field(*record, "fin_exp_int_exp").or_else(|| field(*record, "int_exp"))))
}

#[derive(Default)]
struct EarningsWindow {
    values: VecDeque<Option<f64>>,
    n: usize,
    total: f64,
    squares: f64,
}

impl EarningsWindow {
    fn push(&mut self, value: Option<f64>) -> Option<f64> {
        if self.values.len() == 252 {
            if let Some(old) = self.values.pop_front().flatten() {
                self.n -= 1;
                self.total -= old;
                self.squares -= old * old;
            }
        }
        self.values.push_back(value);
        let value = value?;
        self.n += 1;
        self.total += value;
        self.squares += value * value;
        if self.n < 60 {
            return None;
        }
        let n = self.n as f64;
        let average = self.total / n;
        let scale = (self.squares / n - average * average).max(0.0).sqrt();
        ratio(Some(value - average), Some(scale))
    }
}

struct StockState {
    events: Vec<i32>,
    cursor: usize,
    snapshot: FinancialSnapshot,
    ep: EarningsWindow,
    last_session: usize,
}

impl StockState {
    fn advance(&mut self, statements: &Statements<'_>, stock: &str, date: i32, session: usize) {
        let next = self.cursor + self.events[self.cursor..].partition_point(|event| *event <= date);
        if next != self.cursor {
            self.snapshot = statements.snapshot(stock, date);
            self.cursor = next;
        }
        let skipped = session.saturating_sub(self.last_session + 1);
        if skipped >= 252 {
            self.ep = EarningsWindow::default();
        } else {
            for _ in 0..skipped {
                self.ep.push(None);
            }
        }
        self.last_session = session;
    }
}

fn statements_index(
    loader: &MarketDataLoader,
    dataset: DatasetId,
    fields: &[&str],
    start: i32,
    end: i32,
    cache: &mut DisclosureTableCache,
) -> Result<FinancialPitIndex> {
    let sources =
        loader.load_financial_sources_cached(dataset, &columns(fields), start, end, 13, cache)?;
    if sources.is_empty() {
        return Err(err(format!(
            "no LFQ {} statement partitions",
            dataset.as_str()
        )));
    }
    FinancialPitIndex::from_source_tables(sources, Some(end))
}

fn daily_output(
    date: i32,
    session: usize,
    market: &Table,
    statements: &Statements<'_>,
    states: &mut HashMap<String, StockState>,
    end: i32,
    analysts: &AnalystHistory,
    funds: &FundHistory,
    analyst_reference: Option<(i32, i32, i32)>,
    emit: bool,
) -> Result<Option<Table>> {
    let codes = market.required_utf8("ts_code")?;
    let date_column = market.required_i32_date_cast("trade_date")?;
    let mv = market.required_f64_cast("total_mv")?;
    let dividend = market.required_f64_cast("dv_ttm")?;
    let shares = market.required_f64_cast("total_share")?;
    let mut output: Option<[Vec<Option<f32>>; 56]> =
        emit.then(|| std::array::from_fn(|_| Vec::with_capacity(market.len)));
    let mut seen = BTreeSet::new();
    for (i, code) in codes.iter().enumerate() {
        let code = code
            .as_deref()
            .filter(|code| !code.is_empty())
            .ok_or_else(|| err(format!("empty LFQ stock key at {date}")))?;
        if date_column[i] != Some(date) || !seen.insert(code) {
            return Err(err(format!(
                "invalid or duplicate LFQ market key {date}/{code}"
            )));
        }
        let state = match states.get_mut(code) {
            Some(state) => state,
            None => states.entry(code.to_owned()).or_insert_with(|| StockState {
                events: statements.events(code, date, end),
                cursor: 0,
                snapshot: statements.snapshot(code, date),
                ep: EarningsWindow::default(),
                last_session: session,
            }),
        };
        state.advance(statements, code, date, session);
        let market_value = number(mv[i])
            .filter(|value| *value > 0.0)
            .map(|value| value * 10_000.0);
        let ep = ratio(state.snapshot.metrics[0], market_value);
        let ep_zscore = state.ep.push(ep);
        let Some(output) = &mut output else { continue };
        let mut values = state.snapshot.metrics;
        for index in [3, 4, 5, 6] {
            values[index] = ratio(values[index], market_value);
        }
        values[0] = ep;
        values[1] = ep_zscore;
        values[2] = number(dividend[i]).map(|value| value / 100.0);
        let ev = market_value.and_then(|value| {
            difference(
                sum([Some(value), state.snapshot.interest_debt]),
                state.snapshot.cash,
            )
        });
        values[7] = ratio(state.snapshot.ebit, ev.filter(|value| *value > 0.0));
        let [holding, change] =
            funds.metrics(code, number(shares[i]).map(|value| value * 10_000.0));
        values[49] = holding;
        values[55] = change;
        values[50..55].copy_from_slice(&analysts.metrics(code, date, analyst_reference));
        for (column, value) in output.iter_mut().zip(values) {
            column.push(
                number(value)
                    .map(|value| value as f32)
                    .filter(|value| value.is_finite()),
            );
        }
    }
    let Some(output) = output else {
        return Ok(None);
    };
    let mut columns = BTreeMap::from([
        (
            "trade_date".into(),
            ColumnData::I32(vec![Some(date); market.len]),
        ),
        ("ts_code".into(), ColumnData::Utf8(codes.to_vec())),
    ]);
    for (name, values) in COLUMNS.into_iter().zip(output) {
        columns.insert(name.into(), ColumnData::F32(values));
    }
    Table::new(columns).map(Some)
}

pub fn derive_fundamental_lfq(
    config: &EngineConfig,
    request: &FundamentalLfqRequest,
) -> Result<()> {
    if request.start_date > request.end_date || request.date_batch_size == 0 {
        return Err(err("invalid LFQ date range or date-batch-size"));
    }
    configure_threads(crate::resources::compute_threads(request.threads)?);
    let calendar = TradingCalendar::load(&config.data_root, &config.stock_calendar_exchange)?;
    let first = calendar
        .first_open_on_or_after(request.start_date)
        .ok_or_else(|| err("no LFQ trading dates"))?;
    if first > request.end_date {
        return Err(err("no LFQ trading dates"));
    }
    // The source library starts in 2009. Early runs use the available EP history.
    let warmup = calendar.warmup_start(first, 251).max(20090105);
    let dates = calendar.open_dates_between(warmup, request.end_date);
    let end = *dates.last().ok_or_else(|| err("no LFQ trading dates"))?;
    let catalog = DataCatalog::new(config.data_root.clone());
    let loader = MarketDataLoader::new(catalog);
    let mut cache = DisclosureTableCache::default();
    let income = statements_index(
        &loader,
        DatasetId::StockIncome,
        INCOME,
        warmup,
        end,
        &mut cache,
    )?;
    let balance = statements_index(
        &loader,
        DatasetId::StockBalanceSheet,
        BALANCE,
        warmup,
        end,
        &mut cache,
    )?;
    let cashflow = statements_index(
        &loader,
        DatasetId::StockCashFlow,
        CASHFLOW,
        warmup,
        end,
        &mut cache,
    )?;
    let statements = Statements {
        income: income.reader(ReportTypePreference::income_single_quarter()),
        balance: balance.reader(ReportTypePreference::balance_sheet_consolidated()),
        cashflow: cashflow.reader(ReportTypePreference::income_single_quarter()),
    };
    drop(cache);

    let analyst_dates =
        calendar.open_dates_between(calendar.warmup_start(first, 120), request.end_date);
    let analyst_start = *analyst_dates.first().unwrap_or(&first);
    let analysts = AnalystHistory::load(&config.data_root, analyst_start, end)?;
    let mut funds = FundHistory::load(&config.data_root, end)?;
    let root = config.data_root.join("derived/stock/astgnn_lfq");
    let manifest = root.join("manifest.json");
    let expected = serde_json::json!({
        "version": VERSION, "columns": COLUMNS.as_slice(), "profit_proxy": "n_income_attr_p",
        "frequency": "daily", "preprocess": "raw_fundamental_metrics"
    })
    .to_string();
    let has_manifest = check_derived_manifest(&manifest, &expected)?;
    let basic = config.data_root.join("chn_stock_data/daily/basic");
    let fields = columns(&["trade_date", "ts_code", "total_mv", "total_share", "dv_ttm"]);
    let mut states = HashMap::new();
    let mut session = 0;
    let progress = crate::progress::ProgressBar::new("derive-fundamental-lfq", dates.len(), true);
    for batch in dates.chunks(request.date_batch_size) {
        let files: Vec<(i32, PathBuf)> = batch
            .iter()
            .map(|date| {
                (
                    *date,
                    basic
                        .join((date / 10_000).to_string())
                        .join(format!("{date}.parquet")),
                )
            })
            .collect();
        let mut outputs = Vec::new();
        visit_filtered_inputs(
            &basic,
            &files,
            |(date, path)| Ok((*date, read_parquet(path, Some(&fields))?)),
            |(date, market)| {
                if market.len == 0 {
                    return Err(err(format!("empty daily basic input {date}")));
                }
                if date >= first {
                    funds.advance(date)?;
                }
                let index = analyst_dates.partition_point(|day| *day < date);
                let reference = match index.checked_sub(60) {
                    Some(prior) => Some((
                        analyst_dates[index + 1 - 60],
                        analyst_dates[prior],
                        analyst_dates[prior.saturating_sub(59)],
                    )),
                    None => None,
                };
                let table = daily_output(
                    date,
                    session,
                    &market,
                    &statements,
                    &mut states,
                    end,
                    &analysts,
                    &funds,
                    reference,
                    date >= first,
                )?;
                if let Some(table) = table {
                    outputs.push((date, table));
                } else {
                    progress.tick(format!("warmup={date}"));
                }
                session += 1;
                Ok(())
            },
        )?;
        for (date, rows) in write_files(&root, outputs, |(date, table)| {
            let path = root
                .join((date / 10_000).to_string())
                .join(format!("{date}.parquet"));
            write_derived_parquet(&path, &table)?;
            Ok((date, table.len))
        })? {
            progress.tick(format!("date={date} rows={rows}"));
        }
    }
    if !has_manifest {
        // Concurrent yearly runs publish the same immutable dataset identity.
        crate::artifact_io::write(&manifest, expected)?;
    }
    progress.finish();
    Ok(())
}

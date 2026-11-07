use std::any::Any;
use std::collections::{BTreeMap, HashMap};

use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::{loader::IndexMembership, DataPool};
use crate::error::{err, Result};
use crate::factor::common::financial::previous_quarter_end_date;
use crate::factor::common::vector::clean;
use crate::factor::common::{
    ClassificationLevel, ClassificationMap, FinancialPitReader, ReportTypePreference,
};
use crate::factor::{Factor, FactorUpdatePolicy};
use crate::operators::{cs_neutralize_regression, cs_pctrank};

const ID: &str = "crr_mod2";
const INDEX: &str = "000985.CSI";
const WINDOW: usize = 16;
const MIN_PERIODS: usize = 8;
const HISTORY: usize = WINDOW + 4;
const CFO: &str = "n_cashflow_act";
const OP: &str = "operate_profit";
const EQUITY: &str = "total_hldr_eqy_inc_min_int";

pub struct StockDailyCrrMod2;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyCrrMod2)
}

#[derive(Default)]
struct State {
    month_end: Option<i32>,
    values: HashMap<String, Option<f64>>,
}

impl Factor for StockDailyCrrMod2 {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: ID.into(),
            aliases: vec!["CRR_MOD2".into()],
            name: "Modified Cash Flow Realization Rate 2".into(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: "0.1.0".into(),
            tags: ["TFZQ", "fundamental", "financial", "pit", "monthly_update", "neutralize", "barra", "size", "sector", "daily"]
                .into_iter().map(str::to_string).collect(),
            description: "Month-end PIT cash-flow realization. Over 16 consecutive quarter slots ending at the latest visible cash-flow quarter, fit CFO(t) on operating profit(t-1) and CFO(t)/positive prior-quarter total equity on market CFOOE, both with intercept and at least 8 finite pairs. Single-quarter flows prefer reported values then consolidated YTD differences. Market CFOOE uses the latest 000985.CSI constituent snapshot on or before each fiscal quarter end and matched sums of available CFO and positive prior-quarter total equity, using only statements visible by the calculation date. Sum ascending full-market CRR percentile and descending residual population-std percentile within CRR terciles. Both legs required; no winsorization or fill. SW L1 and Barra SIZE neutralize at month-end; hold final cross-section daily by stock code.".into(),
            dependencies: vec![
                DataRequest::financial_quarters(DatasetId::StockIncome, &[OP], HISTORY),
                DataRequest::financial_quarters(DatasetId::StockCashFlow, &[CFO], HISTORY),
                DataRequest::financial_quarters(DatasetId::StockBalanceSheet, &[EQUITY], HISTORY),
                DataRequest::index_monthly_members(INDEX, HISTORY),
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
        self.compute_panel(data, &mut State::default())
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
            .ok_or_else(|| err("crr_mod2 state mismatch"))?;
        Ok(vec![self.compute_panel(data, state)?])
    }
}

impl StockDailyCrrMod2 {
    fn compute_panel(&self, data: &DataPool, state: &mut State) -> Result<FactorSeries> {
        let panel = data.stock_universe_panel()?;
        let calendar = data.trading_calendar()?;
        let membership = data.index_monthly_members(INDEX)?;
        let size = panel.column_from_table(data.daily(DatasetId::StockBarraDaily)?, "SIZE")?;
        let sector = ClassificationMap::from_table(
            data.daily(DatasetId::StockSwClassification)?,
            ClassificationLevel::Sector,
        )?;
        let readers = Statements {
            income: Flow::new(data, DatasetId::StockIncome, OP)?,
            cash: Flow::new(data, DatasetId::StockCashFlow, CFO)?,
            balance: data.financial_reader(
                DatasetId::StockBalanceSheet,
                ReportTypePreference::balance_sheet_consolidated(),
            )?,
        };
        let codes = panel.instruments();
        let n = codes.len();
        let mut output = vec![None; panel.shape_len()];
        for (day, &date) in panel.dates().iter().enumerate() {
            if !panel.is_target_date(date) {
                continue;
            }
            let Some(anchor) = calendar.month_end_on_or_before(date) else {
                continue;
            };
            if state.month_end != Some(anchor) {
                let anchor_day = panel
                    .dates()
                    .binary_search(&anchor)
                    .map_err(|_| err("crr_mod2 month-end missing from warmup panel"))?;
                let offset = anchor_day * n;
                let mut market = BTreeMap::new();
                let mut betas = Vec::with_capacity(n);
                let mut vols = Vec::with_capacity(n);
                for (j, code) in codes.iter().enumerate() {
                    let (beta, vol) = if panel.is_present_offset(offset + j) {
                        readers.legs(code, anchor, membership, &mut market)
                    } else {
                        (None, None)
                    };
                    betas.push(beta);
                    vols.push(vol);
                }
                let raw = composite(&betas, &vols);
                let groups = sector.groups_for(anchor, codes);
                let exposure = &size.values()[offset..offset + n];
                let neutral = cs_neutralize_regression(&raw, &[exposure], Some(&groups), None);
                state.values = codes.iter().cloned().zip(neutral).collect();
                state.month_end = Some(anchor);
            }
            for (j, code) in codes.iter().enumerate() {
                let idx = day * n + j;
                if panel.is_present_offset(idx) {
                    output[idx] = state.values.get(code).copied().flatten();
                }
            }
        }
        Ok(panel
            .column_from_values(output)?
            .to_factor_series(self.spec()))
    }
}

struct Flow<'a> {
    single: FinancialPitReader<'a>,
    cumulative: FinancialPitReader<'a>,
    field: &'static str,
}

impl<'a> Flow<'a> {
    fn new(data: &'a DataPool, dataset: DatasetId, field: &'static str) -> Result<Self> {
        Ok(Self {
            single: data
                .financial_reader(dataset, ReportTypePreference::income_single_quarter())?,
            cumulative: data.financial_reader(dataset, ReportTypePreference::consolidated())?,
            field,
        })
    }

    fn value(&self, code: &str, date: i32, end: i32) -> Option<f64> {
        if let Some(v) = self
            .single
            .record_for_end_date(code, date, end)
            .and_then(|r| clean(r.column(self.field)))
        {
            return Some(v);
        }
        let cum = |q| {
            self.cumulative
                .record_for_end_date(code, date, q)
                .and_then(|r| clean(r.column(self.field)))
        };
        let value = cum(end)?;
        if end % 10_000 == 331 {
            Some(value)
        } else {
            clean(Some(value - cum(previous_quarter_end_date(end)?)?))
        }
    }

    fn latest_end(&self, code: &str, date: i32) -> Option<i32> {
        self.single
            .latest_quarter_end_date(code, date)
            .max(self.cumulative.latest_quarter_end_date(code, date))
    }
}

struct Statements<'a> {
    income: Flow<'a>,
    cash: Flow<'a>,
    balance: FinancialPitReader<'a>,
}

impl Statements<'_> {
    fn equity(&self, code: &str, date: i32, end: i32) -> Option<f64> {
        clean(
            self.balance
                .record_for_end_date(code, date, end)?
                .column(EQUITY),
        )
        .filter(|v| *v > 0.0)
    }

    fn market_cfooe(
        &self,
        date: i32,
        end: i32,
        prev: i32,
        membership: &IndexMembership,
    ) -> Option<f64> {
        let (_, codes) = membership.range(..=end).next_back()?;
        let (mut cfo, mut equity) = (0.0, 0.0);
        for code in codes {
            if let (Some(y), Some(b)) = (
                self.cash.value(code, date, end),
                self.equity(code, date, prev),
            ) {
                cfo += y;
                equity += b;
            }
        }
        if equity <= 0.0 {
            return None;
        }
        clean(Some(cfo / equity))
    }

    fn legs(
        &self,
        code: &str,
        date: i32,
        membership: &IndexMembership,
        market: &mut BTreeMap<i32, Option<f64>>,
    ) -> (Option<f64>, Option<f64>) {
        let Some(mut end) = self.cash.latest_end(code, date) else {
            return (None, None);
        };
        let mut crr = Vec::with_capacity(WINDOW);
        let mut resvol = Vec::with_capacity(WINDOW);
        for _ in 0..WINDOW {
            let Some(prev) = previous_quarter_end_date(end) else {
                break;
            };
            if let Some(cfo) = self.cash.value(code, date, end) {
                if let Some(op) = self.income.value(code, date, prev) {
                    crr.push((op, cfo));
                }
                if let Some(equity) = self.equity(code, date, prev) {
                    let benchmark = *market
                        .entry(end)
                        .or_insert_with(|| self.market_cfooe(date, end, prev, membership));
                    if let (Some(mkt), Some(stock)) = (benchmark, clean(Some(cfo / equity))) {
                        resvol.push((mkt, stock));
                    }
                }
            }
            end = prev;
        }
        (
            ols(&crr).map(|(beta, _)| beta),
            ols(&resvol).map(|(_, vol)| vol),
        )
    }
}

// Centered OLS is stable for both cash amounts and small cash-to-equity ratios.
fn ols(pairs: &[(f64, f64)]) -> Option<(f64, f64)> {
    if pairs.len() < MIN_PERIODS {
        return None;
    }
    let n = pairs.len() as f64;
    let mean_x = pairs.iter().map(|p| p.0).sum::<f64>() / n;
    let mean_y = pairs.iter().map(|p| p.1).sum::<f64>() / n;
    let xx = pairs.iter().map(|(x, _)| (x - mean_x).powi(2)).sum::<f64>();
    if xx <= 0.0 || !xx.is_finite() || pairs.iter().all(|p| p.0 == pairs[0].0) {
        return None;
    }
    let xy = pairs
        .iter()
        .map(|(x, y)| (x - mean_x) * (y - mean_y))
        .sum::<f64>();
    let beta = clean(Some(xy / xx))?;
    let ss = pairs
        .iter()
        .map(|(x, y)| (y - mean_y - beta * (x - mean_x)).powi(2))
        .sum::<f64>();
    Some((beta, clean(Some((ss / n).sqrt()))?))
}

fn composite(crr: &[Option<f64>], vol: &[Option<f64>]) -> Vec<Option<f64>> {
    let mut ranked = crr
        .iter()
        .enumerate()
        .filter_map(|(i, v)| v.map(|v| (i, v)))
        .collect::<Vec<_>>();
    ranked.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
    let mut groups: [Vec<usize>; 3] = Default::default();
    let mut output = vec![None; crr.len()];
    if ranked.len() < 3 {
        return output;
    }
    let mut percentile = vec![0.0; crr.len()];
    for (rank, &(i, _)) in ranked.iter().enumerate() {
        groups[rank * 3 / ranked.len()].push(i);
        percentile[i] = rank as f64 / (ranked.len() - 1) as f64;
    }
    for mut group in groups {
        // Keep the usual cross-sectional tie order, independent of CRR within each group.
        group.sort_unstable();
        let values = group.iter().map(|&i| vol[i]).collect::<Vec<_>>();
        for (i, rank) in group.into_iter().zip(cs_pctrank(&values, false)) {
            output[i] = rank.map(|rank| percentile[i] + rank);
        }
    }
    output
}

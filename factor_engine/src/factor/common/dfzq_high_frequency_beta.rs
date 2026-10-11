use std::any::Any;
use std::collections::{BTreeMap, VecDeque};

use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::{ColumnData, DataPool, Table};
use crate::error::{err, Result};
use crate::factor::common::stock_daily_ops::neutralize_size_sector;
use crate::factor::common::DailyPanel;
use crate::factor::Factor;

const WINDOW: usize = 252;
const MIN_DAYS: usize = 126;
const SLOTS: usize = 47;
const MIN_SLOTS: usize = 24;
const PROVIDER: &str = "dfzq_high_frequency_beta_v1";

#[derive(Clone, Copy)]
pub enum Output {
    Continuous,
    Jump,
}
impl Output {
    fn index(self) -> usize {
        match self {
            Self::Continuous => 0,
            Self::Jump => 1,
        }
    }
    fn id(self) -> &'static str {
        match self {
            Self::Continuous => "continuous_beta",
            Self::Jump => "jump_beta",
        }
    }
}
pub struct HighFrequencyBeta(pub Output);
impl Factor for HighFrequencyBeta {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id:self.0.id().into(),aliases:vec![match self.0 {Output::Continuous=>"ContinuousBeta",Output::Jump=>"JumpBeta"}.into()],
            name:self.0.id().into(),asset_class:AssetClass::Stock,frequency:Frequency::Daily,version:"0.1.0".into(),
            tags:["DFZQ","price_volume","beta","intraday","5min","stateful","daily","neutralize","barra","size","sector","deprecated"].into_iter().map(str::to_string).collect(),
            description:"Negative 252-session high-frequency beta against equal-weight non-BJ SH/SZ stock log returns (including self), not CSI800. 09:30 excluded; 48 closes 09:35..11:30/13:05..15:00 give 47 trading-time returns, including 11:30 to 13:05, never overnight. Requires full calendar window and >=126 valid days, >=24 paired slots per valid day. Continuous: separate plus/minus/market RV/BV, preliminary truncation, rolling TOD, sqrt(min(RV,BV)*TOD), refilter entire window daily. Jump: sqrt(sum((stock*market)^2)/sum(market^4)), no daily-beta averaging. Missing slots not filled or bridged; no adjacent BV product across missing slots. Final SW L1 industry and Barra SIZE neutralization; no smoothing or zscore. One-day streaming with shared in-memory state, no raw disk cache; only final factors are persisted.".into(),
            dependencies:vec![DataRequest::new(DatasetId::StockMinute1m, &["close", "vol"]),DataRequest::new(DatasetId::StockBarraDaily, &["SIZE"]),DataRequest::new(DatasetId::StockSwClassification, &["l1_code"])],intraday_raw_dependencies:vec![],lookback:Lookback{trading_days:WINDOW - 1},
        }
    }
    fn streams_minute_state(&self) -> bool {
        true
    }
    fn compute_provider_key(&self) -> String {
        PROVIDER.into()
    }
    fn requirements_for_context(&self, context: &FactorContext) -> Vec<DataRequest> {
        self.spec()
            .dependencies
            .into_iter()
            .filter(|r| !context.target_dates.is_empty() || r.dataset == DatasetId::StockMinute1m)
            .map(|r| r.with_explicit_dates(context.load_dates.clone()))
            .collect()
    }
    fn initial_compute_state(&self, ids: &[String]) -> Box<dyn Any + Send> {
        Box::new(State::new(
            ids.iter().any(|id| id == Output::Continuous.id()),
        ))
    }
    fn compute_many_stateful(
        &self,
        ids: &[String],
        context: &FactorContext,
        data: &DataPool,
        state: &mut (dyn Any + Send),
    ) -> Result<Vec<FactorSeries>> {
        let state = state
            .downcast_mut::<State>()
            .ok_or_else(|| err("HF beta state mismatch"))?;
        if context.load_dates.len() != 1 {
            return Err(err("HF beta requires one streamed date"));
        }
        let date = context.load_dates[0];
        if !context.target_dates.is_empty() && context.target_dates != context.load_dates {
            return Err(err("HF beta target date must match streamed date"));
        }
        let outputs: Vec<_> = [Output::Continuous, Output::Jump]
            .into_iter()
            .filter(|kind| ids.iter().any(|id| id == kind.id()))
            .collect();
        if outputs
            .iter()
            .any(|kind| matches!(kind, Output::Continuous))
            && !state.continuous
        {
            return Err(err("HF beta requested outputs changed within stream"));
        }
        let day = match data.minute(DatasetId::StockMinute1m, date) {
            Some(table) => Day::from_table(table, state.continuous)?,
            None => Day::empty(),
        };
        let codes: Vec<_> = day.stocks.keys().cloned().collect();
        state.push(date, day)?;
        if context.target_dates.is_empty() || codes.is_empty() {
            return Ok(outputs
                .into_iter()
                .map(|kind| FactorSeries {
                    spec: HighFrequencyBeta(kind).spec(),
                    values: Vec::new(),
                })
                .collect());
        }
        let values: Vec<_> = codes
            .into_iter()
            .map(|code| {
                let beta = state.values(&code);
                (code, beta)
            })
            .collect();
        let table = Table::new(BTreeMap::from([
            (
                "trade_date".into(),
                ColumnData::I32(vec![Some(date); values.len()]),
            ),
            (
                "ts_code".into(),
                ColumnData::Utf8(values.iter().map(|(code, _)| Some(code.clone())).collect()),
            ),
        ]))?;
        let panel = DailyPanel::from_table(&table, context)?;
        outputs
            .into_iter()
            .map(|kind| {
                // BTreeMap stock order matches the panel's sorted instrument index.
                let beta = panel.column_from_values(
                    values.iter().map(|(_, beta)| beta[kind.index()]).collect(),
                )?;
                finalize(kind, &panel, &beta, data)
            })
            .collect()
    }
    fn compute(&self, _: &FactorContext, _: &DataPool) -> Result<FactorSeries> {
        Err(err(
            "HF beta requires the stateful minute streaming scheduler",
        ))
    }
}

fn finalize(
    kind: Output,
    panel: &DailyPanel,
    beta: &crate::factor::common::PanelColumn,
    data: &DataPool,
) -> Result<FactorSeries> {
    let raw = beta.map_values(|v| v.filter(|v| v.is_finite()).map(|v| -v));
    Ok(neutralize_size_sector(&raw, panel, data)?.to_factor_series(HighFrequencyBeta(kind).spec()))
}

fn stock_code(code: &str) -> bool {
    code.split_once('.').is_some_and(|(symbol, exchange)| {
        symbol.len() == 6
            && symbol.bytes().all(|b| b.is_ascii_digit())
            && matches!(exchange, "SH" | "SZ")
    })
}
fn endpoint(time: &str) -> Option<usize> {
    let time = time.rsplit([' ', 'T']).next()?;
    if time.len() < 5 || time.get(2..3) != Some(":") {
        return None;
    }
    if time.len() >= 8 && time.get(6..8) != Some("00") {
        return None;
    }
    let m = time.get(0..2)?.parse::<usize>().ok()? * 60 + time.get(3..5)?.parse::<usize>().ok()?;
    if (575..=690).contains(&m) && (m - 575) % 5 == 0 {
        Some((m - 575) / 5)
    } else if (785..=900).contains(&m) && (m - 785) % 5 == 0 {
        Some(24 + (m - 785) / 5)
    } else {
        None
    }
}
fn base_threshold(r: &[f64; SLOTS]) -> f64 {
    let rv = r
        .iter()
        .filter(|v| v.is_finite())
        .map(|v| v * v)
        .sum::<f64>();
    let bv = std::f64::consts::FRAC_PI_2
        * r.windows(2)
            .filter(|p| p[0].is_finite() && p[1].is_finite())
            .map(|p| (p[0] * p[1]).abs())
            .sum::<f64>();
    2.5 * (SLOTS as f64).powf(-0.49) * rv.min(bv).sqrt()
}
#[derive(Clone)]
struct StockDay {
    // Jump-only streams retain sufficient statistics, not intraday arrays.
    returns: Option<Box<[f64; SLOTS]>>,
    base: [f64; 3],
    jump: [f64; 2],
}
#[derive(Clone)]
struct Day {
    market: [f64; SLOTS],
    stocks: BTreeMap<String, StockDay>,
}
impl Day {
    fn empty() -> Self {
        Self {
            market: [f64::NAN; SLOTS],
            stocks: BTreeMap::new(),
        }
    }
    fn from_table(table: &Table, continuous: bool) -> Result<Self> {
        let codes = table.required_utf8("ts_code")?;
        let times = table.required_utf8("trade_time")?;
        let closes = table.required_f64_cast("close")?;
        let volumes = table.required_f64_cast("vol")?;
        let mut anchors: BTreeMap<String, ([f64; 48], f64)> = BTreeMap::new();
        for i in 0..table.len {
            let (Some(code), Some(time)) = (codes[i].as_deref(), times[i].as_deref()) else {
                continue;
            };
            if !stock_code(code) {
                continue;
            }
            let entry = anchors.entry(code.into()).or_insert(([f64::NAN; 48], 0.0));
            if let Some(vol) = volumes[i].filter(|v| v.is_finite() && *v > 0.0) {
                entry.1 += vol;
            }
            if let (Some(slot), Some(close)) = (
                endpoint(time),
                closes[i].filter(|v| v.is_finite() && *v > 0.0),
            ) {
                entry.0[slot] = close;
            }
        }
        let returns = anchors
            .into_iter()
            .filter(|(_, (_, vol))| *vol > 0.0)
            .map(|(code, (prices, _))| {
                let r = std::array::from_fn(|s| {
                    let a = prices[s];
                    let b = prices[s + 1];
                    if a.is_finite() && b.is_finite() {
                        b.ln() - a.ln()
                    } else {
                        f64::NAN
                    }
                });
                (code, r)
            })
            .collect();
        Ok(Self::from_returns(returns, continuous))
    }
    fn from_returns(returns: BTreeMap<String, [f64; SLOTS]>, continuous: bool) -> Self {
        let mut market = [0.0; SLOTS];
        let mut counts = [0; SLOTS];
        for (code, r) in &returns {
            if !stock_code(code) || r.iter().filter(|v| v.is_finite()).count() < MIN_SLOTS {
                continue;
            }
            for s in 0..SLOTS {
                if r[s].is_finite() {
                    market[s] += r[s];
                    counts[s] += 1;
                }
            }
        }
        for s in 0..SLOTS {
            market[s] = if counts[s] >= 2 {
                market[s] / counts[s] as f64
            } else {
                f64::NAN
            };
        }
        let mut stocks = BTreeMap::new();
        for (code, mut r) in returns {
            if !stock_code(&code) {
                continue;
            }
            for s in 0..SLOTS {
                if !market[s].is_finite() {
                    r[s] = f64::NAN;
                }
            }
            if r.iter().filter(|v| v.is_finite()).count() < MIN_SLOTS {
                continue;
            }
            let mut base = [0.0; 3];
            if continuous {
                for j in 0..3 {
                    base[j] = base_threshold(&std::array::from_fn(|s| triple(r[s], market[s])[j]));
                }
            }
            let mut jump = [0.0; 2];
            for s in 0..SLOTS {
                if r[s].is_finite() {
                    jump[0] += (r[s] * market[s]).powi(2);
                    jump[1] += market[s].powi(4);
                }
            }
            stocks.insert(
                code,
                StockDay {
                    returns: continuous.then(|| Box::new(r)),
                    base,
                    jump,
                },
            );
        }
        Self { market, stocks }
    }
}
fn triple(r: f64, m: f64) -> [f64; 3] {
    if r.is_finite() && m.is_finite() {
        [r + m, r - m, m]
    } else {
        [f64::NAN; 3]
    }
}
#[derive(Clone)]
struct Totals {
    q: Option<Box<[[f64; SLOTS]; 3]>>,
    jump: [f64; 2],
    jump_nonzero: [usize; 2],
    days: usize,
}
struct State {
    continuous: bool,
    days: VecDeque<Day>,
    totals: BTreeMap<String, Totals>,
    last_date: Option<i32>,
}
impl State {
    fn new(continuous: bool) -> Self {
        Self {
            continuous,
            days: VecDeque::new(),
            totals: BTreeMap::new(),
            last_date: None,
        }
    }
    fn accumulate(&mut self, day: &Day, add: bool) {
        let sign = if add { 1.0 } else { -1.0 };
        for (code, stock) in &day.stocks {
            let total = self.totals.entry(code.clone()).or_insert_with(|| Totals {
                q: self.continuous.then(|| Box::new([[0.0; SLOTS]; 3])),
                jump: [0.0; 2],
                jump_nonzero: [0; 2],
                days: 0,
            });
            if add {
                total.days += 1;
            } else {
                total.days -= 1;
            }
            for j in 0..2 {
                total.jump[j] += sign * stock.jump[j];
                if stock.jump[j] > 0.0 {
                    if add {
                        total.jump_nonzero[j] += 1;
                    } else {
                        total.jump_nonzero[j] -= 1;
                    }
                }
                if total.jump_nonzero[j] == 0 {
                    total.jump[j] = 0.0;
                }
            }
            if let (Some(q), Some(r)) = (&mut total.q, &stock.returns) {
                for s in 0..SLOTS {
                    for (j, v) in triple(r[s], day.market[s]).into_iter().enumerate() {
                        if v.is_finite() && v.abs() <= stock.base[j] {
                            q[j][s] += sign * v * v;
                        }
                    }
                }
            }
            if total.days == 0 {
                self.totals.remove(code);
            }
        }
    }
    fn push(&mut self, date: i32, day: Day) -> Result<()> {
        if self.last_date.is_some_and(|d| date <= d) {
            return Err(err(
                "HF beta dates must be strictly increasing; reset state for replay",
            ));
        }
        if self.days.len() == WINDOW {
            let old = self.days.pop_front().unwrap();
            self.accumulate(&old, false);
        }
        self.accumulate(&day, true);
        self.days.push_back(day);
        self.last_date = Some(date);
        Ok(())
    }
    fn values(&self, code: &str) -> [Option<f64>; 2] {
        if self.days.len() != WINDOW {
            return [None; 2];
        }
        let Some(total) = self.totals.get(code).filter(|v| v.days >= MIN_DAYS) else {
            return [None; 2];
        };
        let jump = if total.jump[1] > 0.0 && total.jump[0] >= 0.0 {
            finite((total.jump[0] / total.jump[1]).sqrt())
        } else {
            None
        };
        let Some(q) = &total.q else {
            return [None, jump];
        };
        let mut tod = [[0.0; SLOTS]; 3];
        for j in 0..3 {
            let sum = q[j].iter().map(|v| v.max(0.0)).sum::<f64>();
            if sum > 0.0 {
                for s in 0..SLOTS {
                    tod[j][s] = (SLOTS as f64 * q[j][s].max(0.0) / sum).sqrt();
                }
            }
        }
        let mut cv = [0.0; 3];
        // Reclassify historical intervals with the CURRENT window's TOD.
        for day in &self.days {
            if let Some(stock) = day.stocks.get(code) {
                let r = stock.returns.as_ref().unwrap();
                for s in 0..SLOTS {
                    for (j, v) in triple(r[s], day.market[s]).into_iter().enumerate() {
                        if v.is_finite() && v.abs() <= stock.base[j] * tod[j][s] {
                            cv[j] += v * v;
                        }
                    }
                }
            }
        }
        let continuous = if cv[2] > 0.0 {
            finite((cv[0] - cv[1]) / (4.0 * cv[2]))
        } else {
            None
        };
        [continuous, jump]
    }
}
fn finite(v: f64) -> Option<f64> {
    v.is_finite().then_some(v)
}

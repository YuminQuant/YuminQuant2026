use crate::artifact_io::StoragePathExt;
use crate::data::parquet_io::read_parquet;
use crate::calendar::TradingCalendar;
use crate::error::{err, Result};
use crate::strategy::{
    account::AccountState,
    config::{BarFrequency, StrategyRunConfig},
    context::StrategyContext,
    market::{BarEvent, SessionOpenEvent},
    order::{FillEvent, OrderSide},
    strategy::Strategy,
};
use std::collections::{BTreeMap, BTreeSet};
use std::io::{BufWriter, Write};

#[derive(Default)]
struct Leg {
    target: f64,
    filled: f64,
    phase: f64,
}

pub struct Strategy001 {
    calendar: TradingCalendar,
    signal_root: std::path::PathBuf,
    signal_id: String,
    top_n: usize,
    candidate_count: usize,
    hold_days: usize,
    window: usize,
    buffer: f64,
    sessions: usize,
    active: bool,
    ranked: Vec<String>,
    selected: Vec<String>,
    replaced: BTreeSet<String>,
    entry_budget: f64,
    buys: BTreeMap<String, Leg>,
    sells: BTreeMap<String, Leg>,
    reserve: f64,
    net_rebalance: bool,
    rank_carry: bool,
    buy_window_budget: f64,
    buy_window_spent: f64,
    retry_next_day: bool,
    pending_quotes: BTreeMap<String, f64>,
    strict_signal: bool,
    signal_max_age: usize,
    input_days: usize,
    prepared: BTreeMap<i32, Vec<String>>,
    audit: Option<BufWriter<crate::artifact_io::AtomicWriter>>,
    plans: Option<BufWriter<crate::artifact_io::AtomicWriter>>,
}

impl Strategy001 {
    pub fn from_config(config: &StrategyRunConfig) -> Result<Self> {
        if config.bar_frequency != BarFrequency::Minute
            || !config.minute_open_limits
            || !config.respect_price_limits
        {
            return Err(err("strategy_001 requires minute bars, price_limit_mode=minute_open and respect_price_limits=true"));
        }
        let params = &config.strategy_params;
        let integer = |key: &str, default: usize| -> Result<usize> {
            let value = params
                .get(key)
                .map(|v| v.parse::<usize>())
                .transpose()?
                .unwrap_or(default);
            if value == 0 {
                return Err(err(format!("{key} must be positive")));
            }
            Ok(value)
        };
        let window = integer("twap_minutes", 30)?;
        if window > 120 {
            return Err(err("twap_minutes must not exceed the morning session"));
        }
        let buffer = params
            .get("cash_buffer")
            .map(|v| v.parse::<f64>())
            .transpose()?
            .unwrap_or(0.02);
        if !buffer.is_finite() || !(0.0..1.0).contains(&buffer) {
            return Err(err("cash_buffer must be in [0,1)"));
        }
        let mode = params
            .get("rebalance_mode")
            .map(String::as_str)
            .unwrap_or("round_trip");
        if !matches!(mode, "round_trip" | "net") {
            return Err(err("rebalance_mode must be round_trip or net"));
        }
        let hold_days = integer("rebalance_days", 5)?;
        let input_days = integer("input_batch_days", 1)?;
        if hold_days % input_days != 0 {
            return Err(err("input_batch_days must divide rebalance_days"));
        }
        let strict_signal = match params
            .get("missing_signal")
            .map(String::as_str)
            .unwrap_or("exit_only")
        {
            "error" => true,
            "exit_only" => false,
            _ => return Err(err("missing_signal must be error or exit_only")),
        };
        let signal_max_age = params
            .get("signal_max_age_sessions")
            .map(|v| v.parse::<usize>())
            .transpose()?
            .unwrap_or(0);
        let allocation = params
            .get("allocation_mode")
            .map(String::as_str)
            .unwrap_or("equal");
        if !matches!(allocation, "equal" | "rank_carry")
            || (allocation == "rank_carry" && mode != "net")
        {
            return Err(err(
                "allocation_mode must be equal or rank_carry (net mode only)",
            ));
        }
        let top_n = integer("top_n", 50)?;
        let replacements = params
            .get("limit_replacement_candidates")
            .map(|v| v.parse::<usize>())
            .transpose()?
            .unwrap_or(top_n);
        Ok(Self {
            rank_carry: allocation == "rank_carry",
            buy_window_budget: 0.0,
            buy_window_spent: 0.0,
            retry_next_day: match params
                .get("retry_next_day")
                .map(String::as_str)
                .unwrap_or("false")
            {
                "true" => true,
                "false" => false,
                _ => return Err(err("retry_next_day must be true or false")),
            },
            pending_quotes: BTreeMap::new(),
            net_rebalance: mode == "net",
            strict_signal,
            signal_max_age,
            input_days,
            prepared: BTreeMap::new(),
            calendar: TradingCalendar::load(&config.data_root, "SSE")?,
            signal_root: params
                .get("signal_root")
                .map(std::path::PathBuf::from)
                .unwrap_or_else(|| config.model_root.clone()),
            signal_id: params
                .get("signal_id")
                .cloned()
                .unwrap_or_else(|| "mdl_000022".into()),
            top_n,
            candidate_count: top_n.saturating_add(replacements),
            hold_days: integer("rebalance_days", 5)?,
            window,
            buffer,
            sessions: 0,
            active: false,
            ranked: Vec::new(),
            selected: Vec::new(),
            replaced: BTreeSet::new(),
            entry_budget: 0.0,
            buys: BTreeMap::new(),
            sells: BTreeMap::new(),
            reserve: 0.0,
            audit: None,
            plans: None,
        })
    }
    fn reset_buy_window(&mut self, ctx: &StrategyContext) {
        let rate = (ctx.config().buy_commission_bps + ctx.config().slippage_bps) / 10000.0;
        self.buy_window_budget = self
            .buys
            .iter()
            .map(|(s, l)| l.target * ctx.market().open_price(s).unwrap_or(0.0) * (1.0 + rate))
            .sum();
        self.buy_window_spent = 0.0;
    }

    fn replace_buy_slot(
        &mut self,
        ctx: &StrategyContext,
        symbol: &str,
        buy_rate: f64,
    ) -> Option<(String, f64)> {
        let index = self.selected.iter().position(|s| s == symbol)?;
        let lot = ctx.config().lot_size;
        let (backup, target) = self.ranked.iter().skip(self.top_n).find_map(|backup| {
            if self.selected.contains(backup)
                || self.replaced.contains(backup)
                || ctx.account().is_delisted(backup)
            {
                return None;
            }
            let bar = ctx.market().bar(backup)?;
            let target = (self.entry_budget / (bar.open * (1.0 + buy_rate)) / lot).floor() * lot;
            (!bar.is_limit_up && target >= lot).then(|| (backup.clone(), target))
        })?;
        self.replaced
            .insert(std::mem::replace(&mut self.selected[index], backup.clone()));
        let event = &ctx.market().event;
        eprintln!(
            "[strategy001 replacement] date={} time={} from={} to={}",
            event.trade_date,
            event.trade_time.rsplit(' ').next().unwrap_or(&event.trade_time),
            symbol,
            backup
        );
        Some((backup, target))
    }

    fn replace_blocked_orders(&mut self, ctx: &StrategyContext, buy_rate: f64) {
        if self.ranked.len() <= self.top_n {
            return;
        }
        let blocked = self
            .selected
            .iter()
            .filter_map(|symbol| {
                let leg = self.buys.get(symbol)?;
                (leg.target > 0.0
                    && leg.filled == 0.0
                    && ctx.account().position_quantity(symbol) == 0.0
                    && !ctx.account().is_delisted(symbol)
                    && ctx.market().bar(symbol).is_some_and(|bar| bar.open_at_upper_limit))
                    .then(|| (symbol.clone(), leg.phase))
            })
            .collect::<Vec<_>>();
        let lot = ctx.config().lot_size;
        for (symbol, phase) in blocked {
            let Some((backup, target)) = self.replace_buy_slot(ctx, &symbol, buy_rate) else {
                continue;
            };
            self.buys.remove(&symbol);
            let current = ctx.account().position_quantity(&backup);
            let delta = if self.net_rebalance { target - current } else { target };
            if self.net_rebalance {
                self.sells.remove(&backup);
            }
            let quantity = (delta.abs() / lot).floor() * lot;
            if quantity > 0.0 {
                let legs = if delta > 0.0 { &mut self.buys } else { &mut self.sells };
                legs.insert(backup, Leg {
                    target: quantity,
                    filled: 0.0,
                    phase,
                });
            }
        }
    }
}

/// Cumulative equal-time participation, with rank phases to spread indivisible lots.
fn due(leg: &Leg, slot: usize, window: usize, lot: f64) -> f64 {
    let scheduled = if slot + 1 == window {
        leg.target
    } else {
        ((leg.target / lot) * (slot + 1) as f64 / window as f64 + leg.phase).floor() * lot
    };
    (scheduled.min(leg.target) - leg.filled).max(0.0)
}

impl Strategy for Strategy001 {
    fn name(&self) -> &'static str {
        "stock::strategy_001"
    }

    fn input_batch_days(&self) -> usize {
        self.input_days
    }

    fn prepare_run(&mut self, _config: &StrategyRunConfig, dates: &[i32]) -> Result<()> {
        let mut needed = BTreeSet::new();
        let mut candidates = BTreeMap::new();
        for &date in dates.iter().step_by(self.hold_days) {
            let mut end = date;
            let mut preceding = Vec::new();
            for _ in 0..=self.signal_max_age {
                let Some(previous) = self.calendar.last_open_on_or_before(end - 1) else {
                    break;
                };
                needed.insert(previous);
                preceding.push(previous);
                end = previous;
            }
            candidates.insert(date, preceding);
        }
        let available = needed
            .into_iter()
            .filter_map(|date| {
                let path = self
                    .signal_root
                    .join((date / 10000).to_string())
                    .join(format!("{date}.parquet"));
                crate::artifact_io::exists(&path)
                    .map(|yes| yes.then_some((date, path)))
                    .transpose()
            })
            .collect::<Result<Vec<_>>>()?;
        let paths = available.iter().map(|(_, p)| p.clone()).collect::<Vec<_>>();
        let columns = vec![
            "trade_date".into(),
            "ts_code".into(),
            self.signal_id.clone(),
        ];
        let mut signals = BTreeMap::new();
        crate::data::read_ahead::visit_projected_files(&paths, &columns, |path, table| {
            let date = path.file_stem().unwrap().to_string_lossy().parse::<i32>()?;
            let dates = table.required_i32_date_cast("trade_date")?;
            let codes = table.required_utf8("ts_code")?;
            let values = table.required_f64_cast(&self.signal_id)?;
            let mut ranked = (0..table.len)
                .filter_map(|i| {
                    let code = codes[i].clone()?;
                    let value = values[i]?;
                    (dates[i] == Some(date) && value.is_finite() && !code.ends_with(".BJ"))
                        .then_some((code, value))
                })
                .collect::<Vec<_>>();
            ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            signals.insert(
                date,
                ranked
                    .into_iter()
                    .take(self.candidate_count)
                    .map(|(s, _)| s)
                    .collect::<Vec<_>>(),
            );
            Ok(())
        })?;
        for (date, previous) in candidates {
            let selected = previous.iter().find_map(|d| {
                signals
                    .get(d)
                    .filter(|s| !s.is_empty())
                    .map(|s| (*d, s.clone()))
            });
            match selected {
                Some((source, names)) => {
                    if Some(&source) != previous.first() {
                        eprintln!(
                            "[strategy warning] date={date}: using bounded as-of signal {source}"
                        );
                    }
                    self.prepared.insert(date, names);
                }
                None if self.strict_signal => {
                    return Err(err(format!(
                        "{} has no valid preceding signal for {date} within {} sessions",
                        self.signal_id, self.signal_max_age
                    )))
                }
                None => {
                    self.prepared.insert(date, Vec::new());
                }
            }
        }
        Ok(())
    }

    fn prepare_session(
        &mut self,
        _config: &StrategyRunConfig,
        account: &AccountState,
        date: i32,
    ) -> Result<Option<BTreeSet<String>>> {
        let rebalance = self.sessions % self.hold_days == 0;
        self.active = rebalance
            || (self.retry_next_day
                && (self
                    .buys
                    .values()
                    .chain(self.sells.values())
                    .any(|l| l.target - l.filled > 1e-9)
                    || !self.pending_quotes.is_empty()));
        if rebalance {
            self.replaced.clear();
            let previous = self
                .calendar
                .last_open_on_or_before(date - 1)
                .ok_or_else(|| err("missing previous signal date"))?;
            if let Some(names) = self.prepared.get(&date) {
                self.ranked.clone_from(names);
            } else {
                let mut ranked = load_signal(&self.signal_root, previous, &self.signal_id)?
                    .into_iter()
                    .filter(|(s, v)| v.is_finite() && !s.ends_with(".BJ"))
                    .collect::<Vec<_>>();
                ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
                self.ranked = ranked
                    .into_iter()
                    .take(self.candidate_count)
                    .map(|(s, _)| s)
                    .collect();
            }
            self.selected = self.ranked.iter().take(self.top_n).cloned().collect();
            if self.selected.is_empty() && self.strict_signal {
                return Err(err(format!(
                    "missing {} signal before {date}",
                    self.signal_id
                )));
            }
            if self.selected.is_empty() {
                eprintln!("[strategy warning] strategy_001 missing signal on {previous}; exit only on {date}");
            }
        }
        Ok(Some(
            account
                .positions()
                .keys()
                .cloned()
                .chain(self.selected.iter().cloned())
                .chain(
                    self.ranked.iter().skip(self.top_n)
                        .filter(|symbol| self.active && !self.replaced.contains(*symbol))
                        .cloned(),
                )
                .collect(),
        ))
    }

    fn on_start(&mut self, ctx: &mut StrategyContext) -> Result<()> {
        crate::artifact_io::create_dir_all(ctx.config().output_dir())?;
        let mut audit = BufWriter::new(crate::artifact_io::open_writer(
            &ctx.config().output_dir().join("fills.csv"),
        )?);
        writeln!(
            audit,
            "trade_date,trade_time,symbol,side,quantity,price,notional,fee,tax,slippage,cash_after,realized_pnl,net_realized_pnl"
        )?;
        self.audit = Some(audit);
        if self.net_rebalance {
            let mut plans = BufWriter::new(crate::artifact_io::open_writer(
                &ctx.config().output_dir().join("rebalance.csv"),
            )?);
            writeln!(
                plans,
                "trade_date,symbol,current_quantity,target_quantity,requested_delta,equity,reserve"
            )?;
            self.plans = Some(plans);
        }
        Ok(())
    }

    fn on_session_open(
        &mut self,
        ctx: &mut StrategyContext,
        _event: &SessionOpenEvent,
    ) -> Result<()> {
        let rebalance = self.sessions % self.hold_days == 0;
        self.sessions += 1;
        self.buys.retain(|symbol, _| !ctx.account().is_delisted(symbol));
        self.sells.retain(|symbol, _| !ctx.account().is_delisted(symbol));
        self.pending_quotes.retain(|symbol, _| !ctx.account().is_delisted(symbol));
        if self.retry_next_day && !rebalance {
            for leg in self.buys.values_mut().chain(self.sells.values_mut()) {
                leg.target = (leg.target - leg.filled).max(0.0);
                leg.filled = 0.0;
            }
            self.buys.retain(|_, l| l.target > 1e-9);
            self.sells.retain(|_, l| l.target > 1e-9);
            self.reserve = (ctx.account().cash()
                + ctx.account().dividend_receivable()
                + ctx
                    .account()
                    .positions()
                    .iter()
                    .map(|(s, p)| p.quantity * ctx.market().open_price(s).unwrap_or(p.mark_price()))
                    .sum::<f64>())
            .max(0.0)
                * self.buffer;
            let waiting = std::mem::take(&mut self.pending_quotes);
            let rate = (ctx.config().buy_commission_bps + ctx.config().slippage_bps) / 10000.0;
            let lot = ctx.config().lot_size;
            for (symbol, budget) in waiting {
                if let Some(price) = ctx.market().open_price(&symbol) {
                    let target = (budget / (price * (1.0 + rate)) / lot).floor() * lot;
                    let delta = target - ctx.account().position_quantity(&symbol);
                    if delta > 0.0 {
                        self.buys.insert(
                            symbol,
                            Leg {
                                target: (delta / lot).floor() * lot,
                                filled: 0.0,
                                phase: 0.5,
                            },
                        );
                    } else if delta < 0.0 {
                        self.sells.insert(
                            symbol,
                            Leg {
                                target: ((-delta) / lot).floor() * lot,
                                filled: 0.0,
                                phase: 0.5,
                            },
                        );
                    }
                } else {
                    self.pending_quotes.insert(symbol, budget);
                }
            }
            self.reset_buy_window(ctx);
            return Ok(());
        }
        self.pending_quotes.clear();
        self.buys.clear();
        self.sells.clear();
        if !self.active {
            return Ok(());
        }
        let equity = ctx.account().cash()
            + ctx.account().dividend_receivable()
            + ctx
                .account()
                .positions()
                .iter()
                .map(|(s, p)| p.quantity * ctx.market().open_price(s).unwrap_or(p.mark_price()))
                .sum::<f64>();
        self.reserve = equity.max(0.0) * self.buffer;
        self.entry_budget = (equity - self.reserve).max(0.0) / self.selected.len().max(1) as f64;
        let rate = (ctx.config().buy_commission_bps + ctx.config().slippage_bps) / 10000.0;
        if self.ranked.len() > self.top_n {
            // Resolve opening substitutions before net targets and old-position sales are planned.
            for symbol in self.selected.clone() {
                if ctx.account().position_quantity(&symbol) == 0.0
                    && !ctx.account().is_delisted(&symbol)
                    && ctx.market().bar(&symbol).is_some_and(|bar| bar.open_at_upper_limit)
                {
                    self.replace_buy_slot(ctx, &symbol, rate);
                }
            }
            self.selected = self.ranked.iter()
                .filter(|symbol| self.selected.contains(symbol))
                .cloned()
                .collect();
        }
        if self.net_rebalance {
            let lot = ctx.config().lot_size;
            let target_value = self.entry_budget;
            // Reserve a conservative one-way buy fee allowance without resetting retained lots.
            let mut rank_targets = BTreeMap::new();
            if self.rank_carry {
                let mut carry = 0.0;
                for symbol in &self.selected {
                    let budget = target_value + carry;
                    if ctx.account().is_delisted(symbol) {
                        carry = budget;
                        continue;
                    }
                    let current = ctx.account().position_quantity(symbol);
                    let Some(price) = ctx.market().open_price(symbol) else {
                        let retained = ctx
                            .account()
                            .position(symbol)
                            .map_or(0.0, |p| p.market_value());
                        rank_targets.insert(symbol.clone(), current);
                        carry = (budget - retained).max(0.0);
                        continue;
                    };
                    let desired = (budget / (price * (1.0 + rate)) / lot).floor() * lot;
                    let target = if ctx.market().bar(symbol).is_some_and(|b| {
                        b.is_limit_up && !(b.open_at_upper_limit && current == 0.0
                            && self.ranked.len() > self.top_n)
                    }) {
                        desired.min(current)
                    } else {
                        desired
                    };
                    rank_targets.insert(symbol.clone(), target);
                    carry = (budget - target * price * (1.0 + rate)).max(0.0);
                }
            }
            let universe = ctx
                .account()
                .positions()
                .keys()
                .cloned()
                .chain(self.selected.iter().cloned())
                .collect::<BTreeSet<_>>();
            let count = universe.len().max(1);
            for (i, symbol) in universe.iter().enumerate() {
                let current = ctx.account().position_quantity(symbol);
                let selected = self.selected.contains(symbol);
                let target = if ctx.account().is_delisted(symbol) {
                    0.0
                } else if selected && self.rank_carry {
                    rank_targets[symbol]
                } else if selected {
                    let Some(price) = ctx.market().open_price(symbol) else {
                        if self.retry_next_day {
                            self.pending_quotes.insert(symbol.clone(), target_value);
                        }
                        continue;
                    };
                    (target_value / (price * (1.0 + rate)) / lot).floor() * lot
                } else {
                    0.0
                };
                let delta = target - current;
                if let Some(plans) = &mut self.plans {
                    writeln!(
                        plans,
                        "{},{},{},{},{},{},{}",
                        _event.trade_date, symbol, current, target, delta, equity, self.reserve
                    )?;
                }
                if ctx.account().is_delisted(symbol) {
                    continue;
                }
                let phase = (i as f64 + 0.5) / count as f64;
                if delta > 1e-9 {
                    let quantity = (delta / lot).floor() * lot;
                    if quantity > 0.0 {
                        self.buys.insert(
                            symbol.clone(),
                            Leg {
                                target: quantity,
                                filled: 0.0,
                                phase,
                            },
                        );
                    }
                } else if delta < -1e-9 {
                    let excess = if self.retry_next_day {
                        -delta
                    } else {
                        (-delta).min(ctx.account().sellable_quantity(symbol))
                    };
                    // Odd lots are sold only on full exit; retained positions adjust in whole lots.
                    let quantity = if !selected {
                        excess
                    } else {
                        (excess / lot).floor() * lot
                    };
                    if quantity > 0.0 {
                        self.sells.insert(
                            symbol.clone(),
                            Leg {
                                target: quantity,
                                filled: 0.0,
                                phase,
                            },
                        );
                    }
                }
            }
            self.reset_buy_window(ctx);
            return Ok(());
        }
        let count = ctx.account().positions().len().max(1);
        for (i, (symbol, _)) in ctx.account().positions().iter().enumerate() {
            self.sells.insert(
                symbol.clone(),
                Leg {
                    target: ctx.account().sellable_quantity(symbol),
                    filled: 0.0,
                    phase: (i as f64 + 0.5) / count as f64,
                },
            );
        }
        let budget = self.entry_budget;
        let lot = ctx.config().lot_size;
        for (i, symbol) in self.selected.iter().enumerate() {
            if ctx.account().is_delisted(symbol) {
                continue;
            }
            // Missing opening quote: retain that slot as cash, never replace it.
            let Some(price) = ctx.market().open_price(symbol) else {
                continue;
            };
            self.buys.insert(
                symbol.clone(),
                Leg {
                    target: (budget / (price * (1.0 + rate)) / lot).floor() * lot,
                    filled: 0.0,
                    phase: (i as f64 + 0.5) / self.selected.len() as f64,
                },
            );
        }
        Ok(())
    }

    fn on_bar_open(&mut self, ctx: &mut StrategyContext, event: &BarEvent) -> Result<()> {
        if !self.active {
            return Ok(());
        }
        let clock = event.trade_time.rsplit(' ').next().unwrap_or("");
        let (hour, rest) = clock.split_once(':').ok_or_else(|| err("invalid minute timestamp"))?;
        let (minute, _) = rest.split_once(':').ok_or_else(|| err("invalid minute timestamp"))?;
        let minute = hour.parse::<usize>()? * 60 + minute.parse::<usize>()?;
        if !(570..=900).contains(&minute) {
            return Ok(());
        }
        if minute >= 570 + self.window
            && self.buys.values().chain(self.sells.values()).all(|leg| leg.target - leg.filled <= 1e-9)
        {
            self.active = false;
            return Ok(());
        }
        // After the TWAP window, all remaining shares are due on every valid bar.
        let slot = (minute - 570).min(self.window - 1);
        let lot = ctx.config().lot_size;
        let buy_rate = (ctx.config().buy_commission_bps + ctx.config().slippage_bps) / 10000.0;
        self.replace_blocked_orders(ctx, buy_rate);
        let sell_rate = (ctx.config().sell_commission_bps
            + ctx.config().stamp_tax_bps
            + ctx.config().slippage_bps)
            / 10000.0;
        let mut cash = ctx.account().cash();
        for (symbol, leg) in &self.sells {
            let Some(bar) = ctx.market().bar(symbol) else {
                continue;
            };
            if bar.is_limit_down {
                continue;
            }
            // Do not sell shares bought today when the same name is reselected.
            let today_buys = self.buys.get(symbol).map_or(0.0, |l| l.filled);
            let sell_due = if bar.near_price_limit > 0 {
                (leg.target - leg.filled).max(0.0)
            } else {
                due(leg, slot, self.window, lot)
            };
            let quantity = sell_due
                .min((ctx.account().sellable_quantity(symbol) - today_buys).max(0.0));
            if quantity > 0.0 {
                cash += quantity * bar.open * (1.0 - sell_rate);
                ctx.order_quantity(symbol.clone(), -quantity);
            }
        }
        // Sell proceeds are available to buys at the same minute open. The 2% reserve is protected.
        let mut quota = if slot + 1 == self.window {
            f64::INFINITY
        } else {
            (self.buy_window_budget * (slot + 1) as f64 / self.window as f64
                - self.buy_window_spent)
                .max(0.0)
        };
        let buy_symbols = self.ranked.iter().filter(|_| self.rank_carry)
            .chain(self.buys.keys().filter(|_| !self.rank_carry));
        for symbol in buy_symbols {
            let Some(leg) = self.buys.get(symbol) else {
                continue;
            };
            let Some(bar) = ctx.market().bar(symbol) else {
                continue;
            };
            if bar.is_limit_up {
                continue;
            }
            let price = bar.open;
            let affordable =
                ((cash - self.reserve).max(0.0) / (price * (1.0 + buy_rate)) / lot).floor() * lot;
            let quantity = if bar.near_price_limit != 0 {
                (leg.target - leg.filled).max(0.0).min(affordable)
            } else if self.rank_carry {
                let time_limit = (quota / (price * (1.0 + buy_rate)) / lot).floor() * lot;
                (leg.target - leg.filled)
                    .max(0.0)
                    .min(affordable)
                    .min(time_limit)
            } else {
                due(leg, slot, self.window, lot).min(affordable)
            };
            if quantity > 0.0 {
                cash -= quantity * price * (1.0 + buy_rate);
                if self.rank_carry {
                    quota = (quota - quantity * price * (1.0 + buy_rate)).max(0.0);
                }
                ctx.order_quantity(symbol.clone(), quantity);
            } else if self.rank_carry && leg.target - leg.filled > 1e-9 {
                break;
            }
        }
        Ok(())
    }
    fn on_bar(&mut self, _ctx: &mut StrategyContext, _event: &BarEvent) -> Result<()> {
        Ok(())
    }
    fn on_fill(&mut self, _ctx: &mut StrategyContext, fill: &FillEvent) -> Result<()> {
        if self.rank_carry && fill.side == OrderSide::Buy {
            self.buy_window_spent += fill.notional + fill.fee + fill.slippage_cost;
        }
        let leg = match fill.side {
            OrderSide::Buy => self.buys.get_mut(&fill.symbol),
            OrderSide::Sell => self.sells.get_mut(&fill.symbol),
            _ => None,
        };
        if let Some(leg) = leg {
            leg.filled += fill.quantity;
        }
        if let Some(file) = &mut self.audit {
            writeln!(
                file,
                "{},{},{},{},{},{},{},{},{},{},{},{},{}",
                fill.trade_date,
                fill.trade_time,
                fill.symbol,
                fill.side.as_str(),
                fill.quantity,
                fill.fill_price,
                fill.notional,
                fill.fee,
                fill.tax,
                fill.slippage_cost,
                fill.cash_after,
                fill.realized_pnl,
                fill.net_realized_pnl
            )?;
        }
        Ok(())
    }
    fn on_end(&mut self, _ctx: &mut StrategyContext) -> Result<()> {
        if let Some(mut file) = self.plans.take() {
            file.flush()?;
            file.into_inner()
                .map_err(|e| crate::error::err(e.to_string()))?
                .commit()?;
        }
        if let Some(mut file) = self.audit.take() {
            file.flush()?;
            file.into_inner()
                .map_err(|e| crate::error::err(e.to_string()))?
                .commit()?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn twap_lot_schedule_spreads_small_orders_and_catches_up() {
        let mut slots = BTreeSet::new();
        for rank in 0..50 {
            let mut leg = Leg {
                target: 100.0,
                filled: 0.0,
                phase: (rank as f64 + 0.5) / 50.0,
            };
            for slot in 0..30 {
                let q = due(&leg, slot, 30, 100.0);
                if q > 0.0 {
                    slots.insert(slot);
                }
                leg.filled += q;
            }
            assert_eq!(leg.filled, 100.0);
        }
        assert_eq!(slots.len(), 30);
        let leg = Leg {
            target: 3000.0,
            filled: 0.0,
            phase: 0.0,
        };
        assert_eq!(due(&leg, 14, 30, 100.0), 1500.0);
        assert_eq!(due(&leg, 29, 30, 100.0), 3000.0);
    }
}

fn load_signal(
    root: &std::path::Path,
    trade_date: i32,
    signal_id: &str,
) -> Result<BTreeMap<String, f64>> {
    let path = root
        .join((trade_date / 10_000).to_string())
        .join(format!("{trade_date}.parquet"));
    if !path.storage_exists() {
        return Ok(BTreeMap::new());
    }
    let columns = vec![
        "trade_date".to_string(),
        "ts_code".to_string(),
        signal_id.to_string(),
    ];
    let table = read_parquet(&path, Some(&columns))?;
    let dates = table.required_i32_date_cast("trade_date")?;
    let codes = table.required_utf8("ts_code")?;
    let values = table.required_f64_cast(signal_id)?;
    let mut out = BTreeMap::new();
    for idx in 0..table.len {
        if dates[idx] == Some(trade_date) {
            if let (Some(code), Some(value)) = (codes[idx].clone(), values[idx]) {
                if value.is_finite() {
                    out.insert(code, value);
                }
            }
        }
    }
    Ok(out)
}

use super::strategy_001::load_signal;
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

pub struct Strategy002 {
    calendar: TradingCalendar,
    signal_root: std::path::PathBuf,
    signal_id: String,
    top_n: usize,
    hold_days: usize,
    window: usize,
    buffer: f64,
    sessions: usize,
    active: bool,
    selected: Vec<String>,
    buys: BTreeMap<String, Leg>,
    sells: BTreeMap<String, Leg>,
    reserve: f64,
    audit: Option<BufWriter<std::fs::File>>,
}

impl Strategy002 {
    pub fn from_config(config: &StrategyRunConfig) -> Result<Self> {
        if config.bar_frequency != BarFrequency::Minute
            || !config.minute_open_limits
            || !config.respect_price_limits
        {
            return Err(err("strategy_002 requires minute bars, price_limit_mode=minute_open and respect_price_limits=true"));
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
        Ok(Self {
            calendar: TradingCalendar::load(&config.data_root, "SSE")?,
            signal_root: params
                .get("signal_root")
                .map(std::path::PathBuf::from)
                .unwrap_or_else(|| config.model_root.clone()),
            signal_id: params
                .get("signal_id")
                .cloned()
                .unwrap_or_else(|| "mdl_000009".into()),
            top_n: integer("top_n", 50)?,
            hold_days: integer("rebalance_days", 5)?,
            window,
            buffer,
            sessions: 0,
            active: false,
            selected: Vec::new(),
            buys: BTreeMap::new(),
            sells: BTreeMap::new(),
            reserve: 0.0,
            audit: None,
        })
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

impl Strategy for Strategy002 {
    fn name(&self) -> &'static str {
        "stock::strategy_002"
    }

    fn prepare_session(
        &mut self,
        config: &StrategyRunConfig,
        account: &AccountState,
        date: i32,
    ) -> Result<Option<BTreeSet<String>>> {
        self.active = self.sessions % self.hold_days == 0;
        self.selected.clear();
        if self.active {
            let previous = self
                .calendar
                .last_open_on_or_before(date - 1)
                .ok_or_else(|| err("missing previous signal date"))?;
            let mut ranked = load_signal(&self.signal_root, previous, &self.signal_id)?
                .into_iter()
                .filter(|(s, v)| v.is_finite() && !s.ends_with(".BJ"))
                .collect::<Vec<_>>();
            ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            self.selected = ranked
                .into_iter()
                .take(self.top_n)
                .map(|(s, _)| s)
                .collect();
            if self.selected.is_empty() {
                eprintln!("[strategy warning] strategy_002 missing signal on {previous}; exit only on {date}");
            }
        }
        let _ = config;
        Ok(Some(
            account
                .positions()
                .keys()
                .cloned()
                .chain(self.selected.iter().cloned())
                .collect(),
        ))
    }

    fn on_start(&mut self, ctx: &mut StrategyContext) -> Result<()> {
        std::fs::create_dir_all(ctx.config().output_dir())?;
        let mut audit = BufWriter::new(std::fs::File::create(
            ctx.config().output_dir().join("fills.csv"),
        )?);
        writeln!(
            audit,
            "trade_date,trade_time,symbol,side,quantity,price,notional,fee,tax,slippage,cash_after,realized_pnl,net_realized_pnl"
        )?;
        self.audit = Some(audit);
        Ok(())
    }

    fn on_session_open(
        &mut self,
        ctx: &mut StrategyContext,
        _event: &SessionOpenEvent,
    ) -> Result<()> {
        self.sessions += 1;
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
        let budget = (equity - self.reserve).max(0.0) / self.selected.len().max(1) as f64;
        let lot = ctx.config().lot_size;
        let rate = (ctx.config().buy_commission_bps + ctx.config().slippage_bps) / 10000.0;
        for (i, symbol) in self.selected.iter().enumerate() {
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
        let parts = clock.split(':').collect::<Vec<_>>();
        if parts.len() != 3 {
            return Err(err("invalid minute timestamp"));
        }
        let minute = parts[0].parse::<usize>()? * 60 + parts[1].parse::<usize>()?;
        if minute < 570 || minute >= 570 + self.window {
            return Ok(());
        }
        let slot = minute - 570;
        let lot = ctx.config().lot_size;
        let buy_rate = (ctx.config().buy_commission_bps + ctx.config().slippage_bps) / 10000.0;
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
            let quantity = due(leg, slot, self.window, lot)
                .min((ctx.account().sellable_quantity(symbol) - today_buys).max(0.0));
            if quantity > 0.0 {
                cash += quantity * bar.open * (1.0 - sell_rate);
                ctx.order_quantity(symbol.clone(), -quantity);
            }
        }
        // Sell proceeds are available to buys at the same minute open. The 2% reserve is protected.
        for (symbol, leg) in &self.buys {
            let Some(bar) = ctx.market().bar(symbol) else {
                continue;
            };
            if bar.is_limit_up {
                continue;
            }
            let price = bar.open;
            let affordable =
                ((cash - self.reserve).max(0.0) / (price * (1.0 + buy_rate)) / lot).floor() * lot;
            let quantity = due(leg, slot, self.window, lot).min(affordable);
            if quantity > 0.0 {
                cash -= quantity * price * (1.0 + buy_rate);
                ctx.order_quantity(symbol.clone(), quantity);
            }
        }
        Ok(())
    }
    fn on_bar(&mut self, _ctx: &mut StrategyContext, _event: &BarEvent) -> Result<()> {
        Ok(())
    }
    fn on_fill(&mut self, _ctx: &mut StrategyContext, fill: &FillEvent) -> Result<()> {
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
        if let Some(file) = &mut self.audit {
            file.flush()?;
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

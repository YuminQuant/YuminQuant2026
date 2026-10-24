use crate::artifact_io::StoragePathExt;
use std::cmp::Ordering;
use std::collections::BTreeMap;

use crate::calendar::TradingCalendar;
use crate::data::parquet_io::read_parquet;
use crate::error::Result;
use crate::strategy::context::StrategyContext;
use crate::strategy::market::{BarEvent, SessionOpenEvent};
use crate::strategy::strategy::Strategy;

#[derive(Clone, Debug)]
pub struct Strategy001 {
    signal_id: String,
    signal_root: std::path::PathBuf,
    rebalance_days: usize,
    top_n: usize,
    exclude_top_n: usize,
    cash_buffer: f64,
    seen_sessions: usize,
    calendar: Option<TradingCalendar>,
}

impl Strategy001 {
    pub fn from_context_config(config: &crate::strategy::config::StrategyRunConfig) -> Self {
        let params = &config.strategy_params;
        let signal_id = params
            .get("signal_id")
            .cloned()
            .unwrap_or_else(|| "mdl_000019".to_string());
        let signal_root = params
            .get("signal_root")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| config.model_root.clone());
        Self {
            signal_id,
            signal_root,
            rebalance_days: parse_usize(params, "rebalance_days", 5).max(1),
            top_n: parse_usize(params, "top_n", 20).max(1),
            exclude_top_n: parse_usize(params, "exclude_top_n", 0),
            cash_buffer: parse_f64(params, "cash_buffer", 0.0).clamp(0.0, 0.99),
            seen_sessions: 0,
            calendar: None,
        }
    }
}

impl Strategy for Strategy001 {
    fn name(&self) -> &'static str {
        "stock::strategy_001"
    }

    fn on_start(&mut self, ctx: &mut StrategyContext) -> Result<()> {
        if self.exclude_top_n >= self.top_n {
            return Err(crate::error::err(
                "exclude_top_n must be smaller than top_n",
            ));
        }
        self.calendar = Some(TradingCalendar::load(&ctx.config().data_root, "SSE")?);
        Ok(())
    }

    fn on_session_open(
        &mut self,
        ctx: &mut StrategyContext,
        event: &SessionOpenEvent,
    ) -> Result<()> {
        let rebalance = self.seen_sessions % self.rebalance_days == 0;
        self.seen_sessions += 1;
        if !rebalance {
            return Ok(());
        }
        // First-day entry uses the previous trading day's signal, never today's close.
        let Some(signal_date) = self
            .calendar
            .as_ref()
            .and_then(|calendar| calendar.last_open_on_or_before(event.trade_date - 1))
        else {
            return Err(crate::error::err(
                "strategy_001 requires a previous trading date for its opening signal",
            ));
        };
        let signal = load_signal(&self.signal_root, signal_date, &self.signal_id)?;
        if signal.is_empty() {
            eprintln!("[strategy warning] strategy_001 missing signal on {signal_date}; closing prior batch without new entries on {}", event.trade_date);
        }
        let mut ranked = signal
            .into_iter()
            .filter(|(symbol, value)| {
                value.is_finite() && !symbol.to_ascii_uppercase().ends_with(".BJ")
            })
            .collect::<Vec<_>>();
        ranked.sort_by(|left, right| right.1.partial_cmp(&left.1).unwrap_or(Ordering::Equal));
        ranked.truncate(self.top_n);
        let symbols = ranked
            .into_iter()
            .skip(self.exclude_top_n)
            .map(|(symbol, _)| symbol)
            .collect::<Vec<_>>();
        for (symbol, quantity) in opening_allocation(ctx, &symbols, self.cash_buffer) {
            ctx.order_quantity(symbol, quantity);
        }
        Ok(())
    }

    fn on_bar(&mut self, _ctx: &mut StrategyContext, _event: &BarEvent) -> Result<()> {
        Ok(())
    }
}

/// Close the preceding five-session batch, then open a fresh equal-budget batch.
/// Unfilled Top-N allocations stay in cash; never redistribute them or replace names.
fn opening_allocation(
    ctx: &StrategyContext,
    selected: &[String],
    cash_buffer: f64,
) -> Vec<(String, f64)> {
    let config = ctx.config();
    let lot = config.lot_size.max(1.0);
    let buy_rate = (config.buy_commission_bps + config.slippage_bps) / 10_000.0;
    let sell_rate =
        (config.sell_commission_bps + config.stamp_tax_bps + config.slippage_bps) / 10_000.0;
    let mut orders = Vec::new();
    let mut cash = ctx.account().cash();
    let mut retained = BTreeMap::new();
    for (symbol, position) in ctx.account().positions() {
        let sellable = ctx
            .market()
            .bar(symbol)
            .is_some_and(|b| !config.respect_price_limits || !b.is_limit_down);
        let quantity = if sellable {
            ctx.account().sellable_quantity(symbol)
        } else {
            0.0
        };
        if let Some(price) = ctx.market().open_price(symbol) {
            if quantity > 0.0 {
                orders.push((symbol.clone(), -quantity));
                cash += quantity * price * (1.0 - sell_rate);
            }
            retained.insert(symbol.clone(), position.quantity - quantity);
        } else {
            retained.insert(symbol.clone(), position.quantity);
        }
    }
    if selected.is_empty() {
        return orders;
    }
    let retained_value: f64 = retained
        .iter()
        .map(|(symbol, quantity)| {
            quantity
                * ctx
                    .market()
                    .open_price(symbol)
                    .unwrap_or_else(|| ctx.account().position(symbol).unwrap().mark_price())
        })
        .sum();
    let equity = cash + retained_value;
    let reserve = equity.max(0.0) * cash_buffer;
    let target_value = (equity - reserve).max(0.0) / selected.len() as f64;
    let available = (cash - reserve).max(0.0);
    let desired = selected
        .iter()
        .filter_map(|symbol| {
            let bar = ctx.market().bar(symbol)?;
            let price = ctx.market().open_price(symbol)?;
            // A failed liquidation remains the old batch; do not add another batch to it.
            let budget = if (config.respect_price_limits && bar.is_limit_up)
                || retained.get(symbol).copied().unwrap_or(0.0) > 1e-9
            {
                0.0
            } else {
                target_value
            };
            Some((symbol, price, budget))
        })
        .collect::<Vec<_>>();
    let total_budget: f64 = desired.iter().map(|(_, _, budget)| budget).sum();
    let scale = if total_budget > available {
        available / total_budget
    } else {
        1.0
    };
    for (symbol, price, budget) in desired {
        let quantity = (budget * scale / (price * (1.0 + buy_rate)) / lot).floor() * lot;
        if quantity > 0.0 {
            orders.push((symbol.clone(), quantity));
        }
    }
    orders
}

pub(super) fn load_signal(
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

fn parse_usize(values: &BTreeMap<String, String>, key: &str, default: usize) -> usize {
    values
        .get(key)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(default)
}

fn parse_f64(values: &BTreeMap<String, String>, key: &str, default: f64) -> f64 {
    values
        .get(key)
        .and_then(|value| value.parse::<f64>().ok())
        .unwrap_or(default)
}

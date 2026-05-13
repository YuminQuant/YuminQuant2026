use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

use crate::data::parquet_io::read_parquet;
use crate::error::Result;
use crate::strategy::context::StrategyContext;
use crate::strategy::market::BarEvent;
use crate::strategy::strategy::Strategy;

#[derive(Clone, Debug)]
pub struct Strategy001 {
    signal_id: String,
    signal_root: std::path::PathBuf,
    rebalance_days: usize,
    top_n: usize,
    cash_buffer: f64,
    seen_sessions: usize,
}

impl Strategy001 {
    pub fn from_context_config(config: &crate::strategy::config::StrategyRunConfig) -> Self {
        let params = &config.strategy_params;
        let signal_id = params
            .get("signal_id")
            .cloned()
            .unwrap_or_else(|| "ml_alpha_xgb".to_string());
        let signal_root = params
            .get("signal_root")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| config.model_root.clone());
        Self {
            signal_id,
            signal_root,
            rebalance_days: parse_usize(params, "rebalance_days", 5).max(1),
            top_n: parse_usize(params, "top_n", 20).max(1),
            cash_buffer: parse_f64(params, "cash_buffer", 0.02).clamp(0.0, 0.99),
            seen_sessions: 0,
        }
    }
}

impl Strategy for Strategy001 {
    fn name(&self) -> &'static str {
        "stock::strategy_001"
    }

    fn on_bar(&mut self, ctx: &mut StrategyContext, event: &BarEvent) -> Result<()> {
        if !event.is_session_end {
            return Ok(());
        }
        self.seen_sessions += 1;
        if (self.seen_sessions - 1) % self.rebalance_days != 0 {
            return Ok(());
        }

        let signal = load_signal(&self.signal_root, event.trade_date, &self.signal_id)?;
        if signal.is_empty() {
            return Ok(());
        }
        let mut ranked = signal
            .into_iter()
            .filter(|(symbol, value)| {
                value.is_finite()
                    && !symbol.to_ascii_uppercase().ends_with(".BJ")
                    && ctx.market().bar(symbol).is_some()
            })
            .collect::<Vec<_>>();
        ranked.sort_by(|left, right| right.1.partial_cmp(&left.1).unwrap_or(Ordering::Equal));
        ranked.truncate(self.top_n);
        let selected = ranked
            .iter()
            .map(|(symbol, _)| symbol.clone())
            .collect::<BTreeSet<_>>();

        for (symbol, position) in ctx.account().positions().clone() {
            if position.quantity > 0.0 && !selected.contains(&symbol) {
                ctx.order_target_quantity(symbol, 0.0);
            }
        }
        if ranked.is_empty() {
            return Ok(());
        }
        let target_value = ctx.account().equity() * (1.0 - self.cash_buffer) / ranked.len() as f64;
        for (symbol, _) in ranked {
            ctx.order_target_value(symbol, target_value);
        }
        Ok(())
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
    if !path.exists() {
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

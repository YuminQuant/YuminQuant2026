use std::collections::{BTreeMap, VecDeque};

use crate::error::Result;
use crate::strategy::config::{future_product_from_symbol, StrategyRunConfig};
use crate::strategy::context::StrategyContext;
use crate::strategy::market::BarEvent;
use crate::strategy::strategy::Strategy;

#[derive(Clone, Debug)]
pub struct SmaStrategy {
    product: String,
    window: usize,
    capital_per_contract: f64,
    add_quantity: f64,
    closes: BTreeMap<String, VecDeque<f64>>,
}

impl SmaStrategy {
    pub fn from_context_config(config: &StrategyRunConfig) -> Self {
        let params = &config.strategy_params;
        Self {
            product: params
                .get("product")
                .cloned()
                .unwrap_or_else(|| "AG".to_string())
                .to_ascii_uppercase(),
            window: parse_usize(params, "window", 20).max(1),
            capital_per_contract: parse_f64(params, "capital_per_contract", 500_000.0).max(0.0),
            add_quantity: parse_f64(params, "add_quantity", 1.0).max(0.0),
            closes: BTreeMap::new(),
        }
    }
}

impl Strategy for SmaStrategy {
    fn name(&self) -> &'static str {
        "future::sma"
    }

    fn on_bar(&mut self, ctx: &mut StrategyContext, _event: &BarEvent) -> Result<()> {
        let symbols = ctx
            .market()
            .symbols()
            .filter(|symbol| future_product_from_symbol(symbol) == self.product)
            .cloned()
            .collect::<Vec<_>>();

        for symbol in symbols {
            let Some(bar) = ctx.market().bar(&symbol) else {
                continue;
            };
            let close = bar.close;
            if !close.is_finite() || close <= 0.0 {
                continue;
            }

            let history = self.closes.entry(symbol.clone()).or_default();
            history.push_back(close);
            while history.len() > self.window {
                history.pop_front();
            }
            if history.len() < self.window {
                continue;
            }
            let sma = history.iter().sum::<f64>() / self.window as f64;
            if !sma.is_finite() || (close - sma).abs() <= 1e-12 {
                continue;
            }

            let base_qty = base_quantity(
                self.capital_per_contract,
                ctx.config().future.max_margin_ratio,
                close,
                bar.multiplier,
                ctx.config().future_margin_ratio(&symbol),
                ctx.config().lot_size,
            );
            if base_qty <= 0.0 {
                continue;
            }

            let current = ctx.account().position_quantity(&symbol);
            let target = if close > sma {
                if current > 0.0 {
                    current + self.add_quantity.max(ctx.config().lot_size)
                } else {
                    base_qty
                }
            } else if current < 0.0 {
                current - self.add_quantity.max(ctx.config().lot_size)
            } else {
                -base_qty
            };
            ctx.order_target_quantity(symbol, target);
        }

        Ok(())
    }
}

fn base_quantity(
    capital_per_contract: f64,
    max_margin_ratio: f64,
    price: f64,
    multiplier: f64,
    margin_ratio: f64,
    lot_size: f64,
) -> f64 {
    let single_margin = price * multiplier.max(1.0) * margin_ratio.max(0.0);
    if single_margin <= 0.0 || !single_margin.is_finite() {
        return 0.0;
    }
    let budget = capital_per_contract.max(0.0) * max_margin_ratio.clamp(0.0, 1.0);
    let lot = lot_size.max(1.0);
    (budget / single_margin / lot).floor() * lot
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

#[cfg(test)]
mod tests {
    use super::base_quantity;

    #[test]
    fn base_quantity_uses_contract_budget_and_lot() {
        let qty = base_quantity(500_000.0, 0.30, 10_000.0, 15.0, 0.12, 1.0);
        assert_eq!(qty, 8.0);
    }
}

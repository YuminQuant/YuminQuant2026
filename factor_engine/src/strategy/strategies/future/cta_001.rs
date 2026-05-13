use crate::error::Result;
use crate::strategy::context::StrategyContext;
use crate::strategy::market::BarEvent;
use crate::strategy::strategy::Strategy;

#[derive(Clone, Debug)]
pub struct Cta001 {
    symbol: String,
    target_quantity: f64,
    rebalance_days: usize,
    seen_sessions: usize,
}

impl Cta001 {
    pub fn from_context_config(config: &crate::strategy::config::StrategyRunConfig) -> Self {
        let params = &config.strategy_params;
        Self {
            symbol: params
                .get("symbol")
                .cloned()
                .unwrap_or_else(|| "IF.CFX".to_string()),
            target_quantity: params
                .get("target_quantity")
                .and_then(|value| value.parse::<f64>().ok())
                .unwrap_or(1.0),
            rebalance_days: params
                .get("rebalance_days")
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(1)
                .max(1),
            seen_sessions: 0,
        }
    }
}

impl Strategy for Cta001 {
    fn name(&self) -> &'static str {
        "future::cta_001"
    }

    fn on_bar(&mut self, ctx: &mut StrategyContext, event: &BarEvent) -> Result<()> {
        if !event.is_session_end {
            return Ok(());
        }
        self.seen_sessions += 1;
        if (self.seen_sessions - 1) % self.rebalance_days != 0 {
            return Ok(());
        }
        if ctx.market().bar(&self.symbol).is_none() {
            return Ok(());
        }
        ctx.order_target_quantity(self.symbol.clone(), self.target_quantity);
        Ok(())
    }
}

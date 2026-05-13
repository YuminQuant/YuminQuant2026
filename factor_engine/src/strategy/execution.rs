use crate::strategy::account::AccountState;
use crate::strategy::config::StrategyRunConfig;
use crate::strategy::market::MarketSnapshot;
use crate::strategy::order::{FillEvent, Order, OrderSide};

#[derive(Clone, Debug)]
pub struct ExecutionEngine {
    fill_id: i64,
}

impl ExecutionEngine {
    pub fn new() -> Self {
        Self { fill_id: 0 }
    }

    pub fn fill_orders(
        &mut self,
        config: &StrategyRunConfig,
        account: &mut AccountState,
        market: &MarketSnapshot,
        orders: Vec<Order>,
    ) -> Vec<FillEvent> {
        let mut fills = Vec::new();
        for order in orders {
            let Some(bar) = market.bar(&order.symbol) else {
                continue;
            };
            let Some(price) = market.open_price(&order.symbol) else {
                continue;
            };
            let side = OrderSide::from_signed_quantity(order.signed_quantity);
            if side == OrderSide::Buy && bar.is_limit_up {
                continue;
            }
            if side == OrderSide::Sell && bar.is_limit_down {
                continue;
            }

            let requested_qty = order.signed_quantity.abs();
            let quantity = match side {
                OrderSide::Buy => {
                    affordable_buy_quantity(account.cash(), requested_qty, price, config)
                }
                OrderSide::Sell => requested_qty.min(account.position_quantity(&order.symbol)),
                _ => 0.0,
            };
            if quantity <= 0.0 {
                continue;
            }

            let notional = quantity * price;
            let fee = notional * config.commission_bps / 10_000.0;
            let tax = if side == OrderSide::Sell {
                notional * config.stamp_tax_bps / 10_000.0
            } else {
                0.0
            };
            let slippage_cost = notional * config.slippage_bps / 10_000.0;
            let total_cost = fee + tax + slippage_cost;
            let realized_pnl = if side == OrderSide::Buy {
                account.apply_buy(&order.symbol, quantity, price, total_cost);
                account.record_buy_cost(total_cost);
                0.0
            } else {
                account.apply_sell(&order.symbol, quantity, price, total_cost)
            };
            account.mark_price(&order.symbol, price);
            self.fill_id += 1;
            let position = account.position(&order.symbol).cloned().unwrap_or_default();
            let fill_time = format!("{} {}", market.event.trade_date, market.event.trade_time);
            fills.push(FillEvent {
                strategy_id: config.strategy_id.clone(),
                asset_class: config.asset_class.as_str().to_string(),
                trade_date: market.event.trade_date,
                trade_time: market.event.trade_time.clone(),
                bar_frequency: market.event.bar_frequency.as_str().to_string(),
                symbol: order.symbol,
                order_id: order.order_id,
                fill_id: self.fill_id,
                side,
                quantity,
                signed_quantity: if side == OrderSide::Buy {
                    quantity
                } else {
                    -quantity
                },
                fill_price: price,
                notional,
                fee,
                tax,
                slippage_cost,
                realized_pnl,
                net_realized_pnl: realized_pnl - total_cost,
                cash_after: account.cash(),
                position_qty_after: position.quantity,
                avg_cost_after: position.avg_cost,
                unrealized_pnl_after: account.total_unrealized_pnl(),
                account_pnl_after: account.account_pnl(),
                signal_time: order.signal_time,
                fill_time,
            });
        }
        fills
    }
}

fn affordable_buy_quantity(
    cash: f64,
    requested_qty: f64,
    price: f64,
    config: &StrategyRunConfig,
) -> f64 {
    let lot = config.lot_size.max(1.0);
    let unit_cost = price * (1.0 + (config.commission_bps + config.slippage_bps) / 10_000.0);
    if unit_cost <= 0.0 || !unit_cost.is_finite() {
        return 0.0;
    }
    let affordable = (cash / unit_cost / lot).floor() * lot;
    let requested = (requested_qty / lot).floor() * lot;
    requested.min(affordable).max(0.0)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::path::PathBuf;

    use crate::strategy::account::AccountState;
    use crate::strategy::config::{BarFrequency, FillPrice, StrategyAssetClass, StrategyRunConfig};
    use crate::strategy::market::{Bar, BarEvent, MarketSnapshot};
    use crate::strategy::order::Order;

    use super::ExecutionEngine;

    fn config() -> StrategyRunConfig {
        StrategyRunConfig {
            asset_class: StrategyAssetClass::Stock,
            strategy_id: "s".to_string(),
            strategy_class: "stock::strategy_001".to_string(),
            start_date: 20200101,
            end_date: 20200102,
            initial_cash: 10_000.0,
            bar_frequency: BarFrequency::Daily,
            fill_price: FillPrice::NextOpen,
            commission_bps: 10.0,
            stamp_tax_bps: 10.0,
            slippage_bps: 0.0,
            lot_size: 100.0,
            data_root: PathBuf::new(),
            model_root: PathBuf::new(),
            factor_root: PathBuf::new(),
            strategy_params: BTreeMap::new(),
        }
    }

    fn market(open: f64) -> MarketSnapshot {
        MarketSnapshot::new(
            BarEvent {
                trade_date: 20200102,
                trade_time: "daily".to_string(),
                bar_frequency: BarFrequency::Daily,
                is_session_first: true,
                is_session_end: true,
            },
            vec![Bar {
                symbol: "a".to_string(),
                trade_date: 20200102,
                trade_time: "daily".to_string(),
                open,
                high: open,
                low: open,
                close: open,
                volume: None,
                amount: None,
                is_limit_up: false,
                is_limit_down: false,
            }],
        )
    }

    #[test]
    fn fills_buy_at_open_with_lot_and_costs() {
        let mut account = AccountState::new(10_000.0);
        let mut execution = ExecutionEngine::new();
        let fills = execution.fill_orders(
            &config(),
            &mut account,
            &market(10.0),
            vec![Order {
                order_id: 1,
                symbol: "a".to_string(),
                signed_quantity: 1_000.0,
                signal_time: "20200101 daily".to_string(),
            }],
        );
        assert_eq!(fills.len(), 1);
        assert_eq!(fills[0].quantity, 900.0);
        assert!(fills[0].net_realized_pnl < 0.0);
    }
}

use crate::strategy::account::AccountState;
use crate::strategy::config::{StrategyAssetClass, StrategyRunConfig};
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
            match config.asset_class {
                StrategyAssetClass::Stock => {
                    self.fill_stock_order(config, account, market, order, &mut fills)
                }
                StrategyAssetClass::Future => {
                    self.fill_future_order(config, account, market, order, &mut fills)
                }
                StrategyAssetClass::MultiAsset => {}
            }
        }
        fills
    }

    fn fill_stock_order(
        &mut self,
        config: &StrategyRunConfig,
        account: &mut AccountState,
        market: &MarketSnapshot,
        order: Order,
        fills: &mut Vec<FillEvent>,
    ) {
        let Some(bar) = market.bar(&order.symbol) else {
            return;
        };
        let Some(price) = market.open_price(&order.symbol) else {
            return;
        };
        let side = OrderSide::from_signed_quantity(order.signed_quantity);
        if side == OrderSide::Buy && bar.is_limit_up {
            return;
        }
        if side == OrderSide::Sell && bar.is_limit_down {
            return;
        }

        let requested_qty = order.signed_quantity.abs();
        let quantity = match side {
            OrderSide::Buy => affordable_buy_quantity(account.cash(), requested_qty, price, config),
            OrderSide::Sell => requested_qty.min(account.position_quantity(&order.symbol)),
            _ => 0.0,
        };
        if quantity <= 0.0 {
            return;
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
        self.push_fill(
            config,
            account,
            market,
            order,
            side,
            quantity,
            if side == OrderSide::Buy {
                quantity
            } else {
                -quantity
            },
            price,
            notional,
            fee,
            tax,
            slippage_cost,
            realized_pnl,
            fills,
        );
    }

    fn fill_future_order(
        &mut self,
        config: &StrategyRunConfig,
        account: &mut AccountState,
        market: &MarketSnapshot,
        order: Order,
        fills: &mut Vec<FillEvent>,
    ) {
        let Some(bar) = market.bar(&order.symbol) else {
            return;
        };
        let Some(price) = market.open_price(&order.symbol) else {
            return;
        };
        let multiplier = bar.multiplier.max(1.0);
        let margin_ratio = config.future_margin_ratio(&order.symbol);
        let lot = config.lot_size.max(1.0);
        let mut remaining = round_down_lot(order.signed_quantity.abs(), lot);
        if remaining <= 0.0 {
            return;
        }

        if order.signed_quantity > 0.0 {
            let current = account.position_quantity(&order.symbol);
            if current < 0.0 {
                let close_qty = remaining.min(-current);
                self.fill_future_segment(
                    config,
                    account,
                    market,
                    &order,
                    OrderSide::Cover,
                    close_qty,
                    close_qty,
                    price,
                    multiplier,
                    margin_ratio,
                    fills,
                );
                remaining -= close_qty;
            }
            let open_qty = affordable_future_open_quantity(
                account,
                remaining,
                price,
                config,
                multiplier,
                margin_ratio,
            );
            if open_qty > 0.0 {
                self.fill_future_segment(
                    config,
                    account,
                    market,
                    &order,
                    OrderSide::Buy,
                    open_qty,
                    open_qty,
                    price,
                    multiplier,
                    margin_ratio,
                    fills,
                );
            }
        } else {
            let current = account.position_quantity(&order.symbol);
            if current > 0.0 {
                let close_qty = remaining.min(current);
                self.fill_future_segment(
                    config,
                    account,
                    market,
                    &order,
                    OrderSide::Sell,
                    close_qty,
                    -close_qty,
                    price,
                    multiplier,
                    margin_ratio,
                    fills,
                );
                remaining -= close_qty;
            }
            let open_qty = affordable_future_open_quantity(
                account,
                remaining,
                price,
                config,
                multiplier,
                margin_ratio,
            );
            if open_qty > 0.0 {
                self.fill_future_segment(
                    config,
                    account,
                    market,
                    &order,
                    OrderSide::Short,
                    open_qty,
                    -open_qty,
                    price,
                    multiplier,
                    margin_ratio,
                    fills,
                );
            }
        }
    }

    fn fill_future_segment(
        &mut self,
        config: &StrategyRunConfig,
        account: &mut AccountState,
        market: &MarketSnapshot,
        order: &Order,
        side: OrderSide,
        quantity: f64,
        signed_quantity: f64,
        price: f64,
        multiplier: f64,
        margin_ratio: f64,
        fills: &mut Vec<FillEvent>,
    ) {
        if quantity <= 0.0 {
            return;
        }
        let notional = quantity * price * multiplier;
        let fee = notional * config.commission_bps / 10_000.0;
        let tax = 0.0;
        let slippage_cost = notional * config.slippage_bps / 10_000.0;
        let total_cost = fee + tax + slippage_cost;
        let realized_pnl = account.apply_derivative_trade(
            &order.symbol,
            signed_quantity,
            price,
            total_cost,
            multiplier,
            margin_ratio,
        );
        account.mark_price_with_spec(&order.symbol, price, multiplier, margin_ratio, false);
        self.push_fill(
            config,
            account,
            market,
            order.clone(),
            side,
            quantity,
            signed_quantity,
            price,
            notional,
            fee,
            tax,
            slippage_cost,
            realized_pnl,
            fills,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn push_fill(
        &mut self,
        config: &StrategyRunConfig,
        account: &AccountState,
        market: &MarketSnapshot,
        order: Order,
        side: OrderSide,
        quantity: f64,
        signed_quantity: f64,
        price: f64,
        notional: f64,
        fee: f64,
        tax: f64,
        slippage_cost: f64,
        realized_pnl: f64,
        fills: &mut Vec<FillEvent>,
    ) {
        let total_cost = fee + tax + slippage_cost;
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
            signed_quantity,
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

fn affordable_future_open_quantity(
    account: &AccountState,
    requested_qty: f64,
    price: f64,
    config: &StrategyRunConfig,
    multiplier: f64,
    margin_ratio: f64,
) -> f64 {
    let lot = config.lot_size.max(1.0);
    let requested = round_down_lot(requested_qty, lot);
    if requested <= 0.0 {
        return 0.0;
    }
    let unit_notional = price * multiplier.max(1.0);
    let unit_need = unit_notional * margin_ratio.max(0.0)
        + unit_notional * (config.commission_bps + config.slippage_bps) / 10_000.0;
    if unit_need <= 0.0 || !unit_need.is_finite() {
        return requested;
    }
    let affordable = round_down_lot(account.available_margin().max(0.0) / unit_need, lot);
    requested.min(affordable).max(0.0)
}

fn round_down_lot(quantity: f64, lot: f64) -> f64 {
    if quantity <= 0.0 || !quantity.is_finite() || lot <= 0.0 || !lot.is_finite() {
        return 0.0;
    }
    (quantity / lot).floor() * lot
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
            future: Default::default(),
            strategy_params: BTreeMap::new(),
        }
    }

    fn future_config(initial_cash: f64) -> StrategyRunConfig {
        StrategyRunConfig {
            asset_class: StrategyAssetClass::Future,
            strategy_id: "cta".to_string(),
            strategy_class: "future::cta_001".to_string(),
            start_date: 20200101,
            end_date: 20200102,
            initial_cash,
            bar_frequency: BarFrequency::Daily,
            fill_price: FillPrice::NextOpen,
            commission_bps: 0.0,
            stamp_tax_bps: 0.0,
            slippage_bps: 0.0,
            lot_size: 1.0,
            data_root: PathBuf::new(),
            model_root: PathBuf::new(),
            factor_root: PathBuf::new(),
            future: Default::default(),
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
                settle: None,
                multiplier: 1.0,
                volume: None,
                amount: None,
                is_limit_up: false,
                is_limit_down: false,
            }],
        )
    }

    fn future_market(open: f64) -> MarketSnapshot {
        MarketSnapshot::new(
            BarEvent {
                trade_date: 20200102,
                trade_time: "daily".to_string(),
                bar_frequency: BarFrequency::Daily,
                is_session_first: true,
                is_session_end: true,
            },
            vec![Bar {
                symbol: "IF2301.CFX".to_string(),
                trade_date: 20200102,
                trade_time: "daily".to_string(),
                open,
                high: open,
                low: open,
                close: open,
                settle: Some(open),
                multiplier: 300.0,
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

    #[test]
    fn futures_short_cover_and_reversal_split_fills() {
        let mut account = AccountState::new(100_000.0);
        let mut execution = ExecutionEngine::new();
        let cfg = future_config(100_000.0);
        let fills = execution.fill_orders(
            &cfg,
            &mut account,
            &future_market(100.0),
            vec![Order {
                order_id: 1,
                symbol: "IF2301.CFX".to_string(),
                signed_quantity: -1.0,
                signal_time: "20200101 daily".to_string(),
            }],
        );
        assert_eq!(fills.len(), 1);
        assert_eq!(fills[0].side, crate::strategy::order::OrderSide::Short);
        assert_eq!(account.position_quantity("IF2301.CFX"), -1.0);

        let fills = execution.fill_orders(
            &cfg,
            &mut account,
            &future_market(90.0),
            vec![Order {
                order_id: 2,
                symbol: "IF2301.CFX".to_string(),
                signed_quantity: 2.0,
                signal_time: "20200102 daily".to_string(),
            }],
        );
        assert_eq!(fills.len(), 2);
        assert_eq!(fills[0].side, crate::strategy::order::OrderSide::Cover);
        assert_eq!(fills[1].side, crate::strategy::order::OrderSide::Buy);
        assert!((fills[0].realized_pnl - 3_000.0).abs() < 1e-9);
        assert_eq!(account.position_quantity("IF2301.CFX"), 1.0);
        assert!((account.position("IF2301.CFX").unwrap().avg_cost - 90.0).abs() < 1e-9);
    }

    #[test]
    fn futures_opening_order_is_scaled_by_margin() {
        let mut account = AccountState::new(10_000.0);
        let mut execution = ExecutionEngine::new();
        let cfg = future_config(10_000.0);
        let fills = execution.fill_orders(
            &cfg,
            &mut account,
            &future_market(100.0),
            vec![Order {
                order_id: 1,
                symbol: "IF2301.CFX".to_string(),
                signed_quantity: 10.0,
                signal_time: "20200101 daily".to_string(),
            }],
        );
        assert_eq!(fills.len(), 1);
        assert_eq!(fills[0].quantity, 2.0);
        assert_eq!(account.position_quantity("IF2301.CFX"), 2.0);
        assert!((account.margin_required() - 7_200.0).abs() < 1e-9);
    }
}

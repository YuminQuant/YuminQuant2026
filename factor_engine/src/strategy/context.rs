use crate::strategy::account::{AccountState, Position};
use crate::strategy::config::StrategyRunConfig;
use crate::strategy::market::MarketSnapshot;
use crate::strategy::order::{Order, OrderSide};

pub struct StrategyContext<'a> {
    config: &'a StrategyRunConfig,
    account: &'a AccountState,
    market: &'a MarketSnapshot,
    orders: Vec<Order>,
    next_order_id: &'a mut i64,
}

impl<'a> StrategyContext<'a> {
    pub fn new(
        config: &'a StrategyRunConfig,
        account: &'a AccountState,
        market: &'a MarketSnapshot,
        next_order_id: &'a mut i64,
    ) -> Self {
        Self {
            config,
            account,
            market,
            orders: Vec::new(),
            next_order_id,
        }
    }

    pub fn config(&self) -> &StrategyRunConfig {
        self.config
    }

    pub fn account(&self) -> &AccountState {
        self.account
    }

    pub fn position(&self, symbol: &str) -> Option<&Position> {
        self.account.position(symbol)
    }

    pub fn market(&self) -> &MarketSnapshot {
        self.market
    }

    pub fn order_quantity(&mut self, symbol: impl Into<String>, signed_quantity: f64) {
        self.submit(symbol.into(), signed_quantity, None, self.signal_time());
    }

    pub fn order_future_side(
        &mut self,
        symbol: &str,
        side: OrderSide,
        quantity: f64,
        signal_time: String,
    ) {
        if !quantity.is_finite() || quantity <= 0.0 {
            return;
        }
        let signed = if matches!(side, OrderSide::Buy | OrderSide::Cover) {
            quantity
        } else {
            -quantity
        };
        self.submit(symbol.to_owned(), signed, Some(side), signal_time);
    }

    pub fn signal_time(&self) -> String {
        format!(
            "{} {}",
            self.market.event.trade_date, self.market.event.trade_time
        )
    }

    fn submit(
        &mut self,
        symbol: String,
        signed_quantity: f64,
        future_side: Option<OrderSide>,
        signal_time: String,
    ) {
        if !signed_quantity.is_finite() || signed_quantity.abs() <= 1e-9 {
            return;
        }
        *self.next_order_id += 1;
        self.orders.push(Order {
            order_id: *self.next_order_id,
            symbol,
            signed_quantity,
            signal_time,
            future_side,
        });
    }

    pub fn order_target_quantity(&mut self, symbol: impl Into<String>, target_quantity: f64) {
        let symbol = symbol.into();
        let current = self.account.position_quantity(&symbol);
        self.order_quantity(symbol, target_quantity - current);
    }

    pub fn order_value(&mut self, symbol: impl Into<String>, signed_value: f64) {
        let symbol = symbol.into();
        let Some(price) = self.market.close_price(&symbol) else {
            return;
        };
        self.order_quantity(symbol, signed_value / price);
    }

    pub fn order_target_value(&mut self, symbol: impl Into<String>, target_value: f64) {
        let symbol = symbol.into();
        let Some(price) = self.market.close_price(&symbol) else {
            return;
        };
        self.order_target_quantity(symbol, target_value / price);
    }

    pub fn take_orders(self) -> Vec<Order> {
        self.orders
    }
}

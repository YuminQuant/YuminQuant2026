use std::collections::BTreeMap;

#[derive(Clone, Debug, Default)]
pub struct Position {
    pub quantity: f64,
    pub avg_cost: f64,
    pub last_price: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct AccountState {
    initial_cash: f64,
    cash: f64,
    realized_pnl_cum: f64,
    net_realized_pnl_cum: f64,
    positions: BTreeMap<String, Position>,
}

impl AccountState {
    pub fn new(initial_cash: f64) -> Self {
        Self {
            initial_cash,
            cash: initial_cash,
            realized_pnl_cum: 0.0,
            net_realized_pnl_cum: 0.0,
            positions: BTreeMap::new(),
        }
    }

    pub fn cash(&self) -> f64 {
        self.cash
    }

    pub fn initial_cash(&self) -> f64 {
        self.initial_cash
    }

    pub fn position(&self, symbol: &str) -> Option<&Position> {
        self.positions.get(symbol)
    }

    pub fn position_quantity(&self, symbol: &str) -> f64 {
        self.position(symbol).map(|pos| pos.quantity).unwrap_or(0.0)
    }

    pub fn positions(&self) -> &BTreeMap<String, Position> {
        &self.positions
    }

    pub fn realized_pnl_cum(&self) -> f64 {
        self.realized_pnl_cum
    }

    pub fn net_realized_pnl_cum(&self) -> f64 {
        self.net_realized_pnl_cum
    }

    pub fn mark_price(&mut self, symbol: &str, price: f64) {
        if let Some(position) = self.positions.get_mut(symbol) {
            position.last_price = Some(price);
        }
    }

    pub fn mark_prices<'a>(&mut self, values: impl Iterator<Item = (&'a String, f64)>) {
        for (symbol, price) in values {
            self.mark_price(symbol, price);
        }
    }

    pub fn equity(&self) -> f64 {
        self.cash
            + self
                .positions
                .values()
                .map(|position| {
                    position.quantity * position.last_price.unwrap_or(position.avg_cost)
                })
                .sum::<f64>()
    }

    pub fn total_unrealized_pnl(&self) -> f64 {
        self.positions
            .values()
            .map(|position| {
                position.quantity
                    * (position.last_price.unwrap_or(position.avg_cost) - position.avg_cost)
            })
            .sum()
    }

    pub fn gross_market_value(&self) -> f64 {
        self.positions
            .values()
            .map(|position| {
                (position.quantity * position.last_price.unwrap_or(position.avg_cost)).abs()
            })
            .sum()
    }

    pub fn net_market_value(&self) -> f64 {
        self.positions
            .values()
            .map(|position| position.quantity * position.last_price.unwrap_or(position.avg_cost))
            .sum()
    }

    pub fn account_pnl(&self) -> f64 {
        self.equity() - self.initial_cash
    }

    pub fn apply_buy(&mut self, symbol: &str, quantity: f64, price: f64, total_cost: f64) {
        let notional = quantity * price;
        self.cash -= notional + total_cost;
        let position = self.positions.entry(symbol.to_string()).or_default();
        let new_qty = position.quantity + quantity;
        if new_qty > 0.0 {
            position.avg_cost = if position.quantity > 0.0 {
                (position.avg_cost * position.quantity + notional) / new_qty
            } else {
                price
            };
            position.quantity = new_qty;
            position.last_price = Some(price);
        }
    }

    pub fn apply_sell(&mut self, symbol: &str, quantity: f64, price: f64, total_cost: f64) -> f64 {
        let Some(position) = self.positions.get_mut(symbol) else {
            return 0.0;
        };
        let sell_qty = quantity.min(position.quantity).max(0.0);
        let realized = (price - position.avg_cost) * sell_qty;
        self.cash += sell_qty * price - total_cost;
        position.quantity -= sell_qty;
        position.last_price = Some(price);
        if position.quantity <= 1e-9 {
            self.positions.remove(symbol);
        }
        self.realized_pnl_cum += realized;
        self.net_realized_pnl_cum += realized - total_cost;
        realized
    }

    pub fn record_buy_cost(&mut self, total_cost: f64) {
        self.net_realized_pnl_cum -= total_cost;
    }
}

#[cfg(test)]
mod tests {
    use super::AccountState;

    #[test]
    fn buy_and_sell_update_cash_position_and_realized_pnl() {
        let mut account = AccountState::new(10_000.0);
        account.apply_buy("a", 100.0, 10.0, 1.0);
        assert_eq!(account.position_quantity("a"), 100.0);
        assert!((account.cash() - 8_999.0).abs() < 1e-9);

        let realized = account.apply_sell("a", 40.0, 12.0, 1.0);
        assert!((realized - 80.0).abs() < 1e-9);
        assert_eq!(account.position_quantity("a"), 60.0);
        assert!((account.cash() - 9_478.0).abs() < 1e-9);
    }
}

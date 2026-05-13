use std::collections::BTreeMap;

#[derive(Clone, Debug, Default)]
pub struct Position {
    pub quantity: f64,
    pub avg_cost: f64,
    pub last_price: Option<f64>,
    pub multiplier: f64,
    pub margin_ratio: f64,
    pub cash_settled: bool,
}

impl Position {
    pub fn direction(&self) -> &'static str {
        if self.quantity < 0.0 {
            "short"
        } else {
            "long"
        }
    }

    pub fn mark_price(&self) -> f64 {
        self.last_price.unwrap_or(self.avg_cost)
    }

    pub fn market_value(&self) -> f64 {
        self.quantity * self.mark_price() * self.multiplier.max(1.0)
    }

    pub fn unrealized_pnl(&self) -> f64 {
        self.quantity * (self.mark_price() - self.avg_cost) * self.multiplier.max(1.0)
    }

    pub fn margin_value(&self) -> f64 {
        if self.cash_settled {
            0.0
        } else {
            self.quantity.abs()
                * self.mark_price()
                * self.multiplier.max(1.0)
                * self.margin_ratio.max(0.0)
        }
    }

    fn equity_contribution(&self) -> f64 {
        if self.cash_settled {
            self.market_value()
        } else {
            self.unrealized_pnl()
        }
    }
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

    pub fn mark_price_with_spec(
        &mut self,
        symbol: &str,
        price: f64,
        multiplier: f64,
        margin_ratio: f64,
        cash_settled: bool,
    ) {
        if let Some(position) = self.positions.get_mut(symbol) {
            position.last_price = Some(price);
            position.multiplier = multiplier.max(1.0);
            position.margin_ratio = margin_ratio.max(0.0);
            position.cash_settled = cash_settled;
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
                .map(Position::equity_contribution)
                .sum::<f64>()
    }

    pub fn total_unrealized_pnl(&self) -> f64 {
        self.positions.values().map(Position::unrealized_pnl).sum()
    }

    pub fn gross_market_value(&self) -> f64 {
        self.positions
            .values()
            .map(|position| position.market_value().abs())
            .sum()
    }

    pub fn net_market_value(&self) -> f64 {
        self.positions.values().map(Position::market_value).sum()
    }

    pub fn account_pnl(&self) -> f64 {
        self.equity() - self.initial_cash
    }

    pub fn margin_required(&self) -> f64 {
        self.positions.values().map(Position::margin_value).sum()
    }

    pub fn available_margin(&self) -> f64 {
        self.equity() - self.margin_required()
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
            position.multiplier = 1.0;
            position.margin_ratio = 0.0;
            position.cash_settled = true;
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

    pub fn apply_derivative_trade(
        &mut self,
        symbol: &str,
        signed_quantity: f64,
        price: f64,
        total_cost: f64,
        multiplier: f64,
        margin_ratio: f64,
    ) -> f64 {
        if signed_quantity.abs() <= 1e-9 {
            return 0.0;
        }
        let multiplier = multiplier.max(1.0);
        let margin_ratio = margin_ratio.max(0.0);
        let current_qty = self.position_quantity(symbol);
        if current_qty.abs() <= 1e-9 || current_qty.signum() == signed_quantity.signum() {
            self.cash -= total_cost;
            self.net_realized_pnl_cum -= total_cost;
            let position = self.positions.entry(symbol.to_string()).or_default();
            let old_abs = position.quantity.abs();
            let add_abs = signed_quantity.abs();
            let new_abs = old_abs + add_abs;
            position.avg_cost = if old_abs > 0.0 {
                (position.avg_cost * old_abs + price * add_abs) / new_abs
            } else {
                price
            };
            position.quantity += signed_quantity;
            position.last_price = Some(price);
            position.multiplier = multiplier;
            position.margin_ratio = margin_ratio;
            position.cash_settled = false;
            return 0.0;
        }

        let Some(position) = self.positions.get_mut(symbol) else {
            return 0.0;
        };
        let close_qty = signed_quantity.abs().min(position.quantity.abs());
        let realized = if position.quantity > 0.0 {
            (price - position.avg_cost) * close_qty * position.multiplier.max(1.0)
        } else {
            (position.avg_cost - price) * close_qty * position.multiplier.max(1.0)
        };
        position.quantity += signed_quantity;
        position.last_price = Some(price);
        position.multiplier = multiplier;
        position.margin_ratio = margin_ratio;
        position.cash_settled = false;
        self.cash += realized - total_cost;
        self.realized_pnl_cum += realized;
        self.net_realized_pnl_cum += realized - total_cost;
        if position.quantity.abs() <= 1e-9 {
            self.positions.remove(symbol);
        }
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

    #[test]
    fn derivative_short_cover_and_multiplier_pnl() {
        let mut account = AccountState::new(10_000.0);
        account.apply_derivative_trade("IF001.CFX", -2.0, 100.0, 1.0, 300.0, 0.12);
        assert_eq!(account.position_quantity("IF001.CFX"), -2.0);
        assert!((account.cash() - 9_999.0).abs() < 1e-9);

        account.mark_price_with_spec("IF001.CFX", 90.0, 300.0, 0.12, false);
        assert!((account.total_unrealized_pnl() - 6_000.0).abs() < 1e-9);
        assert!((account.equity() - 15_999.0).abs() < 1e-9);

        let realized = account.apply_derivative_trade("IF001.CFX", 1.0, 90.0, 1.0, 300.0, 0.12);
        assert!((realized - 3_000.0).abs() < 1e-9);
        assert_eq!(account.position_quantity("IF001.CFX"), -1.0);
        assert!((account.cash() - 12_998.0).abs() < 1e-9);
    }
}

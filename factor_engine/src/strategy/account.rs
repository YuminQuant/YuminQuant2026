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

/// A single-pass valuation of the current positions; never cached across trades/bars.
#[derive(Clone, Copy, Debug, Default)]
pub struct AccountSummary {
    pub equity: f64,
    pub unrealized_pnl: f64,
    pub gross_market_value: f64,
    pub net_market_value: f64,
    pub margin_required: f64,
}

#[derive(Clone, Debug)]
pub struct AccountState {
    initial_cash: f64,
    cash: f64,
    realized_pnl_cum: f64,
    net_realized_pnl_cum: f64,
    positions: BTreeMap<String, Position>,
    dividend_receivables: Vec<(i32, f64)>,
    locked_shares: Vec<(i32, String, f64)>,
    dividend_income: f64,
}

impl AccountState {
    pub fn new(initial_cash: f64) -> Self {
        Self {
            initial_cash,
            cash: initial_cash,
            realized_pnl_cum: 0.0,
            net_realized_pnl_cum: 0.0,
            positions: BTreeMap::new(),
            dividend_receivables: Vec::new(),
            locked_shares: Vec::new(),
            dividend_income: 0.0,
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

    pub(crate) fn positions_mut(&mut self) -> impl Iterator<Item = (&String, &mut Position)> {
        self.positions.iter_mut()
    }

    pub fn summary(&self) -> AccountSummary {
        let mut totals = AccountSummary::default();
        for position in self.positions.values() {
            let market_value = position.market_value();
            let unrealized_pnl = position.unrealized_pnl();
            totals.equity += if position.cash_settled {
                market_value
            } else {
                unrealized_pnl
            };
            totals.unrealized_pnl += unrealized_pnl;
            totals.gross_market_value += market_value.abs();
            totals.net_market_value += market_value;
            totals.margin_required += position.margin_value();
        }
        totals.equity += self.cash + self.dividend_receivable();
        totals
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
            + self.dividend_receivable()
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

    pub fn dividend_receivable(&self) -> f64 {
        self.dividend_receivables
            .iter()
            .map(|(_, amount)| amount)
            .sum()
    }

    pub fn dividend_income(&self) -> f64 {
        self.dividend_income
    }

    pub fn sellable_quantity(&self, symbol: &str) -> f64 {
        let locked: f64 = self
            .locked_shares
            .iter()
            .filter(|(_, code, _)| code == symbol)
            .map(|(_, _, qty)| qty)
            .sum();
        (self.position_quantity(symbol) - locked).max(0.0)
    }

    pub fn settle_stock_entitlements(&mut self, date: i32) {
        self.dividend_receivables.retain(|(pay_date, amount)| {
            if *pay_date <= date {
                self.cash += amount;
                false
            } else {
                true
            }
        });
        self.locked_shares
            .retain(|(listing_date, _, _)| *listing_date > date);
    }

    /// Recognize the registered holder's distribution on ex-date. Cash stays
    /// receivable until payment; bonus shares stay unsellable until listing.
    pub fn apply_stock_distribution(
        &mut self,
        symbol: &str,
        entitled: f64,
        stock_rate: f64,
        cash_rate: f64,
        pay_date: i32,
        listing_date: i32,
        date: i32,
        record_mark: f64,
    ) {
        let bonus = (entitled * stock_rate + 1e-6).floor();
        let cash_income = entitled * cash_rate;
        if bonus > 0.0 {
            let position = self.positions.entry(symbol.to_string()).or_default();
            let basis = position.quantity * position.avg_cost;
            position.quantity += bonus;
            position.avg_cost = basis / position.quantity;
            position.multiplier = 1.0;
            position.cash_settled = true;
            if listing_date > date {
                self.locked_shares
                    .push((listing_date, symbol.to_string(), bonus));
            }
        }
        // Correct stale marks as well, e.g. an ex-date during a suspension.
        if let Some(position) = self.positions.get_mut(symbol) {
            position.last_price = Some(((record_mark - cash_rate) / (1.0 + stock_rate)).max(0.0));
        }
        if cash_income > 0.0 {
            self.dividend_income += cash_income;
            self.realized_pnl_cum += cash_income;
            self.net_realized_pnl_cum += cash_income;
            if pay_date <= date {
                self.cash += cash_income;
            } else {
                self.dividend_receivables.push((pay_date, cash_income));
            }
        }
    }
}

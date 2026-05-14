#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OrderSide {
    Buy,
    Sell,
    Short,
    Cover,
}

impl OrderSide {
    pub fn from_signed_quantity(signed_quantity: f64) -> Self {
        if signed_quantity >= 0.0 {
            Self::Buy
        } else {
            Self::Sell
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Buy => "buy",
            Self::Sell => "sell",
            Self::Short => "short",
            Self::Cover => "cover",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Order {
    pub order_id: i64,
    pub symbol: String,
    pub signed_quantity: f64,
    pub signal_time: String,
}

#[derive(Clone, Debug)]
pub struct FillEvent {
    pub strategy_id: String,
    pub asset_class: String,
    pub trade_date: i32,
    pub trade_time: String,
    pub bar_frequency: String,
    pub symbol: String,
    pub order_id: i64,
    pub fill_id: i64,
    pub side: OrderSide,
    pub quantity: f64,
    pub signed_quantity: f64,
    pub fill_price: f64,
    pub notional: f64,
    pub fee: f64,
    pub tax: f64,
    pub slippage_cost: f64,
    pub realized_pnl: f64,
    pub net_realized_pnl: f64,
    pub cash_after: f64,
    pub position_qty_after: f64,
    pub avg_cost_after: f64,
    pub unrealized_pnl_after: f64,
    pub account_pnl_after: f64,
    pub signal_time: String,
    pub fill_time: String,
}

#[derive(Clone, Debug)]
pub struct HoldingSnapshot {
    pub strategy_id: String,
    pub asset_class: String,
    pub trade_date: i32,
    pub trade_time: String,
    pub cash: f64,
    pub account_pnl: f64,
    pub realized_pnl_cum: f64,
    pub net_realized_pnl_cum: f64,
    pub unrealized_pnl: f64,
    pub gross_market_value: f64,
    pub net_market_value: f64,
    pub margin_required: f64,
    pub available_margin: f64,
    pub position_count: i64,
    pub trade_count: i64,
    pub symbols_json: String,
    pub quantities_json: String,
    pub signed_quantities_json: String,
    pub directions_json: String,
    pub avg_costs_json: String,
    pub prices_json: String,
    pub market_values_json: String,
    pub unrealized_pnls_json: String,
    pub multipliers_json: String,
    pub margin_ratios_json: String,
    pub margin_values_json: String,
    pub trade_symbols_json: String,
    pub trade_sides_json: String,
    pub trade_quantities_json: String,
    pub trade_signed_quantities_json: String,
    pub trade_prices_json: String,
    pub trade_realized_pnls_json: String,
    pub trade_net_pnls_json: String,
    pub trade_fill_ids_json: String,
}

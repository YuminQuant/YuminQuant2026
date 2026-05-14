use std::collections::BTreeMap;
use std::path::PathBuf;

use crate::calendar::TradingCalendar;
use crate::config::EngineConfig;
use crate::error::{err, Result};
use crate::progress::ProgressBar;
use crate::strategy::account::AccountState;
use crate::strategy::config::{BarFrequency, StrategyAssetClass, StrategyRunConfig};
use crate::strategy::context::StrategyContext;
use crate::strategy::execution::ExecutionEngine;
use crate::strategy::market::{
    load_future_daily_frames, load_future_metadata, load_future_minute_frames,
    load_stock_daily_frames, load_stock_minute_frames, Bar, BarEvent, FutureContractMeta,
    MarketFrame, MarketSnapshot, SessionOpenEvent,
};
use crate::strategy::order::{FillEvent, HoldingSnapshot, Order, OrderSide};
use crate::strategy::request::StrategyRunRequest;
use crate::strategy::storage::write_holdings;
use crate::strategy::strategies::future::cta_001::Cta001;
use crate::strategy::strategies::future::sma::SmaStrategy;
use crate::strategy::strategies::stock::strategy_001::Strategy001;
use crate::strategy::strategy::Strategy;

#[derive(Debug)]
pub struct StrategyEngine {
    config: EngineConfig,
}

#[derive(Clone, Debug)]
pub struct StrategyRunReport {
    pub strategy_id: String,
    pub asset_class: String,
    pub trade_count: usize,
    pub output_files: Vec<PathBuf>,
}

impl StrategyEngine {
    pub fn from_request(request: &StrategyRunRequest) -> Result<Self> {
        Ok(Self {
            config: EngineConfig::discover(request.project_config_path.clone())?,
        })
    }

    pub fn run(&self, request: &StrategyRunRequest) -> Result<StrategyRunReport> {
        let run_config = StrategyRunConfig::load(
            &request.config_path,
            self.config.data_root.clone(),
            self.config.factor_root.clone(),
        )?;
        let calendar =
            TradingCalendar::load(&self.config.data_root, &self.config.stock_calendar_exchange)?;
        let dates = calendar.open_dates_between(run_config.start_date, run_config.end_date);
        if dates.is_empty() {
            return Err(err("no trading dates in strategy range"));
        }
        let mut strategy = create_strategy(&run_config)?;
        let mut account = AccountState::new(run_config.initial_cash);
        let mut execution = ExecutionEngine::new();
        let mut pending_orders = Vec::<Order>::new();
        let mut holdings = Vec::<HoldingSnapshot>::new();
        let mut trade_count = 0_usize;
        let mut next_order_id = 0_i64;
        let progress = ProgressBar::new("strategy-run", dates.len(), true);
        let future_meta = if run_config.asset_class == StrategyAssetClass::Future {
            load_future_metadata(&run_config.data_root)?
        } else {
            BTreeMap::new()
        };

        let mut started = false;
        for trade_date in dates {
            let frames = load_frames(&run_config, trade_date, &future_meta)?;
            if frames.is_empty() {
                progress.tick(format!("date={trade_date} frames=0 trades={}", trade_count));
                continue;
            }
            for frame in frames {
                let market = MarketSnapshot::new(frame.event.clone(), frame.bars);
                let mut bar_fills = Vec::<FillEvent>::new();
                if !started {
                    call_start(
                        &mut *strategy,
                        &run_config,
                        &account,
                        &market,
                        &mut next_order_id,
                    )?;
                    started = true;
                }
                if !pending_orders.is_empty() {
                    let fills = execution.fill_orders(
                        &run_config,
                        &mut account,
                        &market,
                        std::mem::take(&mut pending_orders),
                    );
                    call_fills(
                        &mut *strategy,
                        &run_config,
                        &account,
                        &market,
                        &mut next_order_id,
                        &fills,
                    )?;
                    trade_count += fills.len();
                    bar_fills.extend(fills);
                }
                if market.event.is_session_first {
                    let event = SessionOpenEvent {
                        trade_date: market.event.trade_date,
                        trade_time: market.event.trade_time.clone(),
                        bar_frequency: market.event.bar_frequency,
                    };
                    let mut ctx =
                        StrategyContext::new(&run_config, &account, &market, &mut next_order_id);
                    strategy.on_session_open(&mut ctx, &event)?;
                    let open_orders = ctx.take_orders();
                    if !open_orders.is_empty() {
                        let fills =
                            execution.fill_orders(&run_config, &mut account, &market, open_orders);
                        call_fills(
                            &mut *strategy,
                            &run_config,
                            &account,
                            &market,
                            &mut next_order_id,
                            &fills,
                        )?;
                        trade_count += fills.len();
                        bar_fills.extend(fills);
                    }
                }

                mark_account_positions(&run_config, &mut account, &market);
                let mut ctx =
                    StrategyContext::new(&run_config, &account, &market, &mut next_order_id);
                strategy.on_bar(&mut ctx, &market.event)?;
                pending_orders.extend(ctx.take_orders());
                holdings.push(holding_snapshot(&run_config, &account, &market, &bar_fills));
            }
            progress.tick(format!("date={trade_date} trades={trade_count}"));
        }
        if started {
            let end_frame = terminal_frame(&run_config);
            let mut ctx =
                StrategyContext::new(&run_config, &account, &end_frame, &mut next_order_id);
            strategy.on_end(&mut ctx)?;
        }
        progress.finish();
        let output = write_holdings(&run_config.output_dir(), &holdings)?;
        Ok(StrategyRunReport {
            strategy_id: run_config.strategy_id,
            asset_class: run_config.asset_class.as_str().to_string(),
            trade_count,
            output_files: vec![output],
        })
    }
}

fn load_frames(
    config: &StrategyRunConfig,
    trade_date: i32,
    future_meta: &BTreeMap<String, FutureContractMeta>,
) -> Result<Vec<MarketFrame>> {
    let dates = [trade_date];
    match (&config.asset_class, config.bar_frequency) {
        (StrategyAssetClass::Stock, BarFrequency::Daily) => {
            load_stock_daily_frames(&config.data_root, &dates)
        }
        (StrategyAssetClass::Stock, BarFrequency::Minute) => {
            load_stock_minute_frames(&config.data_root, &dates)
        }
        (StrategyAssetClass::Future, BarFrequency::Daily) => {
            load_future_daily_frames(&config.data_root, &dates, future_meta)
        }
        (StrategyAssetClass::Future, BarFrequency::Minute) => {
            load_future_minute_frames(&config.data_root, &dates, future_meta)
        }
        (StrategyAssetClass::MultiAsset, _) => {
            Err(err("multi_asset strategy execution is not implemented"))
        }
    }
}

fn create_strategy(config: &StrategyRunConfig) -> Result<Box<dyn Strategy>> {
    match config.strategy_class.as_str() {
        "stock::strategy_001" | "stock::top_signal_equal_weight" => {
            Ok(Box::new(Strategy001::from_context_config(config)))
        }
        "future::cta_001" => Ok(Box::new(Cta001::from_context_config(config))),
        "future::sma" => Ok(Box::new(SmaStrategy::from_context_config(config))),
        other => Err(err(format!("unknown strategy_class: {other}"))),
    }
}

fn call_start(
    strategy: &mut dyn Strategy,
    config: &StrategyRunConfig,
    account: &AccountState,
    market: &MarketSnapshot,
    next_order_id: &mut i64,
) -> Result<()> {
    let mut ctx = StrategyContext::new(config, account, market, next_order_id);
    strategy.on_start(&mut ctx)
}

fn call_fills(
    strategy: &mut dyn Strategy,
    config: &StrategyRunConfig,
    account: &AccountState,
    market: &MarketSnapshot,
    next_order_id: &mut i64,
    fills: &[FillEvent],
) -> Result<()> {
    for fill in fills {
        let mut ctx = StrategyContext::new(config, account, market, next_order_id);
        strategy.on_fill(&mut ctx, fill)?;
    }
    Ok(())
}

fn terminal_frame(config: &StrategyRunConfig) -> MarketSnapshot {
    MarketSnapshot::new(
        BarEvent {
            trade_date: config.end_date,
            trade_time: "end".to_string(),
            bar_frequency: config.bar_frequency,
            is_session_first: false,
            is_session_end: true,
        },
        Vec::new(),
    )
}

fn holding_snapshot(
    config: &StrategyRunConfig,
    account: &AccountState,
    market: &MarketSnapshot,
    fills: &[FillEvent],
) -> HoldingSnapshot {
    let mut symbols = Vec::new();
    let mut quantities = Vec::new();
    let mut signed_quantities = Vec::new();
    let mut directions = Vec::new();
    let mut avg_costs = Vec::new();
    let mut prices = Vec::new();
    let mut market_values = Vec::new();
    let mut unrealized_pnls = Vec::new();
    let mut multipliers = Vec::new();
    let mut margin_ratios = Vec::new();
    let mut margin_values = Vec::new();
    for (symbol, position) in account.positions() {
        if position.quantity.abs() <= 1e-9 {
            continue;
        }
        let price = position.mark_price();
        symbols.push(symbol.clone());
        quantities.push(position.quantity.abs());
        signed_quantities.push(position.quantity);
        directions.push(position_direction_code(position.quantity));
        avg_costs.push(position.avg_cost);
        prices.push(price);
        market_values.push(position.market_value());
        unrealized_pnls.push(position.unrealized_pnl());
        multipliers.push(position.multiplier.max(1.0));
        margin_ratios.push(position.margin_ratio.max(0.0));
        margin_values.push(position.margin_value());
    }

    HoldingSnapshot {
        strategy_id: config.strategy_id.clone(),
        asset_class: config.asset_class.as_str().to_string(),
        trade_date: market.event.trade_date,
        trade_time: market.event.trade_time.clone(),
        cash: round2(account.cash()),
        account_pnl: account.account_pnl(),
        realized_pnl_cum: account.realized_pnl_cum(),
        net_realized_pnl_cum: account.net_realized_pnl_cum(),
        unrealized_pnl: round2(account.total_unrealized_pnl()),
        gross_market_value: round2(account.gross_market_value()),
        net_market_value: account.net_market_value(),
        margin_required: account.margin_required(),
        available_margin: account.available_margin(),
        position_count: symbols.len() as i64,
        trade_count: fills.len() as i64,
        symbols_json: json_string_array(&symbols),
        quantities_json: json_f64_array(&quantities),
        signed_quantities_json: json_f64_array(&signed_quantities),
        directions_json: json_i64_array(&directions),
        avg_costs_json: json_f64_array_rounded(&avg_costs, 2),
        prices_json: json_f64_array(&prices),
        market_values_json: json_f64_array(&market_values),
        unrealized_pnls_json: json_f64_array_rounded(&unrealized_pnls, 2),
        multipliers_json: json_f64_array(&multipliers),
        margin_ratios_json: json_f64_array(&margin_ratios),
        margin_values_json: json_f64_array(&margin_values),
        trade_symbols_json: json_string_array(
            &fills
                .iter()
                .map(|fill| fill.symbol.clone())
                .collect::<Vec<_>>(),
        ),
        trade_sides_json: json_i64_array(
            &fills
                .iter()
                .map(|fill| trade_side_code(fill.side))
                .collect::<Vec<_>>(),
        ),
        trade_quantities_json: json_f64_array(
            &fills.iter().map(|fill| fill.quantity).collect::<Vec<_>>(),
        ),
        trade_signed_quantities_json: json_f64_array(
            &fills
                .iter()
                .map(|fill| fill.signed_quantity)
                .collect::<Vec<_>>(),
        ),
        trade_prices_json: json_f64_array_rounded(
            &fills.iter().map(|fill| fill.fill_price).collect::<Vec<_>>(),
            2,
        ),
        trade_realized_pnls_json: json_f64_array_rounded(
            &fills
                .iter()
                .map(|fill| fill.realized_pnl)
                .collect::<Vec<_>>(),
            2,
        ),
        trade_net_pnls_json: json_f64_array(
            &fills
                .iter()
                .map(|fill| fill.net_realized_pnl)
                .collect::<Vec<_>>(),
        ),
        trade_fill_ids_json: json_i64_array(
            &fills.iter().map(|fill| fill.fill_id).collect::<Vec<_>>(),
        ),
    }
}

fn mark_account_positions(
    config: &StrategyRunConfig,
    account: &mut AccountState,
    market: &MarketSnapshot,
) {
    for symbol in market.symbols() {
        let Some(bar) = market.bar(symbol) else {
            continue;
        };
        let Some(price) = mark_price_for(config, bar) else {
            continue;
        };
        let cash_settled = config.asset_class == StrategyAssetClass::Stock;
        let margin_ratio = if cash_settled {
            0.0
        } else {
            config.future_margin_ratio(symbol)
        };
        account.mark_price_with_spec(
            symbol,
            price,
            bar.multiplier.max(1.0),
            margin_ratio,
            cash_settled,
        );
    }
}

fn mark_price_for(config: &StrategyRunConfig, bar: &Bar) -> Option<f64> {
    match config.asset_class {
        StrategyAssetClass::Future => match config.future.mark_price {
            crate::strategy::config::FutureMarkPrice::Close => Some(bar.close),
            crate::strategy::config::FutureMarkPrice::Settle => bar.settle.or(Some(bar.close)),
        },
        _ => Some(bar.close),
    }
    .filter(|value| value.is_finite() && *value > 0.0)
}

fn json_string_array(values: &[String]) -> String {
    if values.is_empty() {
        return "[]".to_string();
    }
    format!(
        "[{}]",
        values
            .iter()
            .map(|value| format!("\"{}\"", json_escape(value)))
            .collect::<Vec<_>>()
            .join(",")
    )
}

fn json_f64_array(values: &[f64]) -> String {
    if values.is_empty() {
        return "[]".to_string();
    }
    format!(
        "[{}]",
        values
            .iter()
            .map(|value| {
                if value.is_finite() {
                    format!("{value:.12}")
                        .trim_end_matches('0')
                        .trim_end_matches('.')
                        .to_string()
                } else {
                    "null".to_string()
                }
            })
            .collect::<Vec<_>>()
            .join(",")
    )
}

fn json_f64_array_rounded(values: &[f64], decimals: usize) -> String {
    if values.is_empty() {
        return "[]".to_string();
    }
    let scale = 10_f64.powi(decimals as i32);
    format!(
        "[{}]",
        values
            .iter()
            .map(|value| {
                if value.is_finite() {
                    let mut rounded = (value * scale).round() / scale;
                    if rounded == 0.0 {
                        rounded = 0.0;
                    }
                    format!("{rounded:.precision$}", precision = decimals)
                } else {
                    "null".to_string()
                }
            })
            .collect::<Vec<_>>()
            .join(",")
    )
}

fn json_i64_array(values: &[i64]) -> String {
    if values.is_empty() {
        return "[]".to_string();
    }
    format!(
        "[{}]",
        values
            .iter()
            .map(|value| value.to_string())
            .collect::<Vec<_>>()
            .join(",")
    )
}

fn json_escape(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"")
}

fn round2(value: f64) -> f64 {
    if value.is_finite() {
        let rounded = (value * 100.0).round() / 100.0;
        if rounded == 0.0 {
            0.0
        } else {
            rounded
        }
    } else {
        value
    }
}

fn position_direction_code(quantity: f64) -> i64 {
    if quantity < 0.0 {
        -1
    } else {
        1
    }
}

fn trade_side_code(side: OrderSide) -> i64 {
    match side {
        OrderSide::Buy => 1,
        OrderSide::Sell => -2,
        OrderSide::Short => -1,
        OrderSide::Cover => 2,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::path::PathBuf;

    use crate::strategy::account::AccountState;
    use crate::strategy::config::{BarFrequency, FillPrice, StrategyAssetClass, StrategyRunConfig};
    use crate::strategy::market::{Bar, BarEvent, MarketSnapshot};
    use crate::strategy::order::{FillEvent, OrderSide};

    use super::holding_snapshot;

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
            commission_bps: 0.0,
            buy_commission_bps: 0.0,
            sell_commission_bps: 0.0,
            short_commission_bps: 0.0,
            cover_commission_bps: 0.0,
            stamp_tax_bps: 0.0,
            slippage_bps: 0.0,
            lot_size: 100.0,
            data_root: PathBuf::new(),
            model_root: PathBuf::new(),
            factor_root: PathBuf::new(),
            future: Default::default(),
            strategy_params: BTreeMap::new(),
        }
    }

    fn market(price: f64) -> MarketSnapshot {
        MarketSnapshot::new(
            BarEvent {
                trade_date: 20200102,
                trade_time: "daily".to_string(),
                bar_frequency: BarFrequency::Daily,
                is_session_first: true,
                is_session_end: true,
            },
            vec![Bar {
                symbol: "000001.SZ".to_string(),
                trade_date: 20200102,
                trade_time: "daily".to_string(),
                open: price,
                high: price,
                low: price,
                close: price,
                settle: None,
                multiplier: 1.0,
                volume: None,
                amount: None,
                is_limit_up: false,
                is_limit_down: false,
            }],
        )
    }

    #[test]
    fn holding_snapshot_uses_empty_json_arrays_without_positions_or_trades() {
        let account = AccountState::new(10_000.0);
        let snapshot = holding_snapshot(&config(), &account, &market(10.0), &[]);

        assert_eq!(snapshot.position_count, 0);
        assert_eq!(snapshot.trade_count, 0);
        assert_eq!(snapshot.symbols_json, "[]");
        assert_eq!(snapshot.trade_symbols_json, "[]");
        assert_eq!(snapshot.account_pnl, 0.0);
    }

    #[test]
    fn holding_snapshot_records_positions_and_current_bar_trades_as_json() {
        let mut account = AccountState::new(10_000.0);
        account.apply_buy("000001.SZ", 100.0, 10.0, 1.0);
        account.record_buy_cost(1.0);
        account.mark_price("000001.SZ", 12.0);
        let fill = FillEvent {
            strategy_id: "s".to_string(),
            asset_class: "stock".to_string(),
            trade_date: 20200102,
            trade_time: "daily".to_string(),
            bar_frequency: "daily".to_string(),
            symbol: "000001.SZ".to_string(),
            order_id: 1,
            fill_id: 1,
            side: OrderSide::Buy,
            quantity: 100.0,
            signed_quantity: 100.0,
            fill_price: 10.0,
            notional: 1_000.0,
            fee: 1.0,
            tax: 0.0,
            slippage_cost: 0.0,
            realized_pnl: 0.0,
            net_realized_pnl: -1.0,
            cash_after: account.cash(),
            position_qty_after: 100.0,
            avg_cost_after: 10.0,
            unrealized_pnl_after: 200.0,
            account_pnl_after: account.account_pnl(),
            signal_time: "20200101 daily".to_string(),
            fill_time: "20200102 daily".to_string(),
        };
        let snapshot = holding_snapshot(&config(), &account, &market(12.0), &[fill]);

        assert_eq!(snapshot.position_count, 1);
        assert_eq!(snapshot.trade_count, 1);
        assert_eq!(snapshot.symbols_json, "[\"000001.SZ\"]");
        assert_eq!(snapshot.quantities_json, "[100]");
        assert_eq!(snapshot.signed_quantities_json, "[100]");
        assert_eq!(snapshot.directions_json, "[1]");
        assert_eq!(snapshot.avg_costs_json, "[10.00]");
        assert_eq!(snapshot.prices_json, "[12]");
        assert_eq!(snapshot.unrealized_pnl, 200.0);
        assert_eq!(snapshot.unrealized_pnls_json, "[200.00]");
        assert_eq!(snapshot.multipliers_json, "[1]");
        assert_eq!(snapshot.margin_values_json, "[0]");
        assert_eq!(snapshot.trade_symbols_json, "[\"000001.SZ\"]");
        assert_eq!(snapshot.trade_sides_json, "[1]");
        assert_eq!(snapshot.trade_quantities_json, "[100]");
        assert_eq!(snapshot.trade_prices_json, "[10.00]");
        assert_eq!(snapshot.trade_realized_pnls_json, "[0.00]");
        assert_eq!(snapshot.trade_net_pnls_json, "[-1]");
    }
}

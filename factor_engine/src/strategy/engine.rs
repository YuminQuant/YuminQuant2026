use std::collections::BTreeMap;
use std::path::PathBuf;
use std::collections::BTreeSet;
use std::sync::Arc;

use crate::calendar::TradingCalendar;
use crate::config::EngineConfig;
use crate::error::{err, Result};
use crate::progress::ProgressBar;
use crate::strategy::account::AccountState;
use crate::strategy::config::{BarFrequency, StrategyAssetClass, StrategyRunConfig};
use crate::strategy::context::StrategyContext;
use crate::strategy::execution::ExecutionEngine;
use crate::strategy::market::{
    load_future_metadata, load_selected_future_minute_frames, load_stock_daily_frames,
    Bar, BarEvent, FutureContractMeta, FutureDailyCache, MarketFrame,
    MarketSnapshot, SessionOpenEvent,
};
use crate::strategy::order::{FillEvent, HoldingSnapshot, Order, OrderSide};
use crate::strategy::product_pnl::ProductPnl;
use crate::strategy::request::StrategyRunRequest;
use crate::strategy::session_io;
use crate::strategy::storage::HoldingsWriter;
use crate::strategy::future_market::FutureReferenceData;
use crate::strategy::strategies::future::mdl_100000::ModelCta;
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
        let mut run_config = StrategyRunConfig::load(
            &request.config_path,
            self.config.data_root.clone(),
            self.config.factor_root.clone(),
        )?;
        if let Some(detail) = request.detail {
            run_config.detail = detail;
        }
        let exchange = if run_config.asset_class == StrategyAssetClass::Future {
            &self.config.future_calendar_exchange
        } else {
            &self.config.stock_calendar_exchange
        };
        let calendar = TradingCalendar::load(&self.config.data_root, exchange)?;
        let dates = calendar.open_dates_between(run_config.start_date, run_config.end_date);
        if dates.is_empty() {
            return Err(err("no trading dates in strategy range"));
        }
        let mut strategy = create_strategy(&run_config)?;
        let mut account = AccountState::new(run_config.initial_cash);
        let mut corporate_actions = if run_config.asset_class == StrategyAssetClass::Stock {
            super::corporate_actions::CorporateActions::load(
                &run_config.data_root,
                run_config.start_date,
                run_config.end_date,
            )?
        } else {
            super::corporate_actions::CorporateActions::default()
        };
        let mut execution = ExecutionEngine::new();
        let mut lifecycle = super::lifecycle::StockLifecycle::load(&run_config)?;
        let mut pending_orders = Vec::<Order>::new();
        let mut holdings = HoldingsWriter::new(&run_config.output_dir())?;
        let mut product_pnl = run_config.future.product_pnl
            .then(|| ProductPnl::new(&run_config.output_dir().join("product_pnl.parquet"), run_config.initial_cash))
            .transpose()?;
        let mut daily_cache = FutureDailyCache::default();
        let mut future_reference = FutureReferenceData::default();
        let mut trade_count = 0_usize;
        let mut next_order_id = 0_i64;
        let progress = ProgressBar::new("strategy-run", dates.len(), true);
        let future_meta = if run_config.asset_class == StrategyAssetClass::Future {
            load_future_metadata(&run_config.data_root)?
        } else {
            BTreeMap::new()
        };

        let mut started = false;
        strategy.prepare_run(&run_config, &dates)?;
        if run_config.asset_class == StrategyAssetClass::Future {
            strategy.prepare_future_run(&run_config, &calendar, &dates)?;
        }
        let batch_days = if run_config.asset_class == StrategyAssetClass::Stock
            && run_config.bar_frequency == BarFrequency::Minute
            && run_config.minute_open_limits
            && run_config.session_inputs.is_empty()
        {
            strategy.input_batch_days().max(1)
        } else {
            1
        };
        for batch in dates.chunks(batch_days) {
            let future_universe = if run_config.asset_class == StrategyAssetClass::Future {
                strategy.select_future_contracts(&run_config, batch[0], &future_meta)?
            } else {
                None
            };
            let future_market_universe = future_universe.as_ref().map(|selected| {
                let mut symbols = selected.clone();
                symbols.extend(account.futures_positions().map(|(symbol, _)| symbol.clone()));
                symbols
            });
            let universe = if run_config.asset_class == StrategyAssetClass::Future {
                None
            } else {
                strategy.prepare_session(&run_config, &account, batch[0])?
            };
            let parallel_stock_batch = batch.len() > 1;
            let (frames, future_day) = if !parallel_stock_batch {
                let mut files = strategy.session_files(
                    &run_config, batch[0], future_universe.as_ref().or(universe.as_ref()),
                );
                for (name, template) in &run_config.session_inputs {
                    files.push(template.for_date(&run_config.data_root, name, batch[0]));
                }
                let (frames, other_inputs) = session_io::join(
                    || load_session_frames(&run_config, batch[0], &future_meta, &mut daily_cache,
                        future_market_universe.as_ref(), universe.as_ref()),
                    || session_io::join(
                        || if run_config.asset_class == StrategyAssetClass::Future && run_config.future.daily_costs {
                            future_reference.load_day(&run_config, batch[0], calendar.previous_open(batch[0]),
                                &future_meta, future_market_universe.as_ref()).map(|day| Some(Arc::new(day)))
                        } else {
                            Ok(None)
                        },
                        || session_io::load_files(files),
                    ),
                )?;
                let (future_day, inputs) = other_inputs?;
                let (frames, future_day, inputs) = (frames?, future_day?, inputs?);
                strategy.on_session_inputs(batch[0], inputs)?;
                (frames, future_day)
            } else {
                (Vec::new(), None)
            };
            let mut consume = |trade_date: i32, mut frames: Vec<MarketFrame>| -> Result<()> {
                if parallel_stock_batch {
                    let files = strategy.session_files(&run_config, trade_date, universe.as_ref());
                    strategy.on_session_inputs(trade_date, session_io::load_files(files)?)?;
                }
                corporate_actions.begin_session(&mut account, trade_date);
                let exit_time = lifecycle.begin_session(&run_config, &mut account, trade_date, &mut frames);
                strategy.prepare_session(&run_config, &account, trade_date)?;
                if let Some(report) = &mut product_pnl {
                    if let Some(symbols) = &future_universe {
                        for symbol in symbols {
                            report.observe(symbol);
                        }
                    } else {
                        for bar in frames.iter().flat_map(|frame| &frame.bars) {
                            report.observe(&bar.symbol);
                        }
                    }
                }
                if frames.is_empty() {
                    if run_config.asset_class == StrategyAssetClass::Future && future_universe.as_ref().is_some_and(|codes| !codes.is_empty()) {
                        return Err(err(format!("missing selected futures minute bars for {trade_date}")));
                    }
                    corporate_actions.end_session(&account, trade_date);
                    if let Some(report) = &mut product_pnl {
                        report.close_day(trade_date, &account)?;
                    }
                    eprintln!(
                        "[strategy warning] date={trade_date}: no market frames; skipping date"
                    );
                    progress.tick(format!("date={trade_date} frames=0 trades={}", trade_count));
                    return Ok(());
                }
                let mut daily_side_counts = SideCounts::default();
                let frame_count = frames.len();
                for (frame_index, frame) in frames.into_iter().enumerate() {
                    let mut market = MarketSnapshot::new(frame.event.clone(), frame.bars);
                    market.future_day = future_day.clone();
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
                        daily_side_counts.add_fills(&fills);
                        bar_fills.extend(fills);
                    }
                    if market.event.is_session_first {
                        let event = SessionOpenEvent {
                            trade_date: market.event.trade_date,
                            trade_time: market.event.trade_time.clone(),
                            bar_frequency: market.event.bar_frequency,
                        };
                        let mut ctx = StrategyContext::new(
                            &run_config,
                            &account,
                            &market,
                            &mut next_order_id,
                        );
                        strategy.on_session_open(&mut ctx, &event)?;
                        let open_orders = ctx.take_orders();
                        if !open_orders.is_empty() {
                            let fills = execution.fill_orders(
                                &run_config,
                                &mut account,
                                &market,
                                open_orders,
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
                            daily_side_counts.add_fills(&fills);
                            bar_fills.extend(fills);
                        }
                    }

                    let mut ctx =
                        StrategyContext::new(&run_config, &account, &market, &mut next_order_id);
                    strategy.on_bar_open(&mut ctx, &market.event)?;
                    let orders = ctx.take_orders();
                    if !orders.is_empty() {
                        let fills =
                            execution.fill_orders(&run_config, &mut account, &market, orders);
                        call_fills(
                            &mut *strategy,
                            &run_config,
                            &account,
                            &market,
                            &mut next_order_id,
                            &fills,
                        )?;
                        trade_count += fills.len();
                        daily_side_counts.add_fills(&fills);
                        bar_fills.extend(fills);
                    }
                    mark_account_positions(&run_config, &mut account, &market);
                    if exit_time.as_ref() == Some(&market.event.trade_time) {
                        let fills = lifecycle.fill_exits(&run_config, &mut account, &market, &mut execution, &mut next_order_id)?;
                        call_fills(&mut *strategy, &run_config, &account, &market, &mut next_order_id, &fills)?;
                        trade_count += fills.len();
                        daily_side_counts.add_fills(&fills);
                        bar_fills.extend(fills);
                    }
                    let mut ctx =
                        StrategyContext::new(&run_config, &account, &market, &mut next_order_id);
                    strategy.on_bar(&mut ctx, &market.event)?;
                    pending_orders.extend(ctx.take_orders());
                    if market.event.is_session_end {
                        let mut ctx = StrategyContext::new(&run_config, &account, &market, &mut next_order_id);
                        strategy.on_session_end(&mut ctx, &market.event)?;
                    }
                    if let Some(report) = &mut product_pnl {
                        report.record(&bar_fills);
                    }
                    let daily_only =
                        run_config.bar_frequency == BarFrequency::Minute && !run_config.detail;
                    if !daily_only || frame_index + 1 == frame_count {
                        // Daily mode never serializes the discarded intraday snapshots/trades.
                        let mut snapshot = holding_snapshot(
                            &run_config,
                            &account,
                            &market,
                            if daily_only { &[] } else { &bar_fills },
                        );
                        if daily_only {
                            daily_side_counts.apply_to_snapshot(&mut snapshot);
                        }
                        holdings.push(snapshot)?;
                    }
                }
                corporate_actions.end_session(&account, trade_date);
                if let Some(report) = &mut product_pnl {
                    report.close_day(trade_date, &account)?;
                }
                progress.tick(format!("date={trade_date} trades={trade_count}"));
                Ok(())
            };
            if parallel_stock_batch {
                super::market::visit_stock_minute_sessions(
                    &run_config.data_root,
                    batch,
                    universe.as_ref(),
                    true,
                    &mut consume,
                )?;
            } else {
                consume(batch[0], frames)?;
            }
        }

        if started {
            let end_frame = terminal_frame(&run_config);
            let mut ctx =
                StrategyContext::new(&run_config, &account, &end_frame, &mut next_order_id);
            strategy.on_end(&mut ctx)?;
        }
        progress.finish();
        lifecycle.finish()?;
        let mut output_files = vec![holdings.finish()?];
        if let Some(report) = product_pnl {
            output_files.push(report.finish()?);
        }
        crate::artifact_io::write(
            run_config.output_dir().join("run_config.toml"),
            std::fs::read(&request.config_path)?,
        )?;
        Ok(StrategyRunReport {
            strategy_id: run_config.strategy_id,
            asset_class: run_config.asset_class.as_str().to_string(),
            trade_count,
            output_files,
        })
    }
}

fn load_session_frames(
    config: &StrategyRunConfig,
    trade_date: i32,
    future_meta: &BTreeMap<String, FutureContractMeta>,
    daily_cache: &mut FutureDailyCache,
    future_universe: Option<&BTreeSet<String>>,
    stock_universe: Option<&BTreeSet<String>>,
) -> Result<Vec<MarketFrame>> {
    let dates = [trade_date];
    match (&config.asset_class, config.bar_frequency) {
        (StrategyAssetClass::Stock, BarFrequency::Daily) => {
            load_stock_daily_frames(&config.data_root, &dates)
        }
        (StrategyAssetClass::Stock, BarFrequency::Minute) => {
            super::market::load_stock_minute_frames_for_run(
                &config.data_root, &dates, stock_universe, config.minute_open_limits,
            )
        }
        (StrategyAssetClass::Future, BarFrequency::Daily) => {
            daily_cache.load(&config.data_root, &dates, future_meta)
        }
        (StrategyAssetClass::Future, BarFrequency::Minute) => load_selected_future_minute_frames(
            &config.data_root,
            &dates,
            future_meta,
            &config.market_products,
            &config.market_symbols,
            future_universe,
        ),
        (StrategyAssetClass::MultiAsset, _) => {
            Err(err("multi_asset strategy execution is not implemented"))
        }
    }
}

fn create_strategy(config: &StrategyRunConfig) -> Result<Box<dyn Strategy>> {
    match config.strategy_class.as_str() {
        "stock::strategy_001" => Ok(Box::new(Strategy001::from_config(config)?)),
        "future::mdl_100000" => Ok(Box::new(ModelCta::from_config(config)?)),
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
    let side_counts = SideCounts::from_fills(fills);
    let mut symbols = Vec::new();
    let mut quantities = Vec::new();
    let mut signed_quantities = Vec::new();
    let mut directions = Vec::new();
    let mut avg_costs = Vec::new();
    let mut market_values = Vec::new();
    let mut unrealized_pnls = Vec::new();
    let mut multipliers = Vec::new();
    let mut margin_ratios = Vec::new();
    let mut margin_values = Vec::new();
    for (symbol, position) in account.all_positions() {
        if position.quantity.abs() <= 1e-9 {
            continue;
        }
        symbols.push(symbol.clone());
        quantities.push(position.quantity.abs());
        signed_quantities.push(position.quantity);
        directions.push(position_direction_code(position.quantity));
        avg_costs.push(position.avg_cost);
        market_values.push(position.market_value());
        unrealized_pnls.push(position.unrealized_pnl());
        multipliers.push(position.multiplier.max(1.0));
        margin_ratios.push(position.margin_ratio.max(0.0));
        margin_values.push(position.margin_value());
    }

    let totals = account.summary();
    HoldingSnapshot {
        strategy_id: config.strategy_id.clone(),
        asset_class: config.asset_class.as_str().to_string(),
        trade_date: market.event.trade_date,
        trade_time: market.event.trade_time.clone(),
        cash: round2(account.cash()),
        dividend_receivable: round2(account.dividend_receivable()),
        dividend_income_cum: round2(account.dividend_income()),
        account_pnl: round2(totals.equity - account.initial_cash()),
        realized_pnl_cum: round2(account.realized_pnl_cum()),
        net_realized_pnl_cum: round2(account.net_realized_pnl_cum()),
        unrealized_pnl: round2(totals.unrealized_pnl),
        gross_market_value: round2(totals.gross_market_value),
        net_market_value: round2(totals.net_market_value),
        margin_required: round2(totals.margin_required),
        available_margin: round2(totals.equity - totals.margin_required),
        position_count: symbols.len() as i64,
        buy_count: side_counts.buy,
        sell_count: side_counts.sell,
        short_count: side_counts.short,
        cover_count: side_counts.cover,
        symbols_json: json_string_array(&symbols),
        quantities_json: json_f64_array(&quantities),
        signed_quantities_json: json_f64_array(&signed_quantities),
        directions_json: json_i64_array(&directions),
        avg_costs_json: json_f64_array_rounded(&avg_costs, 2),
        market_values_json: json_f64_array_rounded(&market_values, 2),
        unrealized_pnls_json: json_f64_array_rounded(&unrealized_pnls, 2),
        multipliers_json: json_f64_array(&multipliers),
        margin_ratios_json: json_f64_array(&margin_ratios),
        margin_values_json: json_f64_array_rounded(&margin_values, 2),
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
        trade_net_pnls_json: json_f64_array_rounded(
            &fills
                .iter()
                .map(|fill| fill.net_realized_pnl)
                .collect::<Vec<_>>(),
            2,
        ),
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct SideCounts {
    buy: i64,
    sell: i64,
    short: i64,
    cover: i64,
}

impl SideCounts {
    fn from_fills(fills: &[FillEvent]) -> Self {
        let mut counts = Self::default();
        counts.add_fills(fills);
        counts
    }

    fn add_fills(&mut self, fills: &[FillEvent]) {
        for fill in fills {
            match fill.side {
                OrderSide::Buy => self.buy += 1,
                OrderSide::Sell => self.sell += 1,
                OrderSide::Short => self.short += 1,
                OrderSide::Cover => self.cover += 1,
            }
        }
    }

    fn apply_to_snapshot(self, snapshot: &mut HoldingSnapshot) {
        snapshot.buy_count = self.buy;
        snapshot.sell_count = self.sell;
        snapshot.short_count = self.short;
        snapshot.cover_count = self.cover;
    }
}

fn mark_account_positions(
    config: &StrategyRunConfig,
    account: &mut AccountState,
    market: &MarketSnapshot,
) {
    for (symbol, position) in account.positions_mut() {
        let Some(bar) = market.bar(symbol) else {
            continue;
        };
        let Some(price) = mark_price_for(config, bar) else {
            continue;
        };
        let cash_settled = config.asset_class == StrategyAssetClass::Stock;
        let margin_ratio = if cash_settled {
            0.0
        } else if let Some(spec) = market.future_day.as_ref().and_then(|day| day.spec(symbol)) {
            spec.margin(if position.quantity > 0.0 { OrderSide::Buy } else { OrderSide::Short })
        } else {
            config.future_margin_ratio(symbol)
        };
        position.last_price = Some(price);
        position.multiplier = bar.multiplier.max(1.0);
        position.margin_ratio = margin_ratio.max(0.0);
        position.cash_settled = cash_settled;
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
#[path = "tests.rs"]
mod tests;

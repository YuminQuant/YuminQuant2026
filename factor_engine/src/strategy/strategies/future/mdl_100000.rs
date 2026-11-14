use std::collections::{BTreeMap, BTreeSet};

use crate::calendar::TradingCalendar;
use crate::data::Table;
use crate::error::{err, Result};
use crate::factor::common::intraday_raw::quantile_linear;
use crate::factor::future::minute_input::{product, FutureMinuteInput, FutureMinuteOptions};
use crate::strategy::config::{
    future_product_from_symbol, BarFrequency, StrategyAssetClass, StrategyRunConfig,
};
use crate::strategy::context::StrategyContext;
use crate::strategy::market::{clock_minute, BarEvent, FutureContractMeta};
use crate::strategy::order::{FillEvent, OrderSide};
use crate::strategy::session_io::{SessionFile, SessionInputs};
use crate::strategy::strategy::Strategy;

type Signals = BTreeMap<String, BTreeMap<String, f64>>;
type Thresholds = BTreeMap<String, (f64, f64)>;

pub struct ModelCta {
    mappings: Option<FutureMinuteInput>,
    dates: Vec<i32>,
    history: BTreeMap<i32, Thresholds>,
    close_minutes: BTreeMap<String, u16>,
    thresholds: Thresholds,
    signals: Signals,
    entries: BTreeMap<String, (OrderSide, String)>,
    overnight: BTreeSet<String>,
}

impl ModelCta {
    pub fn from_config(config: &StrategyRunConfig) -> Result<Self> {
        if config.asset_class != StrategyAssetClass::Future
            || config.bar_frequency != BarFrequency::Minute
            || !config.future.daily_costs
            || !config.lot_size.is_finite()
            || config.lot_size.fract() != 0.0
        {
            return Err(err(
                "mdl_100000 CTA requires futures minute bars, daily costs and integer lots",
            ));
        }
        Ok(Self {
            mappings: None,
            dates: Vec::new(),
            history: BTreeMap::new(),
            close_minutes: BTreeMap::new(),
            thresholds: BTreeMap::new(),
            signals: BTreeMap::new(),
            entries: BTreeMap::new(),
            overnight: BTreeSet::new(),
        })
    }

    fn selected(&self, config: &StrategyRunConfig, date: i32) -> BTreeSet<String> {
        self.mappings
            .as_ref()
            .and_then(|map| map.selection(date))
            .into_iter()
            .flat_map(|map| map.values())
            .filter(|symbol| !future_product_from_symbol(symbol).ends_with("TAS"))
            .filter(|symbol| {
                (config.market_products.is_empty() && config.market_symbols.is_empty())
                    || config.market_symbols.contains(symbol)
                    || config
                        .market_products
                        .contains(&future_product_from_symbol(symbol))
            })
            .cloned()
            .collect()
    }

    fn prior_dates(&self, date: i32) -> &[i32] {
        let index = self.dates.partition_point(|&day| day < date);
        &self.dates[index.saturating_sub(5)..index]
    }

    fn refresh_thresholds(&mut self, date: i32) {
        let prior = self.prior_dates(date).to_vec();
        self.history.retain(|day, _| prior.contains(day));
        self.thresholds.clear();
        if prior.len() != 5 {
            return;
        }
        for key in self.history[&prior[0]].keys() {
            let pair = prior.iter().try_fold((0.0, 0.0), |(low, high), day| {
                let (lo, hi) = self.history[day].get(key)?;
                Some((low + lo, high + hi))
            });
            if let Some((low, high)) = pair {
                if low < high {
                    self.thresholds.insert(key.clone(), (low / 5.0, high / 5.0));
                }
            }
        }
    }
}

impl Strategy for ModelCta {
    fn name(&self) -> &'static str {
        "future::mdl_100000"
    }

    fn prepare_future_run(
        &mut self,
        config: &StrategyRunConfig,
        calendar: &TradingCalendar,
        dates: &[i32],
    ) -> Result<()> {
        let start = calendar.warmup_start(dates[0], 5);
        self.dates = calendar.open_dates_between(start, config.end_date);
        self.mappings = Some(FutureMinuteInput::load(
            FutureMinuteOptions::default(),
            &config.data_root,
            &self.dates,
        )?);
        Ok(())
    }

    fn select_future_contracts(
        &mut self,
        config: &StrategyRunConfig,
        date: i32,
        contracts: &BTreeMap<String, FutureContractMeta>,
    ) -> Result<Option<BTreeSet<String>>> {
        self.entries.clear();
        self.close_minutes.clear();
        let selected = self.selected(config, date);
        if self
            .mappings
            .as_ref()
            .and_then(|map| map.selection(date))
            .is_none()
        {
            eprintln!("[strategy warning] {date}: missing CTA main-contract mapping");
        }
        for symbol in &selected {
            let meta = contracts
                .get(symbol)
                .ok_or_else(|| err(format!("missing contract metadata for {symbol} on {date}")))?;
            let mut minute = meta
                .daily_close_minute
                .ok_or_else(|| err(format!("missing daily close time for {symbol} on {date}")))?;
            let product = future_product_from_symbol(symbol);
            if meta.delist_date == Some(date)
                && matches!(product.as_str(), "T" | "TF" | "TS" | "TL")
            {
                minute = 11 * 60 + 30;
            }
            self.close_minutes.insert(symbol.clone(), minute);
        }
        Ok(Some(selected))
    }

    fn session_files(
        &self,
        config: &StrategyRunConfig,
        date: i32,
        selected: Option<&BTreeSet<String>>,
    ) -> Vec<SessionFile> {
        let mut files = Vec::new();
        if let Some(selected) = selected.filter(|symbols| !symbols.is_empty()) {
            files.push(signal_file(config, date, selected.clone()));
        }
        for &day in self.prior_dates(date) {
            if !self.history.contains_key(&day) {
                let selected = self.selected(config, day);
                if !selected.is_empty() {
                    files.push(signal_file(config, day, selected));
                }
            }
        }
        files
    }

    fn on_session_inputs(&mut self, date: i32, mut inputs: SessionInputs) -> Result<()> {
        for day in self.prior_dates(date).to_vec() {
            if !self.history.contains_key(&day) {
                let samples = parse_signals(day, inputs.remove(&signal_name(day)).flatten())?;
                self.history.insert(day, daily_quantiles(&samples));
            }
        }
        self.refresh_thresholds(date);
        self.signals = parse_signals(date, inputs.remove(&signal_name(date)).flatten())?;
        if !self.close_minutes.is_empty() && self.signals.is_empty() {
            eprintln!(
                "[strategy warning] {date}: no usable mdl_100000 signals for selected contracts"
            );
        }
        self.history.insert(date, daily_quantiles(&self.signals));
        Ok(())
    }

    fn on_bar_open(&mut self, ctx: &mut StrategyContext, event: &BarEvent) -> Result<()> {
        let minute = clock_minute(&event.trade_time).ok_or_else(|| {
            err(format!(
                "invalid futures minute timestamp {}",
                event.trade_time
            ))
        })?;
        let symbols: Vec<_> = ctx.market().symbols().cloned().collect();
        for symbol in &symbols {
            if !self.overnight.contains(symbol)
                && !self.close_minutes.get(symbol).is_some_and(|&close| liquidation_window(minute, close))
            {
                continue;
            }
            for side in [OrderSide::Sell, OrderSide::Cover] {
                if let Some(quantity) = ctx
                    .account()
                    .future_position(symbol, side)
                    .map(|p| p.quantity.abs())
                {
                    ctx.order_future_side(symbol, side, quantity, ctx.signal_time());
                }
            }
        }
        for symbol in symbols {
            let Some((side, signal_time)) = self.entries.remove(&symbol) else {
                continue;
            };
            if self.overnight.contains(&symbol)
                || self.close_minutes.get(&symbol).is_none_or(|&close| after_liquidation_cutoff(minute, close))
                || ctx.account().future_position(&symbol, side).is_some()
            {
                continue;
            }
            if let Some(lots) = ctx
                .market()
                .future_day
                .as_ref()
                .and_then(|day| day.spec(&symbol))
                .map(|spec| spec.min_lots)
            {
                ctx.order_future_side(&symbol, side, lots, signal_time);
            }
        }
        Ok(())
    }

    fn on_bar(&mut self, ctx: &mut StrategyContext, event: &BarEvent) -> Result<()> {
        let Some(minute) = clock_minute(&event.trade_time) else {
            return Ok(());
        };
        for symbol in ctx.market().symbols() {
            if self.overnight.contains(symbol) {
                continue;
            }
            let Some(&close) = self.close_minutes.get(symbol) else {
                continue;
            };
            if after_liquidation_cutoff(minute, close) || event.is_session_end {
                continue;
            }
            let Some(score) = self
                .signals
                .get(symbol)
                .and_then(|scores| scores.get(&event.trade_time))
            else {
                continue;
            };
            let Some((low, high)) = product(symbol).and_then(|key| self.thresholds.get(&key))
            else {
                continue;
            };
            let side = if score > high {
                OrderSide::Buy
            } else if score < low {
                OrderSide::Short
            } else {
                continue;
            };
            if ctx.account().future_position(symbol, side).is_none() {
                self.entries
                    .insert(symbol.clone(), (side, ctx.signal_time()));
            }
        }
        Ok(())
    }

    fn on_fill(&mut self, ctx: &mut StrategyContext, fill: &FillEvent) -> Result<()> {
        if self.overnight.contains(&fill.symbol)
            && [OrderSide::Sell, OrderSide::Cover].into_iter()
                .all(|side| ctx.account().future_position(&fill.symbol, side).is_none())
        {
            self.overnight.remove(&fill.symbol);
        }
        Ok(())
    }

    fn on_session_end(&mut self, ctx: &mut StrategyContext, event: &BarEvent) -> Result<()> {
        self.entries.clear();
        self.overnight = ctx
            .account()
            .futures_positions()
            .map(|(symbol, _)| symbol.clone())
            .collect();
        if !self.overnight.is_empty() {
            eprintln!("[strategy warning] {} deferred close to next trading day: {:?}",
                event.trade_date, self.overnight);
        }
        Ok(())
    }
}

fn liquidation_window(minute: u16, close: u16) -> bool {
    (close.saturating_sub(5)..=close).contains(&minute)
}

fn after_liquidation_cutoff(minute: u16, close: u16) -> bool {
    // Night trading belongs to the next trading date. Its 21:00 open is not a day close.
    minute >= close.saturating_sub(5) && minute < 18 * 60
}

fn signal_name(date: i32) -> String {
    format!("mdl_100000:{date}")
}

fn signal_file(config: &StrategyRunConfig, date: i32, selected: BTreeSet<String>) -> SessionFile {
    SessionFile {
        name: signal_name(date),
        path: config
            .model_root
            .join("future/minute_1m")
            .join((date / 10000).to_string())
            .join(format!("{date}.parquet")),
        columns: ["trade_date", "ts_code", "trade_time", "mdl_100000"]
            .map(str::to_owned)
            .to_vec(),
        symbols: Some(selected),
        optional: true,
    }
}

fn parse_signals(date: i32, table: Option<Table>) -> Result<Signals> {
    let mut signals = Signals::new();
    let Some(table) = table else {
        return Ok(signals);
    };
    let dates = table.required_i32_date_cast("trade_date")?;
    let symbols = table.required_utf8("ts_code")?;
    let times = table.required_utf8("trade_time")?;
    let values = table.required_f64_cast("mdl_100000")?;
    let mut keys = BTreeSet::new();
    for i in 0..table.len {
        let (Some(symbol), Some(time)) = (&symbols[i], &times[i]) else {
            return Err(err(format!("invalid mdl_100000 row on {date}")));
        };
        if dates[i] != Some(date) || !keys.insert((symbol, time)) {
            return Err(err(format!(
                "invalid or duplicate mdl_100000 key: {date} {symbol} {time}"
            )));
        }
        if let Some(value) = values[i].filter(|v| v.is_finite()) {
            signals
                .entry(symbol.clone())
                .or_default()
                .insert(time.clone(), value);
        }
    }
    Ok(signals)
}

fn daily_quantiles(signals: &Signals) -> Thresholds {
    signals
        .iter()
        .filter_map(|(symbol, scores)| {
            let key = product(symbol)?;
            let mut values: Vec<_> = scores.values().copied().collect();
            let low = quantile_linear(&mut values, 0.05)?;
            let high = quantile_linear(&mut values, 0.95)?;
            Some((key, (low, high)))
        })
        .collect()
}

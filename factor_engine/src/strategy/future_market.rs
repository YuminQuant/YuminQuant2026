use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use crate::artifact_io::StoragePathExt;
use crate::data::parquet_io::read_parquet;
use crate::error::{err, Result};

use super::config::{future_product_from_symbol, StrategyRunConfig};
use super::market::FutureContractMeta;
use super::order::OrderSide;

/// Exchange reference data frozen for one trading date.
#[derive(Debug, Default)]
pub struct FutureMarketDay {
    contracts: BTreeMap<String, FutureTradingSpec>,
}

#[derive(Debug)]
pub struct FutureTradingSpec {
    pub min_lots: f64,
    pub multiplier: f64,
    pub cost_date: i32,
    long_margin: f64,
    short_margin: f64,
    fee_per_lot: f64,
    fee_rate: f64,
    close_today_fee: Option<f64>,
    lower_limit: Option<f64>,
    upper_limit: Option<f64>,
}

impl FutureTradingSpec {
    pub fn margin(&self, side: OrderSide) -> f64 {
        match side {
            OrderSide::Buy | OrderSide::Sell => self.long_margin,
            OrderSide::Short | OrderSide::Cover => self.short_margin,
        }
    }

    pub fn fee(&self, side: OrderSide, price: f64, lots: f64) -> f64 {
        if matches!(side, OrderSide::Sell | OrderSide::Cover) {
            if let Some(fee) = self.close_today_fee {
                return lots * fee;
            }
        }
        lots * (self.fee_per_lot + price * self.multiplier * self.fee_rate)
    }

    pub fn allows_price(&self, side: OrderSide, price: f64) -> bool {
        let tolerance = price * 1e-7;
        match side {
            OrderSide::Buy | OrderSide::Cover => self
                .upper_limit
                .is_none_or(|upper| price < upper - tolerance),
            OrderSide::Sell | OrderSide::Short => self
                .lower_limit
                .is_none_or(|lower| price > lower + tolerance),
        }
    }
}

impl FutureMarketDay {
    pub fn spec(&self, symbol: &str) -> Option<&FutureTradingSpec> {
        self.contracts.get(symbol)
    }
}

#[derive(Default)]
pub(super) struct FutureReferenceData {
    limit_year: Option<(i32, DailyLimits)>,
}

impl FutureReferenceData {
    pub fn load_day(
        &mut self,
        config: &StrategyRunConfig,
        date: i32,
        previous_date: Option<i32>,
        contracts: &BTreeMap<String, FutureContractMeta>,
        universe: Option<&BTreeSet<String>>,
    ) -> Result<FutureMarketDay> {
        let Some(cost_date) = previous_date else {
            return Ok(FutureMarketDay::default());
        };
        let year = date / 10000;
        let exchanges: BTreeSet<_> = contracts
            .iter()
            .filter(|(symbol, meta)| {
                !meta.exchange.is_empty() && universe.is_none_or(|codes| codes.contains(*symbol))
            })
            .map(|(_, meta)| meta.exchange.as_str())
            .collect();
        let mut day = FutureMarketDay::default();
        let root = config
            .future
            .settle_root
            .clone()
            .unwrap_or_else(|| config.data_root.join("future_data/basic/settle"));
        let columns = [
            "ts_code",
            "trade_date",
            "exchange",
            "trading_fee",
            "trading_fee_rate",
            "offset_today_fee",
            "long_margin_rate",
            "short_margin_rate",
        ]
        .map(str::to_owned);
        let needs_limits = self.limit_year.as_ref().is_none_or(|(old, _)| *old != year);
        let (limits, settlements) = super::session_io::join(
            || {
                needs_limits
                    .then(|| load_limits(&config.data_root, year))
                    .transpose()
            },
            || {
                super::session_io::map(exchanges.into_iter().collect(), |exchange| {
                    let path = settlement_path(&root, exchange, cost_date);
                    let table = path
                        .storage_exists()
                        .then(|| read_parquet(&path, Some(&columns)))
                        .transpose()?;
                    Ok((exchange, path, table))
                })
            },
        )?;
        if let Some(limits) = limits? {
            self.limit_year = Some((year, limits));
        }
        let limits = self
            .limit_year
            .as_ref()
            .and_then(|(_, days)| days.get(&date));
        for (exchange, path, table) in settlements? {
            let Some(table) = table else {
                eprintln!(
                    "[strategy warning] {date}: missing futures cost snapshot {}",
                    path.display()
                );
                continue;
            };
            let symbols = table.required_utf8("ts_code")?;
            let dates = table.required_i32_date_cast("trade_date")?;
            let source_exchanges = table.required_utf8("exchange")?;
            let fees = table.required_f64_cast("trading_fee")?;
            let rates = table.required_f64_cast("trading_fee_rate")?;
            let close_today = table.required_f64_cast("offset_today_fee")?;
            let long = table.required_f64_cast("long_margin_rate")?;
            let short = table.required_f64_cast("short_margin_rate")?;
            let mut seen = BTreeSet::new();
            for i in 0..table.len {
                let Some(symbol) = symbols[i].as_deref() else {
                    return Err(err("null futures cost contract"));
                };
                if dates[i] != Some(cost_date)
                    || source_exchanges[i].as_deref() != Some(exchange)
                    || !seen.insert(symbol)
                {
                    return Err(err(format!(
                        "invalid or duplicate futures cost row {symbol}: {}",
                        path.display()
                    )));
                }
                let Some(meta) = contracts.get(symbol) else {
                    continue;
                };
                if meta.multiplier <= 0.0
                    || meta.exchange != exchange
                    || universe.is_some_and(|codes| !codes.contains(symbol))
                {
                    continue;
                }
                let Some((long_margin, short_margin)) =
                    margin(long[i], exchange).zip(margin(short[i], exchange))
                else {
                    continue;
                };
                let Some((fee_per_lot, fee_rate)) = fee_terms(symbol, exchange, fees[i], rates[i])
                else {
                    continue;
                };
                let (lower_limit, upper_limit) = limits
                    .and_then(|entries| entries.get(symbol))
                    .copied()
                    .unwrap_or_default();
                let spec = FutureTradingSpec {
                    min_lots: config.future_min_lots(symbol),
                    multiplier: meta.multiplier,
                    cost_date,
                    long_margin,
                    short_margin,
                    fee_per_lot,
                    fee_rate,
                    // CZCE publishes the absolute per-lot close-today fee.
                    close_today_fee: (exchange == "CZCE")
                        .then(|| nonnegative(close_today[i]))
                        .flatten(),
                    lower_limit,
                    upper_limit,
                };
                day.contracts.insert(symbol.to_owned(), spec);
            }
        }
        if let Some(symbols) = universe {
            let missing: Vec<_> = symbols
                .iter()
                .filter(|symbol| day.spec(symbol).is_none())
                .collect();
            if !missing.is_empty() {
                eprintln!("[strategy warning] {date}: contracts without usable costs or metadata: {missing:?}");
            }
        }
        Ok(day)
    }
}

fn settlement_path(root: &Path, exchange: &str, date: i32) -> PathBuf {
    root.join(exchange)
        .join((date / 10000).to_string())
        .join(format!("{date}.parquet"))
}

fn nonnegative(value: Option<f64>) -> Option<f64> {
    value.filter(|value| value.is_finite() && *value >= 0.0)
}

fn margin(value: Option<f64>, exchange: &str) -> Option<f64> {
    let value = nonnegative(value)?;
    let ratio = if exchange == "CFFEX" {
        value / 100.0
    } else {
        value
    };
    (ratio > 0.0 && ratio <= 1.0).then_some(ratio)
}

fn fee_terms(
    symbol: &str,
    exchange: &str,
    amount: Option<f64>,
    rate: Option<f64>,
) -> Option<(f64, f64)> {
    let amount = nonnegative(amount);
    let rate = nonnegative(rate);
    if exchange == "CFFEX" {
        let product = future_product_from_symbol(symbol);
        return match product.as_str() {
            "IF" | "IH" | "IC" | "IM" => Some((0.0, rate? / 10000.0)),
            "T" | "TF" | "TS" | "TL" => Some((amount?, 0.0)),
            _ => None,
        };
    }
    if amount.is_none() && rate.is_none() {
        return None;
    }
    Some((amount.unwrap_or(0.0), rate.unwrap_or(0.0) / 10000.0))
}

type DailyLimits = BTreeMap<i32, BTreeMap<String, (Option<f64>, Option<f64>)>>;

fn load_limits(root: &Path, year: i32) -> Result<DailyLimits> {
    let path = root
        .join("future_data/daily_limit")
        .join(format!("{year}.parquet"));
    if !path.storage_exists() {
        eprintln!(
            "[strategy warning] futures price limits unavailable for {year}: {}",
            path.display()
        );
        return Ok(BTreeMap::new());
    }
    let columns = ["trade_date", "ts_code", "down_limit", "up_limit"].map(str::to_owned);
    let table = read_parquet(&path, Some(&columns))?;
    let dates = table.required_i32_date_cast("trade_date")?;
    let symbols = table.required_utf8("ts_code")?;
    let down = table.required_f64_cast("down_limit")?;
    let up = table.required_f64_cast("up_limit")?;
    let mut limits = DailyLimits::new();
    for i in 0..table.len {
        if let (Some(date), Some(symbol)) = (dates[i], &symbols[i]) {
            limits
                .entry(date)
                .or_default()
                .insert(symbol.clone(), (nonnegative(down[i]), nonnegative(up[i])));
        }
    }
    Ok(limits)
}

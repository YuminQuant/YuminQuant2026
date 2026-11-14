use std::io::{BufWriter, Write};

use crate::artifact_io::{self, AtomicWriter, StoragePathExt};
use crate::data::parquet_io::read_parquet;
use crate::error::{err, Result};

use super::account::AccountState;
use super::config::{BarFrequency, StrategyAssetClass, StrategyRunConfig};
use super::execution::ExecutionEngine;
use super::market::{BarEvent, MarketFrame, MarketSnapshot};
use super::order::{FillEvent, Order};

pub(super) struct StockLifecycle {
    dates: Vec<(i32, String)>,
    next: usize,
    audit: Option<BufWriter<AtomicWriter>>,
}

impl StockLifecycle {
    pub fn load(config: &StrategyRunConfig) -> Result<Self> {
        let mut book = Self {
            dates: Vec::new(),
            next: 0,
            audit: None,
        };
        if config.asset_class != StrategyAssetClass::Stock || !config.force_delisting_exit {
            return Ok(book);
        }
        let path = config
            .data_root
            .join("chn_stock_data/info/stock_basic.parquet");
        if !path.storage_exists() {
            eprintln!(
                "[strategy warning] missing stock_basic.parquet: delisting dates unavailable"
            );
            return Ok(book);
        }
        let columns = vec!["ts_code".into(), "delist_date".into()];
        let table = read_parquet(&path, Some(&columns))?;
        let symbols = table.required_utf8("ts_code")?;
        let dates = table.required_i32_date_cast("delist_date")?;
        for (symbol, date) in symbols.iter().zip(dates) {
            if let (Some(symbol), Some(date)) = (symbol, date) {
                if date > 0 && date <= config.end_date {
                    book.dates.push((date, symbol.clone()));
                }
            }
        }
        book.dates.sort();
        book.dates.dedup();
        artifact_io::create_dir_all(config.output_dir())?;
        let mut audit = BufWriter::new(artifact_io::open_writer(
            &config.output_dir().join("lifecycle.csv"),
        )?);
        writeln!(
            audit,
            "trade_date,trade_time,symbol,delist_date,quantity,price,price_source,net_proceeds"
        )?;
        book.audit = Some(audit);
        Ok(book)
    }

    /// Register dated events once and ensure the scheduled cutoff exists even without a quote.
    pub fn begin_session(
        &mut self,
        config: &StrategyRunConfig,
        account: &mut AccountState,
        date: i32,
        frames: &mut Vec<MarketFrame>,
    ) -> Option<String> {
        while self.next < self.dates.len() && self.dates[self.next].0 <= date {
            account.register_delisting(self.dates[self.next].1.clone());
            self.next += 1;
        }
        if self.next == 0 || !account.positions().keys().any(|s| account.is_delisted(s)) {
            return None;
        }
        let time = match config.bar_frequency {
            BarFrequency::Daily => "daily".to_owned(),
            BarFrequency::Minute => format!(
                "{:04}-{:02}-{:02} {}:00",
                date / 10000,
                date / 100 % 100,
                date % 100,
                config.delisting_exit_time
            ),
        };
        if let Err(index) = frames.binary_search_by(|frame| frame.event.trade_time.cmp(&time)) {
            if index == 0 {
                if let Some(first) = frames.first_mut() {
                    first.event.is_session_first = false;
                }
            }
            if index == frames.len() {
                if let Some(last) = frames.last_mut() {
                    last.event.is_session_end = false;
                }
            }
            frames.insert(
                index,
                MarketFrame {
                    event: BarEvent {
                        trade_date: date,
                        trade_time: time.clone(),
                        bar_frequency: config.bar_frequency,
                        is_session_first: index == 0,
                        is_session_end: index == frames.len(),
                    },
                    bars: Vec::new(),
                },
            );
        }
        Some(time)
    }

    pub fn fill_exits(
        &mut self,
        config: &StrategyRunConfig,
        account: &mut AccountState,
        market: &MarketSnapshot,
        execution: &mut ExecutionEngine,
        next_order_id: &mut i64,
    ) -> Result<Vec<FillEvent>> {
        let mut exits = Vec::new();
        for (symbol, position) in account.positions() {
            if !account.is_delisted(symbol) {
                continue;
            }
            // The engine has already marked this bar. A missing bar leaves the previous mark intact.
            let price = position.mark_price();
            if !price.is_finite() || price <= 0.0 {
                return Err(err(format!(
                    "no valid price for forced delisting exit: {symbol}"
                )));
            }
            exits.push((symbol.clone(), position.quantity, price));
        }
        let mut fills = Vec::with_capacity(exits.len());
        for (symbol, quantity, price) in exits {
            let delist_date = self.dates.iter().find(|(_, s)| s == &symbol).unwrap().0;
            let source = if market.close_price(&symbol).is_some() {
                "bar_close"
            } else {
                "last_mark"
            };
            *next_order_id += 1;
            let order = Order {
                order_id: *next_order_id,
                symbol,
                signed_quantity: -quantity,
                signal_time: market.event.trade_time.clone(),
                future_side: None,
            };
            let sales = execution.force_delisting_sale(config, account, market, order, price);
            if let Some(audit) = &mut self.audit {
                for fill in &sales {
                    let proceeds = fill.notional - fill.fee - fill.tax - fill.slippage_cost;
                    writeln!(
                        audit,
                        "{},{},{},{delist_date},{quantity},{price},{source},{proceeds}",
                        fill.trade_date, fill.trade_time, fill.symbol
                    )?;
                }
            }
            fills.extend(sales);
        }
        Ok(fills)
    }

    pub fn finish(self) -> Result<()> {
        if let Some(mut audit) = self.audit {
            audit.flush()?;
            audit
                .into_inner()
                .map_err(|error| err(error.to_string()))?
                .commit()?;
        }
        Ok(())
    }
}

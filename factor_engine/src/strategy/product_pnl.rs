use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::data::parquet_io::ParquetStreamWriter;
use crate::data::{ColumnData, Table};
use crate::error::{err, Result};

use super::account::AccountState;
use super::config::future_product_from_symbol;
use super::order::FillEvent;

#[derive(Default)]
struct ProductDay {
    realized: f64,
    fee: f64,
    slippage: f64,
    trade_count: i64,
    traded_lots: f64,
    turnover_notional: f64,
}

struct ProductRow {
    trade_date: i32,
    product: String,
    gross_pnl: f64,
    net_pnl: f64,
    cumulative_net_pnl: f64,
    turnover_rate: Option<f64>,
    day: ProductDay,
}

pub(super) struct ProductPnl {
    writer: ParquetStreamWriter,
    day: BTreeMap<String, ProductDay>,
    previous_unrealized: BTreeMap<String, f64>,
    cumulative: BTreeMap<String, f64>,
    previous_equity: f64,
    rows: Vec<ProductRow>,
}

impl ProductPnl {
    pub fn new(path: &Path, initial_cash: f64) -> Result<Self> {
        Ok(Self {
            writer: ParquetStreamWriter::new(path, &table(&[])?)?,
            day: BTreeMap::new(),
            previous_unrealized: BTreeMap::new(),
            cumulative: BTreeMap::new(),
            previous_equity: initial_cash,
            rows: Vec::new(),
        })
    }

    pub fn observe(&mut self, symbol: &str) {
        self.day
            .entry(future_product_from_symbol(symbol))
            .or_default();
    }

    pub fn record(&mut self, fills: &[FillEvent]) {
        for fill in fills {
            let row = self
                .day
                .entry(future_product_from_symbol(&fill.symbol))
                .or_default();
            row.realized += fill.realized_pnl;
            row.fee += fill.fee;
            row.slippage += fill.slippage_cost;
            row.trade_count += 1;
            row.traded_lots += fill.quantity;
            row.turnover_notional += fill.notional;
        }
    }

    pub fn close_day(&mut self, date: i32, account: &AccountState) -> Result<()> {
        let mut unrealized = BTreeMap::<String, f64>::new();
        for (symbol, position) in account.all_positions() {
            *unrealized
                .entry(future_product_from_symbol(symbol))
                .or_default() += position.unrealized_pnl();
        }
        for product in self.previous_unrealized.keys().chain(unrealized.keys()) {
            self.day.entry(product.clone()).or_default();
        }
        let mut daily_net = 0.0;
        for (product, day) in std::mem::take(&mut self.day) {
            let gross_pnl = day.realized + unrealized.get(&product).copied().unwrap_or(0.0)
                - self
                    .previous_unrealized
                    .get(&product)
                    .copied()
                    .unwrap_or(0.0);
            let net_pnl = gross_pnl - day.fee - day.slippage;
            daily_net += net_pnl;
            let cumulative = self.cumulative.entry(product.clone()).or_default();
            *cumulative += net_pnl;
            self.rows.push(ProductRow {
                trade_date: date,
                product,
                gross_pnl,
                net_pnl,
                cumulative_net_pnl: *cumulative,
                turnover_rate: (self.previous_equity > 0.0)
                    .then(|| day.turnover_notional / self.previous_equity),
                day,
            });
        }
        let equity = account.equity();
        let tolerance = 1e-7 + equity.abs().max(self.previous_equity.abs()) * 1e-12;
        if !daily_net.is_finite()
            || !equity.is_finite()
            || (equity - self.previous_equity - daily_net).abs() > tolerance
        {
            return Err(err(format!(
                "{date}: product net PnL {daily_net} does not match account change {}",
                equity - self.previous_equity
            )));
        }
        self.previous_unrealized = unrealized;
        self.previous_equity = equity;
        if self.rows.len() >= 4096 {
            self.flush()?;
        }
        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        if !self.rows.is_empty() {
            self.writer.write(&table(&self.rows)?)?;
            self.rows.clear();
        }
        Ok(())
    }

    pub fn finish(mut self) -> Result<PathBuf> {
        self.flush()?;
        self.writer.finish()
    }
}

fn table(rows: &[ProductRow]) -> Result<Table> {
    let number = |value: fn(&ProductRow) -> f64| {
        ColumnData::F64(rows.iter().map(|row| Some(value(row))).collect())
    };
    Table::new(BTreeMap::from([
        (
            "trade_date".into(),
            ColumnData::I32(rows.iter().map(|row| Some(row.trade_date)).collect()),
        ),
        (
            "product".into(),
            ColumnData::Utf8(rows.iter().map(|row| Some(row.product.clone())).collect()),
        ),
        ("gross_pnl".into(), number(|row| row.gross_pnl)),
        ("fee".into(), number(|row| row.day.fee)),
        ("slippage".into(), number(|row| row.day.slippage)),
        ("net_pnl".into(), number(|row| row.net_pnl)),
        (
            "cumulative_net_pnl".into(),
            number(|row| row.cumulative_net_pnl),
        ),
        (
            "trade_count".into(),
            ColumnData::I64(rows.iter().map(|row| Some(row.day.trade_count)).collect()),
        ),
        ("traded_lots".into(), number(|row| row.day.traded_lots)),
        (
            "turnover_notional".into(),
            number(|row| row.day.turnover_notional),
        ),
        (
            "turnover_rate".into(),
            ColumnData::F64(rows.iter().map(|row| row.turnover_rate).collect()),
        ),
    ]))
}

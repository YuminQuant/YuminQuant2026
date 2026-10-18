use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::data::parquet_io::ParquetStreamWriter;
use crate::data::{ColumnData, Table};
use crate::error::Result;
use crate::strategy::order::HoldingSnapshot;

/// Bound snapshot buffering by both row count and approximate string bytes.
pub struct HoldingsWriter {
    writer: ParquetStreamWriter,
    rows: Vec<HoldingSnapshot>,
    bytes: usize,
    legacy: PathBuf,
}

impl HoldingsWriter {
    pub fn new(output_dir: &Path) -> Result<Self> {
        Ok(Self {
            writer: ParquetStreamWriter::new(
                &output_dir.join("holdings.parquet"),
                &holdings_table(&[])?,
            )?,
            rows: Vec::new(),
            bytes: 0,
            legacy: output_dir.join("trades.parquet"),
        })
    }

    pub fn push(&mut self, row: HoldingSnapshot) -> Result<()> {
        self.bytes += snapshot_bytes(&row);
        self.rows.push(row);
        if self.rows.len() >= 256 || self.bytes >= 8 * 1024 * 1024 {
            self.flush()?;
        }
        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        if !self.rows.is_empty() {
            self.writer.write(&holdings_table(&self.rows)?)?;
            self.rows.clear();
            self.bytes = 0;
        }
        Ok(())
    }

    pub fn finish(mut self) -> Result<PathBuf> {
        self.flush()?;
        let path = self.writer.finish()?;
        if self.legacy.exists() {
            std::fs::remove_file(self.legacy)?;
        }
        Ok(path)
    }
}

pub fn write_holdings(output_dir: &Path, rows: &[HoldingSnapshot]) -> Result<PathBuf> {
    let mut writer = HoldingsWriter::new(output_dir)?;
    for row in rows {
        writer.push(row.clone())?;
    }
    writer.finish()
}

fn snapshot_bytes(row: &HoldingSnapshot) -> usize {
    std::mem::size_of::<HoldingSnapshot>()
        + [
            &row.strategy_id,
            &row.asset_class,
            &row.trade_time,
            &row.symbols_json,
            &row.quantities_json,
            &row.signed_quantities_json,
            &row.directions_json,
            &row.avg_costs_json,
            &row.market_values_json,
            &row.unrealized_pnls_json,
            &row.multipliers_json,
            &row.margin_ratios_json,
            &row.margin_values_json,
            &row.trade_symbols_json,
            &row.trade_sides_json,
            &row.trade_quantities_json,
            &row.trade_signed_quantities_json,
            &row.trade_prices_json,
            &row.trade_realized_pnls_json,
            &row.trade_net_pnls_json,
        ]
        .iter()
        .map(|value| value.len())
        .sum::<usize>()
}

fn holdings_table(rows: &[HoldingSnapshot]) -> Result<Table> {
    Table::new(BTreeMap::from([
        (
            "strategy_id".to_string(),
            utf8(rows.iter().map(|row| Some(row.strategy_id.clone()))),
        ),
        (
            "asset_class".to_string(),
            utf8(rows.iter().map(|row| Some(row.asset_class.clone()))),
        ),
        (
            "trade_date".to_string(),
            i32_col(rows.iter().map(|row| Some(row.trade_date))),
        ),
        (
            "trade_time".to_string(),
            utf8(rows.iter().map(|row| Some(row.trade_time.clone()))),
        ),
        (
            "dividend_receivable".to_string(),
            f64_col(rows.iter().map(|row| Some(row.dividend_receivable))),
        ),
        (
            "dividend_income_cum".to_string(),
            f64_col(rows.iter().map(|row| Some(row.dividend_income_cum))),
        ),
        (
            "cash".to_string(),
            f64_col(rows.iter().map(|row| Some(row.cash))),
        ),
        (
            "account_pnl".to_string(),
            f64_col(rows.iter().map(|row| Some(row.account_pnl))),
        ),
        (
            "realized_pnl_cum".to_string(),
            f64_col(rows.iter().map(|row| Some(row.realized_pnl_cum))),
        ),
        (
            "net_realized_pnl_cum".to_string(),
            f64_col(rows.iter().map(|row| Some(row.net_realized_pnl_cum))),
        ),
        (
            "unrealized_pnl".to_string(),
            f64_col(rows.iter().map(|row| Some(row.unrealized_pnl))),
        ),
        (
            "gross_market_value".to_string(),
            f64_col(rows.iter().map(|row| Some(row.gross_market_value))),
        ),
        (
            "net_market_value".to_string(),
            f64_col(rows.iter().map(|row| Some(row.net_market_value))),
        ),
        (
            "margin_required".to_string(),
            f64_col(rows.iter().map(|row| Some(row.margin_required))),
        ),
        (
            "available_margin".to_string(),
            f64_col(rows.iter().map(|row| Some(row.available_margin))),
        ),
        (
            "position_count".to_string(),
            i64_col(rows.iter().map(|row| Some(row.position_count))),
        ),
        (
            "buy_count".to_string(),
            i64_col(rows.iter().map(|row| Some(row.buy_count))),
        ),
        (
            "sell_count".to_string(),
            i64_col(rows.iter().map(|row| Some(row.sell_count))),
        ),
        (
            "short_count".to_string(),
            i64_col(rows.iter().map(|row| Some(row.short_count))),
        ),
        (
            "cover_count".to_string(),
            i64_col(rows.iter().map(|row| Some(row.cover_count))),
        ),
        (
            "symbols_json".to_string(),
            utf8(rows.iter().map(|row| Some(row.symbols_json.clone()))),
        ),
        (
            "quantities_json".to_string(),
            utf8(rows.iter().map(|row| Some(row.quantities_json.clone()))),
        ),
        (
            "signed_quantities_json".to_string(),
            utf8(
                rows.iter()
                    .map(|row| Some(row.signed_quantities_json.clone())),
            ),
        ),
        (
            "directions_json".to_string(),
            utf8(rows.iter().map(|row| Some(row.directions_json.clone()))),
        ),
        (
            "avg_costs_json".to_string(),
            utf8(rows.iter().map(|row| Some(row.avg_costs_json.clone()))),
        ),
        (
            "market_values_json".to_string(),
            utf8(rows.iter().map(|row| Some(row.market_values_json.clone()))),
        ),
        (
            "unrealized_pnls_json".to_string(),
            utf8(
                rows.iter()
                    .map(|row| Some(row.unrealized_pnls_json.clone())),
            ),
        ),
        (
            "multipliers_json".to_string(),
            utf8(rows.iter().map(|row| Some(row.multipliers_json.clone()))),
        ),
        (
            "margin_ratios_json".to_string(),
            utf8(rows.iter().map(|row| Some(row.margin_ratios_json.clone()))),
        ),
        (
            "margin_values_json".to_string(),
            utf8(rows.iter().map(|row| Some(row.margin_values_json.clone()))),
        ),
        (
            "trade_symbols_json".to_string(),
            utf8(rows.iter().map(|row| Some(row.trade_symbols_json.clone()))),
        ),
        (
            "trade_sides_json".to_string(),
            utf8(rows.iter().map(|row| Some(row.trade_sides_json.clone()))),
        ),
        (
            "trade_quantities_json".to_string(),
            utf8(
                rows.iter()
                    .map(|row| Some(row.trade_quantities_json.clone())),
            ),
        ),
        (
            "trade_signed_quantities_json".to_string(),
            utf8(
                rows.iter()
                    .map(|row| Some(row.trade_signed_quantities_json.clone())),
            ),
        ),
        (
            "trade_prices_json".to_string(),
            utf8(rows.iter().map(|row| Some(row.trade_prices_json.clone()))),
        ),
        (
            "trade_realized_pnls_json".to_string(),
            utf8(
                rows.iter()
                    .map(|row| Some(row.trade_realized_pnls_json.clone())),
            ),
        ),
        (
            "trade_net_pnls_json".to_string(),
            utf8(rows.iter().map(|row| Some(row.trade_net_pnls_json.clone()))),
        ),
    ]))
}

fn utf8<I>(values: I) -> ColumnData
where
    I: Iterator<Item = Option<String>>,
{
    ColumnData::Utf8(values.collect())
}

fn i32_col<I>(values: I) -> ColumnData
where
    I: Iterator<Item = Option<i32>>,
{
    ColumnData::I32(values.collect())
}

fn i64_col<I>(values: I) -> ColumnData
where
    I: Iterator<Item = Option<i64>>,
{
    ColumnData::I64(values.collect())
}

fn f64_col<I>(values: I) -> ColumnData
where
    I: Iterator<Item = Option<f64>>,
{
    ColumnData::F64(values.collect())
}

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::data::parquet_io::write_parquet;
use crate::data::{ColumnData, Table};
use crate::error::Result;
use crate::strategy::order::HoldingSnapshot;

pub fn write_holdings(output_dir: &Path, rows: &[HoldingSnapshot]) -> Result<PathBuf> {
    let path = output_dir.join("holdings.parquet");
    let legacy = output_dir.join("trades.parquet");
    if legacy.exists() {
        std::fs::remove_file(legacy)?;
    }
    write_parquet(&path, &holdings_table(rows)?)?;
    Ok(path)
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
            "bar_frequency".to_string(),
            utf8(rows.iter().map(|row| Some(row.bar_frequency.clone()))),
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
            "trade_count".to_string(),
            i64_col(rows.iter().map(|row| Some(row.trade_count))),
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
            "prices_json".to_string(),
            utf8(rows.iter().map(|row| Some(row.prices_json.clone()))),
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
        (
            "trade_order_ids_json".to_string(),
            utf8(
                rows.iter()
                    .map(|row| Some(row.trade_order_ids_json.clone())),
            ),
        ),
        (
            "trade_fill_ids_json".to_string(),
            utf8(rows.iter().map(|row| Some(row.trade_fill_ids_json.clone()))),
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

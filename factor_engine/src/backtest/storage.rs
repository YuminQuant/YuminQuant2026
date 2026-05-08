use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::backtest::ic::IcObservation;
use crate::backtest::metrics::{
    FactorStatsSummary, IcSummary, PerformancePoint, PerformanceSummary,
};
use crate::data::parquet_io::write_parquet;
use crate::data::{ColumnData, Table};
use crate::error::Result;

#[derive(Clone, Debug, Default)]
pub struct BacktestOutputFiles {
    pub summary_files: Vec<PathBuf>,
    pub detail_files: Vec<PathBuf>,
}

pub fn write_summary_outputs(
    output_dir: &Path,
    performance: &[PerformanceSummary],
    ic: &[IcSummary],
    ic_decay: &[IcSummary],
    factor_stats: &[FactorStatsSummary],
) -> Result<Vec<PathBuf>> {
    std::fs::create_dir_all(output_dir)?;
    let mut written = Vec::new();
    let path = output_dir.join("performance_summary.parquet");
    write_parquet(&path, &performance_summary_table(performance)?)?;
    written.push(path);
    let path = output_dir.join("ic_summary.parquet");
    write_parquet(&path, &ic_summary_table(ic)?)?;
    written.push(path);
    let path = output_dir.join("ic_decay_summary.parquet");
    write_parquet(&path, &ic_summary_table(ic_decay)?)?;
    written.push(path);
    let path = output_dir.join("factor_stats.parquet");
    write_parquet(&path, &factor_stats_table(factor_stats)?)?;
    written.push(path);
    Ok(written)
}

pub fn write_detail_outputs(
    output_dir: &Path,
    returns: &[PerformancePoint],
    daily_ic: &[IcObservation],
    ic_decay: &[IcObservation],
) -> Result<Vec<PathBuf>> {
    let detail_dir = output_dir.join("detail");
    std::fs::create_dir_all(&detail_dir)?;
    let mut written = Vec::new();
    let path = detail_dir.join("daily_returns.parquet");
    write_parquet(&path, &returns_table(returns)?)?;
    written.push(path);
    let path = detail_dir.join("daily_ic.parquet");
    write_parquet(&path, &ic_observation_table(daily_ic)?)?;
    written.push(path);
    let path = detail_dir.join("ic_decay.parquet");
    write_parquet(&path, &ic_observation_table(ic_decay)?)?;
    written.push(path);
    Ok(written)
}

fn performance_summary_table(rows: &[PerformanceSummary]) -> Result<Table> {
    table_from_columns(BTreeMap::from([
        (
            "factor_id",
            utf8(rows.iter().map(|row| Some(row.factor_id.clone()))),
        ),
        (
            "scope",
            utf8(rows.iter().map(|row| Some(row.scope.clone()))),
        ),
        ("year", i32_col(rows.iter().map(|row| row.year))),
        (
            "portfolio",
            utf8(rows.iter().map(|row| Some(row.portfolio.clone()))),
        ),
        (
            "observations",
            i64_col(rows.iter().map(|row| Some(row.observations))),
        ),
        (
            "mean_return",
            f64_col(rows.iter().map(|row| row.mean_return)),
        ),
        ("std_return", f64_col(rows.iter().map(|row| row.std_return))),
        (
            "cumulative_return",
            f64_col(rows.iter().map(|row| row.cumulative_return)),
        ),
        (
            "annualized_return",
            f64_col(rows.iter().map(|row| row.annualized_return)),
        ),
        (
            "annualized_volatility",
            f64_col(rows.iter().map(|row| row.annualized_volatility)),
        ),
        ("sharpe", f64_col(rows.iter().map(|row| row.sharpe))),
        (
            "max_drawdown",
            f64_col(rows.iter().map(|row| row.max_drawdown)),
        ),
        (
            "avg_turnover",
            f64_col(rows.iter().map(|row| row.avg_turnover)),
        ),
    ]))
}

fn factor_stats_table(rows: &[FactorStatsSummary]) -> Result<Table> {
    table_from_columns(BTreeMap::from([
        (
            "factor_id",
            utf8(rows.iter().map(|row| Some(row.factor_id.clone()))),
        ),
        (
            "scope",
            utf8(rows.iter().map(|row| Some(row.scope.clone()))),
        ),
        ("year", i32_col(rows.iter().map(|row| row.year))),
        (
            "observations",
            i64_col(rows.iter().map(|row| Some(row.observations))),
        ),
        ("mean", f64_col(rows.iter().map(|row| row.mean))),
        ("std", f64_col(rows.iter().map(|row| row.std))),
        ("min", f64_col(rows.iter().map(|row| row.min))),
        ("p25", f64_col(rows.iter().map(|row| row.p25))),
        ("median", f64_col(rows.iter().map(|row| row.median))),
        ("p75", f64_col(rows.iter().map(|row| row.p75))),
        ("max", f64_col(rows.iter().map(|row| row.max))),
        (
            "coverage_mean",
            f64_col(rows.iter().map(|row| row.coverage_mean)),
        ),
        (
            "inf_rate_mean",
            f64_col(rows.iter().map(|row| row.inf_rate_mean)),
        ),
    ]))
}

fn ic_summary_table(rows: &[IcSummary]) -> Result<Table> {
    table_from_columns(BTreeMap::from([
        (
            "factor_id",
            utf8(rows.iter().map(|row| Some(row.factor_id.clone()))),
        ),
        (
            "scope",
            utf8(rows.iter().map(|row| Some(row.scope.clone()))),
        ),
        ("year", i32_col(rows.iter().map(|row| row.year))),
        (
            "horizon",
            i32_col(rows.iter().map(|row| row.horizon.map(|value| value as i32))),
        ),
        (
            "observations",
            i64_col(rows.iter().map(|row| Some(row.observations))),
        ),
        ("ic_mean", f64_col(rows.iter().map(|row| row.ic_mean))),
        ("ic_std", f64_col(rows.iter().map(|row| row.ic_std))),
        ("icir", f64_col(rows.iter().map(|row| row.icir))),
        (
            "ic_abs_mean",
            f64_col(rows.iter().map(|row| row.ic_abs_mean)),
        ),
        ("ic_abs_std", f64_col(rows.iter().map(|row| row.ic_abs_std))),
        ("icir_abs", f64_col(rows.iter().map(|row| row.icir_abs))),
        (
            "rank_ic_mean",
            f64_col(rows.iter().map(|row| row.rank_ic_mean)),
        ),
        (
            "rank_ic_std",
            f64_col(rows.iter().map(|row| row.rank_ic_std)),
        ),
        ("rank_icir", f64_col(rows.iter().map(|row| row.rank_icir))),
        (
            "rank_ic_abs_mean",
            f64_col(rows.iter().map(|row| row.rank_ic_abs_mean)),
        ),
        (
            "rank_ic_abs_std",
            f64_col(rows.iter().map(|row| row.rank_ic_abs_std)),
        ),
        (
            "rank_icir_abs",
            f64_col(rows.iter().map(|row| row.rank_icir_abs)),
        ),
        (
            "coverage_mean",
            f64_col(rows.iter().map(|row| row.coverage_mean)),
        ),
        (
            "inf_rate_mean",
            f64_col(rows.iter().map(|row| row.inf_rate_mean)),
        ),
    ]))
}

fn returns_table(rows: &[PerformancePoint]) -> Result<Table> {
    table_from_columns(BTreeMap::from([
        (
            "factor_id",
            utf8(rows.iter().map(|row| Some(row.factor_id.clone()))),
        ),
        (
            "factor_date",
            i32_col(rows.iter().map(|row| Some(row.factor_date))),
        ),
        ("trade_date", i32_col(rows.iter().map(|row| row.trade_date))),
        (
            "settle_date",
            i32_col(rows.iter().map(|row| row.settle_date)),
        ),
        (
            "portfolio",
            utf8(rows.iter().map(|row| Some(row.portfolio.clone()))),
        ),
        ("return", f64_col(rows.iter().map(|row| row.return_value))),
        ("nav", f64_col(rows.iter().map(|row| row.nav))),
        ("turnover", f64_col(rows.iter().map(|row| row.turnover))),
    ]))
}

fn ic_observation_table(rows: &[IcObservation]) -> Result<Table> {
    table_from_columns(BTreeMap::from([
        (
            "factor_id",
            utf8(rows.iter().map(|row| Some(row.factor_id.clone()))),
        ),
        (
            "factor_date",
            i32_col(rows.iter().map(|row| Some(row.factor_date))),
        ),
        (
            "label_date",
            i32_col(rows.iter().map(|row| Some(row.label_date))),
        ),
        (
            "settle_date",
            i32_col(rows.iter().map(|row| row.settle_date)),
        ),
        (
            "horizon",
            i32_col(rows.iter().map(|row| row.horizon.map(|value| value as i32))),
        ),
        ("ic", f64_col(rows.iter().map(|row| row.ic))),
        ("rank_ic", f64_col(rows.iter().map(|row| row.rank_ic))),
        (
            "pair_count",
            i64_col(rows.iter().map(|row| Some(row.pair_count as i64))),
        ),
        (
            "coverage",
            f64_col(rows.iter().map(|row| Some(row.coverage))),
        ),
        (
            "inf_rate",
            f64_col(rows.iter().map(|row| Some(row.inf_rate))),
        ),
    ]))
}

fn table_from_columns(columns: BTreeMap<&str, ColumnData>) -> Result<Table> {
    Table::new(
        columns
            .into_iter()
            .map(|(name, column)| (name.to_string(), column))
            .collect(),
    )
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

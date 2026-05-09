use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::backtest::ic::IcObservation;
use crate::backtest::metrics::{FactorStatsSummary, PerformancePoint};
use crate::data::parquet_io::write_parquet;
use crate::data::{ColumnData, Table};
use crate::error::Result;

pub fn write_backtest_outputs(
    output_dir: &Path,
    returns: &[PerformancePoint],
    daily_ic: &[IcObservation],
    factor_stats: &[FactorStatsSummary],
) -> Result<Vec<PathBuf>> {
    std::fs::create_dir_all(output_dir)?;
    remove_legacy_outputs(output_dir)?;
    let mut written = Vec::new();

    let returns_dir = output_dir.join("returns");
    std::fs::create_dir_all(&returns_dir)?;
    for (factor_id, rows) in group_returns_by_factor(returns) {
        let path = returns_dir.join(format!("{}.parquet", safe_file_stem(&factor_id)));
        write_parquet(&path, &returns_table(&rows)?)?;
        written.push(path);
    }

    let ic_dir = output_dir.join("ic");
    std::fs::create_dir_all(&ic_dir)?;
    for (factor_id, rows) in group_ic_by_factor(daily_ic) {
        let path = ic_dir.join(format!("{}.parquet", safe_file_stem(&factor_id)));
        write_parquet(&path, &ic_observation_table(&rows)?)?;
        written.push(path);
    }

    let factor_stats_dir = output_dir.join("factor_stats");
    std::fs::create_dir_all(&factor_stats_dir)?;
    for (factor_id, rows) in group_factor_stats_by_factor(factor_stats) {
        let path = factor_stats_dir.join(format!("{}.parquet", safe_file_stem(&factor_id)));
        write_parquet(&path, &factor_stats_table(&rows)?)?;
        written.push(path);
    }
    Ok(written)
}

fn remove_legacy_outputs(output_dir: &Path) -> Result<()> {
    for name in [
        "performance_summary.parquet",
        "ic_summary.parquet",
        "ic_decay_summary.parquet",
        "factor_stats.parquet",
    ] {
        let path = output_dir.join(name);
        if path.exists() {
            std::fs::remove_file(path)?;
        }
    }

    let legacy_detail = output_dir.join("detail");
    for name in [
        "daily_returns.parquet",
        "daily_ic.parquet",
        "ic_decay.parquet",
    ] {
        let path = legacy_detail.join(name);
        if path.exists() {
            std::fs::remove_file(path)?;
        }
    }
    Ok(())
}

fn group_returns_by_factor(rows: &[PerformancePoint]) -> BTreeMap<String, Vec<PerformancePoint>> {
    let mut grouped = BTreeMap::<String, Vec<PerformancePoint>>::new();
    for row in rows {
        grouped
            .entry(row.factor_id.clone())
            .or_default()
            .push(row.clone());
    }
    grouped
}

fn group_ic_by_factor(rows: &[IcObservation]) -> BTreeMap<String, Vec<IcObservation>> {
    let mut grouped = BTreeMap::<String, Vec<IcObservation>>::new();
    for row in rows {
        grouped
            .entry(row.factor_id.clone())
            .or_default()
            .push(row.clone());
    }
    grouped
}

fn group_factor_stats_by_factor(
    rows: &[FactorStatsSummary],
) -> BTreeMap<String, Vec<FactorStatsSummary>> {
    let mut grouped = BTreeMap::<String, Vec<FactorStatsSummary>>::new();
    for row in rows {
        grouped
            .entry(row.factor_id.clone())
            .or_default()
            .push(row.clone());
    }
    grouped
}

fn safe_file_stem(value: &str) -> String {
    value
        .chars()
        .map(|ch| match ch {
            '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|' => '_',
            _ => ch,
        })
        .collect()
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

use crate::artifact_io::StoragePathExt;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::backtest::ic::IcObservation;
use crate::backtest::metrics::{
    BarraExposureRecord, FactorStatsDaily, HoldingWeight, IndexGroupReturnPoint, IndustryWeight,
    PerformancePoint,
};
use crate::data::parquet_io::write_parquet;
use crate::data::{ColumnData, Table};
use crate::error::Result;

pub fn write_backtest_outputs(
    output_dir: &Path,
    returns: &[PerformancePoint],
    index_group_returns: &[IndexGroupReturnPoint],
    daily_ic: &[IcObservation],
    factor_stats: &[FactorStatsDaily],
    holdings: &[HoldingWeight],
    industry_weights: &[IndustryWeight],
    barra_exposure: &[BarraExposureRecord],
) -> Result<Vec<PathBuf>> {
    crate::artifact_io::create_dir_all(output_dir)?;
    remove_legacy_outputs(output_dir)?;
    let mut jobs: Vec<Box<dyn FnOnce() -> Result<PathBuf> + Send>> = Vec::new();

    for (factor_id, rows) in group_by_factor(returns, |row| &row.factor_id) {
        jobs.push(Box::new(move || {
            let path = factor_metric_path(output_dir, &factor_id, "returns.parquet")?;
            write_parquet(&path, &returns_table(&rows)?)?;
            Ok(path)
        }));
    }

    for (factor_id, rows) in group_by_factor(index_group_returns, |row| &row.factor_id) {
        jobs.push(Box::new(move || {
            let path = factor_metric_path(output_dir, &factor_id, "index_group_returns.parquet")?;
            write_parquet(&path, &index_group_returns_table(&rows)?)?;
            Ok(path)
        }));
    }

    for (factor_id, rows) in group_by_factor(daily_ic, |row| &row.factor_id) {
        jobs.push(Box::new(move || {
            let path = factor_metric_path(output_dir, &factor_id, "ic.parquet")?;
            write_parquet(&path, &ic_observation_table(&rows)?)?;
            Ok(path)
        }));
    }

    for (factor_id, rows) in group_by_factor(factor_stats, |row| &row.factor_id) {
        jobs.push(Box::new(move || {
            let path = factor_metric_path(output_dir, &factor_id, "factor_stats.parquet")?;
            write_parquet(&path, &factor_stats_table(&rows)?)?;
            Ok(path)
        }));
    }

    if !holdings.is_empty() {
        for (factor_id, rows) in group_by_factor(holdings, |row| &row.factor_id) {
            jobs.push(Box::new(move || {
                let path = factor_metric_path(output_dir, &factor_id, "holdings.parquet")?;
                write_parquet(&path, &holdings_table(&rows)?)?;
                Ok(path)
            }));
        }
    }

    if !industry_weights.is_empty() {
        for (factor_id, rows) in group_by_factor(industry_weights, |row| &row.factor_id) {
            jobs.push(Box::new(move || {
                let path = factor_metric_path(output_dir, &factor_id, "industry_weights.parquet")?;
                write_parquet(&path, &industry_weights_table(&rows)?)?;
                Ok(path)
            }));
        }
    }
    if !barra_exposure.is_empty() {
        for (factor_id, rows) in group_by_factor(barra_exposure, |row| &row.factor_id) {
            jobs.push(Box::new(move || {
                let path = factor_metric_path(output_dir, &factor_id, "barra_exposure.parquet")?;
                write_parquet(&path, &barra_exposure_table(&rows)?)?;
                Ok(path)
            }));
        }
    }
    crate::data::read_ahead::write_files(output_dir, jobs, |job| job())
}

fn factor_metric_path(output_dir: &Path, factor_id: &str, metric_file: &str) -> Result<PathBuf> {
    let factor_dir = output_dir.join(safe_file_stem(factor_id));
    crate::artifact_io::create_dir_all(&factor_dir)?;
    Ok(factor_dir.join(metric_file))
}

fn remove_legacy_outputs(output_dir: &Path) -> Result<()> {
    for name in [
        "performance_summary.parquet",
        "ic_summary.parquet",
        "ic_decay_summary.parquet",
        "factor_stats.parquet",
    ] {
        let path = output_dir.join(name);
        if path.storage_exists() {
            crate::artifact_io::remove_file(path)?;
        }
    }

    let legacy_detail = output_dir.join("detail");
    for name in [
        "daily_returns.parquet",
        "daily_ic.parquet",
        "ic_decay.parquet",
    ] {
        let path = legacy_detail.join(name);
        if path.storage_exists() {
            crate::artifact_io::remove_file(path)?;
        }
    }
    Ok(())
}

fn group_by_factor<T>(
    rows: &[T],
    factor_id: impl Fn(&T) -> &str,
) -> BTreeMap<&str, Vec<&T>> {
    let mut grouped = BTreeMap::<&str, Vec<&T>>::new();
    for row in rows {
        grouped.entry(factor_id(row)).or_default().push(row);
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

pub(crate) fn factor_stats_table(rows: &[&FactorStatsDaily]) -> Result<Table> {
    table_from_columns(BTreeMap::from([
        (
            "factor_id",
            utf8(rows.iter().map(|row| Some(row.factor_id.clone()))),
        ),
        (
            "trade_date",
            i32_col(rows.iter().map(|row| Some(row.trade_date))),
        ),
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
            "coverage",
            f64_col(rows.iter().map(|row| Some(row.coverage))),
        ),
        (
            "inf_rate",
            f64_col(rows.iter().map(|row| Some(row.inf_rate))),
        ),
    ]))
}

fn returns_table(rows: &[&PerformancePoint]) -> Result<Table> {
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
        (
            "benchmark_return",
            f64_col(rows.iter().map(|row| row.benchmark_return)),
        ),
        (
            "excess_return",
            f64_col(rows.iter().map(|row| row.excess_return)),
        ),
        ("turnover", f64_col(rows.iter().map(|row| row.turnover))),
    ]))
}

fn index_group_returns_table(rows: &[&IndexGroupReturnPoint]) -> Result<Table> {
    table_from_columns(BTreeMap::from([
        (
            "factor_id",
            utf8(rows.iter().map(|row| Some(row.factor_id.clone()))),
        ),
        (
            "index_id",
            utf8(rows.iter().map(|row| Some(row.index_id.clone()))),
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
        (
            "benchmark_return",
            f64_col(rows.iter().map(|row| row.benchmark_return)),
        ),
        (
            "excess_return",
            f64_col(rows.iter().map(|row| row.excess_return)),
        ),
        ("turnover", f64_col(rows.iter().map(|row| row.turnover))),
        (
            "member_count",
            i64_col(rows.iter().map(|row| Some(row.member_count))),
        ),
        (
            "benchmark_count",
            i64_col(rows.iter().map(|row| Some(row.benchmark_count))),
        ),
    ]))
}

fn ic_observation_table(rows: &[&IcObservation]) -> Result<Table> {
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

fn holdings_table(rows: &[&HoldingWeight]) -> Result<Table> {
    table_from_columns(BTreeMap::from([
        (
            "factor_id",
            utf8(rows.iter().map(|row| Some(row.factor_id.clone()))),
        ),
        (
            "rebalance_date",
            i32_col(rows.iter().map(|row| Some(row.rebalance_date))),
        ),
        (
            "portfolio",
            utf8(rows.iter().map(|row| Some(row.portfolio.clone()))),
        ),
        (
            "rank_ic_sign",
            f64_col(rows.iter().map(|row| Some(row.rank_ic_sign))),
        ),
        (
            "ts_code",
            utf8(rows.iter().map(|row| Some(row.ts_code.clone()))),
        ),
        ("weight", f64_col(rows.iter().map(|row| Some(row.weight)))),
    ]))
}

fn industry_weights_table(rows: &[&IndustryWeight]) -> Result<Table> {
    table_from_columns(BTreeMap::from([
        (
            "factor_id",
            utf8(rows.iter().map(|row| Some(row.factor_id.clone()))),
        ),
        (
            "rebalance_date",
            i32_col(rows.iter().map(|row| Some(row.rebalance_date))),
        ),
        (
            "portfolio",
            utf8(rows.iter().map(|row| Some(row.portfolio.clone()))),
        ),
        (
            "rank_ic_sign",
            f64_col(rows.iter().map(|row| Some(row.rank_ic_sign))),
        ),
        (
            "sector_source",
            utf8(rows.iter().map(|row| Some(row.sector_source.clone()))),
        ),
        (
            "sector_code",
            utf8(rows.iter().map(|row| Some(row.sector_code.clone()))),
        ),
        ("weight", f64_col(rows.iter().map(|row| Some(row.weight)))),
        (
            "stock_count",
            i64_col(rows.iter().map(|row| Some(row.stock_count))),
        ),
    ]))
}

fn barra_exposure_table(rows: &[&BarraExposureRecord]) -> Result<Table> {
    table_from_columns(BTreeMap::from([
        (
            "factor_id",
            utf8(rows.iter().map(|row| Some(row.factor_id.clone()))),
        ),
        ("trade_date", i32_col(rows.iter().map(|row| row.trade_date))),
        (
            "metric",
            utf8(rows.iter().map(|row| Some(row.metric.clone()))),
        ),
        (
            "barra_factor",
            utf8(rows.iter().map(|row| Some(row.barra_factor.clone()))),
        ),
        (
            "selected_group",
            utf8(rows.iter().map(|row| row.selected_group.clone())),
        ),
        (
            "rank_ic_sign",
            f64_col(rows.iter().map(|row| row.rank_ic_sign)),
        ),
        ("value", f64_col(rows.iter().map(|row| row.value))),
        ("pair_count", i64_col(rows.iter().map(|row| row.pair_count))),
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

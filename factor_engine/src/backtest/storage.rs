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
    std::fs::create_dir_all(output_dir)?;
    remove_legacy_outputs(output_dir)?;
    let mut written = Vec::new();

    for (factor_id, rows) in group_returns_by_factor(returns) {
        let path = factor_metric_path(output_dir, &factor_id, "returns.parquet")?;
        write_parquet(&path, &returns_table(&rows)?)?;
        written.push(path);
    }

    for (factor_id, rows) in group_index_group_returns_by_factor(index_group_returns) {
        let path = factor_metric_path(output_dir, &factor_id, "index_group_returns.parquet")?;
        write_parquet(&path, &index_group_returns_table(&rows)?)?;
        written.push(path);
    }

    for (factor_id, rows) in group_ic_by_factor(daily_ic) {
        let path = factor_metric_path(output_dir, &factor_id, "ic.parquet")?;
        write_parquet(&path, &ic_observation_table(&rows)?)?;
        written.push(path);
    }

    for (factor_id, rows) in group_factor_stats_by_factor(factor_stats) {
        let path = factor_metric_path(output_dir, &factor_id, "factor_stats.parquet")?;
        write_parquet(&path, &factor_stats_table(&rows)?)?;
        written.push(path);
    }

    if !holdings.is_empty() {
        for (factor_id, rows) in group_holdings_by_factor(holdings) {
            let path = factor_metric_path(output_dir, &factor_id, "holdings.parquet")?;
            write_parquet(&path, &holdings_table(&rows)?)?;
            written.push(path);
        }
    }

    if !industry_weights.is_empty() {
        for (factor_id, rows) in group_industry_weights_by_factor(industry_weights) {
            let path = factor_metric_path(output_dir, &factor_id, "industry_weights.parquet")?;
            write_parquet(&path, &industry_weights_table(&rows)?)?;
            written.push(path);
        }
    }
    if !barra_exposure.is_empty() {
        for (factor_id, rows) in group_barra_exposure_by_factor(barra_exposure) {
            let path = factor_metric_path(output_dir, &factor_id, "barra_exposure.parquet")?;
            write_parquet(&path, &barra_exposure_table(&rows)?)?;
            written.push(path);
        }
    }
    Ok(written)
}

fn factor_metric_path(output_dir: &Path, factor_id: &str, metric_file: &str) -> Result<PathBuf> {
    let factor_dir = output_dir.join(safe_file_stem(factor_id));
    std::fs::create_dir_all(&factor_dir)?;
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

fn group_index_group_returns_by_factor(
    rows: &[IndexGroupReturnPoint],
) -> BTreeMap<String, Vec<IndexGroupReturnPoint>> {
    let mut grouped = BTreeMap::<String, Vec<IndexGroupReturnPoint>>::new();
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
    rows: &[FactorStatsDaily],
) -> BTreeMap<String, Vec<FactorStatsDaily>> {
    let mut grouped = BTreeMap::<String, Vec<FactorStatsDaily>>::new();
    for row in rows {
        grouped
            .entry(row.factor_id.clone())
            .or_default()
            .push(row.clone());
    }
    grouped
}

fn group_holdings_by_factor(rows: &[HoldingWeight]) -> BTreeMap<String, Vec<HoldingWeight>> {
    let mut grouped = BTreeMap::<String, Vec<HoldingWeight>>::new();
    for row in rows {
        grouped
            .entry(row.factor_id.clone())
            .or_default()
            .push(row.clone());
    }
    grouped
}

fn group_industry_weights_by_factor(
    rows: &[IndustryWeight],
) -> BTreeMap<String, Vec<IndustryWeight>> {
    let mut grouped = BTreeMap::<String, Vec<IndustryWeight>>::new();
    for row in rows {
        grouped
            .entry(row.factor_id.clone())
            .or_default()
            .push(row.clone());
    }
    grouped
}

fn group_barra_exposure_by_factor(
    rows: &[BarraExposureRecord],
) -> BTreeMap<String, Vec<BarraExposureRecord>> {
    let mut grouped = BTreeMap::<String, Vec<BarraExposureRecord>>::new();
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

fn factor_stats_table(rows: &[FactorStatsDaily]) -> Result<Table> {
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

fn index_group_returns_table(rows: &[IndexGroupReturnPoint]) -> Result<Table> {
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

fn holdings_table(rows: &[HoldingWeight]) -> Result<Table> {
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

fn industry_weights_table(rows: &[IndustryWeight]) -> Result<Table> {
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

fn barra_exposure_table(rows: &[BarraExposureRecord]) -> Result<Table> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backtest::ic::IcObservation;
    use crate::backtest::metrics::{
        BarraExposureRecord, FactorStatsDaily, IndexGroupReturnPoint, PerformancePoint,
    };
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn write_backtest_outputs_uses_factor_first_layout() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let output_dir = std::env::temp_dir().join(format!(
            "yq_factor_engine_storage_layout_{}_{}",
            std::process::id(),
            nonce
        ));

        let returns = vec![PerformancePoint {
            factor_id: "factor:a".to_string(),
            factor_date: 20240102,
            trade_date: Some(20240103),
            settle_date: Some(20240104),
            portfolio: "group_1".to_string(),
            return_value: Some(0.01),
            benchmark_return: Some(0.0),
            excess_return: Some(0.01),
            turnover: Some(0.5),
        }];
        let index_returns = vec![IndexGroupReturnPoint {
            factor_id: "factor:a".to_string(),
            index_id: "000300.SH".to_string(),
            factor_date: 20240102,
            trade_date: Some(20240103),
            settle_date: Some(20240104),
            portfolio: "group_1".to_string(),
            return_value: Some(0.01),
            benchmark_return: Some(0.0),
            excess_return: Some(0.01),
            turnover: Some(0.5),
            member_count: 10,
            benchmark_count: 100,
        }];
        let daily_ic = vec![IcObservation {
            factor_id: "factor:a".to_string(),
            factor_date: 20240102,
            label_date: 20240103,
            settle_date: Some(20240104),
            horizon: Some(1),
            ic: Some(0.1),
            rank_ic: Some(0.2),
            pair_count: 10,
            coverage: 1.0,
            inf_rate: 0.0,
        }];
        let factor_stats = vec![FactorStatsDaily {
            factor_id: "factor:a".to_string(),
            trade_date: 20240102,
            observations: 10,
            mean: Some(0.0),
            std: Some(1.0),
            min: Some(-1.0),
            p25: Some(-0.5),
            median: Some(0.0),
            p75: Some(0.5),
            max: Some(1.0),
            coverage: 1.0,
            inf_rate: 0.0,
        }];
        let barra = vec![BarraExposureRecord {
            factor_id: "factor:a".to_string(),
            trade_date: Some(20240102),
            metric: "barra_ic_mean".to_string(),
            barra_factor: "SIZE".to_string(),
            selected_group: None,
            rank_ic_sign: None,
            value: Some(0.1),
            pair_count: Some(10),
        }];

        let written = write_backtest_outputs(
            &output_dir,
            &returns,
            &index_returns,
            &daily_ic,
            &factor_stats,
            &[],
            &[],
            &barra,
        )
        .unwrap();

        let factor_dir = output_dir.join("factor_a");
        assert!(factor_dir.join("returns.parquet").exists());
        assert!(factor_dir.join("index_group_returns.parquet").exists());
        assert!(factor_dir.join("ic.parquet").exists());
        assert!(factor_dir.join("factor_stats.parquet").exists());
        assert!(factor_dir.join("barra_exposure.parquet").exists());
        assert_eq!(written.len(), 5);
        assert!(!output_dir.join("returns").exists());
        assert!(!output_dir.join("ic").exists());
        assert!(!output_dir.join("factor_stats").exists());
        assert!(!output_dir.join("index_group_returns").exists());
        assert!(!output_dir.join("barra_exposure").exists());

        let _ = std::fs::remove_dir_all(output_dir);
    }
}

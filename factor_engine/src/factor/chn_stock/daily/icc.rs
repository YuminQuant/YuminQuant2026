use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::stock_daily_ops::{is_bj_stock, neutralize_size_sector};
use crate::factor::common::PanelColumn;
use crate::factor::Factor;

pub struct StockDailyIcc;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyIcc)
}

impl Factor for StockDailyIcc {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "icc".into(),
            aliases: vec!["ICC".into(), "Implied Cost of Capital".into()],
            name: "Implied Cost of Capital".into(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: "0.2.0".into(),
            tags: ["DFZQ", "analyst", "consensus", "valuation", "daily", "neutralize", "barra", "size", "sector"]
                .into_iter().map(str::to_string).collect(),
            description: "Zero-dividend Easton raw ICC: sqrt((con_eps_fy2-con_eps_fy1)/close). EPS and unadjusted close are yuan/share; raw is a decimal rate. Uses the consensus dataset's PIT fiscal-year anchor. Requires finite EPS, strictly positive EPS increase and close; negative EPS levels are allowed. Stock universe panel, excludes BJ. Final output is the daily SW L1 industry and Barra SIZE regression residual, which may be negative; missing SIZE/industry stays null. No rank transform, winsorization or final zscore.".into(),
            dependencies: vec![
                DataRequest::new(DatasetId::StockConsensus, &["con_eps_fy1", "con_eps_fy2"]),
                DataRequest::new(DatasetId::StockDailyPv, &["close"]),
                DataRequest::new(DatasetId::StockBarraDaily, &["SIZE"]),
                DataRequest::new(DatasetId::StockSwClassification, &["l1_code"]),
            ],
            intraday_raw_dependencies: vec![],
            lookback: Lookback { trading_days: 0 },
        }
    }

    fn compute(&self, _: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let raw = raw_icc(data)?;
        Ok(
            neutralize_size_sector(&raw, data.stock_universe_panel()?, data)?
                .to_factor_series(self.spec()),
        )
    }
}

fn raw_icc(data: &DataPool) -> Result<PanelColumn> {
    let panel = data.stock_universe_panel()?;
    let consensus = data.daily(DatasetId::StockConsensus)?;
    let eps1 = panel.column_from_table(consensus, "con_eps_fy1")?;
    let eps2 = panel.column_from_table(consensus, "con_eps_fy2")?;
    let close = panel.column_from_table(data.daily(DatasetId::StockDailyPv)?, "close")?;
    let n = panel.instruments().len();
    let mut values = vec![None; panel.shape_len()];
    for (day, &date) in panel.dates().iter().enumerate() {
        if !panel.is_target_date(date) {
            continue;
        }
        for (i, code) in panel.instruments().iter().enumerate() {
            let offset = day * n + i;
            if panel.is_present_offset(offset) && !is_bj_stock(code) {
                values[offset] = icc(
                    eps1.values()[offset],
                    eps2.values()[offset],
                    close.values()[offset],
                );
            }
        }
    }
    panel.column_from_values(values)
}

fn icc(eps1: Option<f64>, eps2: Option<f64>, close: Option<f64>) -> Option<f64> {
    let eps1 = eps1.filter(|v| v.is_finite())?;
    let eps2 = eps2.filter(|v| v.is_finite())?;
    let close = close.filter(|v| v.is_finite() && *v > 0.0)?;
    let increase = eps2 - eps1;
    if !increase.is_finite() || increase <= 0.0 {
        return None;
    }
    let result = (increase / close).sqrt();
    (result.is_finite() && result > 0.0).then_some(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::FactorRowKey;
    use crate::data::{ColumnData, Table};
    use std::collections::{BTreeMap, HashMap};

    #[test]
    fn icc_formula_and_invalid_inputs() {
        assert_eq!(icc(Some(1.0), Some(2.0), Some(25.0)), Some(0.2));
        assert_eq!(icc(Some(-2.0), Some(-1.0), Some(25.0)), Some(0.2));
        for invalid in [None, Some(f64::NAN), Some(f64::INFINITY)] {
            assert_eq!(icc(invalid, Some(2.0), Some(25.0)), None);
            assert_eq!(icc(Some(1.0), invalid, Some(25.0)), None);
            assert_eq!(icc(Some(1.0), Some(2.0), invalid), None);
        }
        for price in [0.0, -1.0] {
            assert_eq!(icc(Some(1.0), Some(2.0), Some(price)), None);
        }
        assert_eq!(icc(Some(2.0), Some(2.0), Some(25.0)), None);
        assert_eq!(icc(Some(2.0), Some(1.0), Some(25.0)), None);
        assert_eq!(icc(Some(-f64::MAX), Some(f64::MAX), Some(25.0)), None);
        assert_eq!(
            icc(Some(0.0), Some(f64::MAX), Some(f64::MIN_POSITIVE)),
            None
        );
    }

    #[test]
    fn icc_metadata() {
        let spec = StockDailyIcc.spec();
        assert_eq!(spec.id, "icc");
        assert!(spec.tags.contains(&"DFZQ".into()));
        assert!(spec.tags.contains(&"analyst".into()));
        assert!(!spec.tags.contains(&"fundamental".into()));
        assert_eq!(spec.dependencies.len(), 4);
        assert!(spec.tags.contains(&"neutralize".into()));
        assert!(spec
            .dependencies
            .iter()
            .any(|d| d.dataset == DatasetId::StockBarraDaily));
        assert!(spec
            .dependencies
            .iter()
            .any(|d| d.dataset == DatasetId::StockSwClassification));
        assert_eq!(spec.dependencies[0].columns, ["con_eps_fy1", "con_eps_fy2"]);
        assert_eq!(spec.lookback.trading_days, 0);
    }

    #[test]
    #[ignore = "reads local market data; never writes factor files"]
    fn icc_local_single_day_load_and_compute() {
        use crate::config::EngineConfig;
        use crate::data::{DataCatalog, MarketDataLoader};
        let config = EngineConfig::discover(None).unwrap();
        let loader = MarketDataLoader::new(DataCatalog::new(config.data_root));
        let context = FactorContext {
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            start_date: 20260424,
            end_date: 20260424,
            load_start_date: 20260424,
            load_dates: vec![20260424],
            target_dates: vec![20260424],
        };
        let requests = StockDailyIcc.spec().dependencies;
        let pool = DataPool::load(&loader, &requests, &context).unwrap();
        let view = pool.view_for_requests(&requests, &context);
        let result = StockDailyIcc.compute(&context, &view).unwrap();
        let valid = result.values.iter().filter(|v| v.value.is_some()).count();
        println!("ICC 20260424: rows={}, valid={valid}", result.values.len());
        assert!(valid > 0);
        assert!(result.values.iter().all(|v| v.key.trade_date() == 20260424));
    }

    #[test]
    fn icc_final_residual_removes_size_and_sector_and_masks_missing() {
        let context = FactorContext {
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            start_date: 20260424,
            end_date: 20260424,
            load_start_date: 20260424,
            load_dates: vec![20260424],
            target_dates: vec![20260424],
        };
        let codes = || ColumnData::Utf8((1..=10).map(|i| Some(format!("{i:06}.SZ"))).collect());
        let dates = || ColumnData::I32(vec![Some(20260424); 10]);
        let basic = Table::new(BTreeMap::from([
            ("ts_code".into(), codes()),
            (
                "list_date".into(),
                ColumnData::I32(vec![Some(20100101); 10]),
            ),
            ("delist_date".into(), ColumnData::I32(vec![None; 10])),
        ]))
        .unwrap();
        let consensus = Table::new(BTreeMap::from([
            ("ts_code".into(), codes()),
            ("trade_date".into(), dates()),
            ("con_eps_fy1".into(), ColumnData::F64(vec![Some(1.0); 10])),
            (
                "con_eps_fy2".into(),
                ColumnData::F64(
                    (0..10)
                        .map(|i| {
                            let raw = 0.1
                                + 0.01 * i as f64
                                + 0.02 * (i / 4) as f64
                                + 0.003 * (i as f64).sin();
                            Some(1.0 + 25.0 * raw * raw)
                        })
                        .collect(),
                ),
            ),
        ]))
        .unwrap();
        let pv = Table::new(BTreeMap::from([
            ("ts_code".into(), codes()),
            ("trade_date".into(), dates()),
            ("close".into(), ColumnData::F64(vec![Some(25.0); 10])),
        ]))
        .unwrap();
        let size = Table::new(BTreeMap::from([
            ("ts_code".into(), codes()),
            ("trade_date".into(), dates()),
            (
                "SIZE".into(),
                ColumnData::F64((0..10).map(|i| (i != 8).then_some(i as f64)).collect()),
            ),
        ]))
        .unwrap()
        .take(&(0..10).rev().collect::<Vec<_>>())
        .unwrap();
        let sector = Table::new(BTreeMap::from([
            ("ts_code".into(), codes()),
            ("in_date".into(), ColumnData::I32(vec![Some(20100101); 10])),
            ("out_date".into(), ColumnData::I32(vec![None; 10])),
            (
                "l1_code".into(),
                ColumnData::Utf8(
                    (0..10)
                        .map(|i| (i != 9).then(|| format!("{}", i / 4)))
                        .collect(),
                ),
            ),
        ]))
        .unwrap();
        let data = DataPool::from_daily_tables_for_test(
            HashMap::from([
                (DatasetId::StockBasic, basic),
                (DatasetId::StockConsensus, consensus),
                (DatasetId::StockDailyPv, pv),
                (DatasetId::StockBarraDaily, size),
                (DatasetId::StockSwClassification, sector),
            ]),
            &context,
        )
        .unwrap();
        let result = StockDailyIcc.compute(&context, &data).unwrap();
        let residuals: Vec<_> = result.values.iter().map(|v| v.value).collect();
        assert_eq!(residuals.len(), 10);
        assert_eq!(&residuals[8..], &[None, None]);
        assert!(residuals[..8].iter().all(Option::is_some));
        assert!(residuals[..8].iter().flatten().any(|r| *r < 0.0));
        for group in residuals[..8].chunks(4) {
            assert!(group.iter().flatten().sum::<f64>().abs() < 1e-8);
        }
        assert!(
            residuals[..8]
                .iter()
                .enumerate()
                .map(|(i, r)| i as f64 * r.unwrap())
                .sum::<f64>()
                .abs()
                < 1e-8
        );
    }

    #[test]
    fn icc_aligns_dates_codes_and_preserves_missing_stocks() {
        let context = FactorContext {
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            start_date: 20250507,
            end_date: 20250507,
            load_start_date: 20250506,
            load_dates: vec![20250506, 20250507],
            target_dates: vec![20250507],
        };
        let strings = |s: &[&str]| ColumnData::Utf8(s.iter().map(|s| Some((*s).into())).collect());
        let basic = Table::new(BTreeMap::from([
            (
                "ts_code".into(),
                strings(&["000003.SZ", "430001.BJ", "000001.SZ", "000002.SZ"]),
            ),
            ("list_date".into(), ColumnData::I32(vec![Some(20100101); 4])),
            ("delist_date".into(), ColumnData::I32(vec![None; 4])),
        ]))
        .unwrap();
        let consensus = Table::new(BTreeMap::from([
            (
                "ts_code".into(),
                strings(&[
                    "000001.SZ",
                    "000003.SZ",
                    "000002.SZ",
                    "430001.BJ",
                    "000001.SZ",
                    "A00001.SZ",
                ]),
            ),
            (
                "trade_date".into(),
                ColumnData::I32(vec![
                    Some(20250506),
                    Some(20250507),
                    Some(20250507),
                    Some(20250507),
                    Some(20250507),
                    Some(20250507),
                ]),
            ),
            ("con_eps_fy1".into(), ColumnData::F64(vec![Some(1.0); 6])),
            (
                "con_eps_fy2".into(),
                ColumnData::F64(vec![
                    Some(10.0),
                    Some(2.0),
                    None,
                    Some(2.0),
                    Some(2.0),
                    Some(2.0),
                ]),
            ),
        ]))
        .unwrap();
        let pv = Table::new(BTreeMap::from([
            (
                "ts_code".into(),
                strings(&["430001.BJ", "000002.SZ", "000001.SZ", "000001.SZ"]),
            ),
            (
                "trade_date".into(),
                ColumnData::I32(vec![
                    Some(20250507),
                    Some(20250507),
                    Some(20250507),
                    Some(20250506),
                ]),
            ),
            ("close".into(), ColumnData::F64(vec![Some(25.0); 4])),
        ]))
        .unwrap();
        let data = DataPool::from_daily_tables_for_test(
            HashMap::from([
                (DatasetId::StockBasic, basic),
                (DatasetId::StockConsensus, consensus),
                (DatasetId::StockDailyPv, pv),
            ]),
            &context,
        )
        .unwrap();
        let result = raw_icc(&data)
            .unwrap()
            .to_factor_series(StockDailyIcc.spec());
        assert_eq!(result.values.len(), 4);
        for value in result.values {
            let FactorRowKey::Daily {
                trade_date,
                ts_code,
            } = value.key
            else {
                panic!("daily key expected")
            };
            assert_eq!(trade_date, 20250507);
            assert_eq!(
                value.value,
                if ts_code == "000001.SZ" {
                    Some(0.2)
                } else {
                    None
                }
            );
        }
    }
}

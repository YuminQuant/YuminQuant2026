use std::any::Any;
use std::collections::BTreeMap;

use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::vector::clean;
use crate::factor::common::{
    cached_financial_stock_snapshots_for_date, ClassificationLevel, ClassificationMap, DailyPanel,
    FinancialEventMarkerBuilder, FinancialEventSchedule, FinancialStatementDataset,
    InstrumentAlignedSnapshotCache, PanelColumn, ReportTypePreference,
};
use crate::factor::{Factor, FactorUpdatePolicy};
use crate::operators::{cs_zscore, ts_sum};

pub const MAINBZ_MOM_RESVOL_ID: &str = "mainbz_mom_resvol";
pub const PROVIDER_KEY: &str = "stock|daily|mainbz_mom_resvol";
const MOM_WINDOW: usize = 20;
const MOM_MIN_PERIODS: usize = 10;
const RESVOL_WINDOW: usize = 20;
const RESVOL_MIN_PERIODS: usize = 10;
const OTHER_MAX_RATIO: f64 = 0.30;
const PURE_BUSINESS_RATIO: f64 = 0.50;

#[derive(Clone, Debug)]
struct BusinessSnapshot {
    ratios: Vec<(String, f64)>,
}

#[derive(Default)]
pub struct MainbzMomResvolState {
    snapshot_cache: InstrumentAlignedSnapshotCache<BusinessSnapshot>,
    current_instruments: Vec<String>,
    current_snapshots: Vec<Option<BusinessSnapshot>>,
    last_processed_trade_date: Option<i32>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct StockDailyMainbzMomResvol;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyMainbzMomResvol)
}

pub fn spec_for() -> FactorSpec {
    let dependencies = vec![
        DataRequest::new(DatasetId::StockDailyPv, &["close", "pre_close"]),
        DataRequest::new(DatasetId::StockDailyBasic, &["circ_mv"]),
        DataRequest::financial_quarters(DatasetId::StockBalanceSheet, &["total_assets"], 8),
        DataRequest::financial_quarters(
            DatasetId::StockMainBusiness,
            &["bz_type", "bz_item", "bz_sales", "update_flag"],
            8,
        ),
        DataRequest::new(DatasetId::StockBarraDaily, &["SIZE"]),
        DataRequest::new(DatasetId::StockSwClassification, &["l1_code"]),
    ];
    FactorSpec {
        id: MAINBZ_MOM_RESVOL_ID.to_string(),
        aliases: vec!["product_mom_resvol".to_string()],
        name: MAINBZ_MOM_RESVOL_ID.to_string(),
        asset_class: AssetClass::Stock,
        frequency: Frequency::Daily,
        version: "0.1.0".to_string(),
        tags: vec![
            "DBZQ",
            "financial",
            "fundamental",
            "mainbz",
            "business_momentum",
            "residual_volatility",
            "neutralize",
            "barra",
            "size",
            "sector",
            "daily",
        ]
        .into_iter()
        .map(|tag| tag.to_string())
        .collect(),
        description: "DBZQ main business composite factor. It uses positive-sales fina_mainbz type=I update_flag=0 rows gated by balance-sheet PIT announcements, builds business item returns, combines 20-day business excess momentum with residual volatility, and neutralizes SIZE plus SW sector.".to_string(),
        dependencies,
        intraday_raw_dependencies: Vec::new(),
        lookback: Lookback {
            trading_days: RESVOL_WINDOW - 1,
        },
    }
}

impl Factor for StockDailyMainbzMomResvol {
    fn spec(&self) -> FactorSpec {
        spec_for()
    }

    fn compute_provider_key(&self) -> String {
        PROVIDER_KEY.to_string()
    }

    fn update_policy(&self) -> FactorUpdatePolicy {
        FactorUpdatePolicy::FinancialEventStateDailyFast
    }

    fn initial_compute_state(&self, _requested_ids: &[String]) -> Box<dyn Any + Send> {
        Box::<MainbzMomResvolState>::default()
    }

    fn compute(&self, context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let mut state = MainbzMomResvolState::default();
        self.compute_with_state(context, data, &mut state)
    }

    fn compute_many_stateful(
        &self,
        requested_ids: &[String],
        context: &FactorContext,
        data: &DataPool,
        state: &mut (dyn Any + Send),
    ) -> Result<Vec<FactorSeries>> {
        if !requested_ids.iter().any(|id| id == MAINBZ_MOM_RESVOL_ID) {
            return Ok(Vec::new());
        }
        let state = state
            .downcast_mut::<MainbzMomResvolState>()
            .expect("mainbz_mom_resvol state type");
        self.compute_requested_with_state(requested_ids, context, data, state)
    }
}

impl StockDailyMainbzMomResvol {
    fn compute_with_state(
        &self,
        context: &FactorContext,
        data: &DataPool,
        state: &mut MainbzMomResvolState,
    ) -> Result<FactorSeries> {
        let requested = [self.spec().id];
        self.compute_requested_with_state(&requested, context, data, state)
            .map(|mut series| series.remove(0))
    }

    fn compute_requested_with_state(
        &self,
        requested_ids: &[String],
        _context: &FactorContext,
        data: &DataPool,
        state: &mut MainbzMomResvolState,
    ) -> Result<Vec<FactorSeries>> {
        let want_composite = requested_ids.iter().any(|id| id == MAINBZ_MOM_RESVOL_ID);
        if !want_composite {
            return Ok(Vec::new());
        }
        let panel = data.daily_panel(DatasetId::StockDailyPv)?;
        let balance = data.financial_reader(
            DatasetId::StockBalanceSheet,
            ReportTypePreference::balance_sheet_consolidated(),
        )?;
        let mainbz = data.main_business_reader()?;
        let schedule = FinancialEventSchedule::from_pit_readers(std::slice::from_ref(&balance));

        let close = panel.column_from_table(data.daily(DatasetId::StockDailyPv)?, "close")?;
        let pre_close =
            panel.column_from_table(data.daily(DatasetId::StockDailyPv)?, "pre_close")?;
        let circ_mv =
            panel.column_from_table(data.daily(DatasetId::StockDailyBasic)?, "circ_mv")?;
        let stock_ret = close.zip_binary(&pre_close, simple_return)?;

        let (product_abs_mom, product_mom_daily) = business_weighted_return_panels(
            panel, &stock_ret, &circ_mv, &balance, &mainbz, &schedule, state,
        )?;
        let product_mom_20d =
            product_mom_daily.ts(|values| ts_sum(values, MOM_WINDOW, MOM_MIN_PERIODS))?;
        let product_resvol = stock_ret.ts_binary(&product_abs_mom, residual_std_rolling)?;

        let product_mom_20d = mask_non_standard_sh_sz(&product_mom_20d, panel)?;
        let product_resvol = mask_non_standard_sh_sz(&product_resvol, panel)?;
        let z_mom = product_mom_20d.cs(cs_zscore)?;
        let z_resvol = product_resvol.cs(cs_zscore)?;
        let composite =
            z_mom.zip_binary(&z_resvol, |mom, resvol| match (clean(mom), clean(resvol)) {
                (Some(mom), Some(resvol)) => Some(0.5 * mom - 0.5 * resvol),
                _ => None,
            })?;

        let barra = data.daily(DatasetId::StockBarraDaily)?;
        let size = panel.column_from_table(barra, "SIZE")?;
        let sector_map = ClassificationMap::from_table(
            data.daily(DatasetId::StockSwClassification)?,
            ClassificationLevel::Sector,
        )?;
        let neutralized = composite.cs_neutralize_regression_by_group(
            &[&size],
            None,
            |trade_date, ts_codes| sector_map.groups_for(trade_date, ts_codes),
        )?;
        let neutralized = mask_non_standard_sh_sz(&neutralized, panel)?;
        Ok(vec![neutralized.to_factor_series(spec_for())])
    }
}

fn is_standard_sh_sz_stock(ts_code: &str) -> bool {
    let bytes = ts_code.as_bytes();
    bytes.len() == 9
        && bytes[..6].iter().all(u8::is_ascii_digit)
        && bytes[6] == b'.'
        && matches!(&bytes[7..], b"SH" | b"SZ")
}

fn mask_non_standard_sh_sz(column: &PanelColumn, panel: &DailyPanel) -> Result<PanelColumn> {
    let instrument_count = panel.instruments().len();
    let mut values = column.values().to_vec();
    for date_idx in 0..panel.dates().len() {
        let date_offset = date_idx * instrument_count;
        for (instrument_idx, ts_code) in panel.instruments().iter().enumerate() {
            if !is_standard_sh_sz_stock(ts_code) {
                values[date_offset + instrument_idx] = None;
            }
        }
    }
    panel.column_from_values(values)
}

fn business_weighted_return_panels(
    panel: &DailyPanel,
    stock_ret: &PanelColumn,
    circ_mv: &PanelColumn,
    balance: &crate::factor::common::FinancialPitReader<'_>,
    mainbz: &crate::factor::common::MainBusinessReader<'_>,
    schedule: &FinancialEventSchedule,
    state: &mut MainbzMomResvolState,
) -> Result<(PanelColumn, PanelColumn)> {
    let instrument_count = panel.instruments().len();
    let mut abs_output = vec![None; panel.shape_len()];
    let mut excess_output = vec![None; panel.shape_len()];
    let instruments_changed = state.current_instruments.as_slice() != panel.instruments();
    if instruments_changed {
        state.current_instruments = panel.instruments().to_vec();
        state.current_snapshots.clear();
        state.last_processed_trade_date = None;
    }

    for (date_idx, trade_date) in panel.dates().iter().copied().enumerate() {
        let should_update = state.current_snapshots.len() != instrument_count
            || state.last_processed_trade_date.is_none()
            || schedule.has_event_after_until(state.last_processed_trade_date, trade_date);
        if should_update {
            state.current_snapshots = update_business_snapshots_for_date(
                panel,
                trade_date,
                balance,
                mainbz,
                &mut state.snapshot_cache,
            );
        }
        let date_offset = date_idx * instrument_count;
        let product_returns = product_returns_for_date(
            panel,
            date_offset,
            &state.current_snapshots,
            stock_ret.values(),
            circ_mv.values(),
        );
        for instrument_idx in 0..instrument_count {
            let offset = date_offset + instrument_idx;
            if !panel.is_present_offset(offset) {
                continue;
            }
            if !is_standard_sh_sz_stock(&panel.instruments()[instrument_idx]) {
                continue;
            }
            let Some(snapshot) = state
                .current_snapshots
                .get(instrument_idx)
                .and_then(|value| value.as_ref())
            else {
                continue;
            };
            let Some(ret) = clean(stock_ret.values()[offset]) else {
                continue;
            };
            let Some((abs_value, excess_value)) =
                business_weighted_values(snapshot, &product_returns, ret)
            else {
                continue;
            };
            abs_output[offset] = Some(abs_value);
            excess_output[offset] = Some(excess_value);
        }
        state.last_processed_trade_date = Some(trade_date);
    }
    Ok((
        panel.column_from_values(abs_output)?,
        panel.column_from_values(excess_output)?,
    ))
}

fn update_business_snapshots_for_date(
    panel: &DailyPanel,
    trade_date: i32,
    balance: &crate::factor::common::FinancialPitReader<'_>,
    mainbz: &crate::factor::common::MainBusinessReader<'_>,
    cache: &mut InstrumentAlignedSnapshotCache<BusinessSnapshot>,
) -> Vec<Option<BusinessSnapshot>> {
    cached_financial_stock_snapshots_for_date(
        panel,
        trade_date,
        cache,
        |_, ts_code, offset| !is_standard_sh_sz_stock(ts_code) || !panel.is_present_offset(offset),
        |trade_date, ts_code, _| {
            let balance_end_date = balance.latest_quarter_end_date(ts_code, trade_date)?;
            let mainbz_end_date =
                mainbz.latest_industry_update0_end_date(ts_code, balance_end_date);
            let mut marker = FinancialEventMarkerBuilder::new();
            marker.include_reader_record_for_end_date(
                FinancialStatementDataset::BalanceSheet,
                balance,
                ts_code,
                trade_date,
                balance_end_date,
            );
            if let Some(mainbz_end_date) = mainbz_end_date {
                marker.include_main_business_end_date(mainbz, ts_code, mainbz_end_date);
            }
            marker.build()
        },
        |trade_date, ts_code, _| {
            let balance_end_date = balance.latest_quarter_end_date(ts_code, trade_date)?;
            let mainbz_end_date =
                mainbz.latest_industry_update0_end_date(ts_code, balance_end_date)?;
            business_snapshot(mainbz, ts_code, mainbz_end_date)
        },
    )
}

fn business_snapshot(
    mainbz: &crate::factor::common::MainBusinessReader<'_>,
    ts_code: &str,
    end_date: i32,
) -> Option<BusinessSnapshot> {
    let records = mainbz.industry_update0_records(ts_code, end_date);
    if records.is_empty() {
        return None;
    }
    let mut by_item = BTreeMap::<String, f64>::new();
    for record in records {
        let Some(item) = record.bz_item().map(str::trim) else {
            continue;
        };
        if item.is_empty() {
            continue;
        }
        let Some(sales) = clean(record.bz_sales()).filter(|value| *value > 0.0) else {
            continue;
        };
        *by_item.entry(item.to_string()).or_default() += sales;
    }
    business_snapshot_from_item_sales(&by_item)
}

fn business_snapshot_from_item_sales(by_item: &BTreeMap<String, f64>) -> Option<BusinessSnapshot> {
    let positive_items = by_item
        .iter()
        .filter(|(_, sales)| sales.is_finite() && **sales > 0.0)
        .collect::<Vec<_>>();
    let total: f64 = positive_items.iter().map(|(_, sales)| **sales).sum();
    if !total.is_finite() || total <= 0.0 {
        return None;
    }
    let other_sales: f64 = by_item
        .iter()
        .filter(|(_, sales)| sales.is_finite() && **sales > 0.0)
        .filter(|(item, _)| is_other_or_internal(item))
        .map(|(_, sales)| *sales)
        .sum();
    if other_sales / total > OTHER_MAX_RATIO {
        return None;
    }
    let clean_total: f64 = by_item
        .iter()
        .filter(|(_, sales)| sales.is_finite() && **sales > 0.0)
        .filter(|(item, _)| !is_other_or_internal(item))
        .map(|(_, sales)| *sales)
        .sum();
    if !clean_total.is_finite() || clean_total <= 0.0 {
        return None;
    }
    let ratios = by_item
        .iter()
        .filter(|(_, sales)| sales.is_finite() && **sales > 0.0)
        .filter(|(item, _)| !is_other_or_internal(item))
        .filter_map(|(item, sales)| {
            let ratio = *sales / clean_total;
            (ratio.is_finite() && ratio > 0.0).then_some((item.clone(), ratio))
        })
        .collect::<Vec<_>>();
    (!ratios.is_empty()).then_some(BusinessSnapshot { ratios })
}

fn business_weighted_values(
    snapshot: &BusinessSnapshot,
    product_returns: &BTreeMap<String, f64>,
    stock_ret: f64,
) -> Option<(f64, f64)> {
    let available_weight = snapshot
        .ratios
        .iter()
        .filter(|(item, _)| product_returns.contains_key(item))
        .map(|(_, ratio)| *ratio)
        .sum::<f64>();
    if !available_weight.is_finite() || available_weight <= 0.0 {
        return None;
    }
    let mut abs_value = 0.0;
    let mut excess_value = 0.0;
    for (item, ratio) in &snapshot.ratios {
        let Some(product_ret) = product_returns.get(item).copied() else {
            continue;
        };
        let normalized_ratio = ratio / available_weight;
        abs_value += normalized_ratio * product_ret;
        excess_value += normalized_ratio * (product_ret - stock_ret);
    }
    (abs_value.is_finite() && excess_value.is_finite()).then_some((abs_value, excess_value))
}

fn is_other_or_internal(item: &str) -> bool {
    item.contains('\u{5176}') && item.contains('\u{4ed6}')
        || item.contains('\u{5185}')
            && item.contains('\u{90e8}')
            && item.contains('\u{62b5}')
            && item.contains('\u{6d88}')
}

fn product_returns_for_date(
    panel: &DailyPanel,
    date_offset: usize,
    snapshots: &[Option<BusinessSnapshot>],
    stock_ret: &[Option<f64>],
    circ_mv: &[Option<f64>],
) -> BTreeMap<String, f64> {
    let mut sums = BTreeMap::<String, (f64, f64)>::new();
    for (instrument_idx, snapshot) in snapshots.iter().enumerate() {
        let offset = date_offset + instrument_idx;
        if !panel.is_present_offset(offset)
            || !is_standard_sh_sz_stock(&panel.instruments()[instrument_idx])
        {
            continue;
        }
        let (Some(ret), Some(weight), Some(snapshot)) = (
            clean(stock_ret[offset]),
            clean(circ_mv[offset]).filter(|value| *value > 0.0),
            snapshot.as_ref(),
        ) else {
            continue;
        };
        for (item, ratio) in &snapshot.ratios {
            if *ratio > PURE_BUSINESS_RATIO {
                let entry = sums.entry(item.clone()).or_default();
                entry.0 += weight * ret;
                entry.1 += weight;
            }
        }
    }
    sums.into_iter()
        .filter_map(|(item, (numerator, denominator))| {
            (denominator > 0.0).then_some((item, numerator / denominator))
        })
        .collect()
}

fn residual_std_rolling(y: &[Option<f64>], x: &[Option<f64>]) -> Vec<Option<f64>> {
    let mut output = vec![None; y.len()];
    for end in 0..y.len() {
        let start = (end + 1).saturating_sub(RESVOL_WINDOW);
        let pairs = (start..=end)
            .filter_map(|idx| match (clean(y[idx]), clean(x[idx])) {
                (Some(y), Some(x)) => Some((y, x)),
                _ => None,
            })
            .collect::<Vec<_>>();
        if pairs.len() < RESVOL_MIN_PERIODS {
            continue;
        }
        output[end] = regression_residual_sample_std(&pairs);
    }
    output
}

fn regression_residual_sample_std(pairs: &[(f64, f64)]) -> Option<f64> {
    if pairs.len() < 2 {
        return None;
    }
    let n = pairs.len() as f64;
    let mean_y = pairs.iter().map(|(y, _)| *y).sum::<f64>() / n;
    let mean_x = pairs.iter().map(|(_, x)| *x).sum::<f64>() / n;
    let variance_x = pairs
        .iter()
        .map(|(_, x)| {
            let diff = *x - mean_x;
            diff * diff
        })
        .sum::<f64>();
    if variance_x <= f64::EPSILON {
        return None;
    }
    let covariance = pairs
        .iter()
        .map(|(y, x)| (*y - mean_y) * (*x - mean_x))
        .sum::<f64>();
    let beta = covariance / variance_x;
    let alpha = mean_y - beta * mean_x;
    let rss = pairs
        .iter()
        .map(|(y, x)| {
            let residual = *y - alpha - beta * *x;
            residual * residual
        })
        .sum::<f64>();
    let std = (rss / (pairs.len() as f64 - 1.0)).sqrt();
    std.is_finite().then_some(std)
}

fn simple_return(close: Option<f64>, pre_close: Option<f64>) -> Option<f64> {
    match (clean(close), clean(pre_close).filter(|value| *value > 0.0)) {
        (Some(close), Some(pre_close)) => Some(close / pre_close - 1.0),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{AssetClass, FactorContext, Frequency};
    use crate::data::{ColumnData, Table};

    fn assert_close(actual: Option<f64>, expected: f64) {
        let actual = actual.expect("some value");
        assert!(
            (actual - expected).abs() < 1e-10,
            "expected {expected}, got {actual}"
        );
    }

    fn test_context(target_dates: Vec<i32>) -> FactorContext {
        let start_date = *target_dates.first().unwrap();
        let end_date = *target_dates.last().unwrap();
        FactorContext {
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            start_date,
            end_date,
            load_start_date: start_date,
            load_dates: target_dates.clone(),
            target_dates,
        }
    }

    fn panel_for_one_date() -> DailyPanel {
        let table = Table::new(BTreeMap::from([
            (
                "trade_date".to_string(),
                ColumnData::I32(vec![Some(20260105), Some(20260105)]),
            ),
            (
                "ts_code".to_string(),
                ColumnData::Utf8(vec![
                    Some("000001.SZ".to_string()),
                    Some("000002.SZ".to_string()),
                ]),
            ),
            (
                "close".to_string(),
                ColumnData::F64(vec![Some(1.0), Some(1.0)]),
            ),
        ]))
        .expect("table");
        DailyPanel::from_table(&table, &test_context(vec![20260105])).expect("panel")
    }

    #[test]
    fn business_snapshot_filters_other_and_internal_then_reweights() {
        let by_item = BTreeMap::from([
            ("bank".to_string(), 60.0),
            ("insurance".to_string(), 20.0),
            ("zero".to_string(), 0.0),
            ("negative".to_string(), -10.0),
            ("\u{5176}\u{4ed6}".to_string(), 10.0),
        ]);
        let snapshot = business_snapshot_from_item_sales(&by_item).expect("snapshot");
        assert_eq!(snapshot.ratios.len(), 2);
        assert!((snapshot.ratios[0].1 + snapshot.ratios[1].1 - 1.0).abs() < 1e-12);
    }

    #[test]
    fn business_snapshot_rejects_large_other_share() {
        let by_item = BTreeMap::from([
            ("bank".to_string(), 60.0),
            ("\u{5176}\u{4ed6}".to_string(), 40.1),
        ]);
        assert!(business_snapshot_from_item_sales(&by_item).is_none());
    }

    #[test]
    fn standard_sh_sz_filter_rejects_non_six_digit_codes() {
        assert!(is_standard_sh_sz_stock("000001.SZ"));
        assert!(is_standard_sh_sz_stock("600000.SH"));
        assert!(!is_standard_sh_sz_stock("A26018.SZ"));
        assert!(!is_standard_sh_sz_stock("000001.BJ"));
        assert!(!is_standard_sh_sz_stock("000001.sz"));
    }

    #[test]
    fn missing_product_return_reweights_available_business_items() {
        let panel = panel_for_one_date();
        let snapshots = vec![
            Some(BusinessSnapshot {
                ratios: vec![("A".to_string(), 0.6), ("B".to_string(), 0.4)],
            }),
            Some(BusinessSnapshot {
                ratios: vec![("A".to_string(), 1.0)],
            }),
        ];
        let stock_ret = vec![Some(0.10), Some(0.20)];
        let circ_mv = vec![Some(100.0), Some(100.0)];
        let product_returns = product_returns_for_date(&panel, 0, &snapshots, &stock_ret, &circ_mv);
        assert_close(product_returns.get("A").copied(), 0.15);
        assert_eq!(product_returns.get("B"), None);
        let (abs_value, excess_value) =
            business_weighted_values(snapshots[0].as_ref().unwrap(), &product_returns, 0.10)
                .expect("weighted values");
        assert!((abs_value - 0.15).abs() < 1e-10);
        assert!((excess_value - 0.05).abs() < 1e-10);
    }

    #[test]
    fn residual_std_requires_enough_valid_pairs_and_nonzero_x_variance() {
        let y = (0..20).map(|idx| Some(idx as f64)).collect::<Vec<_>>();
        let x = (0..20)
            .map(|idx| Some(idx as f64 * 2.0))
            .collect::<Vec<_>>();
        let std = residual_std_rolling(&y, &x);
        assert!(std[8].is_none());
        assert_close(std[19], 0.0);

        let flat_x = vec![Some(1.0); 20];
        assert!(residual_std_rolling(&y, &flat_x)[19].is_none());
    }
}

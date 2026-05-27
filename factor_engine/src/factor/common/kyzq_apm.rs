use std::collections::BTreeMap;

use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorRowKey, FactorSeries, FactorSpec,
    FactorValue, Frequency, IntradayDailyRawRequest, IntradayDailyRawSeries, IntradayDailyRawSpec,
    Lookback,
};
use crate::data::{DataPool, Table};
use crate::error::Result;
use crate::factor::common::stock_daily_ops::{
    adjusted_20d_return, mask_bj, neutralize_size_sector,
};
use crate::factor::common::{
    clean_intraday_value, intraday_time_in_range, stock_minute_raw_spec, RequestedRawIds,
};
use crate::factor::common::{DailyPanel, PanelColumn};
use crate::operators::cs_regression_residual;

pub const PROVIDER_KEY: &str = "kyzq_apm_provider";
pub const RAW_VERSION: &str = "0.2.0";
pub const VERSION: &str = "0.1.0";

pub const APM_AM_RET_RAW_ID: &str = "daily_kyzq_apm_am_ret_raw";
pub const APM_PM_RET_RAW_ID: &str = "daily_kyzq_apm_pm_ret_raw";

const APM_WINDOW: usize = 20;
const APM_RAW_WINDOW_DAYS: usize = 1;
const EPS: f64 = f64::EPSILON;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KyzqApmKind {
    Apm,
    ApmNew,
}

#[derive(Clone, Copy, Debug)]
pub struct KyzqApmFactorDef {
    pub id: &'static str,
    pub alias: &'static str,
    pub name: &'static str,
    pub kind: KyzqApmKind,
}

#[derive(Clone, Copy, Debug, Default)]
struct SessionReturns {
    am: Option<f64>,
    pm: Option<f64>,
}

pub fn raw_spec(raw_id: &str) -> IntradayDailyRawSpec {
    stock_minute_raw_spec(raw_id, RAW_VERSION, &["close"], APM_RAW_WINDOW_DAYS)
}

pub fn raw_specs() -> Vec<IntradayDailyRawSpec> {
    vec![raw_spec(APM_AM_RET_RAW_ID), raw_spec(APM_PM_RET_RAW_ID)]
}

pub fn raw_specs_for_kind(kind: KyzqApmKind) -> Vec<IntradayDailyRawSpec> {
    raw_ids_for_kind(kind)
        .iter()
        .map(|raw_id| raw_spec(raw_id))
        .collect()
}

pub fn factor_spec(def: KyzqApmFactorDef) -> FactorSpec {
    FactorSpec {
        id: def.id.to_string(),
        aliases: vec![def.alias.to_string()],
        name: def.name.to_string(),
        asset_class: AssetClass::Stock,
        frequency: Frequency::Daily,
        version: VERSION.to_string(),
        tags: tags(def.kind),
        description: description(def),
        dependencies: dependencies(def.kind),
        intraday_raw_dependencies: raw_ids_for_kind(def.kind)
            .iter()
            .map(|raw_id| IntradayDailyRawRequest::new(raw_id, APM_WINDOW - 1))
            .collect(),
        lookback: Lookback {
            trading_days: APM_WINDOW - 1,
        },
    }
}

pub fn compute_factor(def: KyzqApmFactorDef, data: &DataPool) -> Result<FactorSeries> {
    let factor = compute_apm(data, matches!(def.kind, KyzqApmKind::ApmNew))?;
    Ok(factor.to_factor_series(factor_spec(def)))
}

pub fn minute_compute_many(
    raw_ids: &[String],
    context: &FactorContext,
    data: &DataPool,
) -> Result<Vec<IntradayDailyRawSeries>> {
    let requested = RequestedRawIds::new(raw_ids, &raw_ids_for_provider());
    if requested.is_empty() {
        return Ok(Vec::new());
    }
    let need_am = requested.contains(APM_AM_RET_RAW_ID);
    let need_pm = requested.contains(APM_PM_RET_RAW_ID);
    let trade_date = *context
        .target_dates
        .first()
        .expect("raw materialization provides one or more target dates");
    let returns = match data.minute(DatasetId::StockMinute1m, trade_date) {
        Some(table) => session_returns_from_table(table, need_am, need_pm)?,
        None => BTreeMap::new(),
    };

    let mut am_values = Vec::new();
    let mut pm_values = Vec::new();
    for (ts_code, values) in returns {
        let key = FactorRowKey::Daily {
            trade_date,
            ts_code,
        };
        if need_am {
            am_values.push(FactorValue {
                key: key.clone(),
                value: values.am,
            });
        }
        if need_pm {
            pm_values.push(FactorValue {
                key,
                value: values.pm,
            });
        }
    }

    let mut output = Vec::new();
    if need_am {
        output.push(IntradayDailyRawSeries {
            spec: raw_spec(APM_AM_RET_RAW_ID),
            values: am_values,
        });
    }
    if need_pm {
        output.push(IntradayDailyRawSeries {
            spec: raw_spec(APM_PM_RET_RAW_ID),
            values: pm_values,
        });
    }
    Ok(output)
}

fn compute_apm(data: &DataPool, use_overnight: bool) -> Result<PanelColumn> {
    let panel = if use_overnight {
        data.intraday_daily_raw_panel(APM_PM_RET_RAW_ID)?
    } else {
        data.intraday_daily_raw_panel(APM_AM_RET_RAW_ID)?
    };
    let left = if use_overnight {
        overnight_return_column(panel, data)?
    } else {
        panel.column(APM_AM_RET_RAW_ID)?
    };
    let left_market = market_mean_column(panel, &left)?;
    let pm = panel.column(APM_PM_RET_RAW_ID)?;
    let pm_market = market_mean_column(panel, &pm)?;
    let stat = apm_stat_column(panel, &left, &left_market, &pm, &pm_market)?;
    let ret20 = adjusted_20d_return(data, panel)?;
    let deret20 = stat.cs_binary(&ret20, cs_regression_residual)?;
    neutralize_size_sector(&deret20, panel, data)
}

fn overnight_return_column(panel: &DailyPanel, data: &DataPool) -> Result<PanelColumn> {
    let pv = data.daily(DatasetId::StockDailyPv)?;
    let open = panel.column_from_table(pv, "open")?;
    let pre_close = panel.column_from_table(pv, "pre_close")?;
    let overnight = open.zip_binary(&pre_close, overnight_return)?;
    mask_bj(&overnight, panel)
}

fn session_returns_from_table(
    table: &Table,
    need_am: bool,
    need_pm: bool,
) -> Result<BTreeMap<String, SessionReturns>> {
    let ts_codes = table.required_utf8("ts_code")?;
    let trade_times = table.required_utf8("trade_time")?;
    let close = table.required_f64_cast("close")?;

    let mut grouped = BTreeMap::<String, Vec<usize>>::new();
    for idx in 0..table.len {
        let Some(ts_code) = ts_codes[idx].clone() else {
            continue;
        };
        if !is_sh_sz_stock(&ts_code) || trade_times[idx].is_none() {
            continue;
        }
        grouped.entry(ts_code).or_default().push(idx);
    }

    let mut output = BTreeMap::new();
    for (ts_code, mut indices) in grouped {
        indices.sort_by(|left, right| trade_times[*left].cmp(&trade_times[*right]));
        output.insert(
            ts_code,
            SessionReturns {
                am: need_am
                    .then(|| session_return(&indices, &trade_times, &close, "09:30:00", "11:30:00"))
                    .flatten(),
                pm: need_pm
                    .then(|| session_return(&indices, &trade_times, &close, "13:00:00", "15:00:00"))
                    .flatten(),
            },
        );
    }
    Ok(output)
}

fn session_return(
    indices: &[usize],
    trade_times: &[Option<String>],
    close: &[Option<f64>],
    start: &str,
    end: &str,
) -> Option<f64> {
    let first = session_start_close(indices, trade_times, close, start, end)?;
    let last = session_end_close(indices, trade_times, close, start, end)?;
    if first <= EPS {
        return None;
    }
    finite_option(Some(last / first - 1.0))
}

fn session_start_close(
    indices: &[usize],
    trade_times: &[Option<String>],
    close: &[Option<f64>],
    start: &str,
    end: &str,
) -> Option<f64> {
    for idx in indices {
        let Some(trade_time) = trade_times[*idx].as_deref() else {
            continue;
        };
        if intraday_time_in_range(trade_time, start, end) {
            if let Some(value) = clean_positive(close[*idx]) {
                return Some(value);
            }
        }
    }
    None
}

fn session_end_close(
    indices: &[usize],
    trade_times: &[Option<String>],
    close: &[Option<f64>],
    start: &str,
    end: &str,
) -> Option<f64> {
    let mut output = None;
    for idx in indices {
        let Some(trade_time) = trade_times[*idx].as_deref() else {
            continue;
        };
        if intraday_time_in_range(trade_time, start, end) {
            if let Some(value) = clean_positive(close[*idx]) {
                output = Some(value);
            }
        }
    }
    output
}

fn market_mean_column(panel: &DailyPanel, values: &PanelColumn) -> Result<PanelColumn> {
    let instrument_count = panel.instruments().len();
    let date_count = panel.dates().len();
    let mut output = vec![None; panel.shape_len()];
    for date_idx in 0..date_count {
        let mut sum = 0.0;
        let mut count = 0usize;
        for instrument_idx in 0..instrument_count {
            let ts_code = &panel.instruments()[instrument_idx];
            if !is_sh_sz_stock(ts_code) {
                continue;
            }
            let offset = date_idx * instrument_count + instrument_idx;
            if let Some(value) = finite_option(values.values()[offset]) {
                sum += value;
                count += 1;
            }
        }
        if count == 0 {
            continue;
        }
        let mean = sum / count as f64;
        for instrument_idx in 0..instrument_count {
            if is_sh_sz_stock(&panel.instruments()[instrument_idx]) {
                output[date_idx * instrument_count + instrument_idx] = Some(mean);
            }
        }
    }
    panel.column_from_values(output)
}

fn apm_stat_column(
    panel: &DailyPanel,
    left: &PanelColumn,
    left_market: &PanelColumn,
    pm: &PanelColumn,
    pm_market: &PanelColumn,
) -> Result<PanelColumn> {
    let instrument_count = panel.instruments().len();
    let date_count = panel.dates().len();
    let mut output = vec![None; panel.shape_len()];
    for instrument_idx in 0..instrument_count {
        for date_idx in 0..date_count {
            let start = (date_idx + 1).saturating_sub(APM_WINDOW);
            let mut y = Vec::new();
            let mut x = Vec::new();
            for window_date_idx in start..=date_idx {
                let offset = window_date_idx * instrument_count + instrument_idx;
                if let (Some(y_value), Some(x_value)) = (
                    finite_option(left.values()[offset]),
                    finite_option(left_market.values()[offset]),
                ) {
                    y.push(y_value);
                    x.push(x_value);
                }
                if let (Some(y_value), Some(x_value)) = (
                    finite_option(pm.values()[offset]),
                    finite_option(pm_market.values()[offset]),
                ) {
                    y.push(y_value);
                    x.push(x_value);
                }
            }
            let Some((alpha, beta)) = ols_intercept_beta(&y, &x) else {
                continue;
            };

            let mut deltas = Vec::new();
            for window_date_idx in start..=date_idx {
                let offset = window_date_idx * instrument_count + instrument_idx;
                let left_residual = match (
                    finite_option(left.values()[offset]),
                    finite_option(left_market.values()[offset]),
                ) {
                    (Some(y_value), Some(x_value)) => Some(y_value - (alpha + beta * x_value)),
                    _ => None,
                };
                let pm_residual = match (
                    finite_option(pm.values()[offset]),
                    finite_option(pm_market.values()[offset]),
                ) {
                    (Some(y_value), Some(x_value)) => Some(y_value - (alpha + beta * x_value)),
                    _ => None,
                };
                if let (Some(left_residual), Some(pm_residual)) = (left_residual, pm_residual) {
                    deltas.push(left_residual - pm_residual);
                }
            }
            output[date_idx * instrument_count + instrument_idx] = t_stat(&deltas);
        }
    }
    panel.column_from_values(output)
}

fn ols_intercept_beta(y: &[f64], x: &[f64]) -> Option<(f64, f64)> {
    if y.len() != x.len() || y.len() < 2 {
        return None;
    }
    let mean_y = y.iter().sum::<f64>() / y.len() as f64;
    let mean_x = x.iter().sum::<f64>() / x.len() as f64;
    let var_x = x
        .iter()
        .map(|value| {
            let diff = value - mean_x;
            diff * diff
        })
        .sum::<f64>();
    let cov_xy = y
        .iter()
        .zip(x)
        .map(|(y_value, x_value)| (x_value - mean_x) * (y_value - mean_y))
        .sum::<f64>();
    let beta = if var_x.abs() <= EPS {
        0.0
    } else {
        cov_xy / var_x
    };
    let alpha = mean_y - beta * mean_x;
    finite_option(Some(alpha)).zip(finite_option(Some(beta)))
}

fn t_stat(values: &[f64]) -> Option<f64> {
    if values.len() < 2 {
        return None;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|value| {
            let diff = value - mean;
            diff * diff
        })
        .sum::<f64>()
        / values.len() as f64;
    let std_dev = variance.sqrt();
    if std_dev <= EPS || !std_dev.is_finite() {
        return None;
    }
    finite_option(Some(mean / (std_dev / (values.len() as f64).sqrt())))
}

fn raw_ids_for_provider() -> [&'static str; 2] {
    [APM_AM_RET_RAW_ID, APM_PM_RET_RAW_ID]
}

fn raw_ids_for_kind(kind: KyzqApmKind) -> &'static [&'static str] {
    match kind {
        KyzqApmKind::Apm => &[APM_AM_RET_RAW_ID, APM_PM_RET_RAW_ID],
        KyzqApmKind::ApmNew => &[APM_PM_RET_RAW_ID],
    }
}

fn dependencies(kind: KyzqApmKind) -> Vec<DataRequest> {
    let pv_columns = match kind {
        KyzqApmKind::Apm => vec!["close"],
        KyzqApmKind::ApmNew => vec!["open", "pre_close", "close"],
    };
    vec![
        DataRequest::new(DatasetId::StockBarraDaily, &["SIZE"]),
        DataRequest::new(DatasetId::StockSwClassification, &["l1_code"]),
        DataRequest::new(DatasetId::StockDailyPv, &pv_columns),
        DataRequest::new(DatasetId::StockAdjFactor, &["adj_factor"]),
    ]
}

fn tags(kind: KyzqApmKind) -> Vec<String> {
    let mut tags = vec![
        "KYZQ",
        "price_volume",
        "return",
        "intraday",
        "minute_agg",
        "neutralize",
        "barra",
        "size",
        "sector",
        "daily",
        "apm",
        "residual",
        "ret20",
    ];
    match kind {
        KyzqApmKind::Apm => tags.push("deprecated"),
        KyzqApmKind::ApmNew => tags.push("overnight"),
    }
    tags.into_iter().map(str::to_string).collect()
}

fn description(def: KyzqApmFactorDef) -> String {
    match def.kind {
        KyzqApmKind::Apm => format!(
            "{} KYZQ APM factor from independent AM/PM minute return raw, 20-day residual t-stat, Ret20 residualized with intercept, then neutralized by Barra SIZE and SW sector.",
            def.name
        ),
        KyzqApmKind::ApmNew => format!(
            "{} KYZQ improved APM factor from daily open/pre-close overnight return plus PM minute return raw, 20-day residual t-stat, Ret20 residualized with intercept, then neutralized by Barra SIZE and SW sector.",
            def.name
        ),
    }
}

fn is_sh_sz_stock(ts_code: &str) -> bool {
    let upper = ts_code.to_ascii_uppercase();
    upper.ends_with(".SH") || upper.ends_with(".SZ")
}

fn clean_positive(value: Option<f64>) -> Option<f64> {
    clean_intraday_value(value).filter(|value| *value > 0.0)
}

fn finite_option(value: Option<f64>) -> Option<f64> {
    value.filter(|value| value.is_finite())
}

fn overnight_return(open: Option<f64>, pre_close: Option<f64>) -> Option<f64> {
    match (finite_option(open), finite_option(pre_close)) {
        (Some(open), Some(pre_close)) if pre_close.abs() > EPS => {
            finite_option(Some(open / pre_close - 1.0))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use crate::core::FactorContext;
    use crate::data::{ColumnData, Table};

    use super::*;

    fn assert_close(actual: Option<f64>, expected: f64) {
        let actual = actual.expect("value");
        assert!(
            (actual - expected).abs() < 1e-10,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn kyzq_apm_ols_uses_intercept() {
        let y = vec![2.0, 4.0, 6.0];
        let x = vec![1.0, 2.0, 3.0];
        let (alpha, beta) = ols_intercept_beta(&y, &x).expect("ols");
        assert!((alpha.abs()) < 1e-10);
        assert!((beta - 2.0).abs() < 1e-10);

        let y = vec![3.0, 5.0, 7.0];
        let (alpha, beta) = ols_intercept_beta(&y, &x).expect("ols");
        assert!((alpha - 1.0).abs() < 1e-10);
        assert!((beta - 2.0).abs() < 1e-10);
    }

    #[test]
    fn kyzq_apm_session_returns_exclude_bj_and_use_close_endpoints() {
        let table = Table::new(BTreeMap::from([
            (
                "trade_time".to_string(),
                ColumnData::Utf8(vec![
                    Some("09:30:00".to_string()),
                    Some("11:30:00".to_string()),
                    Some("13:00:00".to_string()),
                    Some("15:00:00".to_string()),
                    Some("09:30:00".to_string()),
                    Some("11:30:00".to_string()),
                ]),
            ),
            (
                "ts_code".to_string(),
                ColumnData::Utf8(vec![
                    Some("000001.SZ".to_string()),
                    Some("000001.SZ".to_string()),
                    Some("000001.SZ".to_string()),
                    Some("000001.SZ".to_string()),
                    Some("920001.BJ".to_string()),
                    Some("920001.BJ".to_string()),
                ]),
            ),
            (
                "close".to_string(),
                ColumnData::F64(vec![
                    Some(10.0),
                    Some(11.0),
                    Some(20.0),
                    Some(22.0),
                    Some(1.0),
                    Some(2.0),
                ]),
            ),
        ]))
        .expect("table");

        let returns = session_returns_from_table(&table, true, true).expect("returns");

        assert_eq!(returns.len(), 1);
        let values = returns.get("000001.SZ").expect("sz");
        assert_close(values.am, 0.1);
        assert_close(values.pm, 0.1);
        assert!(!returns.contains_key("920001.BJ"));
    }

    #[test]
    fn kyzq_apm_session_returns_only_compute_requested_sessions() {
        let table = Table::new(BTreeMap::from([
            (
                "trade_time".to_string(),
                ColumnData::Utf8(vec![
                    Some("09:30:00".to_string()),
                    Some("11:30:00".to_string()),
                    Some("13:00:00".to_string()),
                    Some("15:00:00".to_string()),
                ]),
            ),
            (
                "ts_code".to_string(),
                ColumnData::Utf8(vec![
                    Some("000001.SZ".to_string()),
                    Some("000001.SZ".to_string()),
                    Some("000001.SZ".to_string()),
                    Some("000001.SZ".to_string()),
                ]),
            ),
            (
                "close".to_string(),
                ColumnData::F64(vec![Some(10.0), Some(11.0), Some(20.0), Some(22.0)]),
            ),
        ]))
        .expect("table");

        let pm_only = session_returns_from_table(&table, false, true).expect("returns");
        let values = pm_only.get("000001.SZ").expect("sz");
        assert_eq!(values.am, None);
        assert_close(values.pm, 0.1);

        let am_only = session_returns_from_table(&table, true, false).expect("returns");
        let values = am_only.get("000001.SZ").expect("sz");
        assert_close(values.am, 0.1);
        assert_eq!(values.pm, None);
    }

    #[test]
    fn kyzq_apm_factor_specs_have_kyzq_tag() {
        let spec = factor_spec(KyzqApmFactorDef {
            id: "apm_new",
            alias: "APMnew",
            name: "APM New",
            kind: KyzqApmKind::ApmNew,
        });
        assert_eq!(spec.id, "apm_new");
        assert!(spec.tags.iter().any(|tag| tag == "KYZQ"));
        assert!(spec.tags.iter().any(|tag| tag == "overnight"));
        assert_eq!(spec.intraday_raw_dependencies[0].raw_id, APM_PM_RET_RAW_ID);
        assert_eq!(spec.lookback.trading_days, 19);
    }

    #[test]
    fn kyzq_apm_new_overnight_uses_daily_open_preclose() {
        assert_close(overnight_return(Some(11.0), Some(10.0)), 0.1);
        assert_eq!(overnight_return(Some(11.0), Some(0.0)), None);
    }

    #[test]
    fn kyzq_apm_market_mean_uses_cross_section_valid_values() {
        let context = FactorContext {
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            start_date: 20260101,
            end_date: 20260101,
            load_start_date: 20260101,
            load_dates: vec![20260101],
            target_dates: vec![20260101],
        };
        let table = Table::new(BTreeMap::from([
            (
                "trade_date".to_string(),
                ColumnData::I32(vec![Some(20260101), Some(20260101), Some(20260101)]),
            ),
            (
                "ts_code".to_string(),
                ColumnData::Utf8(vec![
                    Some("000001.SZ".to_string()),
                    Some("600000.SH".to_string()),
                    Some("920001.BJ".to_string()),
                ]),
            ),
            (
                APM_AM_RET_RAW_ID.to_string(),
                ColumnData::F64(vec![Some(0.01), Some(0.03), Some(0.99)]),
            ),
        ]))
        .expect("table");
        let panel = DailyPanel::from_table(&table, &context).expect("panel");
        let raw = panel.column(APM_AM_RET_RAW_ID).expect("raw");
        let mean = market_mean_column(&panel, &raw).expect("mean");

        assert_eq!(mean.values()[0], Some(0.02));
        assert_eq!(mean.values()[1], Some(0.02));
        assert_eq!(mean.values()[2], None);
    }
}

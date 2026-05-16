use std::collections::{BTreeMap, BTreeSet};

use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorRowKey, FactorSeries, FactorSpec,
    FactorValue, Frequency, IntradayDailyRawRequest, IntradayDailyRawSeries, IntradayDailyRawSpec,
    Lookback,
};
use crate::data::{DataPool, Table};
use crate::error::Result;
use crate::factor::common::stock_daily_ops::is_bj_stock;
use crate::factor::common::stock_daily_raw_ids::{
    HF_RDKURT_RAW_ID, HF_RDSKEW_RAW_ID, HF_RDVOL_RAW_ID,
};
use crate::factor::common::vector::clean;
use crate::factor::common::{clean_intraday_value, stock_minute_raw_spec, DailyPanel, PanelColumn};

pub const VERSION: &str = "0.1.0";
pub const RAW_VERSION: &str = "0.1.0";
pub const PROVIDER_KEY: &str = "gfzq_hf_resid_stats_provider";
pub const REGRESSION_WINDOW: usize = 20;
pub const RAW_DAILY_LOOKBACK: usize = REGRESSION_WINDOW;
pub const MIN_OBS: usize = 5;

const RAW_WINDOW_DAYS: usize = 1;
const REGRESSOR_COUNT: usize = 5;
const EPS: f64 = 1e-12;

#[derive(Clone, Copy, Debug)]
pub struct GfzqHfResidStdFactorDef {
    pub id: &'static str,
    pub alias: &'static str,
    pub name: &'static str,
}

#[derive(Clone, Copy, Debug, Default)]
struct DailyHfStats {
    rdvol: Option<f64>,
    rdskew: Option<f64>,
    rdkurt: Option<f64>,
}

pub const DEF: GfzqHfResidStdFactorDef = GfzqHfResidStdFactorDef {
    id: "hf_resid_std",
    alias: "HighFreqResidualStd",
    name: "High Frequency Residual Std",
};

pub fn all_raw_ids() -> [&'static str; 3] {
    [HF_RDVOL_RAW_ID, HF_RDSKEW_RAW_ID, HF_RDKURT_RAW_ID]
}

pub fn raw_spec(raw_id: &str) -> IntradayDailyRawSpec {
    stock_minute_raw_spec(raw_id, RAW_VERSION, &["close"], RAW_WINDOW_DAYS)
}

pub fn raw_specs() -> Vec<IntradayDailyRawSpec> {
    all_raw_ids()
        .iter()
        .map(|raw_id| raw_spec(raw_id))
        .collect()
}

pub fn factor_spec() -> FactorSpec {
    FactorSpec {
        id: DEF.id.to_string(),
        aliases: vec![DEF.alias.to_string()],
        name: DEF.name.to_string(),
        asset_class: AssetClass::Stock,
        frequency: Frequency::Daily,
        version: VERSION.to_string(),
        tags: tags(),
        description: "GFZQ high-frequency residual standard deviation factor from 1-minute realized volatility, skewness, and kurtosis changes in a 20-day time-series regression.".to_string(),
        dependencies: vec![DataRequest::new(
            DatasetId::StockDailyPv,
            &["close", "pre_close"],
        )],
        intraday_raw_dependencies: all_raw_ids()
            .iter()
            .map(|raw_id| IntradayDailyRawRequest::new(raw_id, RAW_DAILY_LOOKBACK))
            .collect(),
        lookback: Lookback {
            trading_days: REGRESSION_WINDOW,
        },
    }
}

pub fn minute_compute_many(
    raw_ids: &[String],
    context: &FactorContext,
    data: &DataPool,
) -> Result<Vec<IntradayDailyRawSeries>> {
    let requested = raw_ids
        .iter()
        .map(String::as_str)
        .filter(|raw_id| all_raw_ids().contains(raw_id))
        .collect::<BTreeSet<_>>();
    if requested.is_empty() {
        return Ok(Vec::new());
    }

    let mut values = all_raw_ids()
        .iter()
        .map(|raw_id| (*raw_id, Vec::<FactorValue>::new()))
        .collect::<BTreeMap<_, _>>();

    for trade_date in &context.target_dates {
        let Some(table) = data.minute(DatasetId::StockMinute1m, *trade_date) else {
            continue;
        };
        let day_values = daily_hf_stats_from_table(table)?;
        for (ts_code, stats) in day_values {
            let key = FactorRowKey::Daily {
                trade_date: *trade_date,
                ts_code,
            };
            push_requested(&mut values, &requested, HF_RDVOL_RAW_ID, &key, stats.rdvol);
            push_requested(
                &mut values,
                &requested,
                HF_RDSKEW_RAW_ID,
                &key,
                stats.rdskew,
            );
            push_requested(
                &mut values,
                &requested,
                HF_RDKURT_RAW_ID,
                &key,
                stats.rdkurt,
            );
        }
    }

    let mut output = Vec::new();
    for raw_id in all_raw_ids() {
        if !requested.contains(raw_id) {
            continue;
        }
        output.push(IntradayDailyRawSeries {
            spec: raw_spec(raw_id),
            values: values.remove(raw_id).unwrap_or_default(),
        });
    }
    Ok(output)
}

pub fn compute_factor(data: &DataPool) -> Result<FactorSeries> {
    let panel = data.intraday_daily_raw_panel(HF_RDVOL_RAW_ID)?;
    let rdvol = panel.column(HF_RDVOL_RAW_ID)?;
    let rdskew = panel.column(HF_RDSKEW_RAW_ID)?;
    let rdkurt = panel.column(HF_RDKURT_RAW_ID)?;
    let delta_vol = rdvol.ts(delta_series)?;
    let delta_skew = rdskew.ts(delta_series)?;
    let delta_kurt = rdkurt.ts(delta_series)?;

    let close = panel.column_from_table(data.daily(DatasetId::StockDailyPv)?, "close")?;
    let pre_close = panel.column_from_table(data.daily(DatasetId::StockDailyPv)?, "pre_close")?;
    let stock_return = close.zip_binary(&pre_close, daily_return)?;
    let market_return = expand_market_returns_ex_bj(panel, &stock_return)?;

    let factor = residual_std_column(
        panel,
        &stock_return,
        &market_return,
        &delta_vol,
        &delta_skew,
        &delta_kurt,
    )?;
    Ok(factor.to_factor_series(factor_spec()))
}

fn daily_hf_stats_from_table(table: &Table) -> Result<BTreeMap<String, DailyHfStats>> {
    let ts_codes = table.required_utf8("ts_code")?;
    let trade_times = table.required_utf8("trade_time")?;
    let close = table.required_f64_cast("close")?;

    let mut grouped = BTreeMap::<String, Vec<usize>>::new();
    for idx in 0..table.len {
        let Some(ts_code) = ts_codes[idx].clone() else {
            continue;
        };
        if trade_times[idx].is_none() {
            continue;
        }
        grouped.entry(ts_code).or_default().push(idx);
    }

    let mut output = BTreeMap::new();
    for (ts_code, indices) in grouped {
        let close_by_minute = close_by_minute(&indices, trade_times, &close);
        let returns = one_minute_log_returns(&close_by_minute);
        output.insert(ts_code, daily_hf_stats(&returns));
    }
    Ok(output)
}

pub fn one_minute_log_returns(close_by_minute: &BTreeMap<i32, f64>) -> Vec<f64> {
    let anchors = minute_anchors();
    let mut output = Vec::with_capacity(240);
    for pair in anchors.windows(2) {
        let (Some(previous), Some(current)) =
            (close_by_minute.get(&pair[0]), close_by_minute.get(&pair[1]))
        else {
            continue;
        };
        if *previous <= 0.0 || *current <= 0.0 {
            continue;
        }
        let value = current.ln() - previous.ln();
        if value.is_finite() {
            output.push(value);
        }
    }
    output
}

fn daily_hf_stats(returns: &[f64]) -> DailyHfStats {
    let Some(rdvar) = finite_value(returns.iter().map(|value| value * value).sum::<f64>()) else {
        return DailyHfStats::default();
    };
    if returns.is_empty() {
        return DailyHfStats::default();
    }
    let rdvol = finite_value(rdvar.sqrt());
    if rdvar <= EPS {
        return DailyHfStats {
            rdvol,
            ..DailyHfStats::default()
        };
    }
    let n = returns.len() as f64;
    let sum3 = returns.iter().map(|value| value.powi(3)).sum::<f64>();
    let sum4 = returns.iter().map(|value| value.powi(4)).sum::<f64>();
    DailyHfStats {
        rdvol,
        rdskew: finite_value(n.sqrt() * sum3 / rdvar.powf(1.5)),
        rdkurt: finite_value(n * sum4 / rdvar.powi(2)),
    }
}

fn residual_std_column(
    panel: &DailyPanel,
    stock_return: &PanelColumn,
    market_return: &PanelColumn,
    delta_vol: &PanelColumn,
    delta_skew: &PanelColumn,
    delta_kurt: &PanelColumn,
) -> Result<PanelColumn> {
    let date_count = panel.dates().len();
    let instrument_count = panel.instruments().len();
    let mut output = vec![None; panel.shape_len()];

    for instrument_idx in 0..instrument_count {
        for end in 0..date_count {
            let start = (end + 1).saturating_sub(REGRESSION_WINDOW);
            let mut y = Vec::with_capacity(REGRESSION_WINDOW);
            let mut x = Vec::with_capacity(REGRESSION_WINDOW);
            for date_idx in start..=end {
                let offset = date_idx * instrument_count + instrument_idx;
                let (
                    Some(y_value),
                    Some(mkt_value),
                    Some(vol_value),
                    Some(skew_value),
                    Some(kurt_value),
                ) = (
                    finite(stock_return.values()[offset]),
                    finite(market_return.values()[offset]),
                    finite(delta_vol.values()[offset]),
                    finite(delta_skew.values()[offset]),
                    finite(delta_kurt.values()[offset]),
                )
                else {
                    continue;
                };
                y.push(y_value);
                x.push([1.0, mkt_value, vol_value, skew_value, kurt_value]);
            }
            output[end * instrument_count + instrument_idx] = regression_residual_std(&y, &x);
        }
    }

    panel.column_from_values(output)
}

fn tags() -> Vec<String> {
    [
        "GFZQ",
        "price_volume",
        "intraday",
        "realized_moments",
        "residual_volatility",
        "regression",
        "daily",
    ]
    .iter()
    .map(|value| value.to_string())
    .collect()
}

fn close_by_minute(
    indices: &[usize],
    trade_times: &[Option<String>],
    close: &[Option<f64>],
) -> BTreeMap<i32, f64> {
    let mut output = BTreeMap::new();
    for idx in indices {
        let (Some(minute), Some(close_value)) = (
            trade_times[*idx].as_deref().and_then(time_to_minute),
            clean_intraday_value(close[*idx]).filter(|value| *value > 0.0),
        ) else {
            continue;
        };
        if is_sample_anchor(minute) {
            output.insert(minute, close_value);
        }
    }
    output
}

fn minute_anchors() -> Vec<i32> {
    let mut anchors = Vec::with_capacity(241);
    anchors.push(9 * 60 + 30);
    let mut minute = 9 * 60 + 31;
    while minute <= 11 * 60 + 30 {
        anchors.push(minute);
        minute += 1;
    }
    minute = 13 * 60 + 1;
    while minute <= 15 * 60 {
        anchors.push(minute);
        minute += 1;
    }
    anchors
}

fn is_sample_anchor(minute: i32) -> bool {
    minute == 9 * 60 + 30
        || minute == 11 * 60 + 30
        || ((9 * 60 + 31)..=(11 * 60 + 30)).contains(&minute)
        || ((13 * 60 + 1)..=(15 * 60)).contains(&minute)
}

fn time_to_minute(value: &str) -> Option<i32> {
    let value = value.trim();
    if value.is_empty() {
        return None;
    }
    let value = value
        .rsplit_once(' ')
        .map(|(_, right)| right)
        .or_else(|| value.rsplit_once('T').map(|(_, right)| right))
        .unwrap_or(value)
        .trim();
    if value.contains(':') {
        let parts = value.split(':').collect::<Vec<_>>();
        if parts.len() < 2 {
            return None;
        }
        let hour = parts[0].parse::<i32>().ok()?;
        let minute = parts[1].parse::<i32>().ok()?;
        return Some(hour * 60 + minute);
    }
    if value.len() >= 4 {
        let hour = value.get(0..2)?.parse::<i32>().ok()?;
        let minute = value.get(2..4)?.parse::<i32>().ok()?;
        return Some(hour * 60 + minute);
    }
    None
}

fn push_requested(
    values: &mut BTreeMap<&'static str, Vec<FactorValue>>,
    requested: &BTreeSet<&str>,
    raw_id: &'static str,
    key: &FactorRowKey,
    value: Option<f64>,
) {
    if !requested.contains(raw_id) {
        return;
    }
    values.entry(raw_id).or_default().push(FactorValue {
        key: key.clone(),
        value,
    });
}

fn expand_market_returns_ex_bj(
    panel: &DailyPanel,
    stock_return: &PanelColumn,
) -> Result<PanelColumn> {
    let instrument_count = panel.instruments().len();
    let instruments = panel.instruments();
    let mut values = Vec::with_capacity(panel.shape_len());
    for date_idx in 0..panel.dates().len() {
        let mut sum = 0.0;
        let mut count = 0usize;
        for instrument_idx in 0..instrument_count {
            if is_bj_stock(&instruments[instrument_idx]) {
                continue;
            }
            let offset = date_idx * instrument_count + instrument_idx;
            if let Some(value) = finite(stock_return.values()[offset]) {
                sum += value;
                count += 1;
            }
        }
        let mean = (count > 0).then_some(sum / count as f64);
        for _ in 0..instrument_count {
            values.push(mean);
        }
    }
    panel.column_from_values(values)
}

fn delta_series(values: &[Option<f64>]) -> Vec<Option<f64>> {
    let mut output = vec![None; values.len()];
    for idx in 1..values.len() {
        match (finite(values[idx]), finite(values[idx - 1])) {
            (Some(current), Some(previous)) => output[idx] = finite_value(current - previous),
            _ => {}
        }
    }
    output
}

fn daily_return(close: Option<f64>, pre_close: Option<f64>) -> Option<f64> {
    match (finite(close), finite(pre_close)) {
        (Some(close), Some(pre_close)) if pre_close.abs() > EPS => {
            finite_value(close / pre_close - 1.0)
        }
        _ => None,
    }
}

fn regression_residual_std(y: &[f64], x: &[[f64; REGRESSOR_COUNT]]) -> Option<f64> {
    if y.len() != x.len() || y.len() < MIN_OBS {
        return None;
    }
    let beta = ols_beta(y, x)?;
    let mut residual_sum_squares = 0.0;
    for (row, y_value) in x.iter().zip(y) {
        let fitted = row
            .iter()
            .zip(beta.iter())
            .map(|(x_value, beta)| x_value * beta)
            .sum::<f64>();
        residual_sum_squares += (y_value - fitted).powi(2);
    }
    finite_value((residual_sum_squares / y.len() as f64).sqrt())
}

fn ols_beta(y: &[f64], x: &[[f64; REGRESSOR_COUNT]]) -> Option<[f64; REGRESSOR_COUNT]> {
    if y.len() != x.len() || y.len() < REGRESSOR_COUNT {
        return None;
    }
    let mut xtx = vec![vec![0.0; REGRESSOR_COUNT]; REGRESSOR_COUNT];
    let mut xty = vec![0.0; REGRESSOR_COUNT];
    for (row, y_value) in x.iter().zip(y.iter()) {
        for i in 0..REGRESSOR_COUNT {
            xty[i] += row[i] * y_value;
            for j in 0..REGRESSOR_COUNT {
                xtx[i][j] += row[i] * row[j];
            }
        }
    }
    let beta = solve_linear_system(xtx, xty)?;
    let mut output = [0.0; REGRESSOR_COUNT];
    for (idx, value) in beta.into_iter().enumerate() {
        output[idx] = value;
    }
    Some(output)
}

fn solve_linear_system(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    for pivot in 0..n {
        let mut max_row = pivot;
        let mut max_value = a[pivot][pivot].abs();
        for (row_idx, row) in a.iter().enumerate().skip(pivot + 1) {
            let value = row[pivot].abs();
            if value > max_value {
                max_value = value;
                max_row = row_idx;
            }
        }
        if max_value <= EPS {
            return None;
        }
        if max_row != pivot {
            a.swap(max_row, pivot);
            b.swap(max_row, pivot);
        }

        let pivot_value = a[pivot][pivot];
        for col in pivot..n {
            a[pivot][col] /= pivot_value;
        }
        b[pivot] /= pivot_value;

        for row_idx in 0..n {
            if row_idx == pivot {
                continue;
            }
            let factor = a[row_idx][pivot];
            if factor.abs() <= EPS {
                continue;
            }
            for col in pivot..n {
                a[row_idx][col] -= factor * a[pivot][col];
            }
            b[row_idx] -= factor * b[pivot];
        }
    }
    Some(b)
}

fn finite(value: Option<f64>) -> Option<f64> {
    clean(value).filter(|value| value.is_finite())
}

fn finite_value(value: f64) -> Option<f64> {
    value.is_finite().then_some(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: Option<f64>, expected: f64) {
        let actual = actual.expect("expected value");
        assert!(
            (actual - expected).abs() < 1e-10,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn gfzq_hf_minute_log_returns_use_0930_and_1130_anchors() {
        let mut close = BTreeMap::new();
        close.insert(9 * 60 + 30, 100.0);
        for minute in (9 * 60 + 31)..=(11 * 60 + 30) {
            close.insert(minute, 100.0 + (minute - (9 * 60 + 30)) as f64);
        }
        close.insert(13 * 60 + 1, 300.0);
        for minute in (13 * 60 + 2)..=(15 * 60) {
            close.insert(minute, 300.0 + (minute - (13 * 60 + 1)) as f64);
        }

        let returns = one_minute_log_returns(&close);

        assert_eq!(returns.len(), 240);
        assert!((returns[0] - (101.0_f64 / 100.0).ln()).abs() < 1e-12);
        assert!((returns[120] - (300.0_f64 / 220.0).ln()).abs() < 1e-12);
    }

    #[test]
    fn gfzq_hf_daily_stats_match_realized_moment_formulas() {
        let returns = vec![0.01, -0.02, 0.03];
        let stats = daily_hf_stats(&returns);
        let rdvar = 0.01_f64.powi(2) + (-0.02_f64).powi(2) + 0.03_f64.powi(2);
        let sum3 = 0.01_f64.powi(3) + (-0.02_f64).powi(3) + 0.03_f64.powi(3);
        let sum4 = 0.01_f64.powi(4) + (-0.02_f64).powi(4) + 0.03_f64.powi(4);

        assert_close(stats.rdvol, rdvar.sqrt());
        assert_close(stats.rdskew, 3.0_f64.sqrt() * sum3 / rdvar.powf(1.5));
        assert_close(stats.rdkurt, 3.0 * sum4 / rdvar.powi(2));
    }

    #[test]
    fn gfzq_hf_zero_variance_keeps_vol_and_drops_shape() {
        let stats = daily_hf_stats(&[0.0, 0.0]);

        assert_close(stats.rdvol, 0.0);
        assert_eq!(stats.rdskew, None);
        assert_eq!(stats.rdkurt, None);
    }

    #[test]
    fn gfzq_hf_delta_series_uses_previous_day() {
        assert_eq!(
            delta_series(&[Some(1.0), Some(1.5), None, Some(3.0)]),
            vec![None, Some(0.5), None, None]
        );
    }

    #[test]
    fn gfzq_hf_market_return_excludes_bj() {
        let panel = test_panel(
            vec![20260102],
            vec!["000001.SZ".to_string(), "430001.BJ".to_string()],
        );
        let returns = panel
            .column_from_values(vec![Some(0.02), Some(0.50)])
            .unwrap();

        let market = expand_market_returns_ex_bj(&panel, &returns).unwrap();

        assert_eq!(market.values(), &[Some(0.02), Some(0.02)]);
    }

    #[test]
    fn gfzq_hf_residual_std_recovers_zero_for_exact_model() {
        let panel = test_panel((0..6).collect(), vec!["000001.SZ".to_string()]);
        let mut y = Vec::new();
        let mut mkt = Vec::new();
        let mut vol = Vec::new();
        let mut skew = Vec::new();
        let mut kurt = Vec::new();
        for idx in 0..6 {
            let mkt_value = idx as f64;
            let vol_value = (idx % 2) as f64;
            let skew_value = (idx % 3) as f64;
            let kurt_value = (idx % 5) as f64;
            y.push(Some(
                1.0 + 0.1 * mkt_value + 0.2 * vol_value - 0.3 * skew_value + 0.4 * kurt_value,
            ));
            mkt.push(Some(mkt_value));
            vol.push(Some(vol_value));
            skew.push(Some(skew_value));
            kurt.push(Some(kurt_value));
        }
        let y = panel.column_from_values(y).unwrap();
        let mkt = panel.column_from_values(mkt).unwrap();
        let vol = panel.column_from_values(vol).unwrap();
        let skew = panel.column_from_values(skew).unwrap();
        let kurt = panel.column_from_values(kurt).unwrap();

        let residual = residual_std_column(&panel, &y, &mkt, &vol, &skew, &kurt).unwrap();

        assert!(residual.values()[5].unwrap().abs() < 1e-10);
    }

    #[test]
    fn gfzq_hf_residual_std_rejects_singular_design() {
        let panel = test_panel((0..5).collect(), vec!["000001.SZ".to_string()]);
        let y = panel.column_from_values(vec![Some(1.0); 5]).unwrap();
        let x = panel.column_from_values(vec![Some(1.0); 5]).unwrap();

        let residual = residual_std_column(&panel, &y, &x, &x, &x, &x).unwrap();

        assert_eq!(residual.values()[4], None);
    }

    fn test_panel(dates: Vec<i32>, instruments: Vec<String>) -> DailyPanel {
        let present = vec![true; dates.len() * instruments.len()];
        DailyPanel::from_index(dates.clone(), instruments, &dates, present).unwrap()
    }
}

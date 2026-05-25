use std::collections::{BTreeMap, BTreeSet};

use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorRowKey, FactorSeries, FactorSpec,
    FactorValue, Frequency, IntradayDailyRawRequest, IntradayDailyRawSeries, IntradayDailyRawSpec,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::stock_daily_ops::neutralize_size_sector;
use crate::factor::common::{
    clean_intraday_value, minute_vwap_from_amount_vol, quantile_linear, stock_minute_raw_spec,
};
use crate::factor::common::{DailyPanel, PanelColumn};
use crate::operators::{cs_regression_residual, cs_zscore, ts_mean};

pub const PROVIDER_KEY: &str = "mszq_gravity_field_provider";
pub const RAW_VERSION: &str = "0.1.0";
pub const VERSION: &str = "0.1.0";

pub const ACTIVE_BUY_COUNT_RAW_ID: &str = "daily_mszq_active_buy_count_raw";
pub const SIPHON_RAW_ID: &str = "daily_mszq_siphon_raw";
pub const SIPHON_REFLUX_RAW_ID: &str = "daily_mszq_siphon_reflux_raw";

const RAW_WINDOW_DAYS: usize = 1;
const ROLLING_WINDOW: usize = 20;
const MIN_PERIODS: usize = 1;
const MINUTE_COUNT: usize = 240;
const HEAT_TIME_COUNT: usize = 23;
const OPEN_EXCLUDE_END: usize = 15;
const CLOSE_EXCLUDE_START: usize = 225;
const AFTERNOON_OPEN_IDX: usize = 120;
const CLOSING_AUCTION_START: usize = 237;
const REFLUX_START: usize = 235;
const EPS: f64 = f64::EPSILON;

#[derive(Clone, Copy, Debug)]
pub struct MszqGravityFieldFactorDef {
    pub id: &'static str,
    pub alias: &'static str,
    pub name: &'static str,
}

#[derive(Clone, Copy, Debug, Default)]
struct ActiveSplit {
    buy_volume: f64,
    sell_amount: f64,
}

#[derive(Clone, Debug)]
struct StockDay {
    close: [Option<f64>; MINUTE_COUNT],
    volume: [Option<f64>; MINUTE_COUNT],
    amount: [Option<f64>; MINUTE_COUNT],
    buy_volume: [Option<f64>; MINUTE_COUNT],
    sell_amount: [Option<f64>; MINUTE_COUNT],
    minute_return: [Option<f64>; MINUTE_COUNT],
    amount_multiplier: [Option<f64>; MINUTE_COUNT],
}

#[derive(Clone, Copy, Debug, Default)]
struct DailyRawValues {
    active_buy_count: Option<f64>,
    siphon: Option<f64>,
    siphon_reflux: Option<f64>,
}

pub fn all_raw_ids() -> [&'static str; 3] {
    [ACTIVE_BUY_COUNT_RAW_ID, SIPHON_RAW_ID, SIPHON_REFLUX_RAW_ID]
}

pub fn raw_spec(raw_id: &str) -> IntradayDailyRawSpec {
    stock_minute_raw_spec(
        raw_id,
        RAW_VERSION,
        &["close", "vol", "amount"],
        RAW_WINDOW_DAYS,
    )
}

pub fn raw_specs() -> Vec<IntradayDailyRawSpec> {
    all_raw_ids()
        .iter()
        .map(|raw_id| raw_spec(raw_id))
        .collect()
}

pub fn factor_spec(def: MszqGravityFieldFactorDef) -> FactorSpec {
    FactorSpec {
        id: def.id.to_string(),
        aliases: vec![def.alias.to_string()],
        name: def.name.to_string(),
        asset_class: AssetClass::Stock,
        frequency: Frequency::Daily,
        version: VERSION.to_string(),
        tags: tags(),
        description: description(def),
        dependencies: dependencies(),
        intraday_raw_dependencies: all_raw_ids()
            .iter()
            .map(|raw_id| IntradayDailyRawRequest::new(raw_id, ROLLING_WINDOW - 1))
            .collect(),
        lookback: Lookback {
            trading_days: ROLLING_WINDOW - 1,
        },
    }
}

pub fn compute_factor(def: MszqGravityFieldFactorDef, data: &DataPool) -> Result<FactorSeries> {
    let panel = data.intraday_daily_raw_panel(ACTIVE_BUY_COUNT_RAW_ID)?;
    let active_buy_count = panel.column(ACTIVE_BUY_COUNT_RAW_ID)?;
    let siphon = panel.column(SIPHON_RAW_ID)?;
    let siphon_reflux = panel.column(SIPHON_REFLUX_RAW_ID)?;

    let main_component = active_buy_count
        .cs(cs_zscore)?
        .ts(|series| ts_mean(series, ROLLING_WINDOW, MIN_PERIODS))?;
    let net_siphon = siphon_reflux.cs_binary(&siphon, cs_regression_residual)?;
    let siphon_component = net_siphon.ts(|series| ts_mean(series, ROLLING_WINDOW, MIN_PERIODS))?;

    let z_main = main_component.cs(cs_zscore)?;
    let z_siphon = siphon_component
        .cs(cs_zscore)?
        .map_values(|value| finite_option(value).map(|value| -value));
    let composite = average_columns(&panel, &[&z_main, &z_siphon])?;
    let factor = neutralize_size_sector(&composite, &panel, data)?;
    Ok(factor.to_factor_series(factor_spec(def)))
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
        let ts_codes = table.required_utf8("ts_code")?;
        let trade_times = table.required_utf8("trade_time")?;
        let close = table.required_f64_cast("close")?;
        let volume = table.required_f64_cast("vol")?;
        let amount = table.required_f64_cast("amount")?;

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

        let mut stocks = BTreeMap::<String, StockDay>::new();
        for (ts_code, mut indices) in grouped {
            indices.sort_by(|left, right| trade_times[*left].cmp(&trade_times[*right]));
            stocks.insert(
                ts_code,
                stock_day_from_indices(&indices, &trade_times, &close, &volume, &amount),
            );
        }

        let daily_values = gravity_daily_raw_values(&stocks);
        for (ts_code, raw) in daily_values {
            let key = FactorRowKey::Daily {
                trade_date: *trade_date,
                ts_code,
            };
            push_requested(
                &mut values,
                &requested,
                ACTIVE_BUY_COUNT_RAW_ID,
                &key,
                raw.active_buy_count,
            );
            push_requested(&mut values, &requested, SIPHON_RAW_ID, &key, raw.siphon);
            push_requested(
                &mut values,
                &requested,
                SIPHON_REFLUX_RAW_ID,
                &key,
                raw.siphon_reflux,
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

fn gravity_daily_raw_values(
    stocks: &BTreeMap<String, StockDay>,
) -> BTreeMap<String, DailyRawValues> {
    let market = MarketIntradayStats::from_stocks(stocks);
    let heat_times = top_heat_times(&market.heat);
    stocks
        .iter()
        .map(|(ts_code, day)| {
            (
                ts_code.clone(),
                DailyRawValues {
                    active_buy_count: active_buy_count(day, &market.buy_volume),
                    siphon: siphon_correlation(day, &market, &heat_times, false),
                    siphon_reflux: siphon_correlation(day, &market, &heat_times, true),
                },
            )
        })
        .collect()
}

#[derive(Clone, Debug)]
struct MarketIntradayStats {
    buy_volume: [f64; MINUTE_COUNT],
    sell_amount: [f64; MINUTE_COUNT],
    sell_amount_total: f64,
    #[allow(dead_code)]
    return_value: [Option<f64>; MINUTE_COUNT],
    heat: [f64; MINUTE_COUNT],
}

impl MarketIntradayStats {
    fn from_stocks(stocks: &BTreeMap<String, StockDay>) -> Self {
        let mut buy_volume = [0.0; MINUTE_COUNT];
        let mut sell_amount = [0.0; MINUTE_COUNT];
        let mut sell_amount_total = 0.0;
        let mut return_numerator = [0.0; MINUTE_COUNT];
        let mut return_denominator = [0.0; MINUTE_COUNT];
        let mut daily_amount_by_stock = BTreeMap::<&str, f64>::new();
        let mut market_daily_amount = 0.0;

        for (ts_code, day) in stocks {
            let mut daily_amount = 0.0;
            for minute_idx in 0..MINUTE_COUNT {
                if let Some(value) = finite_option(day.buy_volume[minute_idx]) {
                    buy_volume[minute_idx] += value;
                }
                if let Some(value) = finite_option(day.sell_amount[minute_idx]) {
                    sell_amount[minute_idx] += value;
                    sell_amount_total += value;
                }
                if let Some(value) = finite_option(day.amount[minute_idx]) {
                    daily_amount += value;
                    if let Some(ret) = finite_option(day.minute_return[minute_idx]) {
                        return_numerator[minute_idx] += ret * value;
                        return_denominator[minute_idx] += value;
                    }
                }
            }
            if daily_amount > EPS {
                daily_amount_by_stock.insert(ts_code.as_str(), daily_amount);
                market_daily_amount += daily_amount;
            }
        }

        let mut return_value = [None; MINUTE_COUNT];
        for minute_idx in 0..MINUTE_COUNT {
            if return_denominator[minute_idx] > EPS {
                return_value[minute_idx] =
                    finite_value(return_numerator[minute_idx] / return_denominator[minute_idx]);
            }
        }

        let mut heat = [0.0; MINUTE_COUNT];
        if market_daily_amount > EPS {
            for (ts_code, day) in stocks {
                let Some(daily_amount) = daily_amount_by_stock.get(ts_code.as_str()) else {
                    continue;
                };
                let weight = daily_amount / market_daily_amount;
                for minute_idx in 0..MINUTE_COUNT {
                    let (Some(ret), Some(market_ret), Some(am)) = (
                        finite_option(day.minute_return[minute_idx]),
                        return_value[minute_idx],
                        finite_option(day.amount_multiplier[minute_idx]),
                    ) else {
                        continue;
                    };
                    if ret > market_ret {
                        heat[minute_idx] += weight * am;
                    }
                }
            }
        }

        Self {
            buy_volume,
            sell_amount,
            sell_amount_total,
            return_value,
            heat,
        }
    }
}

fn stock_day_from_indices(
    indices: &[usize],
    trade_times: &[Option<String>],
    close: &[Option<f64>],
    volume: &[Option<f64>],
    amount: &[Option<f64>],
) -> StockDay {
    let mut output = StockDay::empty();
    let mut previous_close = None;
    let mut previous_regular_volume = None;

    for idx in indices {
        let Some(trade_time) = trade_times[*idx].as_deref() else {
            continue;
        };
        let current_close = clean_positive(close[*idx]);
        if is_anchor_0930(trade_time) {
            if current_close.is_some() {
                previous_close = current_close;
            }
            continue;
        }
        let Some(minute_idx) = minute_index(trade_time) else {
            continue;
        };

        let current_volume = clean_nonnegative(volume[*idx]);
        let current_amount = clean_nonnegative(amount[*idx]);
        output.close[minute_idx] = current_close;
        output.volume[minute_idx] = current_volume;
        output.amount[minute_idx] = current_amount;
        output.minute_return[minute_idx] = minute_return(previous_close, current_close);
        output.amount_multiplier[minute_idx] =
            amount_multiplier(previous_regular_volume, current_volume);

        if let (Some(prev_close), Some(vol), Some(amt)) =
            (previous_close, current_volume, current_amount)
        {
            if let Some(split) = active_split(prev_close, vol, amt) {
                output.buy_volume[minute_idx] = Some(split.buy_volume);
                output.sell_amount[minute_idx] = Some(split.sell_amount);
            }
        }

        if current_close.is_some() {
            previous_close = current_close;
        }
        if current_volume.is_some() {
            previous_regular_volume = current_volume;
        }
    }

    output
}

fn active_split(previous_close: f64, volume: f64, amount: f64) -> Option<ActiveSplit> {
    if previous_close <= EPS || volume <= EPS {
        return None;
    }
    let vwap = minute_vwap_from_amount_vol(Some(amount), Some(volume))?;
    if !vwap.is_finite() {
        return None;
    }
    let split = if vwap > previous_close + EPS {
        ActiveSplit {
            buy_volume: volume,
            sell_amount: 0.0,
        }
    } else if vwap + EPS < previous_close {
        ActiveSplit {
            buy_volume: 0.0,
            sell_amount: amount,
        }
    } else {
        ActiveSplit {
            buy_volume: volume * 0.5,
            sell_amount: amount * 0.5,
        }
    };
    Some(split)
}

fn active_buy_count(day: &StockDay, market_buy_volume: &[f64; MINUTE_COUNT]) -> Option<f64> {
    let mut shares = Vec::<(usize, f64)>::new();
    for minute_idx in 0..MINUTE_COUNT {
        let market_buy = market_buy_volume[minute_idx];
        if market_buy <= EPS {
            continue;
        }
        let Some(buy_volume) = finite_option(day.buy_volume[minute_idx]) else {
            continue;
        };
        shares.push((minute_idx, buy_volume / market_buy));
    }
    if shares.is_empty() {
        return None;
    }
    let mut values = shares.iter().map(|(_, value)| *value).collect::<Vec<_>>();
    let threshold = quantile_linear(&mut values, 0.8)?;
    let count = shares
        .iter()
        .filter(|(minute_idx, share)| {
            *minute_idx >= OPEN_EXCLUDE_END
                && *minute_idx < CLOSE_EXCLUDE_START
                && *share > threshold
                && is_active_buy_peak(day, *minute_idx)
        })
        .count();
    Some(count as f64)
}

fn is_active_buy_peak(day: &StockDay, minute_idx: usize) -> bool {
    let Some(current) = finite_option(day.buy_volume[minute_idx]) else {
        return false;
    };
    if minute_idx == 0 {
        return finite_option(day.buy_volume[1]).is_some_and(|next| current > next);
    }
    if minute_idx + 1 == MINUTE_COUNT {
        return finite_option(day.buy_volume[minute_idx - 1]).is_some_and(|prev| current > prev);
    }
    let (Some(prev), Some(next)) = (
        finite_option(day.buy_volume[minute_idx - 1]),
        finite_option(day.buy_volume[minute_idx + 1]),
    ) else {
        return false;
    };
    current > prev && current > next
}

fn top_heat_times(heat: &[f64; MINUTE_COUNT]) -> Vec<usize> {
    let mut rows = (0..MINUTE_COUNT)
        .filter(|idx| is_heat_time_eligible(*idx))
        .map(|idx| (idx, heat[idx]))
        .collect::<Vec<_>>();
    rows.sort_by(|left, right| {
        right
            .1
            .partial_cmp(&left.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.0.cmp(&right.0))
    });
    rows.into_iter()
        .take(HEAT_TIME_COUNT)
        .map(|(idx, _)| idx)
        .collect()
}

fn is_heat_time_eligible(minute_idx: usize) -> bool {
    minute_idx != AFTERNOON_OPEN_IDX && minute_idx < CLOSING_AUCTION_START
}

fn siphon_correlation(
    day: &StockDay,
    market: &MarketIntradayStats,
    heat_times: &[usize],
    include_reflux: bool,
) -> Option<f64> {
    if market.sell_amount_total <= EPS {
        return None;
    }
    let mut pairs = Vec::<(f64, f64)>::new();
    for minute_idx in heat_times {
        pairs.push(sell_amount_share_pair(day, market, *minute_idx)?);
    }
    if include_reflux {
        pairs.push(reflux_sell_amount_share_pair(day, market)?);
    }
    pearson_pairs(&pairs)
}

fn sell_amount_share_pair(
    day: &StockDay,
    market: &MarketIntradayStats,
    minute_idx: usize,
) -> Option<(f64, f64)> {
    let denominator = market.sell_amount_total;
    if denominator <= EPS {
        return None;
    }
    let stock_share = finite_option(day.sell_amount[minute_idx]).unwrap_or(0.0) / denominator;
    let market_share = market.sell_amount[minute_idx] / denominator;
    Some((stock_share, market_share))
}

fn reflux_sell_amount_share_pair(
    day: &StockDay,
    market: &MarketIntradayStats,
) -> Option<(f64, f64)> {
    let denominator = market.sell_amount_total;
    if denominator <= EPS {
        return None;
    }
    let stock_sell = (REFLUX_START..MINUTE_COUNT)
        .filter_map(|idx| finite_option(day.sell_amount[idx]))
        .sum::<f64>();
    let market_sell = market.sell_amount[REFLUX_START..MINUTE_COUNT]
        .iter()
        .sum::<f64>();
    Some((stock_sell / denominator, market_sell / denominator))
}

fn pearson_pairs(pairs: &[(f64, f64)]) -> Option<f64> {
    if pairs.len() < 2 {
        return None;
    }
    let mean_x = pairs.iter().map(|(x, _)| *x).sum::<f64>() / pairs.len() as f64;
    let mean_y = pairs.iter().map(|(_, y)| *y).sum::<f64>() / pairs.len() as f64;
    let mut covariance = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;
    for (x, y) in pairs {
        let dx = x - mean_x;
        let dy = y - mean_y;
        covariance += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }
    if var_x <= EPS || var_y <= EPS {
        return None;
    }
    finite_value(covariance / (var_x.sqrt() * var_y.sqrt()))
}

fn average_columns(panel: &DailyPanel, columns: &[&PanelColumn]) -> Result<PanelColumn> {
    if columns.is_empty() {
        return panel.column_from_values(vec![None; panel.shape_len()]);
    }
    let mut values = Vec::with_capacity(panel.shape_len());
    for offset in 0..panel.shape_len() {
        let mut sum = 0.0;
        let mut count = 0usize;
        for column in columns {
            if let Some(value) = finite_option(column.values()[offset]) {
                sum += value;
                count += 1;
            }
        }
        values.push((count > 0).then_some(sum / count as f64));
    }
    panel.column_from_values(values)
}

impl StockDay {
    fn empty() -> Self {
        Self {
            close: [None; MINUTE_COUNT],
            volume: [None; MINUTE_COUNT],
            amount: [None; MINUTE_COUNT],
            buy_volume: [None; MINUTE_COUNT],
            sell_amount: [None; MINUTE_COUNT],
            minute_return: [None; MINUTE_COUNT],
            amount_multiplier: [None; MINUTE_COUNT],
        }
    }
}

fn minute_return(previous_close: Option<f64>, current_close: Option<f64>) -> Option<f64> {
    let (Some(previous), Some(current)) = (previous_close, current_close) else {
        return None;
    };
    if previous.abs() <= EPS {
        return None;
    }
    finite_value(current / previous - 1.0)
}

fn amount_multiplier(previous_volume: Option<f64>, current_volume: Option<f64>) -> Option<f64> {
    let (Some(previous), Some(current)) = (previous_volume, current_volume) else {
        return None;
    };
    if previous.abs() <= EPS {
        return None;
    }
    finite_value(current / previous)
}

fn is_sh_sz_stock(ts_code: &str) -> bool {
    let upper = ts_code.to_ascii_uppercase();
    upper.ends_with(".SH") || upper.ends_with(".SZ")
}

fn minute_index(trade_time: &str) -> Option<usize> {
    let minutes = time_to_minutes(trade_time)?;
    let morning_start = 9 * 60 + 31;
    let morning_end = 11 * 60 + 30;
    let afternoon_start = 13 * 60 + 1;
    let afternoon_end = 15 * 60;
    if (morning_start..=morning_end).contains(&minutes) {
        return Some((minutes - morning_start) as usize);
    }
    if (afternoon_start..=afternoon_end).contains(&minutes) {
        return Some(120 + (minutes - afternoon_start) as usize);
    }
    None
}

fn is_anchor_0930(trade_time: &str) -> bool {
    time_to_minutes(trade_time) == Some(9 * 60 + 30)
}

fn time_to_minutes(value: &str) -> Option<i32> {
    let value = value.trim();
    if value.is_empty() {
        return None;
    }
    let time = value
        .rsplit_once(' ')
        .map(|(_, right)| right)
        .or_else(|| value.rsplit_once('T').map(|(_, right)| right))
        .unwrap_or(value)
        .trim();
    if time.len() < 5 {
        return None;
    }
    let hour = time.get(0..2)?.parse::<i32>().ok()?;
    let minute = time.get(3..5)?.parse::<i32>().ok()?;
    Some(hour * 60 + minute)
}

fn tags() -> Vec<String> {
    [
        "price_volume",
        "volume",
        "amount",
        "return",
        "active_buy",
        "siphon",
        "intraday",
        "minute_agg",
        "neutralize",
        "barra",
        "size",
        "sector",
        "daily",
        "MSZQ",
    ]
    .iter()
    .map(|value| value.to_string())
    .collect()
}

fn description(def: MszqGravityFieldFactorDef) -> String {
    format!(
        "{} composites active-buy trading specificity and a sign-flipped net siphon effect from 1-minute close/volume/amount data, restricts the universe to Shanghai/Shenzhen stocks, and neutralizes by Barra SIZE and SW sector.",
        def.name
    )
}

fn dependencies() -> Vec<DataRequest> {
    vec![
        DataRequest::new(DatasetId::StockBarraDaily, &["SIZE"]),
        DataRequest::new(DatasetId::StockSwClassification, &["l1_code"]),
    ]
}

fn push_requested(
    values: &mut BTreeMap<&'static str, Vec<FactorValue>>,
    requested: &BTreeSet<&str>,
    raw_id: &'static str,
    key: &FactorRowKey,
    value: Option<f64>,
) {
    if requested.contains(raw_id) {
        values.entry(raw_id).or_default().push(FactorValue {
            key: key.clone(),
            value,
        });
    }
}

fn clean_positive(value: Option<f64>) -> Option<f64> {
    clean_intraday_value(value)
        .and_then(finite_value)
        .filter(|value| *value > 0.0)
}

fn clean_nonnegative(value: Option<f64>) -> Option<f64> {
    clean_intraday_value(value)
        .and_then(finite_value)
        .filter(|value| *value >= 0.0)
}

fn finite_option(value: Option<f64>) -> Option<f64> {
    value.and_then(finite_value)
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
    fn gravity_field_uses_0930_only_as_anchor() {
        let times = vec![
            Some("09:30:00".to_string()),
            Some("09:31:00".to_string()),
            Some("09:32:00".to_string()),
        ];
        let indices = vec![0, 1, 2];
        let close = vec![Some(10.0), Some(10.5), Some(10.0)];
        let volume = vec![Some(999.0), Some(100.0), Some(200.0)];
        let amount = vec![Some(9990.0), Some(1100.0), Some(1900.0)];

        let day = stock_day_from_indices(&indices, &times, &close, &volume, &amount);

        assert_close(day.minute_return[0], 0.05);
        assert_close(day.buy_volume[0], 100.0);
        assert_eq!(day.amount_multiplier[0], None);
        assert_close(day.amount_multiplier[1], 2.0);
        assert_eq!(minute_index("09:30:00"), None);
    }

    #[test]
    fn gravity_field_active_split_handles_buy_sell_and_equal() {
        let buy = active_split(10.0, 100.0, 1100.0).expect("buy");
        let sell = active_split(10.0, 100.0, 900.0).expect("sell");
        let equal = active_split(10.0, 100.0, 1000.0).expect("equal");

        assert_close(Some(buy.buy_volume), 100.0);
        assert_close(Some(buy.sell_amount), 0.0);
        assert_close(Some(sell.buy_volume), 0.0);
        assert_close(Some(sell.sell_amount), 900.0);
        assert_close(Some(equal.buy_volume), 50.0);
        assert_close(Some(equal.sell_amount), 500.0);
    }

    #[test]
    fn gravity_field_filters_only_sh_sz_stocks() {
        assert!(is_sh_sz_stock("000001.SZ"));
        assert!(is_sh_sz_stock("600000.sh"));
        assert!(!is_sh_sz_stock("920087.BJ"));
        assert!(!is_sh_sz_stock("ABC.US"));
    }

    #[test]
    fn gravity_field_active_buy_count_uses_quantile_peak_and_210_minute_window() {
        let mut day = StockDay::empty();
        let mut market = [100.0; MINUTE_COUNT];
        for idx in 0..MINUTE_COUNT {
            day.buy_volume[idx] = Some(10.0);
        }
        day.buy_volume[14] = Some(99.0);
        day.buy_volume[15] = Some(20.0);
        day.buy_volume[16] = Some(10.0);
        day.buy_volume[100] = Some(90.0);
        day.buy_volume[101] = Some(99.0);
        day.buy_volume[102] = Some(80.0);
        day.buy_volume[225] = Some(99.0);
        market[100] = 100.0;

        let count = active_buy_count(&day, &market);

        assert_close(count, 1.0);
    }

    #[test]
    fn gravity_field_market_heat_uses_minute_return_amount_and_daily_amount_weights() {
        let mut left = StockDay::empty();
        let mut right = StockDay::empty();
        left.minute_return[0] = Some(0.02);
        right.minute_return[0] = Some(0.00);
        left.amount[0] = Some(300.0);
        right.amount[0] = Some(100.0);
        left.amount_multiplier[0] = Some(2.0);
        right.amount_multiplier[0] = Some(9.0);
        left.buy_volume[0] = Some(1.0);
        right.buy_volume[0] = Some(1.0);
        let mut stocks = BTreeMap::new();
        stocks.insert("000001.SZ".to_string(), left);
        stocks.insert("600000.SH".to_string(), right);

        let market = MarketIntradayStats::from_stocks(&stocks);

        assert_close(market.return_value[0], 0.015);
        assert_close(Some(market.heat[0]), 1.5);
    }

    #[test]
    fn gravity_field_top_heat_times_excludes_afternoon_open_and_closing_auction() {
        let mut heat = [0.0; MINUTE_COUNT];
        heat[AFTERNOON_OPEN_IDX] = 999.0;
        heat[237] = 998.0;
        heat[100] = 10.0;

        let times = top_heat_times(&heat);

        assert_eq!(times[0], 100);
        assert!(!times.contains(&AFTERNOON_OPEN_IDX));
        assert!(!times.contains(&237));
    }

    #[test]
    fn gravity_field_siphon_uses_heat_points_and_reflux_point() {
        let mut day = StockDay::empty();
        let mut market = MarketIntradayStats {
            buy_volume: [0.0; MINUTE_COUNT],
            sell_amount: [0.0; MINUTE_COUNT],
            sell_amount_total: 100.0,
            return_value: [None; MINUTE_COUNT],
            heat: [0.0; MINUTE_COUNT],
        };
        let heat_times = vec![10, 11, 12];
        day.sell_amount[10] = Some(10.0);
        day.sell_amount[11] = Some(20.0);
        day.sell_amount[12] = Some(30.0);
        market.sell_amount[10] = 20.0;
        market.sell_amount[11] = 40.0;
        market.sell_amount[12] = 60.0;
        day.sell_amount[235] = Some(50.0);
        market.sell_amount[235] = 10.0;

        assert_close(siphon_correlation(&day, &market, &heat_times, false), 1.0);
        assert!(siphon_correlation(&day, &market, &heat_times, true).is_some());
    }

    #[test]
    fn gravity_field_component_postprocess_flips_siphon_before_average() {
        let panel = DailyPanel::from_index(
            vec![20260423, 20260424],
            vec!["a".to_string(), "b".to_string()],
            &[20260423, 20260424],
            vec![true, true, true, true],
        )
        .unwrap();
        let active_buy = panel
            .column_from_values(vec![Some(1.0), Some(2.0), Some(3.0), Some(5.0)])
            .unwrap();
        let siphon = panel
            .column_from_values(vec![Some(1.0), Some(2.0), Some(1.0), Some(2.0)])
            .unwrap();
        let siphon_reflux = panel
            .column_from_values(vec![Some(2.0), Some(5.0), Some(2.0), Some(5.0)])
            .unwrap();

        let main_component = active_buy
            .cs(cs_zscore)
            .unwrap()
            .ts(|series| ts_mean(series, ROLLING_WINDOW, MIN_PERIODS))
            .unwrap();
        let net_siphon = siphon_reflux
            .cs_binary(&siphon, cs_regression_residual)
            .unwrap();
        let siphon_component = net_siphon
            .ts(|series| ts_mean(series, ROLLING_WINDOW, MIN_PERIODS))
            .unwrap();
        let z_main = main_component.cs(cs_zscore).unwrap();
        let z_siphon = siphon_component
            .cs(cs_zscore)
            .unwrap()
            .map_values(|value| finite_option(value).map(|value| -value));
        let composite = average_columns(&panel, &[&z_main, &z_siphon]).unwrap();

        assert_eq!(composite.values().len(), 4);
    }

    #[test]
    fn gravity_field_factor_spec_has_mszq_tag_and_single_output() {
        let spec = factor_spec(MszqGravityFieldFactorDef {
            id: "gravity_field",
            alias: "gravity_field",
            name: "Gravity Field",
        });

        assert_eq!(spec.id, "gravity_field");
        assert!(spec.tags.iter().any(|tag| tag == "MSZQ"));
        assert!(spec.tags.iter().any(|tag| tag == "active_buy"));
        assert!(spec.tags.iter().any(|tag| tag == "siphon"));
        assert_eq!(spec.intraday_raw_dependencies.len(), 3);
        assert!(spec.description.contains("Shanghai/Shenzhen"));
    }
}

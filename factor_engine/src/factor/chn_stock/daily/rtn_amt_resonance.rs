use std::collections::BTreeMap;

use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorRowKey, FactorSeries, FactorSpec,
    FactorValue, Frequency, IntradayDailyRawRequest, IntradayDailyRawSeries, IntradayDailyRawSpec,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::stock_daily_ops::{is_bj_stock, neutralize_size_sector};
use crate::factor::common::{clean_intraday_value, stock_minute_raw_spec};
use crate::factor::Factor;
use crate::operators::{cs_pctrank, ts_ewm};

const VERSION: &str = "0.1.0";
const RAW_VERSION: &str = "0.1.0";
const RAW_ID: &str = "daily_zszq_rtn_amt_resonance_raw";
const PROVIDER_KEY: &str = "zszq_rtn_amt_resonance_provider";

const MINUTES_PER_DAY: usize = 240;
const AMOUNT_SLOTS_PER_DAY: usize = MINUTES_PER_DAY + 1;
const CO_TOP_N: usize = 5;
const PRE_AMT_TOP_N: usize = 30;
const EMA_SPAN: usize = 20;
const MIN_PERIODS: usize = 1;
const EPS: f64 = 1e-12;

pub struct StockDailyRtnAmtResonance;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyRtnAmtResonance)
}

impl Factor for StockDailyRtnAmtResonance {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "rtn_amt_resonance".to_string(),
            aliases: vec!["RTN_AMT_RESONANCE".to_string()],
            name: "rtn_amt_resonance".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: tags(),
            description: "ZSZQ intraday return and amount resonance factor. It combines market resonance at a stock's top/bottom return minutes with leading previous-minute amount share abnormality, then applies a 20-day EMA and neutralizes by Barra SIZE and SW sector.".to_string(),
            dependencies: vec![
                DataRequest::new(DatasetId::StockBarraDaily, &["SIZE"]),
                DataRequest::new(DatasetId::StockSwClassification, &["l1_code"]),
            ],
            intraday_raw_dependencies: vec![IntradayDailyRawRequest::new(RAW_ID, EMA_SPAN - 1)],
            lookback: Lookback {
                trading_days: EMA_SPAN - 1,
            },
        }
    }

    fn intraday_raw_specs(&self) -> Vec<IntradayDailyRawSpec> {
        vec![raw_spec()]
    }

    fn intraday_raw_provider_key(&self, _raw_id: &str) -> String {
        PROVIDER_KEY.to_string()
    }

    fn minute_compute_many(
        &self,
        raw_ids: &[String],
        context: &FactorContext,
        data: &DataPool,
    ) -> Result<Vec<IntradayDailyRawSeries>> {
        if !raw_ids.iter().any(|raw_id| raw_id == RAW_ID) {
            return Ok(Vec::new());
        }

        let mut values = Vec::<FactorValue>::new();
        for trade_date in &context.target_dates {
            let Some(table) = data.minute(DatasetId::StockMinute1m, *trade_date) else {
                continue;
            };
            let ts_codes = table.required_utf8("ts_code")?;
            let trade_times = table.required_utf8("trade_time")?;
            let close = table.required_f64_cast("close")?;
            let amount = table.required_f64_cast("amount")?;

            let mut stock_days = BTreeMap::<String, MinuteStockDay>::new();
            for idx in 0..table.len {
                let Some(ts_code) = ts_codes[idx].clone() else {
                    continue;
                };
                if is_bj_stock(&ts_code) {
                    continue;
                }
                let Some(trade_time) = trade_times[idx].as_deref() else {
                    continue;
                };
                let day = stock_days.entry(ts_code).or_default();
                let close_value = clean_intraday_value(close[idx]).filter(|value| *value > 0.0);
                let amount_value = clean_intraday_value(amount[idx]).filter(|value| *value >= 0.0);
                if is_anchor_minute(trade_time) {
                    day.anchor_close = close_value;
                    day.anchor_amount = amount_value;
                } else if let Some(minute_idx) = minute_index(trade_time) {
                    day.close[minute_idx] = close_value;
                    day.amount[minute_idx] = amount_value;
                }
            }

            for (ts_code, value) in daily_raw_values(&stock_days) {
                values.push(FactorValue {
                    key: FactorRowKey::Daily {
                        trade_date: *trade_date,
                        ts_code,
                    },
                    value,
                });
            }
        }

        Ok(vec![IntradayDailyRawSeries {
            spec: raw_spec(),
            values,
        }])
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let panel = data.intraday_daily_raw_panel(RAW_ID)?;
        let raw = panel.column(RAW_ID)?;
        let smoothed = raw.ts(|series| ts_ewm(series, EMA_SPAN, MIN_PERIODS))?;
        let factor = neutralize_size_sector(&smoothed, &panel, data)?;
        Ok(factor.to_factor_series(self.spec()))
    }
}

#[derive(Clone, Debug)]
struct MinuteStockDay {
    anchor_close: Option<f64>,
    close: [Option<f64>; MINUTES_PER_DAY],
    anchor_amount: Option<f64>,
    amount: [Option<f64>; MINUTES_PER_DAY],
}

impl Default for MinuteStockDay {
    fn default() -> Self {
        Self {
            anchor_close: None,
            close: [None; MINUTES_PER_DAY],
            anchor_amount: None,
            amount: [None; MINUTES_PER_DAY],
        }
    }
}

#[derive(Clone, Debug)]
struct ComputedStockDay {
    ts_code: String,
    returns: [Option<f64>; MINUTES_PER_DAY],
    amount_shares: [Option<f64>; AMOUNT_SLOTS_PER_DAY],
}

#[derive(Clone, Copy, Debug, Default)]
struct ComponentInputs {
    rm_max: Option<f64>,
    std_max: Option<f64>,
    rm_min: Option<f64>,
    std_min: Option<f64>,
    pre_amt_max: Option<f64>,
    pre_amt_min: Option<f64>,
}

fn raw_spec() -> IntradayDailyRawSpec {
    stock_minute_raw_spec(RAW_ID, RAW_VERSION, &["close", "amount"], 1)
}

fn tags() -> Vec<String> {
    [
        "ZSZQ",
        "return",
        "amount",
        "resonance",
        "intraday",
        "minute_agg",
        "neutralize",
        "barra",
        "size",
        "sector",
        "daily",
    ]
    .iter()
    .map(|value| value.to_string())
    .collect()
}

fn daily_raw_values(stock_days: &BTreeMap<String, MinuteStockDay>) -> Vec<(String, Option<f64>)> {
    let computed = computed_stock_days(stock_days);
    let (market_mean, market_std) = market_return_stats(&computed);
    let inputs = computed
        .iter()
        .map(|day| component_inputs(day, &market_mean, &market_std))
        .collect::<Vec<_>>();
    let raw_values = compose_ranked_raw(&inputs);
    computed
        .into_iter()
        .zip(raw_values)
        .map(|(day, value)| (day.ts_code, value))
        .collect()
}

fn computed_stock_days(stock_days: &BTreeMap<String, MinuteStockDay>) -> Vec<ComputedStockDay> {
    let mut market_amount = [0.0; AMOUNT_SLOTS_PER_DAY];
    let mut rows = Vec::<(
        String,
        [Option<f64>; MINUTES_PER_DAY],
        [Option<f64>; AMOUNT_SLOTS_PER_DAY],
    )>::new();

    for (ts_code, day) in stock_days {
        let returns = minute_returns(day);
        let amount_slots = amount_slots(day);
        for (idx, value) in amount_slots.iter().enumerate() {
            if let Some(value) = clean_intraday_value(*value).filter(|value| *value >= 0.0) {
                market_amount[idx] += value;
            }
        }
        rows.push((ts_code.clone(), returns, amount_slots));
    }

    rows.into_iter()
        .map(|(ts_code, returns, amount_slots)| {
            let mut amount_shares = [None; AMOUNT_SLOTS_PER_DAY];
            for idx in 0..AMOUNT_SLOTS_PER_DAY {
                amount_shares[idx] = match (amount_slots[idx], market_amount[idx]) {
                    (Some(amount), total) if total > EPS && amount.is_finite() => {
                        Some(amount / total)
                    }
                    _ => None,
                };
            }
            ComputedStockDay {
                ts_code,
                returns,
                amount_shares,
            }
        })
        .collect()
}

fn minute_returns(day: &MinuteStockDay) -> [Option<f64>; MINUTES_PER_DAY] {
    let mut output = [None; MINUTES_PER_DAY];
    for idx in 0..MINUTES_PER_DAY {
        let previous = if idx == 0 {
            day.anchor_close
        } else {
            day.close[idx - 1]
        };
        output[idx] = match (previous, day.close[idx]) {
            (Some(previous), Some(current)) if previous > EPS && current > EPS => {
                let value = current / previous - 1.0;
                value.is_finite().then_some(value)
            }
            _ => None,
        };
    }
    output
}

fn amount_slots(day: &MinuteStockDay) -> [Option<f64>; AMOUNT_SLOTS_PER_DAY] {
    let mut output = [None; AMOUNT_SLOTS_PER_DAY];
    output[0] = day.anchor_amount;
    for idx in 0..MINUTES_PER_DAY {
        output[idx + 1] = day.amount[idx];
    }
    output
}

fn market_return_stats(
    rows: &[ComputedStockDay],
) -> (
    [Option<f64>; MINUTES_PER_DAY],
    [Option<f64>; MINUTES_PER_DAY],
) {
    let mut sum = [0.0; MINUTES_PER_DAY];
    let mut sum_sq = [0.0; MINUTES_PER_DAY];
    let mut count = [0usize; MINUTES_PER_DAY];
    for row in rows {
        for idx in 0..MINUTES_PER_DAY {
            if let Some(value) = clean_intraday_value(row.returns[idx]) {
                sum[idx] += value;
                sum_sq[idx] += value * value;
                count[idx] += 1;
            }
        }
    }

    let mut mean = [None; MINUTES_PER_DAY];
    let mut std = [None; MINUTES_PER_DAY];
    for idx in 0..MINUTES_PER_DAY {
        if count[idx] > 0 {
            mean[idx] = Some(sum[idx] / count[idx] as f64);
        }
        if count[idx] > 1 {
            let n = count[idx] as f64;
            let variance = (sum_sq[idx] - sum[idx] * sum[idx] / n) / (n - 1.0);
            std[idx] = Some(variance.max(0.0).sqrt());
        }
    }
    (mean, std)
}

fn component_inputs(
    day: &ComputedStockDay,
    market_mean: &[Option<f64>; MINUTES_PER_DAY],
    market_std: &[Option<f64>; MINUTES_PER_DAY],
) -> ComponentInputs {
    let top5 = select_extreme_indices(&day.returns, CO_TOP_N, true);
    let bottom5 = select_extreme_indices(&day.returns, CO_TOP_N, false);
    let top30 = select_extreme_indices(&day.returns, PRE_AMT_TOP_N, true);
    let bottom30 = select_extreme_indices(&day.returns, PRE_AMT_TOP_N, false);
    let day_amount_share_mean = mean_clean(&day.amount_shares[1..]);

    ComponentInputs {
        rm_max: mean_at_indices(market_mean, &top5),
        std_max: mean_at_indices(market_std, &top5).filter(|value| *value > EPS),
        rm_min: mean_at_indices(market_mean, &bottom5),
        std_min: mean_at_indices(market_std, &bottom5).filter(|value| *value > EPS),
        pre_amt_max: pre_amount_ratio(day, &top30, day_amount_share_mean),
        pre_amt_min: pre_amount_ratio(day, &bottom30, day_amount_share_mean),
    }
}

fn select_extreme_indices(
    returns: &[Option<f64>; MINUTES_PER_DAY],
    n: usize,
    descending: bool,
) -> Vec<usize> {
    let mut values = returns
        .iter()
        .enumerate()
        .filter_map(|(idx, value)| clean_intraday_value(*value).map(|value| (idx, value)))
        .collect::<Vec<_>>();
    values.sort_by(|left, right| {
        let value_order = if descending {
            right.1.total_cmp(&left.1)
        } else {
            left.1.total_cmp(&right.1)
        };
        value_order.then_with(|| left.0.cmp(&right.0))
    });
    if values.len() < n {
        return Vec::new();
    }
    values.into_iter().take(n).map(|(idx, _)| idx).collect()
}

fn mean_at_indices(values: &[Option<f64>; MINUTES_PER_DAY], indices: &[usize]) -> Option<f64> {
    if indices.is_empty() {
        return None;
    }
    let mut sum = 0.0;
    let mut count = 0usize;
    for idx in indices {
        let value = clean_intraday_value(values[*idx])?;
        sum += value;
        count += 1;
    }
    (count == indices.len()).then_some(sum / count as f64)
}

fn mean_clean(values: &[Option<f64>]) -> Option<f64> {
    let mut sum = 0.0;
    let mut count = 0usize;
    for value in values {
        if let Some(value) = clean_intraday_value(*value) {
            sum += value;
            count += 1;
        }
    }
    (count > 0).then_some(sum / count as f64)
}

fn pre_amount_ratio(
    day: &ComputedStockDay,
    indices: &[usize],
    day_amount_share_mean: Option<f64>,
) -> Option<f64> {
    let denominator = day_amount_share_mean.filter(|value| *value > EPS)?;
    let mut sum = 0.0;
    let mut count = 0usize;
    for idx in indices {
        if let Some(value) = clean_intraday_value(day.amount_shares[*idx]) {
            sum += value;
            count += 1;
        }
    }
    (count > 0).then_some((sum / count as f64) / denominator)
}

fn compose_ranked_raw(inputs: &[ComponentInputs]) -> Vec<Option<f64>> {
    let rm_max = inputs.iter().map(|input| input.rm_max).collect::<Vec<_>>();
    let rm_min = inputs.iter().map(|input| input.rm_min).collect::<Vec<_>>();
    let rm_max_rank = cs_pctrank(&rm_max, true);
    let rm_min_rank = cs_pctrank(&rm_min, false);

    let co_max = inputs
        .iter()
        .zip(rm_max_rank.iter())
        .map(|(input, rank)| divide(*rank, input.std_max))
        .collect::<Vec<_>>();
    let co_min = inputs
        .iter()
        .zip(rm_min_rank.iter())
        .map(|(input, rank)| divide(*rank, input.std_min))
        .collect::<Vec<_>>();
    let pre_max = inputs
        .iter()
        .map(|input| input.pre_amt_max)
        .collect::<Vec<_>>();
    let pre_min = inputs
        .iter()
        .map(|input| input.pre_amt_min)
        .collect::<Vec<_>>();

    let co_max_rank = cs_pctrank(&co_max, true);
    let co_min_rank = cs_pctrank(&co_min, true);
    let pre_max_rank = cs_pctrank(&pre_max, true);
    let pre_min_rank = cs_pctrank(&pre_min, true);

    (0..inputs.len())
        .map(|idx| {
            let positive = (clean_intraday_value(co_max_rank[idx])?
                + clean_intraday_value(co_min_rank[idx])?)
                * 0.5;
            let negative = (clean_intraday_value(pre_max_rank[idx])?
                + clean_intraday_value(pre_min_rank[idx])?)
                * 0.5;
            Some(positive - negative)
        })
        .collect()
}

fn divide(numerator: Option<f64>, denominator: Option<f64>) -> Option<f64> {
    let numerator = clean_intraday_value(numerator)?;
    let denominator = clean_intraday_value(denominator)?;
    (denominator.abs() > EPS).then_some(numerator / denominator)
}

fn is_anchor_minute(trade_time: &str) -> bool {
    time_to_minutes(trade_time) == Some(9 * 60 + 30)
}

fn minute_index(trade_time: &str) -> Option<usize> {
    let total = time_to_minutes(trade_time)?;
    let morning_start = 9 * 60 + 31;
    let morning_end = 11 * 60 + 30;
    let afternoon_start = 13 * 60 + 1;
    let afternoon_end = 15 * 60;
    if (morning_start..=morning_end).contains(&total) {
        Some((total - morning_start) as usize)
    } else if (afternoon_start..=afternoon_end).contains(&total) {
        Some(120 + (total - afternoon_start) as usize)
    } else {
        None
    }
}

fn time_to_minutes(value: &str) -> Option<i32> {
    let value = value.trim();
    let value = value
        .rsplit_once(' ')
        .map(|(_, right)| right)
        .or_else(|| value.rsplit_once('T').map(|(_, right)| right))
        .unwrap_or(value)
        .trim();
    if value.len() < 5 {
        return None;
    }
    let hour = value.get(0..2)?.parse::<i32>().ok()?;
    let minute = value.get(3..5)?.parse::<i32>().ok()?;
    Some(hour * 60 + minute)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rtn_amt_resonance_minute_index_uses_regular_session() {
        assert!(is_anchor_minute("2026-04-24 09:30:00"));
        assert_eq!(minute_index("09:31:00"), Some(0));
        assert_eq!(minute_index("11:30:00"), Some(119));
        assert_eq!(minute_index("13:01:00"), Some(120));
        assert_eq!(minute_index("15:00:00"), Some(239));
        assert_eq!(minute_index("09:30:00"), None);
    }

    #[test]
    fn rtn_amt_resonance_pre_amount_uses_previous_minute_share_over_day_mean() {
        let mut returns = [None; MINUTES_PER_DAY];
        let mut shares = [None; AMOUNT_SLOTS_PER_DAY];
        for idx in 0..MINUTES_PER_DAY {
            returns[idx] = Some(idx as f64 / 1000.0);
            shares[idx + 1] = Some(1.0);
        }
        shares[0] = Some(3.0);
        for idx in 210..240 {
            shares[idx] = Some(2.0);
        }
        let day = ComputedStockDay {
            ts_code: "000001.SZ".to_string(),
            returns,
            amount_shares: shares,
        };
        let market_mean = [Some(0.0); MINUTES_PER_DAY];
        let market_std = [Some(1.0); MINUTES_PER_DAY];
        let input = component_inputs(&day, &market_mean, &market_std);
        let day_mean = (210.0 + 30.0 * 2.0) / 240.0;
        assert!((input.pre_amt_max.unwrap() - 2.0 / day_mean).abs() < 1e-12);
        assert!((input.pre_amt_min.unwrap() - ((3.0 + 29.0) / 30.0) / day_mean).abs() < 1e-12);
    }

    #[test]
    fn rtn_amt_resonance_compose_subtracts_negative_pre_amount_components() {
        let inputs = vec![
            ComponentInputs {
                rm_max: Some(0.3),
                std_max: Some(1.0),
                rm_min: Some(-0.1),
                std_min: Some(1.0),
                pre_amt_max: Some(3.0),
                pre_amt_min: Some(3.0),
            },
            ComponentInputs {
                rm_max: Some(0.1),
                std_max: Some(1.0),
                rm_min: Some(-0.3),
                std_min: Some(1.0),
                pre_amt_max: Some(1.0),
                pre_amt_min: Some(1.0),
            },
        ];
        let values = compose_ranked_raw(&inputs);
        assert!(values[0].unwrap() < values[1].unwrap());
    }

    #[test]
    fn rtn_amt_resonance_spec_has_zszq_tag() {
        let spec = StockDailyRtnAmtResonance.spec();
        assert_eq!(spec.id, "rtn_amt_resonance");
        assert!(spec.tags.iter().any(|tag| tag == "ZSZQ"));
        assert_eq!(spec.lookback.trading_days, 19);
    }
}

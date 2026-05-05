use std::collections::{BTreeMap, HashMap};

use crate::core::{
    AssetClass, DatasetId, FactorContext, FactorRowKey, FactorSeries, FactorSpec, FactorValue,
    Frequency, IntradayDailyRawRequest, IntradayDailyRawSeries, IntradayDailyRawSpec, Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::vector::clean;
use crate::factor::common::{clean_intraday_value, intraday_time_in_range, stock_minute_raw_spec};
use crate::factor::Factor;
use crate::operators::ts_sum;

pub const TAIL_CHASE_XY_RAW_ID: &str = "daily_tail_chase_xy";
pub const TAIL_CHASE_X2_RAW_ID: &str = "daily_tail_chase_x2";
pub const TAIL_CHASE_Y2_RAW_ID: &str = "daily_tail_chase_y2";

const RAW_VERSION: &str = "0.1.0";
const VERSION: &str = "0.1.0";
const WINDOW: usize = 20;
const MIN_PERIODS: usize = 1;

pub struct StockDailyTailChasing;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct TailChaseStats {
    count: usize,
    xy: f64,
    x2: f64,
    y2: f64,
}

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyTailChasing)
}

fn raw_spec(raw_id: &str) -> IntradayDailyRawSpec {
    stock_minute_raw_spec(raw_id, RAW_VERSION, &["open", "close"], 1)
}

impl Factor for StockDailyTailChasing {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "tail_chasing".to_string(),
            aliases: vec!["Tail_Chasing".to_string(), "TAIL_CHASING".to_string()],
            name: "Tail Chasing".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: [
                "price_volume",
                "return",
                "intraday",
                "minute_agg",
                "cosine",
                "daily",
                "KYZQ",
            ]
            .iter()
            .map(|value| value.to_string())
            .collect(),
            description: "Late-session chasing factor computed as 20-day cosine similarity between positive excess minute returns and next-minute excess returns.".to_string(),
            dependencies: Vec::new(),
            intraday_raw_dependencies: vec![
                IntradayDailyRawRequest::new(TAIL_CHASE_XY_RAW_ID, WINDOW - 1),
                IntradayDailyRawRequest::new(TAIL_CHASE_X2_RAW_ID, WINDOW - 1),
                IntradayDailyRawRequest::new(TAIL_CHASE_Y2_RAW_ID, WINDOW - 1),
            ],
            lookback: Lookback {
                trading_days: WINDOW - 1,
            },
        }
    }

    fn intraday_raw_specs(&self) -> Vec<IntradayDailyRawSpec> {
        vec![
            raw_spec(TAIL_CHASE_XY_RAW_ID),
            raw_spec(TAIL_CHASE_X2_RAW_ID),
            raw_spec(TAIL_CHASE_Y2_RAW_ID),
        ]
    }

    fn minute_compute_many(
        &self,
        raw_ids: &[String],
        context: &FactorContext,
        data: &DataPool,
    ) -> Result<Vec<IntradayDailyRawSeries>> {
        let wants_xy = raw_ids.iter().any(|raw_id| raw_id == TAIL_CHASE_XY_RAW_ID);
        let wants_x2 = raw_ids.iter().any(|raw_id| raw_id == TAIL_CHASE_X2_RAW_ID);
        let wants_y2 = raw_ids.iter().any(|raw_id| raw_id == TAIL_CHASE_Y2_RAW_ID);
        if !wants_xy && !wants_x2 && !wants_y2 {
            return Ok(Vec::new());
        }

        let mut xy_values = Vec::new();
        let mut x2_values = Vec::new();
        let mut y2_values = Vec::new();
        for trade_date in &context.target_dates {
            let Some(table) = data.minute(DatasetId::StockMinute1m, *trade_date) else {
                continue;
            };
            let ts_codes = table.required_utf8("ts_code")?;
            let trade_times = table.required_utf8("trade_time")?;
            let open = table.required_f64_cast("open")?;
            let close = table.required_f64_cast("close")?;

            let stats = tail_chase_stats_for_day(table.len, ts_codes, trade_times, &open, &close);
            for (ts_code, stats) in stats {
                let xy = stats.filter(|stats| stats.count > 0).map(|stats| stats.xy);
                let x2 = stats.filter(|stats| stats.count > 0).map(|stats| stats.x2);
                let y2 = stats.filter(|stats| stats.count > 0).map(|stats| stats.y2);
                let key = FactorRowKey::Daily {
                    trade_date: *trade_date,
                    ts_code: ts_code.clone(),
                };
                if wants_xy {
                    xy_values.push(FactorValue {
                        key: key.clone(),
                        value: xy,
                    });
                }
                if wants_x2 {
                    x2_values.push(FactorValue {
                        key: key.clone(),
                        value: x2,
                    });
                }
                if wants_y2 {
                    y2_values.push(FactorValue { key, value: y2 });
                }
            }
        }

        let mut output = Vec::new();
        if wants_xy {
            output.push(IntradayDailyRawSeries {
                spec: raw_spec(TAIL_CHASE_XY_RAW_ID),
                values: xy_values,
            });
        }
        if wants_x2 {
            output.push(IntradayDailyRawSeries {
                spec: raw_spec(TAIL_CHASE_X2_RAW_ID),
                values: x2_values,
            });
        }
        if wants_y2 {
            output.push(IntradayDailyRawSeries {
                spec: raw_spec(TAIL_CHASE_Y2_RAW_ID),
                values: y2_values,
            });
        }
        Ok(output)
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let panel = data.intraday_daily_raw_panel(TAIL_CHASE_XY_RAW_ID)?;
        let xy = panel.column(TAIL_CHASE_XY_RAW_ID)?;
        let x2 = panel.column(TAIL_CHASE_X2_RAW_ID)?;
        let y2 = panel.column(TAIL_CHASE_Y2_RAW_ID)?;

        let xy20 = xy.ts(|values| ts_sum(values, WINDOW, MIN_PERIODS))?;
        let x220 = x2.ts(|values| ts_sum(values, WINDOW, MIN_PERIODS))?;
        let y220 = y2.ts(|values| ts_sum(values, WINDOW, MIN_PERIODS))?;
        let factor = xy20.zip_ternary(&x220, &y220, cosine_from_sums)?;
        Ok(factor.to_factor_series(self.spec()))
    }
}

fn tail_chase_stats_for_day(
    row_count: usize,
    ts_codes: &[Option<String>],
    trade_times: &[Option<String>],
    open: &[Option<f64>],
    close: &[Option<f64>],
) -> BTreeMap<String, Option<TailChaseStats>> {
    let mut market_returns_by_time = BTreeMap::<String, Vec<f64>>::new();
    let mut returns_by_stock = BTreeMap::<String, Vec<(String, f64)>>::new();

    for idx in 0..row_count {
        let Some(ts_code) = ts_codes[idx].clone() else {
            continue;
        };
        let Some(trade_time) = trade_times[idx].as_deref() else {
            continue;
        };
        if !is_tail_session_time(trade_time) {
            continue;
        }
        let Some(return_value) = minute_bar_return(open[idx], close[idx]) else {
            continue;
        };
        market_returns_by_time
            .entry(trade_time.to_string())
            .or_default()
            .push(return_value);
        returns_by_stock
            .entry(ts_code)
            .or_default()
            .push((trade_time.to_string(), return_value));
    }

    let market_median_by_time = market_returns_by_time
        .into_iter()
        .filter_map(|(time, values)| median_f64(values).map(|median| (time, median)))
        .collect::<HashMap<_, _>>();

    let mut output = BTreeMap::new();
    for (ts_code, mut rows) in returns_by_stock {
        rows.sort_by(|left, right| left.0.cmp(&right.0));
        output.insert(ts_code, tail_chase_stats(&rows, &market_median_by_time));
    }
    output
}

fn tail_chase_stats(
    rows: &[(String, f64)],
    market_median_by_time: &HashMap<String, f64>,
) -> Option<TailChaseStats> {
    let mut stats = TailChaseStats::default();
    for window in rows.windows(2) {
        let current = &window[0];
        let next = &window[1];
        let (Some(current_median), Some(next_median)) = (
            market_median_by_time.get(&current.0),
            market_median_by_time.get(&next.0),
        ) else {
            continue;
        };
        let current_excess = current.1 - current_median;
        if current_excess <= 0.0 || !current_excess.is_finite() {
            continue;
        }
        let next_excess = next.1 - next_median;
        if !next_excess.is_finite() {
            continue;
        }
        stats.count += 1;
        stats.xy += current_excess * next_excess;
        stats.x2 += current_excess * current_excess;
        stats.y2 += next_excess * next_excess;
    }
    (stats.count > 0).then_some(stats)
}

fn is_tail_session_time(trade_time: &str) -> bool {
    intraday_time_in_range(trade_time, "13:31:00", "15:00:00")
}

fn minute_bar_return(open: Option<f64>, close: Option<f64>) -> Option<f64> {
    let (Some(open), Some(close)) = (clean_intraday_value(open), clean_intraday_value(close))
    else {
        return None;
    };
    if open.abs() <= f64::EPSILON {
        return None;
    }
    Some(close / open - 1.0)
}

fn median_f64(mut values: Vec<f64>) -> Option<f64> {
    values.retain(|value| value.is_finite());
    if values.is_empty() {
        return None;
    }
    values.sort_by(|left, right| left.total_cmp(right));
    let mid = values.len() / 2;
    if values.len() % 2 == 1 {
        Some(values[mid])
    } else {
        Some((values[mid - 1] + values[mid]) / 2.0)
    }
}

fn cosine_from_sums(xy: Option<f64>, x2: Option<f64>, y2: Option<f64>) -> Option<f64> {
    let (Some(xy), Some(x2), Some(y2)) = (clean(xy), clean(x2), clean(y2)) else {
        return None;
    };
    let denominator = x2 * y2;
    if denominator <= f64::EPSILON {
        return None;
    }
    Some(xy / denominator.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1e-12,
            "actual={actual}, expected={expected}"
        );
    }

    #[test]
    fn tail_session_time_keeps_last_ninety_minutes() {
        assert!(!is_tail_session_time("13:30:00"));
        assert!(is_tail_session_time("13:31:00"));
        assert!(is_tail_session_time("15:00:00"));
    }

    #[test]
    fn minute_bar_return_uses_close_over_open() {
        assert_close(
            minute_bar_return(Some(102.0), Some(103.02)).expect("return"),
            0.01,
        );
    }

    #[test]
    fn median_handles_even_and_odd_lengths() {
        assert_close(median_f64(vec![3.0, 1.0, 2.0]).expect("median"), 2.0);
        assert_close(median_f64(vec![4.0, 1.0, 2.0, 3.0]).expect("median"), 2.5);
    }

    #[test]
    fn tail_chase_stats_keep_positive_current_excess_only() {
        let rows = vec![
            ("13:31:00".to_string(), 0.02),
            ("13:32:00".to_string(), 0.01),
            ("13:33:00".to_string(), 0.03),
        ];
        let medians = HashMap::from([
            ("13:31:00".to_string(), 0.01),
            ("13:32:00".to_string(), 0.02),
            ("13:33:00".to_string(), 0.01),
        ]);

        let stats = tail_chase_stats(&rows, &medians).expect("stats");
        assert_eq!(stats.count, 1);
        assert_close(stats.xy, -0.0001);
        assert_close(stats.x2, 0.0001);
        assert_close(stats.y2, 0.0001);
    }

    #[test]
    fn cosine_from_sums_matches_direct_cosine() {
        assert_close(
            cosine_from_sums(Some(6.0), Some(9.0), Some(16.0)).expect("cosine"),
            0.5,
        );
        assert!(cosine_from_sums(Some(1.0), Some(0.0), Some(1.0)).is_none());
    }
}

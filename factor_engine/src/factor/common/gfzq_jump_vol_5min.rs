use std::collections::{BTreeMap, BTreeSet};

use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorRowKey, FactorSeries, FactorSpec,
    FactorValue, Frequency, IntradayDailyRawRequest, IntradayDailyRawSeries, IntradayDailyRawSpec,
    Lookback,
};
use crate::data::{DataPool, Table};
use crate::error::Result;
use crate::factor::common::stock_daily_ops::neutralize_size_sector;
use crate::factor::common::stock_daily_raw_ids::{
    RJVP_5MIN_RAW_ID, RLJVP_5MIN_RAW_ID, SRJV_5MIN_RAW_ID, SRLJV_5MIN_RAW_ID,
};
use crate::factor::common::{clean_intraday_value, stock_minute_raw_spec};
use crate::operators::{cs_zscore, ts_mean};

pub const VERSION: &str = "0.1.0";
pub const RAW_VERSION: &str = "0.1.0";
pub const PROVIDER_KEY: &str = "gfzq_jump_vol_5min_provider";

const RAW_WINDOW_DAYS: usize = 1;
const WEEK_WINDOW: usize = 5;
const MIN_PERIODS: usize = 1;
const FIVE_MINUTE_RETURN_COUNT: usize = 48;
const M: f64 = 2.0 / 3.0;
const MU_M_NEG_2_OVER_M: f64 = 1.9357924048803463;
const LARGE_JUMP_ALPHA: f64 = 4.0;

#[derive(Clone, Copy, Debug)]
pub struct GfzqJumpVolFactorDef {
    pub id: &'static str,
    pub alias: &'static str,
    pub name: &'static str,
    pub raw_id: &'static str,
}

#[derive(Clone, Copy, Debug, Default)]
struct JumpVolValues {
    srjv: Option<f64>,
    rjvp: Option<f64>,
    rljvp: Option<f64>,
    srljv: Option<f64>,
}

pub fn all_raw_ids() -> [&'static str; 4] {
    [
        SRJV_5MIN_RAW_ID,
        RJVP_5MIN_RAW_ID,
        RLJVP_5MIN_RAW_ID,
        SRLJV_5MIN_RAW_ID,
    ]
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

pub fn factor_spec(def: GfzqJumpVolFactorDef) -> FactorSpec {
    FactorSpec {
        id: def.id.to_string(),
        aliases: vec![def.alias.to_string()],
        name: def.name.to_string(),
        asset_class: AssetClass::Stock,
        frequency: Frequency::Daily,
        version: VERSION.to_string(),
        tags: tags(),
        description: format!(
            "{} GFZQ 5-minute jump volatility factor, 5-day averaged, z-scored, and neutralized by Barra SIZE and SW sector.",
            def.name
        ),
        dependencies: vec![
            DataRequest::new(DatasetId::StockBarraDaily, &["SIZE"]),
            DataRequest::new(DatasetId::StockSwClassification, &["l1_code"]),
        ],
        intraday_raw_dependencies: vec![IntradayDailyRawRequest::new(
            def.raw_id,
            WEEK_WINDOW - 1,
        )],
        lookback: Lookback {
            trading_days: WEEK_WINDOW - 1,
        },
    }
}

pub fn compute_factor(def: GfzqJumpVolFactorDef, data: &DataPool) -> Result<FactorSeries> {
    let panel = data.intraday_daily_raw_panel(def.raw_id)?;
    let raw = panel.column(def.raw_id)?;
    let smoothed = raw.ts(|values| ts_mean(values, WEEK_WINDOW, MIN_PERIODS))?;
    let standardized = smoothed.cs(cs_zscore)?;
    let factor = neutralize_size_sector(&standardized, &panel, data)?;
    Ok(factor.to_factor_series(factor_spec(def)))
}

#[macro_export]
macro_rules! define_gfzq_jump_vol_5min_factor {
    ($struct_name:ident, $id:expr, $alias:expr, $name:expr, $raw_id:expr) => {
        const DEF: $crate::factor::common::gfzq_jump_vol_5min::GfzqJumpVolFactorDef =
            $crate::factor::common::gfzq_jump_vol_5min::GfzqJumpVolFactorDef {
                id: $id,
                alias: $alias,
                name: $name,
                raw_id: $raw_id,
            };

        pub struct $struct_name;

        pub fn create() -> Box<dyn $crate::factor::Factor> {
            Box::new($struct_name)
        }

        impl $crate::factor::Factor for $struct_name {
            fn spec(&self) -> $crate::core::FactorSpec {
                $crate::factor::common::gfzq_jump_vol_5min::factor_spec(DEF)
            }

            fn intraday_raw_specs(&self) -> Vec<$crate::core::IntradayDailyRawSpec> {
                vec![$crate::factor::common::gfzq_jump_vol_5min::raw_spec(
                    DEF.raw_id,
                )]
            }

            fn intraday_raw_provider_key(&self, _raw_id: &str) -> String {
                $crate::factor::common::gfzq_jump_vol_5min::PROVIDER_KEY.to_string()
            }

            fn minute_compute_many(
                &self,
                raw_ids: &[String],
                context: &$crate::core::FactorContext,
                data: &$crate::data::DataPool,
            ) -> $crate::error::Result<Vec<$crate::core::IntradayDailyRawSeries>> {
                $crate::factor::common::gfzq_jump_vol_5min::minute_compute_many(
                    raw_ids, context, data,
                )
            }

            fn compute(
                &self,
                _context: &$crate::core::FactorContext,
                data: &$crate::data::DataPool,
            ) -> $crate::error::Result<$crate::core::FactorSeries> {
                $crate::factor::common::gfzq_jump_vol_5min::compute_factor(DEF, data)
            }
        }
    };
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

    let mut output = all_raw_ids()
        .iter()
        .map(|raw_id| (*raw_id, Vec::<FactorValue>::new()))
        .collect::<BTreeMap<_, _>>();

    for trade_date in &context.target_dates {
        let Some(table) = data.minute(DatasetId::StockMinute1m, *trade_date) else {
            continue;
        };
        for (ts_code, returns) in five_minute_log_returns_by_stock(table)? {
            let key = FactorRowKey::Daily {
                trade_date: *trade_date,
                ts_code,
            };
            let values = jump_vol_values(&returns);
            push_raw_value(&mut output, &requested, SRJV_5MIN_RAW_ID, &key, values.srjv);
            push_raw_value(&mut output, &requested, RJVP_5MIN_RAW_ID, &key, values.rjvp);
            push_raw_value(
                &mut output,
                &requested,
                RLJVP_5MIN_RAW_ID,
                &key,
                values.rljvp,
            );
            push_raw_value(
                &mut output,
                &requested,
                SRLJV_5MIN_RAW_ID,
                &key,
                values.srljv,
            );
        }
    }

    Ok(all_raw_ids()
        .iter()
        .filter(|raw_id| requested.contains(**raw_id))
        .map(|raw_id| IntradayDailyRawSeries {
            spec: raw_spec(raw_id),
            values: output.remove(raw_id).unwrap_or_default(),
        })
        .collect())
}

fn tags() -> Vec<String> {
    [
        "GFZQ",
        "price_volume",
        "jump",
        "volatility",
        "intraday",
        "5min",
        "neutralize",
        "daily",
    ]
    .iter()
    .map(|value| value.to_string())
    .collect()
}

fn five_minute_log_returns_by_stock(table: &Table) -> Result<BTreeMap<String, Vec<Option<f64>>>> {
    let ts_codes = table.required_utf8("ts_code")?;
    let trade_times = table.required_utf8("trade_time")?;
    let close = table.required_f64_cast("close")?;

    let mut close_by_stock = BTreeMap::<String, BTreeMap<i32, f64>>::new();
    let anchors = anchor_seconds();
    let anchor_set = anchors.iter().copied().collect::<BTreeSet<_>>();

    for idx in 0..table.len {
        let (Some(ts_code), Some(trade_time)) =
            (ts_codes[idx].clone(), trade_times[idx].as_deref())
        else {
            continue;
        };
        let Some(seconds) = time_to_seconds(trade_time).filter(|value| anchor_set.contains(value))
        else {
            continue;
        };
        let Some(close) = clean_intraday_value(close[idx]).filter(|value| *value > 0.0) else {
            continue;
        };
        close_by_stock
            .entry(ts_code)
            .or_default()
            .insert(seconds, close);
    }

    Ok(close_by_stock
        .into_iter()
        .map(|(ts_code, closes)| {
            let returns = anchors
                .windows(2)
                .map(|pair| {
                    let (Some(previous), Some(current)) =
                        (closes.get(&pair[0]), closes.get(&pair[1]))
                    else {
                        return None;
                    };
                    finite_value(current.ln() - previous.ln())
                })
                .collect::<Vec<_>>();
            (ts_code, returns)
        })
        .collect())
}

fn jump_vol_values(returns: &[Option<f64>]) -> JumpVolValues {
    let valid_returns = returns
        .iter()
        .filter_map(|value| *value)
        .collect::<Vec<_>>();
    let n_valid = valid_returns.len();
    if n_valid == 0 {
        return JumpVolValues::default();
    }

    let rv_pos = valid_returns
        .iter()
        .filter(|value| **value > 0.0)
        .map(|value| value * value)
        .sum::<f64>();
    let rv_neg = valid_returns
        .iter()
        .filter(|value| **value < 0.0)
        .map(|value| value * value)
        .sum::<f64>();
    let Some(iv) = tripower_iv(returns) else {
        return JumpVolValues::default();
    };

    let rjvp = (rv_pos - iv / 2.0).max(0.0);
    let rjvn = (rv_neg - iv / 2.0).max(0.0);
    let srjv = rjvp - rjvn;
    let gamma = LARGE_JUMP_ALPHA * (n_valid as f64).powf(-0.49) * iv.sqrt();
    let large_pos_sum = valid_returns
        .iter()
        .filter(|value| **value >= gamma)
        .map(|value| value * value)
        .sum::<f64>();
    let large_neg_sum = valid_returns
        .iter()
        .filter(|value| **value <= -gamma)
        .map(|value| value * value)
        .sum::<f64>();
    let rljvp = rjvp.min(large_pos_sum);
    let rljvn = rjvn.min(large_neg_sum);
    let srljv = rljvp - rljvn;

    JumpVolValues {
        srjv: finite_value(srjv),
        rjvp: finite_value(rjvp),
        rljvp: finite_value(rljvp),
        srljv: finite_value(srljv),
    }
}

fn tripower_iv(returns: &[Option<f64>]) -> Option<f64> {
    let mut sum = 0.0;
    let mut count = 0usize;
    for idx in 2..returns.len() {
        let (Some(current), Some(previous), Some(previous2)) =
            (returns[idx], returns[idx - 1], returns[idx - 2])
        else {
            continue;
        };
        sum += current.abs().powf(M) * previous.abs().powf(M) * previous2.abs().powf(M);
        count += 1;
    }
    if count == 0 {
        return None;
    }
    finite_value(MU_M_NEG_2_OVER_M * sum)
}

fn push_raw_value(
    output: &mut BTreeMap<&'static str, Vec<FactorValue>>,
    requested: &BTreeSet<&str>,
    raw_id: &'static str,
    key: &FactorRowKey,
    value: Option<f64>,
) {
    if requested.contains(raw_id) {
        output.entry(raw_id).or_default().push(FactorValue {
            key: key.clone(),
            value,
        });
    }
}

fn anchor_seconds() -> Vec<i32> {
    let mut anchors = Vec::with_capacity(FIVE_MINUTE_RETURN_COUNT + 1);
    anchors.push(seconds(9, 30));
    let mut minute = 35;
    while minute <= 150 {
        let (hour, minute_in_hour) = if minute < 60 {
            (9, minute)
        } else {
            (10 + (minute - 60) / 60, (minute - 60) % 60)
        };
        anchors.push(seconds(hour, minute_in_hour));
        minute += 5;
    }
    let mut afternoon_minute = 5;
    while afternoon_minute <= 120 {
        let (hour, minute_in_hour) = if afternoon_minute < 60 {
            (13, afternoon_minute)
        } else {
            (
                14 + (afternoon_minute - 60) / 60,
                (afternoon_minute - 60) % 60,
            )
        };
        anchors.push(seconds(hour, minute_in_hour));
        afternoon_minute += 5;
    }
    anchors
}

fn seconds(hour: i32, minute: i32) -> i32 {
    hour * 3600 + minute * 60
}

fn time_to_seconds(value: &str) -> Option<i32> {
    let time = value.split_whitespace().last().unwrap_or(value).trim();
    let mut parts = time.split(':');
    let hour = parts.next()?.parse::<i32>().ok()?;
    let minute = parts.next()?.parse::<i32>().ok()?;
    Some(seconds(hour, minute))
}

fn finite_value(value: f64) -> Option<f64> {
    value.is_finite().then_some(value)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use crate::data::ColumnData;

    use super::*;

    fn assert_close(actual: Option<f64>, expected: f64) {
        let actual = actual.expect("expected value");
        assert!(
            (actual - expected).abs() < 1e-12,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn gfzq_jump_5min_returns_use_dbzq_anchors_and_lunch_bridge() {
        let table = minute_table(vec![
            ("000001.SZ", "09:30:00", 100.0),
            ("000001.SZ", "09:35:00", 101.0),
            ("000001.SZ", "11:25:00", 102.0),
            ("000001.SZ", "11:30:00", 103.0),
            ("000001.SZ", "13:05:00", 104.0),
            ("000001.SZ", "14:55:00", 105.0),
            ("000001.SZ", "15:00:00", 106.0),
        ]);
        let by_stock = five_minute_log_returns_by_stock(&table).expect("returns");
        let returns = by_stock.get("000001.SZ").expect("stock");
        assert_eq!(returns.len(), FIVE_MINUTE_RETURN_COUNT);
        assert_close(returns[0], (101.0_f64 / 100.0).ln());
        assert_close(returns[23], (103.0_f64 / 102.0).ln());
        assert_close(returns[24], (104.0_f64 / 103.0).ln());
        assert_close(returns[47], (106.0_f64 / 105.0).ln());
    }

    #[test]
    fn gfzq_jump_vol_values_match_manual_rjv_and_large_jump() {
        let returns = vec![Some(0.1), Some(-0.2), Some(0.3), Some(-0.4)];
        let values = jump_vol_values(&returns);
        let iv = MU_M_NEG_2_OVER_M
            * (0.3_f64.abs().powf(M) * 0.2_f64.abs().powf(M) * 0.1_f64.abs().powf(M)
                + 0.4_f64.abs().powf(M) * 0.3_f64.abs().powf(M) * 0.2_f64.abs().powf(M));
        let rv_pos = 0.1_f64.powi(2) + 0.3_f64.powi(2);
        let rv_neg = 0.2_f64.powi(2) + 0.4_f64.powi(2);
        let rjvp = (rv_pos - iv / 2.0).max(0.0);
        let rjvn = (rv_neg - iv / 2.0).max(0.0);
        let gamma = LARGE_JUMP_ALPHA * 4.0_f64.powf(-0.49) * iv.sqrt();
        let large_pos = [0.1_f64, 0.3]
            .iter()
            .filter(|value| **value >= gamma)
            .map(|value| value * value)
            .sum::<f64>();
        let large_neg = [-0.2_f64, -0.4]
            .iter()
            .filter(|value| **value <= -gamma)
            .map(|value| value * value)
            .sum::<f64>();
        assert_close(values.rjvp, rjvp);
        assert_close(values.srjv, rjvp - rjvn);
        assert_close(values.rljvp, rjvp.min(large_pos));
        assert_close(values.srljv, rjvp.min(large_pos) - rjvn.min(large_neg));
    }

    #[test]
    fn gfzq_jump_iv_requires_one_valid_triplet() {
        let returns = vec![Some(0.1), None, Some(0.2), Some(0.3)];
        let values = jump_vol_values(&returns);
        assert_eq!(values.rjvp, None);
        assert_eq!(values.srjv, None);
        assert_eq!(values.rljvp, None);
        assert_eq!(values.srljv, None);
    }

    fn minute_table(rows: Vec<(&str, &str, f64)>) -> Table {
        let len = rows.len();
        Table::new(BTreeMap::from([
            (
                "ts_code".to_string(),
                ColumnData::Utf8(
                    rows.iter()
                        .map(|row| Some(row.0.to_string()))
                        .collect::<Vec<_>>(),
                ),
            ),
            (
                "trade_time".to_string(),
                ColumnData::Utf8(
                    rows.iter()
                        .map(|row| Some(row.1.to_string()))
                        .collect::<Vec<_>>(),
                ),
            ),
            (
                "close".to_string(),
                ColumnData::F64(rows.iter().map(|row| Some(row.2)).collect::<Vec<_>>()),
            ),
        ]))
        .unwrap_or_else(|err| panic!("valid table with {len} rows: {err}"))
    }
}

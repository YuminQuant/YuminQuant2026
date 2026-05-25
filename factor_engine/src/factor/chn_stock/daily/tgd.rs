use std::collections::{BTreeMap, BTreeSet};

use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorRowKey, FactorSeries, FactorSpec,
    FactorValue, Frequency, IntradayDailyRawRequest, IntradayDailyRawSeries, IntradayDailyRawSpec,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::stock_daily_ops::neutralize_size_sector;
use crate::factor::common::{clean_intraday_value, stock_minute_raw_spec};
use crate::factor::Factor;
use crate::operators::{cs_regression_residual, ts_mean};

const VERSION: &str = "0.1.0";
const RAW_VERSION: &str = "0.1.0";
const PROVIDER_KEY: &str = "kyzq_tgd_provider";
const WINDOW: usize = 20;
const RAW_WINDOW_DAYS: usize = 1;

const GU_RAW_ID: &str = "daily_kyzq_tgd_gu_raw";
const GD_RAW_ID: &str = "daily_kyzq_tgd_gd_raw";
const RBAR_U_RAW_ID: &str = "daily_kyzq_tgd_rbar_u_raw";
const RBAR_D_RAW_ID: &str = "daily_kyzq_tgd_rbar_d_raw";
const R1_RAW_ID: &str = "daily_kyzq_tgd_r1_raw";
const R2_RAW_ID: &str = "daily_kyzq_tgd_r2_raw";

pub struct StockDailyTgd;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyTgd)
}

impl Factor for StockDailyTgd {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "tgd".to_string(),
            aliases: vec!["TGD".to_string()],
            name: "tgd".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: tags(),
            description: "KYZQ time-gravity deviation factor from 1-minute up/down return time barycenters, cross-sectionally residualized by return-structure controls and then 20-day averaged, neutralized by Barra SIZE and SW sector.".to_string(),
            dependencies: vec![
                DataRequest::new(DatasetId::StockDailyPv, &["open", "pre_close"]),
                DataRequest::new(DatasetId::StockBarraDaily, &["SIZE"]),
                DataRequest::new(DatasetId::StockSwClassification, &["l1_code"]),
            ],
            intraday_raw_dependencies: all_raw_ids()
                .iter()
                .map(|raw_id| IntradayDailyRawRequest::new(raw_id, WINDOW - 1))
                .collect(),
            lookback: Lookback {
                trading_days: WINDOW - 1,
            },
        }
    }

    fn intraday_raw_specs(&self) -> Vec<IntradayDailyRawSpec> {
        all_raw_ids()
            .iter()
            .map(|raw_id| raw_spec(raw_id))
            .collect()
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

            for (ts_code, mut indices) in grouped {
                indices.sort_by(|left, right| trade_times[*left].cmp(&trade_times[*right]));
                let raw = tgd_daily_raw_from_indices(&indices, &trade_times, &close);
                let key = FactorRowKey::Daily {
                    trade_date: *trade_date,
                    ts_code,
                };
                push_requested(&mut values, &requested, GU_RAW_ID, &key, raw.gu);
                push_requested(&mut values, &requested, GD_RAW_ID, &key, raw.gd);
                push_requested(&mut values, &requested, RBAR_U_RAW_ID, &key, raw.rbar_u);
                push_requested(&mut values, &requested, RBAR_D_RAW_ID, &key, raw.rbar_d);
                push_requested(&mut values, &requested, R1_RAW_ID, &key, raw.r1);
                push_requested(&mut values, &requested, R2_RAW_ID, &key, raw.r2);
            }
        }

        let mut output = Vec::new();
        for raw_id in all_raw_ids() {
            if requested.contains(raw_id) {
                output.push(IntradayDailyRawSeries {
                    spec: raw_spec(raw_id),
                    values: values.remove(raw_id).unwrap_or_default(),
                });
            }
        }
        Ok(output)
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let panel = data.intraday_daily_raw_panel(GU_RAW_ID)?;
        let gu = panel.column(GU_RAW_ID)?;
        let gd = panel.column(GD_RAW_ID)?;
        let rbar_u = panel.column(RBAR_U_RAW_ID)?;
        let rbar_d = panel.column(RBAR_D_RAW_ID)?;
        let r1 = panel.column(R1_RAW_ID)?;
        let r2 = panel.column(R2_RAW_ID)?;
        let open = panel.column_from_table(data.daily(DatasetId::StockDailyPv)?, "open")?;
        let pre_close =
            panel.column_from_table(data.daily(DatasetId::StockDailyPv)?, "pre_close")?;
        let overnight = open.zip_binary(&pre_close, overnight_return)?;

        let epsilon_u = gu.cs_neutralize_regression(&[&rbar_u, &r1, &r2, &overnight], None)?;
        let epsilon_d = gd.cs_neutralize_regression(&[&rbar_d, &r1, &r2, &overnight], None)?;
        let residual = epsilon_d.cs_binary(&epsilon_u, cs_regression_residual)?;
        let raw = residual.ts(|series| ts_mean(series, WINDOW, 1))?;
        let factor = neutralize_size_sector(&raw, &panel, data)?;
        Ok(factor.to_factor_series(self.spec()))
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct TgdRaw {
    gu: Option<f64>,
    gd: Option<f64>,
    rbar_u: Option<f64>,
    rbar_d: Option<f64>,
    r1: Option<f64>,
    r2: Option<f64>,
}

fn all_raw_ids() -> [&'static str; 6] {
    [
        GU_RAW_ID,
        GD_RAW_ID,
        RBAR_U_RAW_ID,
        RBAR_D_RAW_ID,
        R1_RAW_ID,
        R2_RAW_ID,
    ]
}

fn raw_spec(raw_id: &str) -> IntradayDailyRawSpec {
    stock_minute_raw_spec(raw_id, RAW_VERSION, &["close"], RAW_WINDOW_DAYS)
}

fn tags() -> Vec<String> {
    [
        "KYZQ",
        "return",
        "time_gravity",
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

fn tgd_daily_raw_from_indices(
    indices: &[usize],
    trade_times: &[Option<String>],
    close: &[Option<f64>],
) -> TgdRaw {
    let mut returns = [None; 240];
    let mut previous_close = None;
    let mut close_0930 = None;
    let mut close_1000 = None;
    let mut close_1030 = None;

    for idx in indices {
        let current_close = clean_intraday_value(close[*idx]).filter(|value| *value > 0.0);
        if let Some(trade_time) = trade_times[*idx].as_deref() {
            if time_to_minutes(trade_time) == Some(9 * 60 + 30) {
                close_0930 = current_close;
            }
            if time_to_minutes(trade_time) == Some(10 * 60) {
                close_1000 = current_close;
            }
            if time_to_minutes(trade_time) == Some(10 * 60 + 30) {
                close_1030 = current_close;
            }
            if let Some(minute_idx) = minute_index(trade_time) {
                returns[minute_idx] = minute_return(previous_close, current_close);
            }
        }
        if current_close.is_some() {
            previous_close = current_close;
        }
    }

    let r1 = simple_return(close_0930, close_1000);
    let r2 = simple_return(close_1000, close_1030);
    tgd_daily_raw_from_returns(&returns, r1, r2)
}

fn tgd_daily_raw_from_returns(
    returns: &[Option<f64>; 240],
    r1: Option<f64>,
    r2: Option<f64>,
) -> TgdRaw {
    let mut up_weight_sum = 0.0;
    let mut up_time_weight_sum = 0.0;
    let mut up_count = 0usize;
    let mut down_weight_sum = 0.0;
    let mut down_time_weight_sum = 0.0;
    let mut down_count = 0usize;

    for (idx, value) in returns.iter().enumerate() {
        let Some(ret) = clean_intraday_value(*value) else {
            continue;
        };
        if ret > 0.0 {
            let weight = ret.abs();
            up_weight_sum += weight;
            up_time_weight_sum += (idx + 1) as f64 * weight;
            up_count += 1;
        } else if ret < 0.0 {
            let weight = ret.abs();
            down_weight_sum += weight;
            down_time_weight_sum += (idx + 1) as f64 * weight;
            down_count += 1;
        }
    }

    TgdRaw {
        gu: (up_weight_sum > f64::EPSILON).then_some(up_time_weight_sum / up_weight_sum),
        gd: (down_weight_sum > f64::EPSILON).then_some(down_time_weight_sum / down_weight_sum),
        rbar_u: (up_count > 0).then_some(up_weight_sum / up_count as f64),
        rbar_d: (down_count > 0).then_some(down_weight_sum / down_count as f64),
        r1,
        r2,
    }
}

fn overnight_return(open: Option<f64>, pre_close: Option<f64>) -> Option<f64> {
    simple_return(pre_close, open)
}

fn simple_return(start: Option<f64>, end: Option<f64>) -> Option<f64> {
    let (Some(start), Some(end)) = (clean_intraday_value(start), clean_intraday_value(end)) else {
        return None;
    };
    if start.abs() <= f64::EPSILON {
        return None;
    }
    let value = end / start - 1.0;
    value.is_finite().then_some(value)
}

fn minute_return(previous_close: Option<f64>, current_close: Option<f64>) -> Option<f64> {
    simple_return(previous_close, current_close)
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

fn push_requested(
    values: &mut BTreeMap<&'static str, Vec<FactorValue>>,
    requested: &BTreeSet<&str>,
    raw_id: &'static str,
    key: &FactorRowKey,
    value: Option<f64>,
) {
    if requested.contains(raw_id) {
        values
            .get_mut(raw_id)
            .expect("raw id initialized")
            .push(FactorValue {
                key: key.clone(),
                value,
            });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: Option<f64>, expected: f64) {
        let actual = actual.expect("value");
        assert!(
            (actual - expected).abs() < 1e-12,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn kyzq_tgd_raw_uses_one_based_time_gravity() {
        let mut returns = [None; 240];
        returns[0] = Some(0.01);
        returns[2] = Some(-0.02);
        returns[3] = Some(0.03);

        let raw = tgd_daily_raw_from_returns(&returns, Some(0.1), Some(0.2));

        assert_close(raw.gu, (1.0 * 0.01 + 4.0 * 0.03) / 0.04);
        assert_close(raw.gd, 3.0);
        assert_close(raw.rbar_u, 0.02);
        assert_close(raw.rbar_d, 0.02);
        assert_close(raw.r1, 0.1);
        assert_close(raw.r2, 0.2);
    }

    #[test]
    fn kyzq_tgd_minute_index_uses_regular_session_numbering() {
        assert_eq!(minute_index("09:31:00"), Some(0));
        assert_eq!(minute_index("11:30:00"), Some(119));
        assert_eq!(minute_index("13:01:00"), Some(120));
        assert_eq!(minute_index("15:00:00"), Some(239));
        assert_eq!(minute_index("09:30:00"), None);
    }

    #[test]
    fn kyzq_tgd_uses_0930_anchor_for_r1_and_minute_return() {
        assert_close(simple_return(Some(100.0), Some(101.0)), 0.01);
        assert_eq!(simple_return(Some(0.0), Some(101.0)), None);
        assert_close(overnight_return(Some(11.0), Some(10.0)), 0.1);
    }

    #[test]
    fn kyzq_tgd_factor_spec_has_kyzq_tag() {
        let spec = StockDailyTgd.spec();
        assert_eq!(spec.id, "tgd");
        assert!(spec.tags.iter().any(|tag| tag == "KYZQ"));
    }
}

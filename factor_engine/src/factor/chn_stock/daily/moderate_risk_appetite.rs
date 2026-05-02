use std::collections::{BTreeMap, BTreeSet};

use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorRowKey, FactorSeries, FactorSpec,
    FactorValue, Frequency, IntradayDailyRawRequest, IntradayDailyRawSeries, IntradayDailyRawSpec,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::{
    clean_intraday_value, intraday_time_in_range, stock_minute_raw_spec, ClassificationLevel,
    ClassificationMap,
};
use crate::factor::Factor;
use crate::operators::{cs_mean, cs_zscore, ts_mean, ts_std_dev};

pub const SPARKLE_VOLATILITY_RAW_ID: &str = "daily_sparkle_volatility";
pub const SPARKLE_RETURN_RAW_ID: &str = "daily_sparkle_return";

const RAW_VERSION: &str = "0.1.0";
const WINDOW: usize = 20;
const SPARKLE_WINDOW: usize = 5;

pub struct StockDailyModerateRiskAppetite;

#[derive(Clone, Copy, Debug)]
struct SparkleValues {
    volatility: Option<f64>,
    return_mean: Option<f64>,
}

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyModerateRiskAppetite)
}

fn raw_spec(raw_id: &str) -> IntradayDailyRawSpec {
    stock_minute_raw_spec(raw_id, RAW_VERSION, &["close", "vol"], 1)
}

impl Factor for StockDailyModerateRiskAppetite {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "moderate_risk_appetite".to_string(),
            aliases: Vec::new(),
            name: "Moderate Risk Appetite".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: "0.1.0".to_string(),
            tags: [
                "price_volume",
                "return",
                "volume",
                "volatility",
                "intraday",
                "minute_agg",
                "composite",
                "neutralize",
                "barra",
                "size",
                "sector",
                "daily",
                "FZZQ",
            ]
            .iter()
            .map(|value| value.to_string())
            .collect(),
            description: "Composite moderate risk appetite factor from intraday volume-spike return and volatility responses, neutralized by Barra SIZE and SW sector.".to_string(),
            dependencies: vec![
                DataRequest::new(DatasetId::StockBarraDaily, &["SIZE"]),
                DataRequest::new(DatasetId::StockSwClassification, &["l1_code"]),
            ],
            intraday_raw_dependencies: vec![
                IntradayDailyRawRequest::new(SPARKLE_VOLATILITY_RAW_ID, WINDOW - 1),
                IntradayDailyRawRequest::new(SPARKLE_RETURN_RAW_ID, WINDOW - 1),
            ],
            lookback: Lookback {
                trading_days: WINDOW - 1,
            },
        }
    }

    fn intraday_raw_specs(&self) -> Vec<IntradayDailyRawSpec> {
        vec![
            raw_spec(SPARKLE_VOLATILITY_RAW_ID),
            raw_spec(SPARKLE_RETURN_RAW_ID),
        ]
    }

    fn minute_compute(
        &self,
        raw_id: &str,
        context: &FactorContext,
        data: &DataPool,
    ) -> Result<Option<IntradayDailyRawSeries>> {
        let raw_ids = vec![raw_id.to_string()];
        Ok(self
            .minute_compute_many(&raw_ids, context, data)?
            .into_iter()
            .next())
    }

    fn minute_compute_many(
        &self,
        raw_ids: &[String],
        context: &FactorContext,
        data: &DataPool,
    ) -> Result<Vec<IntradayDailyRawSeries>> {
        let requested = raw_ids.iter().map(String::as_str).collect::<BTreeSet<_>>();
        let wants_volatility = requested.contains(SPARKLE_VOLATILITY_RAW_ID);
        let wants_return = requested.contains(SPARKLE_RETURN_RAW_ID);
        if !wants_volatility && !wants_return {
            return Ok(Vec::new());
        }

        let mut volatility_values = Vec::new();
        let mut return_values = Vec::new();
        for trade_date in &context.target_dates {
            let Some(table) = data.minute(DatasetId::StockMinute1m, *trade_date) else {
                continue;
            };
            let ts_codes = table.required_utf8("ts_code")?;
            let trade_times = table.required_utf8("trade_time")?;
            let close = table.required_f64_cast("close")?;
            let volume = table.required_f64_cast("vol")?;
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
                let sparkle = sparkle_values_from_rows(&indices, trade_times, &close, &volume);
                if wants_volatility {
                    volatility_values.push(FactorValue {
                        key: FactorRowKey::Daily {
                            trade_date: *trade_date,
                            ts_code: ts_code.clone(),
                        },
                        value: sparkle.volatility,
                    });
                }
                if wants_return {
                    return_values.push(FactorValue {
                        key: FactorRowKey::Daily {
                            trade_date: *trade_date,
                            ts_code,
                        },
                        value: sparkle.return_mean,
                    });
                }
            }
        }

        let mut output = Vec::new();
        if wants_volatility {
            output.push(IntradayDailyRawSeries {
                spec: raw_spec(SPARKLE_VOLATILITY_RAW_ID),
                values: volatility_values,
            });
        }
        if wants_return {
            output.push(IntradayDailyRawSeries {
                spec: raw_spec(SPARKLE_RETURN_RAW_ID),
                values: return_values,
            });
        }
        Ok(output)
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let sector_map = ClassificationMap::from_table(
            data.daily(DatasetId::StockSwClassification)?,
            ClassificationLevel::Sector,
        )?;
        let panel = data.intraday_daily_raw_panel(SPARKLE_VOLATILITY_RAW_ID)?;
        let size = panel.column_from_table(data.daily(DatasetId::StockBarraDaily)?, "SIZE")?;

        let vol_distance = panel
            .column(SPARKLE_VOLATILITY_RAW_ID)?
            .cs(cs_abs_distance_from_mean)?;
        let ret_distance = panel
            .column(SPARKLE_RETURN_RAW_ID)?
            .cs(cs_abs_distance_from_mean)?;

        let vol_mean20 = vol_distance.ts(|values| ts_mean(values, WINDOW, WINDOW))?;
        let vol_std20 = vol_distance.ts(|values| ts_std_dev(values, WINDOW, WINDOW))?;
        let ret_mean20 = ret_distance.ts(|values| ts_mean(values, WINDOW, WINDOW))?;
        let ret_std20 = ret_distance.ts(|values| ts_std_dev(values, WINDOW, WINDOW))?;

        let vol_component = average_pair(&vol_mean20.cs(cs_zscore)?, &vol_std20.cs(cs_zscore)?)?;
        let ret_component = average_pair(&ret_mean20.cs(cs_zscore)?, &ret_std20.cs(cs_zscore)?)?;
        let raw_factor =
            average_pair(&vol_component.cs(cs_zscore)?, &ret_component.cs(cs_zscore)?)?;
        let neutralized = raw_factor.cs_neutralize_regression_by_group(
            &[&size],
            None,
            |trade_date, ts_codes| sector_map.groups_for(trade_date, ts_codes),
        )?;

        Ok(neutralized.to_factor_series(self.spec()))
    }
}

fn average_pair(
    left: &crate::factor::common::PanelColumn,
    right: &crate::factor::common::PanelColumn,
) -> Result<crate::factor::common::PanelColumn> {
    left.zip_binary(right, |left, right| match (clean(left), clean(right)) {
        (Some(left), Some(right)) => Some((left + right) / 2.0),
        _ => None,
    })
}

fn cs_abs_distance_from_mean(values: &[Option<f64>]) -> Vec<Option<f64>> {
    let means = cs_mean(values);
    values
        .iter()
        .zip(means.iter())
        .map(|(value, mean)| match (clean(*value), clean(*mean)) {
            (Some(value), Some(mean)) => Some((value - mean).abs()),
            _ => None,
        })
        .collect()
}

fn sparkle_values_from_rows(
    indices: &[usize],
    trade_times: &[Option<String>],
    close: &[Option<f64>],
    volume: &[Option<f64>],
) -> SparkleValues {
    let returns = minute_returns(indices, close);
    let mut volume_increases = Vec::<(usize, f64)>::new();
    for (pos, idx) in indices.iter().enumerate() {
        let Some(trade_time) = trade_times[*idx].as_deref() else {
            continue;
        };
        if pos == 0 || !intraday_time_in_range(trade_time, "09:31:00", "14:57:00") {
            continue;
        }
        let prev_idx = indices[pos - 1];
        let (Some(current), Some(previous)) = (
            clean_intraday_value(volume[*idx]),
            clean_intraday_value(volume[prev_idx]),
        ) else {
            continue;
        };
        volume_increases.push((pos, current - previous));
    }

    let Some((increase_mean, increase_std)) =
        mean_std(volume_increases.iter().map(|(_, value)| *value))
    else {
        return SparkleValues {
            volatility: None,
            return_mean: None,
        };
    };
    let threshold = increase_mean + increase_std;

    let mut sparkle_return_sum = 0.0;
    let mut sparkle_return_count = 0usize;
    let mut sparkle_vol_sum = 0.0;
    let mut sparkle_vol_count = 0usize;

    for (pos, increase) in volume_increases {
        if increase <= threshold {
            continue;
        }
        if let Some(ret) = returns[pos] {
            sparkle_return_sum += ret;
            sparkle_return_count += 1;
        }
        if pos + SPARKLE_WINDOW <= indices.len() {
            let window = &returns[pos..pos + SPARKLE_WINDOW];
            if window.iter().all(Option::is_some) {
                let volatility = population_std(window.iter().filter_map(|value| *value));
                sparkle_vol_sum += volatility;
                sparkle_vol_count += 1;
            }
        }
    }

    SparkleValues {
        volatility: (sparkle_vol_count > 0).then_some(sparkle_vol_sum / sparkle_vol_count as f64),
        return_mean: (sparkle_return_count > 0)
            .then_some(sparkle_return_sum / sparkle_return_count as f64),
    }
}

fn minute_returns(indices: &[usize], close: &[Option<f64>]) -> Vec<Option<f64>> {
    let mut returns = vec![None; indices.len()];
    for pos in 1..indices.len() {
        let idx = indices[pos];
        let prev_idx = indices[pos - 1];
        let (Some(current), Some(previous)) = (
            clean_intraday_value(close[idx]),
            clean_intraday_value(close[prev_idx]),
        ) else {
            continue;
        };
        if previous.abs() <= f64::EPSILON {
            continue;
        }
        returns[pos] = Some(current / previous - 1.0);
    }
    returns
}

fn mean_std(values: impl IntoIterator<Item = f64>) -> Option<(f64, f64)> {
    let values = values
        .into_iter()
        .filter(|value| !value.is_nan())
        .collect::<Vec<_>>();
    if values.is_empty() {
        return None;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / values.len() as f64;
    Some((mean, variance.sqrt()))
}

fn population_std(values: impl IntoIterator<Item = f64>) -> f64 {
    mean_std(values)
        .map(|(_, std)| std)
        .expect("population_std requires at least one value")
}

fn clean(value: Option<f64>) -> Option<f64> {
    value.filter(|value| !value.is_nan())
}

#[cfg(test)]
mod tests {
    use super::{average_pair, cs_abs_distance_from_mean, sparkle_values_from_rows};
    use crate::core::{AssetClass, FactorContext, Frequency};
    use crate::factor::common::DailyPanel;

    #[test]
    fn sparkle_values_use_volume_spikes_for_return_and_following_five_minute_volatility() {
        let indices = (0..8).collect::<Vec<_>>();
        let trade_times = vec![
            Some("09:30:00".to_string()),
            Some("09:31:00".to_string()),
            Some("09:32:00".to_string()),
            Some("09:33:00".to_string()),
            Some("09:34:00".to_string()),
            Some("09:35:00".to_string()),
            Some("09:36:00".to_string()),
            Some("14:58:00".to_string()),
        ];
        let close = vec![
            Some(100.0),
            Some(101.0),
            Some(103.02),
            Some(99.9294),
            Some(103.926576),
            Some(98.7302472),
            Some(104.653062032),
            Some(105.0),
        ];
        let volume = vec![
            Some(100.0),
            Some(101.0),
            Some(102.0),
            Some(220.0),
            Some(221.0),
            Some(222.0),
            Some(223.0),
            Some(224.0),
        ];

        let actual = sparkle_values_from_rows(&indices, &trade_times, &close, &volume);

        assert!((actual.return_mean.expect("return") - -0.03).abs() < 1e-12);
        assert!(actual.volatility.expect("volatility") > 0.0);
    }

    #[test]
    fn distance_and_pair_average_require_valid_inputs() {
        let distance = cs_abs_distance_from_mean(&[Some(1.0), Some(3.0), None]);
        assert_eq!(distance, vec![Some(1.0), Some(1.0), None]);

        let panel = DailyPanel::from_index(
            vec![20260102],
            vec!["000001.SZ".to_string(), "000002.SZ".to_string()],
            &[20260102],
            vec![true, true],
        )
        .expect("panel");
        let left = panel
            .column_from_values(vec![Some(1.0), None])
            .expect("left");
        let right = panel
            .column_from_values(vec![Some(3.0), Some(4.0)])
            .expect("right");
        let averaged = average_pair(&left, &right).expect("average");
        assert_eq!(averaged.values(), &[Some(2.0), None]);
    }

    #[allow(dead_code)]
    fn context() -> FactorContext {
        FactorContext {
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            start_date: 20260102,
            end_date: 20260102,
            load_start_date: 20260102,
            load_dates: vec![20260102],
            target_dates: vec![20260102],
        }
    }
}

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
use crate::operators::{cs_demean_abs, cs_zscore, ts_mean, ts_pctchg, ts_std_dev};

pub const SPARKLE_VOLATILITY_RAW_ID: &str = "daily_sparkle_volatility";
pub const SPARKLE_RETURN_RAW_ID: &str = "daily_sparkle_return";

const RAW_VERSION: &str = "0.3.0";
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
            version: "0.3.0".to_string(),
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

        let vol_distance = panel.column(SPARKLE_VOLATILITY_RAW_ID)?.cs(cs_demean_abs)?;
        let ret_distance = panel.column(SPARKLE_RETURN_RAW_ID)?.cs(cs_demean_abs)?;

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

fn sparkle_values_from_rows(
    indices: &[usize],
    trade_times: &[Option<String>],
    close: &[Option<f64>],
    volume: &[Option<f64>],
) -> SparkleValues {
    let close_series = indices
        .iter()
        .map(|idx| clean_intraday_value(close[*idx]))
        .collect::<Vec<_>>();
    let returns = ts_pctchg(&close_series, 1);
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
                let (_, volatility) = mean_std(window.iter().filter_map(|value| *value))
                    .expect("complete sparkle return window has values");
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

fn clean(value: Option<f64>) -> Option<f64> {
    value.filter(|value| !value.is_nan())
}

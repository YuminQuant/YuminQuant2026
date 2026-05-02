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
    ClassificationMap, PanelColumn,
};
use crate::factor::Factor;
use crate::operators::{cs_zscore, ts_mean, ts_pctchg, ts_std_dev};

pub const VAGUE_CORR_RAW_ID: &str = "daily_vague_corr";
pub const VAGUE_AMOUNT_RATIO_RAW_ID: &str = "daily_vague_amount_ratio";
pub const VAGUE_VOLUME_RATIO_RAW_ID: &str = "daily_vague_volume_ratio";

const RAW_VERSION: &str = "0.1.0";
const WINDOW: usize = 20;
const VAGUE_WINDOW: usize = 5;

pub struct StockDailyFogClearing;

#[derive(Clone, Copy, Debug)]
struct VagueMetrics {
    corr: Option<f64>,
    amount_ratio: Option<f64>,
    volume_ratio: Option<f64>,
}

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyFogClearing)
}

fn raw_spec(raw_id: &str) -> IntradayDailyRawSpec {
    stock_minute_raw_spec(raw_id, RAW_VERSION, &["close", "amount", "vol"], 1)
}

impl Factor for StockDailyFogClearing {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "fog_clearing".to_string(),
            aliases: Vec::new(),
            name: "Fog Clearing".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: "0.1.0".to_string(),
            tags: [
                "price_volume",
                "return",
                "volatility",
                "ambiguity",
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
            description: "Fog Clearing factor from intraday return ambiguity, amount response, and modified vague spread, neutralized by Barra SIZE and SW sector.".to_string(),
            dependencies: vec![
                DataRequest::new(DatasetId::StockBarraDaily, &["SIZE"]),
                DataRequest::new(DatasetId::StockSwClassification, &["l1_code"]),
            ],
            intraday_raw_dependencies: vec![
                IntradayDailyRawRequest::new(VAGUE_CORR_RAW_ID, WINDOW - 1),
                IntradayDailyRawRequest::new(VAGUE_AMOUNT_RATIO_RAW_ID, WINDOW - 1),
                IntradayDailyRawRequest::new(VAGUE_VOLUME_RATIO_RAW_ID, WINDOW - 1),
            ],
            lookback: Lookback {
                trading_days: WINDOW - 1,
            },
        }
    }

    fn intraday_raw_specs(&self) -> Vec<IntradayDailyRawSpec> {
        vec![
            raw_spec(VAGUE_CORR_RAW_ID),
            raw_spec(VAGUE_AMOUNT_RATIO_RAW_ID),
            raw_spec(VAGUE_VOLUME_RATIO_RAW_ID),
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
        let wants_corr = requested.contains(VAGUE_CORR_RAW_ID);
        let wants_amount_ratio = requested.contains(VAGUE_AMOUNT_RATIO_RAW_ID);
        let wants_volume_ratio = requested.contains(VAGUE_VOLUME_RATIO_RAW_ID);
        if !wants_corr && !wants_amount_ratio && !wants_volume_ratio {
            return Ok(Vec::new());
        }

        let mut corr_values = Vec::new();
        let mut amount_ratio_values = Vec::new();
        let mut volume_ratio_values = Vec::new();
        for trade_date in &context.target_dates {
            let Some(table) = data.minute(DatasetId::StockMinute1m, *trade_date) else {
                continue;
            };
            let ts_codes = table.required_utf8("ts_code")?;
            let trade_times = table.required_utf8("trade_time")?;
            let close = table.required_f64_cast("close")?;
            let amount = table.required_f64_cast("amount")?;
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
                let metrics =
                    vague_metrics_from_rows(&indices, trade_times, &close, &amount, &volume);
                if wants_corr {
                    corr_values.push(FactorValue {
                        key: FactorRowKey::Daily {
                            trade_date: *trade_date,
                            ts_code: ts_code.clone(),
                        },
                        value: metrics.corr,
                    });
                }
                if wants_amount_ratio {
                    amount_ratio_values.push(FactorValue {
                        key: FactorRowKey::Daily {
                            trade_date: *trade_date,
                            ts_code: ts_code.clone(),
                        },
                        value: metrics.amount_ratio,
                    });
                }
                if wants_volume_ratio {
                    volume_ratio_values.push(FactorValue {
                        key: FactorRowKey::Daily {
                            trade_date: *trade_date,
                            ts_code,
                        },
                        value: metrics.volume_ratio,
                    });
                }
            }
        }

        let mut output = Vec::new();
        if wants_corr {
            output.push(IntradayDailyRawSeries {
                spec: raw_spec(VAGUE_CORR_RAW_ID),
                values: corr_values,
            });
        }
        if wants_amount_ratio {
            output.push(IntradayDailyRawSeries {
                spec: raw_spec(VAGUE_AMOUNT_RATIO_RAW_ID),
                values: amount_ratio_values,
            });
        }
        if wants_volume_ratio {
            output.push(IntradayDailyRawSeries {
                spec: raw_spec(VAGUE_VOLUME_RATIO_RAW_ID),
                values: volume_ratio_values,
            });
        }
        Ok(output)
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let sector_map = ClassificationMap::from_table(
            data.daily(DatasetId::StockSwClassification)?,
            ClassificationLevel::Sector,
        )?;
        let panel = data.intraday_daily_raw_panel(VAGUE_CORR_RAW_ID)?;
        let size = panel.column_from_table(data.daily(DatasetId::StockBarraDaily)?, "SIZE")?;
        let corr = panel.column(VAGUE_CORR_RAW_ID)?;
        let amount_ratio = panel.column(VAGUE_AMOUNT_RATIO_RAW_ID)?;
        let volume_ratio = panel.column(VAGUE_VOLUME_RATIO_RAW_ID)?;

        let corr_component = rolling_component(&corr, WINDOW)?;
        let amount_component = rolling_component(&amount_ratio, WINDOW)?;
        let daily_spread = amount_ratio.zip_binary(&volume_ratio, sub)?;
        let spread_std10 = daily_spread.ts(|values| ts_std_dev(values, 10, 1))?;
        let modified_daily_spread =
            daily_spread.cs_binary(&spread_std10, modified_spread_cross_section)?;
        let modified_spread_component = rolling_component(&modified_daily_spread, WINDOW)?;

        let raw_factor = average_three(
            &corr_component.cs(cs_zscore)?,
            &amount_component.cs(cs_zscore)?,
            &modified_spread_component.cs(cs_zscore)?,
        )?;
        let neutralized = raw_factor.cs_neutralize_regression_by_group(
            &[&size],
            None,
            |trade_date, ts_codes| sector_map.groups_for(trade_date, ts_codes),
        )?;

        Ok(neutralized.to_factor_series(self.spec()))
    }
}

fn rolling_component(values: &PanelColumn, window: usize) -> Result<PanelColumn> {
    let mean = values.ts(|series| ts_mean(series, window, 1))?;
    let std = values.ts(|series| ts_std_dev(series, window, 1))?;
    average_pair(&mean.cs(cs_zscore)?, &std.cs(cs_zscore)?)
}

fn vague_metrics_from_rows(
    indices: &[usize],
    trade_times: &[Option<String>],
    close: &[Option<f64>],
    amount: &[Option<f64>],
    volume: &[Option<f64>],
) -> VagueMetrics {
    let selected = indices
        .iter()
        .filter(|idx| {
            trade_times[**idx]
                .as_deref()
                .is_some_and(|time| intraday_time_in_range(time, "09:31:00", "14:57:00"))
        })
        .copied()
        .collect::<Vec<_>>();
    let close_series = selected
        .iter()
        .map(|idx| clean_intraday_value(close[*idx]))
        .collect::<Vec<_>>();
    let returns = ts_pctchg(&close_series, 1);
    let volatility = ts_std_dev(&returns, VAGUE_WINDOW, VAGUE_WINDOW);
    let vague = ts_std_dev(&volatility, VAGUE_WINDOW, VAGUE_WINDOW);
    let amount_series = selected
        .iter()
        .map(|idx| clean_intraday_value(amount[*idx]))
        .collect::<Vec<_>>();
    let volume_series = selected
        .iter()
        .map(|idx| clean_intraday_value(volume[*idx]))
        .collect::<Vec<_>>();

    VagueMetrics {
        corr: pearson_corr(&vague, &amount_series),
        amount_ratio: ratio_mean_in_fog(&vague, &amount_series),
        volume_ratio: ratio_mean_in_fog(&vague, &volume_series),
    }
}

fn pearson_corr(left: &[Option<f64>], right: &[Option<f64>]) -> Option<f64> {
    let pairs = left
        .iter()
        .zip(right)
        .filter_map(|(left, right)| Some((clean(*left)?, clean(*right)?)))
        .collect::<Vec<_>>();
    if pairs.is_empty() {
        return None;
    }
    let mean_left = pairs.iter().map(|(left, _)| *left).sum::<f64>() / pairs.len() as f64;
    let mean_right = pairs.iter().map(|(_, right)| *right).sum::<f64>() / pairs.len() as f64;
    let cov = pairs
        .iter()
        .map(|(left, right)| (left - mean_left) * (right - mean_right))
        .sum::<f64>()
        / pairs.len() as f64;
    let std_left = (pairs
        .iter()
        .map(|(left, _)| (left - mean_left).powi(2))
        .sum::<f64>()
        / pairs.len() as f64)
        .sqrt();
    let std_right = (pairs
        .iter()
        .map(|(_, right)| (right - mean_right).powi(2))
        .sum::<f64>()
        / pairs.len() as f64)
        .sqrt();
    if std_left <= f64::EPSILON || std_right <= f64::EPSILON {
        return None;
    }
    Some(cov / (std_left * std_right))
}

fn ratio_mean_in_fog(vague: &[Option<f64>], values: &[Option<f64>]) -> Option<f64> {
    let vague_mean = mean(vague.iter().filter_map(|value| clean(*value)))?;
    let all_mean = mean(values.iter().filter_map(|value| clean(*value)))?;
    if all_mean.abs() <= f64::EPSILON {
        return None;
    }
    let fog_mean = mean(vague.iter().zip(values).filter_map(|(vague, value)| {
        let vague = clean(*vague)?;
        let value = clean(*value)?;
        (vague > vague_mean).then_some(value)
    }))?;
    Some(fog_mean / all_mean)
}

fn modified_spread_cross_section(
    spread: &[Option<f64>],
    spread_std10: &[Option<f64>],
) -> Vec<Option<f64>> {
    let s1 = spread
        .iter()
        .filter_map(|value| clean(*value))
        .filter(|value| *value < 0.0)
        .sum::<f64>();
    let intermediate = spread
        .iter()
        .zip(spread_std10)
        .map(|(spread, std)| {
            let spread = clean(*spread)?;
            if spread < 0.0 {
                let std = clean(*std)?;
                if std.abs() <= f64::EPSILON {
                    return None;
                }
                Some(spread / std)
            } else {
                Some(spread)
            }
        })
        .collect::<Vec<_>>();
    let s2 = intermediate
        .iter()
        .filter_map(|value| clean(*value))
        .filter(|value| *value < 0.0)
        .sum::<f64>();

    spread
        .iter()
        .zip(&intermediate)
        .map(|(spread, modified)| {
            let spread = clean(*spread)?;
            let modified = clean(*modified)?;
            if spread < 0.0 {
                if s2.abs() <= f64::EPSILON {
                    return None;
                }
                Some(modified / s2 * s1)
            } else {
                Some(spread)
            }
        })
        .collect()
}

fn average_pair(left: &PanelColumn, right: &PanelColumn) -> Result<PanelColumn> {
    left.zip_binary(right, |left, right| match (clean(left), clean(right)) {
        (Some(left), Some(right)) => Some((left + right) / 2.0),
        _ => None,
    })
}

fn average_three(
    first: &PanelColumn,
    second: &PanelColumn,
    third: &PanelColumn,
) -> Result<PanelColumn> {
    first.zip_ternary(second, third, |first, second, third| {
        match (clean(first), clean(second), clean(third)) {
            (Some(first), Some(second), Some(third)) => Some((first + second + third) / 3.0),
            _ => None,
        }
    })
}

fn sub(left: Option<f64>, right: Option<f64>) -> Option<f64> {
    match (clean(left), clean(right)) {
        (Some(left), Some(right)) => Some(left - right),
        _ => None,
    }
}

fn mean(values: impl IntoIterator<Item = f64>) -> Option<f64> {
    let values = values
        .into_iter()
        .filter(|value| !value.is_nan())
        .collect::<Vec<_>>();
    (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64)
}

fn clean(value: Option<f64>) -> Option<f64> {
    value.filter(|value| !value.is_nan())
}

#[cfg(test)]
mod tests {
    use super::{modified_spread_cross_section, pearson_corr, ratio_mean_in_fog};
    use crate::operators::{ts_pctchg, ts_std_dev};

    #[test]
    fn vague_rolling_std_has_expected_alignment() {
        let close = (100..112)
            .map(|value| Some(value as f64))
            .collect::<Vec<_>>();
        let returns = ts_pctchg(&close, 1);
        let volatility = ts_std_dev(&returns, 5, 5);
        let vague = ts_std_dev(&volatility, 5, 5);

        assert!(vague.iter().take(9).all(Option::is_none));
        assert!(vague[9].is_some());
    }

    #[test]
    fn pearson_corr_skips_missing_and_rejects_zero_std() {
        assert_eq!(
            pearson_corr(
                &[Some(1.0), None, Some(2.0), Some(3.0)],
                &[Some(2.0), Some(9.0), Some(4.0), Some(6.0)],
            ),
            Some(1.0)
        );
        assert_eq!(
            pearson_corr(&[Some(1.0), Some(1.0)], &[Some(2.0), Some(3.0)]),
            None
        );
    }

    #[test]
    fn ratio_mean_in_fog_is_per_stock_mean_ratio() {
        let ratio = ratio_mean_in_fog(
            &[Some(1.0), Some(3.0), Some(5.0)],
            &[Some(10.0), Some(20.0), Some(40.0)],
        )
        .expect("ratio");
        assert!((ratio - 12.0 / 7.0).abs() < 1e-12);
    }

    #[test]
    fn modified_spread_preserves_negative_cross_section_sum() {
        let output = modified_spread_cross_section(
            &[Some(-2.0), Some(-4.0), Some(1.0)],
            &[Some(2.0), Some(4.0), Some(1.0)],
        );
        assert_eq!(output, vec![Some(-3.0), Some(-3.0), Some(1.0)]);
    }
}

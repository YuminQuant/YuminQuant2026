use std::cmp::Ordering;
use std::collections::BTreeMap;

use crate::core::{
    AssetClass, FactorContext, FactorRowKey, FactorSeries, FactorSpec, FactorValue, Frequency,
    IntradayDailyRawRequest, IntradayDailyRawSeries, IntradayDailyRawSpec, Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::{
    clean_intraday_value, intraday_time_in_range, stock_minute_raw_spec, DailyPanel,
};
use crate::factor::Factor;

pub const RAW_ID: &str = "top20_centered_vol_ret_mean";
const RAW_VERSION: &str = "0.1.0";
const TOP_COUNT: usize = 20;
const CENTER_WINDOW: usize = 5;

pub struct StockDailyTop20CenteredVolRetMean;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyTop20CenteredVolRetMean)
}

pub fn raw_spec() -> IntradayDailyRawSpec {
    stock_minute_raw_spec(RAW_ID, RAW_VERSION, &["close", "vol"], 1)
}

impl Factor for StockDailyTop20CenteredVolRetMean {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "top20_centered_vol_ret_mean".to_string(),
            aliases: Vec::new(),
            name: "Stock top centered-volume minute return mean".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: "0.1.0".to_string(),
            tags: [
                "price_volume",
                "return",
                "volume",
                "intraday",
                "minute_agg",
                "daily",
            ]
            .iter()
            .map(|value| value.to_string())
            .collect(),
            description:
                "Mean return of the top 20 minutes ranked by centered 5-minute volume mean."
                    .to_string(),
            dependencies: Vec::new(),
            intraday_raw_dependencies: vec![IntradayDailyRawRequest::new(RAW_ID, 0)],
            lookback: Lookback { trading_days: 0 },
        }
    }

    fn intraday_raw_specs(&self) -> Vec<IntradayDailyRawSpec> {
        vec![raw_spec()]
    }

    fn minute_compute(
        &self,
        raw_id: &str,
        context: &FactorContext,
        data: &DataPool,
    ) -> Result<Option<IntradayDailyRawSeries>> {
        if raw_id != RAW_ID {
            return Ok(None);
        }

        let mut values = Vec::new();
        for trade_date in &context.target_dates {
            let Some(table) = data.minute(raw_spec().source_dataset, *trade_date) else {
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
                values.push(FactorValue {
                    key: FactorRowKey::Daily {
                        trade_date: *trade_date,
                        ts_code,
                    },
                    value: top_centered_vol_return_mean_from_rows(
                        &indices,
                        trade_times,
                        &close,
                        &volume,
                        TOP_COUNT,
                    ),
                });
            }
        }
        Ok(Some(IntradayDailyRawSeries {
            spec: raw_spec(),
            values,
        }))
    }

    fn compute(&self, context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let panel = DailyPanel::from_table(data.intraday_daily_raw(RAW_ID)?, context)?;
        let factor = panel.column(RAW_ID)?;
        Ok(factor.to_factor_series(self.spec()))
    }
}

fn top_centered_vol_return_mean_from_rows(
    indices: &[usize],
    trade_times: &[Option<String>],
    close: &[Option<f64>],
    volume: &[Option<f64>],
    top_count: usize,
) -> Option<f64> {
    let selected_positions = indices
        .iter()
        .enumerate()
        .filter_map(|(pos, idx)| {
            trade_times[*idx]
                .as_deref()
                .is_some_and(|time| intraday_time_in_range(time, "09:31:00", "15:00:00"))
                .then_some(pos)
        })
        .collect::<Vec<_>>();
    if selected_positions.len() < CENTER_WINDOW || top_count == 0 {
        return None;
    }

    let radius = CENTER_WINDOW / 2;
    let mut top = Vec::<(f64, usize, f64)>::with_capacity(top_count);
    for selected_pos_idx in radius..(selected_positions.len() - radius) {
        let center_pos = selected_positions[selected_pos_idx];
        let center_idx = indices[center_pos];
        let prev_idx = indices[center_pos.saturating_sub(1)];
        let (Some(current), Some(previous)) = (
            clean_intraday_value(close[center_idx]),
            clean_intraday_value(close[prev_idx]),
        ) else {
            continue;
        };
        if center_pos == 0 || previous.abs() <= f64::EPSILON {
            continue;
        }
        let ret = current / previous - 1.0;

        let mut volume_sum = 0.0;
        let mut complete = true;
        for window_pos in (selected_pos_idx - radius)..=(selected_pos_idx + radius) {
            let source_idx = indices[selected_positions[window_pos]];
            let Some(vol) = clean_intraday_value(volume[source_idx]) else {
                complete = false;
                break;
            };
            volume_sum += vol;
        }
        if !complete {
            continue;
        }

        let candidate = (volume_sum / CENTER_WINDOW as f64, selected_pos_idx, ret);
        if top.len() < top_count {
            top.push(candidate);
            continue;
        }

        if let Some((worst_idx, _)) = top
            .iter()
            .enumerate()
            .min_by(|(_, left), (_, right)| compare_candidates(left, right))
        {
            if compare_candidates(&candidate, &top[worst_idx]) == Ordering::Greater {
                top[worst_idx] = candidate;
            }
        }
    }
    if top.is_empty() {
        return None;
    }

    Some(top.iter().map(|(_, _, ret)| *ret).sum::<f64>() / top.len() as f64)
}

fn compare_candidates(left: &(f64, usize, f64), right: &(f64, usize, f64)) -> Ordering {
    left.0
        .partial_cmp(&right.0)
        .unwrap_or(Ordering::Equal)
        .then_with(|| right.1.cmp(&left.1))
}

#[cfg(test)]
mod tests {
    use super::top_centered_vol_return_mean_from_rows;

    #[test]
    fn top_centered_vol_return_mean_uses_complete_center_windows() {
        let indices = (0..8).collect::<Vec<_>>();
        let trade_times = vec![
            Some("09:30:00".to_string()),
            Some("09:31:00".to_string()),
            Some("09:32:00".to_string()),
            Some("09:33:00".to_string()),
            Some("09:34:00".to_string()),
            Some("09:35:00".to_string()),
            Some("09:36:00".to_string()),
            Some("09:37:00".to_string()),
        ];
        let close = vec![
            None,
            Some(100.0),
            Some(102.0),
            Some(105.06),
            Some(109.2624),
            Some(114.72552),
            Some(121.6090512),
            Some(130.121684784),
        ];
        let volume = vec![
            Some(1.0),
            Some(1.0),
            Some(2.0),
            Some(100.0),
            Some(2.0),
            Some(1.0),
            Some(50.0),
            Some(1.0),
        ];

        let actual =
            top_centered_vol_return_mean_from_rows(&indices, &trade_times, &close, &volume, 20)
                .expect("factor should be valid");
        assert!((actual - ((0.03 + 0.04 + 0.05) / 3.0)).abs() < 1e-12);
    }

    #[test]
    fn top_centered_vol_return_mean_keeps_earlier_ties_first() {
        let indices = (0..5).collect::<Vec<_>>();
        let trade_times = vec![
            Some("09:31:00".to_string()),
            Some("09:32:00".to_string()),
            Some("09:33:00".to_string()),
            Some("09:34:00".to_string()),
            Some("09:35:00".to_string()),
        ];
        let close = vec![Some(1.0), Some(3.0), Some(12.0), Some(60.0), Some(360.0)];
        let volume = vec![Some(1.0); 5];

        assert_eq!(
            top_centered_vol_return_mean_from_rows(&indices, &trade_times, &close, &volume, 1),
            Some(3.0)
        );
    }
}

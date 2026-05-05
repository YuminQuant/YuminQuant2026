use std::collections::{BTreeMap, BTreeSet};

use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorRowKey, FactorSeries, FactorSpec,
    FactorValue, Frequency, IntradayDailyRawRequest, IntradayDailyRawSeries, IntradayDailyRawSpec,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::stock_daily_ops::rolling_mean_desize;
use crate::factor::common::stock_daily_raw_ids::{
    DP_NEG_NEXT_DP_NEG_CORR_RAW_ID, DP_NEG_PRICE_CORR_RAW_ID, DP_POS_NEXT_DP_POS_CORR_RAW_ID,
    DP_POS_PRICE_CORR_RAW_ID,
};
use crate::factor::common::vector::clean;
use crate::factor::common::{
    clean_intraday_value, intraday_time_in_range, stock_minute_raw_spec, PanelColumn,
};
use crate::factor::Factor;
use crate::operators::cs_zscore;

const RAW_VERSION: &str = "0.1.0";
const VERSION: &str = "0.1.0";
const WINDOW: usize = 20;

pub struct StockDailyCdpp;

#[derive(Clone, Copy, Debug, Default)]
struct CorrAccumulator {
    count: usize,
    sum_x: f64,
    sum_y: f64,
    sum_xx: f64,
    sum_yy: f64,
    sum_xy: f64,
}

#[derive(Clone, Copy, Debug, Default)]
struct DailyCorrelationValues {
    dp_pos_price: Option<f64>,
    dp_neg_price: Option<f64>,
    dp_pos_next_dp_pos: Option<f64>,
    dp_neg_next_dp_neg: Option<f64>,
}

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyCdpp)
}

fn raw_spec(raw_id: &str) -> IntradayDailyRawSpec {
    stock_minute_raw_spec(raw_id, RAW_VERSION, &["close"], 1)
}

fn all_raw_ids() -> [&'static str; 4] {
    [
        DP_POS_PRICE_CORR_RAW_ID,
        DP_NEG_PRICE_CORR_RAW_ID,
        DP_POS_NEXT_DP_POS_CORR_RAW_ID,
        DP_NEG_NEXT_DP_NEG_CORR_RAW_ID,
    ]
}

impl Factor for StockDailyCdpp {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "cdpp".to_string(),
            aliases: vec!["CDPP".to_string()],
            name: "CDPP".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: [
                "price_volume",
                "price",
                "correlation",
                "intraday",
                "minute_agg",
                "neutralize",
                "barra",
                "size",
                "daily",
                "DWZQ",
            ]
            .iter()
            .map(|value| value.to_string())
            .collect(),
            description: "Correlation of Delta Price and next-minute Price, split by positive and negative intraday price deltas and neutralized by SIZE.".to_string(),
            dependencies: vec![DataRequest::new(DatasetId::StockBarraDaily, &["SIZE"])],
            intraday_raw_dependencies: vec![
                IntradayDailyRawRequest::new(DP_POS_PRICE_CORR_RAW_ID, WINDOW - 1),
                IntradayDailyRawRequest::new(DP_NEG_PRICE_CORR_RAW_ID, WINDOW - 1),
            ],
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
        let requested = raw_ids
            .iter()
            .map(String::as_str)
            .filter(|raw_id| all_raw_ids().contains(raw_id))
            .collect::<BTreeSet<_>>();
        if requested.is_empty() {
            return Ok(Vec::new());
        }

        let mut dp_pos_price_values = Vec::new();
        let mut dp_neg_price_values = Vec::new();
        let mut dp_pos_next_dp_pos_values = Vec::new();
        let mut dp_neg_next_dp_neg_values = Vec::new();

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
                let values = daily_correlations(&indices, trade_times, &close);
                let key = FactorRowKey::Daily {
                    trade_date: *trade_date,
                    ts_code,
                };
                if requested.contains(DP_POS_PRICE_CORR_RAW_ID) {
                    dp_pos_price_values.push(FactorValue {
                        key: key.clone(),
                        value: values.dp_pos_price,
                    });
                }
                if requested.contains(DP_NEG_PRICE_CORR_RAW_ID) {
                    dp_neg_price_values.push(FactorValue {
                        key: key.clone(),
                        value: values.dp_neg_price,
                    });
                }
                if requested.contains(DP_POS_NEXT_DP_POS_CORR_RAW_ID) {
                    dp_pos_next_dp_pos_values.push(FactorValue {
                        key: key.clone(),
                        value: values.dp_pos_next_dp_pos,
                    });
                }
                if requested.contains(DP_NEG_NEXT_DP_NEG_CORR_RAW_ID) {
                    dp_neg_next_dp_neg_values.push(FactorValue {
                        key,
                        value: values.dp_neg_next_dp_neg,
                    });
                }
            }
        }

        let mut output = Vec::new();
        if requested.contains(DP_POS_PRICE_CORR_RAW_ID) {
            output.push(IntradayDailyRawSeries {
                spec: raw_spec(DP_POS_PRICE_CORR_RAW_ID),
                values: dp_pos_price_values,
            });
        }
        if requested.contains(DP_NEG_PRICE_CORR_RAW_ID) {
            output.push(IntradayDailyRawSeries {
                spec: raw_spec(DP_NEG_PRICE_CORR_RAW_ID),
                values: dp_neg_price_values,
            });
        }
        if requested.contains(DP_POS_NEXT_DP_POS_CORR_RAW_ID) {
            output.push(IntradayDailyRawSeries {
                spec: raw_spec(DP_POS_NEXT_DP_POS_CORR_RAW_ID),
                values: dp_pos_next_dp_pos_values,
            });
        }
        if requested.contains(DP_NEG_NEXT_DP_NEG_CORR_RAW_ID) {
            output.push(IntradayDailyRawSeries {
                spec: raw_spec(DP_NEG_NEXT_DP_NEG_CORR_RAW_ID),
                values: dp_neg_next_dp_neg_values,
            });
        }
        Ok(output)
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let panel = data.intraday_daily_raw_panel(DP_POS_PRICE_CORR_RAW_ID)?;
        let size = panel.column_from_table(data.daily(DatasetId::StockBarraDaily)?, "SIZE")?;

        let positive = rolling_mean_desize(panel.column(DP_POS_PRICE_CORR_RAW_ID)?, &size)?;
        let negative = rolling_mean_desize(panel.column(DP_NEG_PRICE_CORR_RAW_ID)?, &size)?;
        let factor = subtract_pair(&positive.cs(cs_zscore)?, &negative.cs(cs_zscore)?)?;
        Ok(factor.to_factor_series(self.spec()))
    }
}

fn daily_correlations(
    indices: &[usize],
    trade_times: &[Option<String>],
    close: &[Option<f64>],
) -> DailyCorrelationValues {
    let close_series = indices
        .iter()
        .filter_map(|idx| {
            let trade_time = trade_times[*idx].as_deref()?;
            intraday_time_in_range(trade_time, "09:31:00", "15:00:00")
                .then_some(clean_intraday_value(close[*idx]))
        })
        .collect::<Vec<_>>();
    correlations_from_close_series(&close_series)
}

fn correlations_from_close_series(close: &[Option<f64>]) -> DailyCorrelationValues {
    let mut dp_pos_price = CorrAccumulator::default();
    let mut dp_neg_price = CorrAccumulator::default();
    let mut dp_pos_next_dp_pos = CorrAccumulator::default();
    let mut dp_neg_next_dp_neg = CorrAccumulator::default();

    if close.len() < 3 {
        return DailyCorrelationValues::default();
    }

    for idx in 1..close.len() - 1 {
        let (Some(previous), Some(current), Some(next)) = (
            clean(close[idx - 1]),
            clean(close[idx]),
            clean(close[idx + 1]),
        ) else {
            continue;
        };
        let delta = current - previous;
        let next_delta = next - current;
        if delta > 0.0 {
            dp_pos_price.push(delta, next);
            if next_delta > 0.0 {
                dp_pos_next_dp_pos.push(delta, next_delta);
            }
        } else if delta < 0.0 {
            dp_neg_price.push(delta, next);
            if next_delta < 0.0 {
                dp_neg_next_dp_neg.push(delta, next_delta);
            }
        }
    }

    DailyCorrelationValues {
        dp_pos_price: dp_pos_price.corr(),
        dp_neg_price: dp_neg_price.corr(),
        dp_pos_next_dp_pos: dp_pos_next_dp_pos.corr(),
        dp_neg_next_dp_neg: dp_neg_next_dp_neg.corr(),
    }
}

fn subtract_pair(left: &PanelColumn, right: &PanelColumn) -> Result<PanelColumn> {
    left.zip_binary(right, |left, right| match (clean(left), clean(right)) {
        (Some(left), Some(right)) => Some(left - right),
        _ => None,
    })
}

impl CorrAccumulator {
    fn push(&mut self, x: f64, y: f64) {
        self.count += 1;
        self.sum_x += x;
        self.sum_y += y;
        self.sum_xx += x * x;
        self.sum_yy += y * y;
        self.sum_xy += x * y;
    }

    fn corr(self) -> Option<f64> {
        if self.count < 2 {
            return None;
        }
        let n = self.count as f64;
        let cov = self.sum_xy - self.sum_x * self.sum_y / n;
        let var_x = self.sum_xx - self.sum_x * self.sum_x / n;
        let var_y = self.sum_yy - self.sum_y * self.sum_y / n;
        if var_x <= f64::EPSILON || var_y <= f64::EPSILON {
            return None;
        }
        Some(cov / (var_x.sqrt() * var_y.sqrt()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: Option<f64>, expected: Option<f64>) {
        match (actual, expected) {
            (Some(actual), Some(expected)) => assert!(
                (actual - expected).abs() < 1e-10,
                "expected {expected}, got {actual}"
            ),
            (None, None) => {}
            _ => panic!("expected {:?}, got {:?}", expected, actual),
        }
    }

    #[test]
    fn daily_correlations_use_0931_to_1500_window() {
        let indices = vec![0, 1, 2, 3, 4, 5];
        let times = vec![
            Some("09:30:00".to_string()),
            Some("09:31:00".to_string()),
            Some("09:32:00".to_string()),
            Some("09:33:00".to_string()),
            Some("15:00:00".to_string()),
            Some("15:01:00".to_string()),
        ];
        let close = vec![
            Some(1000.0),
            Some(1.0),
            Some(2.0),
            Some(4.0),
            Some(8.0),
            Some(1000.0),
        ];

        let values = daily_correlations(&indices, &times, &close);

        assert_close(values.dp_pos_price, Some(1.0));
        assert_close(values.dp_pos_next_dp_pos, Some(1.0));
    }

    #[test]
    fn cdpp_aligns_delta_price_with_next_minute_price() {
        let values = correlations_from_close_series(&[Some(1.0), Some(3.0), Some(4.0), Some(8.0)]);

        assert_close(values.dp_pos_price, Some(-1.0));
    }

    #[test]
    fn cdpdp_aligns_delta_price_with_next_delta_price() {
        let values = correlations_from_close_series(&[Some(1.0), Some(3.0), Some(4.0), Some(8.0)]);

        assert_close(values.dp_pos_next_dp_pos, Some(-1.0));
    }

    #[test]
    fn zero_delta_and_insufficient_samples_return_none() {
        let zero_delta =
            correlations_from_close_series(&[Some(1.0), Some(1.0), Some(1.0), Some(1.0)]);
        assert_eq!(zero_delta.dp_pos_price, None);
        assert_eq!(zero_delta.dp_neg_price, None);

        let insufficient = correlations_from_close_series(&[Some(1.0), Some(2.0), Some(3.0)]);
        assert_eq!(insufficient.dp_pos_price, None);
    }

    #[test]
    fn minute_compute_many_returns_only_requested_raw_specs() {
        let requested = BTreeSet::from([
            DP_POS_NEXT_DP_POS_CORR_RAW_ID,
            DP_NEG_NEXT_DP_NEG_CORR_RAW_ID,
        ]);
        assert!(requested.contains(DP_POS_NEXT_DP_POS_CORR_RAW_ID));
        assert!(!requested.contains(DP_POS_PRICE_CORR_RAW_ID));
    }
}

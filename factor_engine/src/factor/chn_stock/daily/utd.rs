use std::collections::{BTreeMap, HashMap};

use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorRowKey, FactorSeries, FactorSpec,
    FactorValue, Frequency, IntradayDailyRawAuxiliaryRequest, IntradayDailyRawRequest,
    IntradayDailyRawSeries, IntradayDailyRawSpec, Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::vector::clean;
use crate::factor::common::{
    clean_intraday_value, intraday_time_in_range, stock_minute_raw_spec, DailyPanel, PanelColumn,
};
use crate::factor::Factor;
use crate::operators::{ts_mean, ts_std_dev};

pub const TURNOVER_VOLATILITY_RAW_ID: &str = "daily_turnover_rate_volatility";

const RAW_VERSION: &str = "0.1.0";
const VERSION: &str = "0.1.0";
const WINDOW: usize = 20;
const FLOAT_SHARE_UNIT: f64 = 10_000.0;

pub struct StockDailyUtd;

#[derive(Clone, Copy, Debug, Default)]
struct MomentAccumulator {
    count: usize,
    sum: f64,
    sum_sq: f64,
}

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyUtd)
}

fn raw_spec() -> IntradayDailyRawSpec {
    stock_minute_raw_spec(TURNOVER_VOLATILITY_RAW_ID, RAW_VERSION, &["vol"], 1)
}

impl Factor for StockDailyUtd {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "utd".to_string(),
            aliases: vec!["UTD".to_string()],
            name: "UTD".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: [
                "price_volume",
                "turnover",
                "volume",
                "distribution",
                "intraday",
                "minute_agg",
                "daily",
            ]
            .iter()
            .map(|value| value.to_string())
            .collect(),
            description: "Uniformity of Turnover Rate Distribution, computed as the 20-day volatility-to-mean ratio of intraday turnover-rate volatility.".to_string(),
            dependencies: Vec::new(),
            intraday_raw_dependencies: vec![IntradayDailyRawRequest::new(
                TURNOVER_VOLATILITY_RAW_ID,
                WINDOW - 1,
            )],
            lookback: Lookback {
                trading_days: WINDOW - 1,
            },
        }
    }

    fn intraday_raw_specs(&self) -> Vec<IntradayDailyRawSpec> {
        vec![raw_spec()]
    }

    fn intraday_raw_auxiliary_requirements(
        &self,
        raw_ids: &[String],
    ) -> Vec<IntradayDailyRawAuxiliaryRequest> {
        if raw_ids
            .iter()
            .any(|raw_id| raw_id == TURNOVER_VOLATILITY_RAW_ID)
        {
            vec![IntradayDailyRawAuxiliaryRequest::new(
                DataRequest::new(DatasetId::StockDailyBasic, &["float_share"]),
                0,
            )]
        } else {
            Vec::new()
        }
    }

    fn minute_compute(
        &self,
        raw_id: &str,
        context: &FactorContext,
        data: &DataPool,
    ) -> Result<Option<IntradayDailyRawSeries>> {
        if raw_id != TURNOVER_VOLATILITY_RAW_ID {
            return Ok(None);
        }

        let float_share = panel_column_map(
            data.daily_panel(DatasetId::StockDailyBasic)?,
            &data
                .daily_panel(DatasetId::StockDailyBasic)?
                .column("float_share")?,
        );

        let mut values = Vec::new();
        for trade_date in &context.target_dates {
            let Some(table) = data.minute(DatasetId::StockMinute1m, *trade_date) else {
                continue;
            };
            let ts_codes = table.required_utf8("ts_code")?;
            let trade_times = table.required_utf8("trade_time")?;
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
                let share = float_share
                    .get(&(*trade_date, ts_code.clone()))
                    .copied()
                    .flatten();
                values.push(FactorValue {
                    key: FactorRowKey::Daily {
                        trade_date: *trade_date,
                        ts_code,
                    },
                    value: daily_turnover_rate_volatility(&indices, trade_times, &volume, share),
                });
            }
        }

        Ok(Some(IntradayDailyRawSeries {
            spec: raw_spec(),
            values,
        }))
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let panel = data.intraday_daily_raw_panel(TURNOVER_VOLATILITY_RAW_ID)?;
        let raw = panel.column(TURNOVER_VOLATILITY_RAW_ID)?;
        let mean20 = raw.ts(|values| ts_mean(values, WINDOW, WINDOW))?;
        let std20 = raw.ts(|values| ts_std_dev(values, WINDOW, WINDOW))?;
        let factor = std20.zip_binary(&mean20, safe_div)?;
        Ok(factor.to_factor_series(self.spec()))
    }
}

fn panel_column_map(
    panel: &DailyPanel,
    column: &PanelColumn,
) -> HashMap<(i32, String), Option<f64>> {
    let mut output = HashMap::new();
    let code_count = panel.instruments().len();
    for (date_idx, trade_date) in panel.dates().iter().enumerate() {
        for (code_idx, ts_code) in panel.instruments().iter().enumerate() {
            output.insert(
                (*trade_date, ts_code.clone()),
                column.values()[date_idx * code_count + code_idx],
            );
        }
    }
    output
}

fn daily_turnover_rate_volatility(
    indices: &[usize],
    trade_times: &[Option<String>],
    volume: &[Option<f64>],
    float_share: Option<f64>,
) -> Option<f64> {
    let float_share = clean(float_share)?;
    if float_share <= 0.0 {
        return None;
    }
    let denominator = float_share * FLOAT_SHARE_UNIT;
    if denominator <= f64::EPSILON {
        return None;
    }

    let mut moments = MomentAccumulator::default();
    for idx in indices {
        let Some(trade_time) = trade_times[*idx].as_deref() else {
            continue;
        };
        if !intraday_time_in_range(trade_time, "09:31:00", "15:00:00") {
            continue;
        }
        let Some(volume) = clean_intraday_value(volume[*idx]) else {
            continue;
        };
        moments.push(volume / denominator);
    }
    moments.std_dev()
}

fn safe_div(numerator: Option<f64>, denominator: Option<f64>) -> Option<f64> {
    match (clean(numerator), clean(denominator)) {
        (Some(numerator), Some(denominator)) if denominator.abs() > f64::EPSILON => {
            Some(numerator / denominator)
        }
        _ => None,
    }
}

impl MomentAccumulator {
    fn push(&mut self, value: f64) {
        self.count += 1;
        self.sum += value;
        self.sum_sq += value * value;
    }

    fn std_dev(self) -> Option<f64> {
        if self.count < 2 {
            return None;
        }
        let n = self.count as f64;
        let variance = (self.sum_sq - self.sum * self.sum / n) / n;
        Some(variance.max(0.0).sqrt())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: Option<f64>, expected: Option<f64>) {
        match (actual, expected) {
            (Some(actual), Some(expected)) => assert!(
                (actual - expected).abs() < 1e-12,
                "expected {expected}, got {actual}"
            ),
            (None, None) => {}
            _ => panic!("expected {:?}, got {:?}", expected, actual),
        }
    }

    #[test]
    fn turnover_volatility_uses_minute_volume_over_float_share_shares() {
        let indices = vec![0, 1, 2, 3, 4];
        let times = vec![
            Some("09:30:00".to_string()),
            Some("09:31:00".to_string()),
            Some("09:32:00".to_string()),
            Some("15:00:00".to_string()),
            Some("15:01:00".to_string()),
        ];
        let volume = vec![
            Some(1_000_000.0),
            Some(10_000.0),
            Some(20_000.0),
            Some(30_000.0),
            Some(100_000_000.0),
        ];

        let actual = daily_turnover_rate_volatility(&indices, &times, &volume, Some(1.0));
        let expected = Some((2.0_f64 / 3.0).sqrt());

        assert_close(actual, expected);
    }

    #[test]
    fn turnover_volatility_rejects_missing_or_zero_float_share() {
        let indices = vec![0, 1];
        let times = vec![Some("09:31:00".to_string()), Some("09:32:00".to_string())];
        let volume = vec![Some(10_000.0), Some(20_000.0)];

        assert_eq!(
            daily_turnover_rate_volatility(&indices, &times, &volume, None),
            None
        );
        assert_eq!(
            daily_turnover_rate_volatility(&indices, &times, &volume, Some(0.0)),
            None
        );
    }

    #[test]
    fn safe_div_rejects_zero_denominator() {
        assert_eq!(safe_div(Some(1.0), Some(0.0)), None);
        assert_close(safe_div(Some(3.0), Some(2.0)), Some(1.5));
    }
}

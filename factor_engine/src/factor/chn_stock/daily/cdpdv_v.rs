use std::collections::{BTreeMap, BTreeSet};

use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorRowKey, FactorSeries, FactorSpec,
    FactorValue, Frequency, IntradayDailyRawRequest, IntradayDailyRawSeries, IntradayDailyRawSpec,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::stock_daily_ops::rolling_mean_desize;
use crate::factor::common::vector::clean;
use crate::factor::common::{
    clean_intraday_value, intraday_time_in_range, stock_minute_raw_spec, PanelColumn,
};
use crate::factor::Factor;
use crate::operators::cs_zscore;

pub const DV_POS_NEXT_DP_POS_CORR_RAW_ID: &str = "daily_dv_pos_next_dp_pos_corr";
pub const DV_POS_NEXT_DP_NEG_CORR_RAW_ID: &str = "daily_dv_pos_next_dp_neg_corr";
pub const DV_NEG_NEXT_DP_POS_CORR_RAW_ID: &str = "daily_dv_neg_next_dp_pos_corr";
pub const DV_NEG_NEXT_DP_NEG_CORR_RAW_ID: &str = "daily_dv_neg_next_dp_neg_corr";

const RAW_VERSION: &str = "0.1.0";
const VERSION: &str = "0.1.0";
const WINDOW: usize = 20;

pub struct StockDailyCdpdvV;

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
    dv_pos_next_dp_pos: Option<f64>,
    dv_pos_next_dp_neg: Option<f64>,
    dv_neg_next_dp_pos: Option<f64>,
    dv_neg_next_dp_neg: Option<f64>,
}

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyCdpdvV)
}

fn raw_spec(raw_id: &str) -> IntradayDailyRawSpec {
    stock_minute_raw_spec(raw_id, RAW_VERSION, &["close", "vol"], 1)
}

fn all_raw_ids() -> [&'static str; 4] {
    [
        DV_POS_NEXT_DP_POS_CORR_RAW_ID,
        DV_POS_NEXT_DP_NEG_CORR_RAW_ID,
        DV_NEG_NEXT_DP_POS_CORR_RAW_ID,
        DV_NEG_NEXT_DP_NEG_CORR_RAW_ID,
    ]
}

impl Factor for StockDailyCdpdvV {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "cdpdv_v".to_string(),
            aliases: vec!["CDPDV_V".to_string()],
            name: "CDPDV_V".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: [
                "price_volume",
                "price",
                "volume",
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
            description: "Correlation of Delta Volume and next-minute Delta Price, split by volume-price sign regimes and neutralized by SIZE.".to_string(),
            dependencies: vec![DataRequest::new(DatasetId::StockBarraDaily, &["SIZE"])],
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

        let mut dv_pos_next_dp_pos_values = Vec::new();
        let mut dv_pos_next_dp_neg_values = Vec::new();
        let mut dv_neg_next_dp_pos_values = Vec::new();
        let mut dv_neg_next_dp_neg_values = Vec::new();

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
                let values = daily_correlations(&indices, trade_times, &close, &volume);
                let key = FactorRowKey::Daily {
                    trade_date: *trade_date,
                    ts_code,
                };
                if requested.contains(DV_POS_NEXT_DP_POS_CORR_RAW_ID) {
                    dv_pos_next_dp_pos_values.push(FactorValue {
                        key: key.clone(),
                        value: values.dv_pos_next_dp_pos,
                    });
                }
                if requested.contains(DV_POS_NEXT_DP_NEG_CORR_RAW_ID) {
                    dv_pos_next_dp_neg_values.push(FactorValue {
                        key: key.clone(),
                        value: values.dv_pos_next_dp_neg,
                    });
                }
                if requested.contains(DV_NEG_NEXT_DP_POS_CORR_RAW_ID) {
                    dv_neg_next_dp_pos_values.push(FactorValue {
                        key: key.clone(),
                        value: values.dv_neg_next_dp_pos,
                    });
                }
                if requested.contains(DV_NEG_NEXT_DP_NEG_CORR_RAW_ID) {
                    dv_neg_next_dp_neg_values.push(FactorValue {
                        key,
                        value: values.dv_neg_next_dp_neg,
                    });
                }
            }
        }

        let mut output = Vec::new();
        if requested.contains(DV_POS_NEXT_DP_POS_CORR_RAW_ID) {
            output.push(IntradayDailyRawSeries {
                spec: raw_spec(DV_POS_NEXT_DP_POS_CORR_RAW_ID),
                values: dv_pos_next_dp_pos_values,
            });
        }
        if requested.contains(DV_POS_NEXT_DP_NEG_CORR_RAW_ID) {
            output.push(IntradayDailyRawSeries {
                spec: raw_spec(DV_POS_NEXT_DP_NEG_CORR_RAW_ID),
                values: dv_pos_next_dp_neg_values,
            });
        }
        if requested.contains(DV_NEG_NEXT_DP_POS_CORR_RAW_ID) {
            output.push(IntradayDailyRawSeries {
                spec: raw_spec(DV_NEG_NEXT_DP_POS_CORR_RAW_ID),
                values: dv_neg_next_dp_pos_values,
            });
        }
        if requested.contains(DV_NEG_NEXT_DP_NEG_CORR_RAW_ID) {
            output.push(IntradayDailyRawSeries {
                spec: raw_spec(DV_NEG_NEXT_DP_NEG_CORR_RAW_ID),
                values: dv_neg_next_dp_neg_values,
            });
        }
        Ok(output)
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let panel = data.intraday_daily_raw_panel(DV_POS_NEXT_DP_POS_CORR_RAW_ID)?;
        let size = panel.column_from_table(data.daily(DatasetId::StockBarraDaily)?, "SIZE")?;

        let pos_pos = rolling_mean_desize(panel.column(DV_POS_NEXT_DP_POS_CORR_RAW_ID)?, &size)?;
        let pos_neg = rolling_mean_desize(panel.column(DV_POS_NEXT_DP_NEG_CORR_RAW_ID)?, &size)?;
        let neg_pos = rolling_mean_desize(panel.column(DV_NEG_NEXT_DP_POS_CORR_RAW_ID)?, &size)?;
        let neg_neg = rolling_mean_desize(panel.column(DV_NEG_NEXT_DP_NEG_CORR_RAW_ID)?, &size)?;
        let factor = signed_sum_four(
            &pos_pos.cs(cs_zscore)?,
            &pos_neg.cs(cs_zscore)?,
            &neg_pos.cs(cs_zscore)?,
            &neg_neg.cs(cs_zscore)?,
        )?;
        Ok(factor.to_factor_series(self.spec()))
    }
}

fn daily_correlations(
    indices: &[usize],
    trade_times: &[Option<String>],
    close: &[Option<f64>],
    volume: &[Option<f64>],
) -> DailyCorrelationValues {
    let (close_series, volume_series): (Vec<_>, Vec<_>) = indices
        .iter()
        .filter_map(|idx| {
            let trade_time = trade_times[*idx].as_deref()?;
            intraday_time_in_range(trade_time, "09:31:00", "15:00:00").then_some((
                clean_intraday_value(close[*idx]),
                clean_intraday_value(volume[*idx]),
            ))
        })
        .unzip();
    correlations_from_close_volume_series(&close_series, &volume_series)
}

fn correlations_from_close_volume_series(
    close: &[Option<f64>],
    volume: &[Option<f64>],
) -> DailyCorrelationValues {
    let mut dv_pos_next_dp_pos = CorrAccumulator::default();
    let mut dv_pos_next_dp_neg = CorrAccumulator::default();
    let mut dv_neg_next_dp_pos = CorrAccumulator::default();
    let mut dv_neg_next_dp_neg = CorrAccumulator::default();

    let len = close.len().min(volume.len());
    if len < 3 {
        return DailyCorrelationValues::default();
    }

    for idx in 1..len - 1 {
        let (Some(previous_volume), Some(current_volume), Some(current_close), Some(next_close)) = (
            clean(volume[idx - 1]),
            clean(volume[idx]),
            clean(close[idx]),
            clean(close[idx + 1]),
        ) else {
            continue;
        };
        let delta_volume = current_volume - previous_volume;
        let next_delta_price = next_close - current_close;
        match (
            delta_volume > 0.0,
            delta_volume < 0.0,
            next_delta_price > 0.0,
            next_delta_price < 0.0,
        ) {
            (true, false, true, false) => {
                dv_pos_next_dp_pos.push(delta_volume, next_delta_price);
            }
            (true, false, false, true) => {
                dv_pos_next_dp_neg.push(delta_volume, next_delta_price);
            }
            (false, true, true, false) => {
                dv_neg_next_dp_pos.push(delta_volume, next_delta_price);
            }
            (false, true, false, true) => {
                dv_neg_next_dp_neg.push(delta_volume, next_delta_price);
            }
            _ => {}
        }
    }

    DailyCorrelationValues {
        dv_pos_next_dp_pos: dv_pos_next_dp_pos.corr(),
        dv_pos_next_dp_neg: dv_pos_next_dp_neg.corr(),
        dv_neg_next_dp_pos: dv_neg_next_dp_pos.corr(),
        dv_neg_next_dp_neg: dv_neg_next_dp_neg.corr(),
    }
}

fn signed_sum_four(
    pos_pos: &PanelColumn,
    pos_neg: &PanelColumn,
    neg_pos: &PanelColumn,
    neg_neg: &PanelColumn,
) -> Result<PanelColumn> {
    pos_pos.zip_quaternary(pos_neg, neg_pos, neg_neg, |a, b, c, d| {
        match (clean(a), clean(b), clean(c), clean(d)) {
            (Some(a), Some(b), Some(c), Some(d)) => Some(a - b - c + d),
            _ => None,
        }
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
        let volume = vec![
            Some(1000.0),
            Some(1.0),
            Some(2.0),
            Some(4.0),
            Some(8.0),
            Some(1000.0),
        ];

        let values = daily_correlations(&indices, &times, &close, &volume);

        assert_close(values.dv_pos_next_dp_pos, Some(1.0));
    }

    #[test]
    fn aligns_delta_volume_with_next_delta_price() {
        let values = correlations_from_close_volume_series(
            &[Some(10.0), Some(11.0), Some(13.0), Some(14.0)],
            &[Some(10.0), Some(11.0), Some(13.0), Some(14.0)],
        );

        assert_close(values.dv_pos_next_dp_pos, Some(-1.0));
    }

    #[test]
    fn splits_all_four_sign_regimes() {
        let pos_pos = correlations_from_close_volume_series(
            &[Some(10.0), Some(11.0), Some(13.0), Some(14.0)],
            &[Some(10.0), Some(11.0), Some(13.0), Some(14.0)],
        );
        let pos_neg = correlations_from_close_volume_series(
            &[Some(10.0), Some(9.0), Some(7.0), Some(6.0)],
            &[Some(10.0), Some(11.0), Some(13.0), Some(14.0)],
        );
        let neg_pos = correlations_from_close_volume_series(
            &[Some(10.0), Some(11.0), Some(13.0), Some(14.0)],
            &[Some(14.0), Some(13.0), Some(11.0), Some(10.0)],
        );
        let neg_neg = correlations_from_close_volume_series(
            &[Some(10.0), Some(9.0), Some(7.0), Some(6.0)],
            &[Some(14.0), Some(13.0), Some(11.0), Some(10.0)],
        );

        assert!(pos_pos.dv_pos_next_dp_pos.is_some());
        assert!(pos_neg.dv_pos_next_dp_neg.is_some());
        assert!(neg_pos.dv_neg_next_dp_pos.is_some());
        assert!(neg_neg.dv_neg_next_dp_neg.is_some());
    }

    #[test]
    fn zero_deltas_do_not_enter_any_regime() {
        let values = correlations_from_close_volume_series(
            &[Some(1.0), Some(1.0), Some(1.0), Some(1.0)],
            &[Some(1.0), Some(1.0), Some(1.0), Some(1.0)],
        );

        assert_eq!(values.dv_pos_next_dp_pos, None);
        assert_eq!(values.dv_pos_next_dp_neg, None);
        assert_eq!(values.dv_neg_next_dp_pos, None);
        assert_eq!(values.dv_neg_next_dp_neg, None);
    }

    #[test]
    fn signed_sum_uses_expected_direction() {
        let table = crate::data::Table::new(std::collections::BTreeMap::from([
            (
                "trade_date".to_string(),
                crate::data::ColumnData::I32(vec![Some(20260101)]),
            ),
            (
                "ts_code".to_string(),
                crate::data::ColumnData::Utf8(vec![Some("000001.SZ".to_string())]),
            ),
            (
                "a".to_string(),
                crate::data::ColumnData::F64(vec![Some(4.0)]),
            ),
            (
                "b".to_string(),
                crate::data::ColumnData::F64(vec![Some(1.0)]),
            ),
            (
                "c".to_string(),
                crate::data::ColumnData::F64(vec![Some(2.0)]),
            ),
            (
                "d".to_string(),
                crate::data::ColumnData::F64(vec![Some(3.0)]),
            ),
        ]))
        .expect("table");
        let context = FactorContext {
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            start_date: 20260101,
            end_date: 20260101,
            load_start_date: 20260101,
            load_dates: vec![20260101],
            target_dates: vec![20260101],
        };
        let panel = crate::factor::common::DailyPanel::from_table(&table, &context).expect("panel");

        let factor = signed_sum_four(
            &panel.column("a").expect("a"),
            &panel.column("b").expect("b"),
            &panel.column("c").expect("c"),
            &panel.column("d").expect("d"),
        )
        .expect("factor");

        assert_close(factor.values()[0], Some(4.0));
    }
}

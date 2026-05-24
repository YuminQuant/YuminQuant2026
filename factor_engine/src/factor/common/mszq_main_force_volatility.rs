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
use crate::factor::common::{DailyPanel, PanelColumn};
use crate::operators::{cs_zscore, ts_std_dev};

pub const PROVIDER_KEY: &str = "mszq_main_force_volatility_provider";
pub const RAW_VERSION: &str = "0.1.0";
pub const VERSION: &str = "0.1.0";

pub const VOLUME_UP_RETURN_RAW_ID: &str = "daily_mszq_volume_up_return_raw";
pub const VOLUME_DOWN_RETURN_RAW_ID: &str = "daily_mszq_volume_down_return_raw";
pub const VOLUME_CONTINUOUS_UP_RETURN_RAW_ID: &str = "daily_mszq_volume_continuous_up_return_raw";
pub const VOLUME_CONTINUOUS_DOWN_RETURN_RAW_ID: &str =
    "daily_mszq_volume_continuous_down_return_raw";

const RAW_WINDOW_DAYS: usize = 1;
const ROLLING_WINDOW: usize = 20;
const MIN_PERIODS: usize = 1;
const EPS: f64 = f64::EPSILON;

#[derive(Clone, Copy, Debug)]
pub struct MszqMainForceVolatilityFactorDef {
    pub id: &'static str,
    pub alias: &'static str,
    pub name: &'static str,
}

#[derive(Clone, Copy, Debug, Default)]
struct DailyStats {
    volume_up_return: Option<f64>,
    volume_down_return: Option<f64>,
    volume_continuous_up_return: Option<f64>,
    volume_continuous_down_return: Option<f64>,
}

#[derive(Clone, Copy, Debug)]
struct MinutePoint {
    close: Option<f64>,
    volume: Option<f64>,
}

pub fn all_raw_ids() -> [&'static str; 4] {
    [
        VOLUME_UP_RETURN_RAW_ID,
        VOLUME_DOWN_RETURN_RAW_ID,
        VOLUME_CONTINUOUS_UP_RETURN_RAW_ID,
        VOLUME_CONTINUOUS_DOWN_RETURN_RAW_ID,
    ]
}

pub fn raw_spec(raw_id: &str) -> IntradayDailyRawSpec {
    stock_minute_raw_spec(raw_id, RAW_VERSION, &["close", "vol"], RAW_WINDOW_DAYS)
}

pub fn raw_specs() -> Vec<IntradayDailyRawSpec> {
    all_raw_ids()
        .iter()
        .map(|raw_id| raw_spec(raw_id))
        .collect()
}

pub fn factor_spec(def: MszqMainForceVolatilityFactorDef) -> FactorSpec {
    FactorSpec {
        id: def.id.to_string(),
        aliases: vec![def.alias.to_string()],
        name: def.name.to_string(),
        asset_class: AssetClass::Stock,
        frequency: Frequency::Daily,
        version: VERSION.to_string(),
        tags: tags(),
        description: description(def),
        dependencies: dependencies(),
        intraday_raw_dependencies: all_raw_ids()
            .iter()
            .map(|raw_id| IntradayDailyRawRequest::new(raw_id, ROLLING_WINDOW - 1))
            .collect(),
        lookback: Lookback {
            trading_days: ROLLING_WINDOW - 1,
        },
    }
}

pub fn compute_factor(
    def: MszqMainForceVolatilityFactorDef,
    data: &DataPool,
) -> Result<FactorSeries> {
    let panel = data.intraday_daily_raw_panel(VOLUME_UP_RETURN_RAW_ID)?;
    let volume_up = panel.column(VOLUME_UP_RETURN_RAW_ID)?;
    let volume_down = panel.column(VOLUME_DOWN_RETURN_RAW_ID)?;
    let continuous_up = panel.column(VOLUME_CONTINUOUS_UP_RETURN_RAW_ID)?;
    let continuous_down = panel.column(VOLUME_CONTINUOUS_DOWN_RETURN_RAW_ID)?;

    let volume_up_component = adjusted_volatility_component(&volume_up)?;
    let volume_down_component = adjusted_volatility_component(&volume_down)?;
    let continuous_up_component = adjusted_volatility_component(&continuous_up)?;
    let continuous_down_component = adjusted_volatility_component(&continuous_down)?;
    let composite = average_columns(
        &panel,
        &[
            &volume_up_component,
            &continuous_up_component,
            &volume_down_component,
            &continuous_down_component,
        ],
    )?;
    let factor = neutralize_size_sector(&composite, &panel, data)?;
    Ok(factor.to_factor_series(factor_spec(def)))
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
            let points = minute_points_from_indices(&indices, &close, &volume);
            let stats = daily_stats(&points);
            let key = FactorRowKey::Daily {
                trade_date: *trade_date,
                ts_code,
            };
            push_requested(
                &mut values,
                &requested,
                VOLUME_UP_RETURN_RAW_ID,
                &key,
                stats.volume_up_return,
            );
            push_requested(
                &mut values,
                &requested,
                VOLUME_DOWN_RETURN_RAW_ID,
                &key,
                stats.volume_down_return,
            );
            push_requested(
                &mut values,
                &requested,
                VOLUME_CONTINUOUS_UP_RETURN_RAW_ID,
                &key,
                stats.volume_continuous_up_return,
            );
            push_requested(
                &mut values,
                &requested,
                VOLUME_CONTINUOUS_DOWN_RETURN_RAW_ID,
                &key,
                stats.volume_continuous_down_return,
            );
        }
    }

    let mut output = Vec::new();
    for raw_id in all_raw_ids() {
        if !requested.contains(raw_id) {
            continue;
        }
        output.push(IntradayDailyRawSeries {
            spec: raw_spec(raw_id),
            values: values.remove(raw_id).unwrap_or_default(),
        });
    }
    Ok(output)
}

fn adjusted_volatility_component(values: &PanelColumn) -> Result<PanelColumn> {
    let standardized = values.cs(cs_zscore)?;
    let absolute = standardized.map_values(|value| finite_option(value.map(f64::abs)));
    let adjusted = absolute.cs(cs_zscore)?;
    adjusted.ts(|series| ts_std_dev(series, ROLLING_WINDOW, MIN_PERIODS))
}

fn average_columns(panel: &DailyPanel, columns: &[&PanelColumn]) -> Result<PanelColumn> {
    if columns.is_empty() {
        return panel.column_from_values(vec![None; panel.shape_len()]);
    }
    let mut values = Vec::with_capacity(panel.shape_len());
    for offset in 0..panel.shape_len() {
        let mut sum = 0.0;
        let mut count = 0usize;
        for column in columns {
            if let Some(value) = finite_option(column.values()[offset]) {
                sum += value;
                count += 1;
            }
        }
        values.push((count > 0).then_some(sum / count as f64));
    }
    panel.column_from_values(values)
}

fn minute_points_from_indices(
    indices: &[usize],
    close: &[Option<f64>],
    volume: &[Option<f64>],
) -> Vec<MinutePoint> {
    indices
        .iter()
        .map(|idx| MinutePoint {
            close: clean_positive(close[*idx]),
            volume: clean_nonnegative(volume[*idx]),
        })
        .collect()
}

fn daily_stats(points: &[MinutePoint]) -> DailyStats {
    DailyStats {
        volume_up_return: simple_volume_return(points, 1),
        volume_down_return: simple_volume_return(points, -1),
        volume_continuous_up_return: continuous_volume_return(points, 1),
        volume_continuous_down_return: continuous_volume_return(points, -1),
    }
}

fn simple_volume_return(points: &[MinutePoint], direction: i8) -> Option<f64> {
    let mut sum = 0.0;
    let mut count = 0usize;
    for idx in 1..points.len() {
        let Some(ret) = minute_return(points[idx - 1].close, points[idx].close) else {
            continue;
        };
        if return_sign(ret) != Some(direction) {
            continue;
        }
        let (Some(current_volume), Some(previous_volume)) =
            (points[idx].volume, points[idx - 1].volume)
        else {
            continue;
        };
        if current_volume > previous_volume {
            sum += ret;
            count += 1;
        }
    }
    (count > 0).then_some(sum).and_then(finite_value)
}

fn continuous_volume_return(points: &[MinutePoint], direction: i8) -> Option<f64> {
    let mut sum = 0.0;
    let mut count = 0usize;
    let mut current_sign: Option<i8> = None;
    let mut has_segment_volume = false;
    let mut segment_min = 0.0;
    let mut segment_max = 0.0;

    for idx in 1..points.len() {
        let Some(ret) = minute_return(points[idx - 1].close, points[idx].close) else {
            current_sign = None;
            has_segment_volume = false;
            continue;
        };
        let Some(sign) = return_sign(ret) else {
            current_sign = None;
            has_segment_volume = false;
            continue;
        };
        if current_sign != Some(sign) {
            current_sign = Some(sign);
            has_segment_volume = false;
        }

        let Some(current_volume) = points[idx].volume else {
            continue;
        };
        if !has_segment_volume {
            segment_min = current_volume;
            segment_max = current_volume;
            has_segment_volume = true;
            continue;
        }

        if sign == direction && current_volume > segment_max {
            sum += ret;
            count += 1;
        }
        if current_volume > segment_max {
            segment_max = current_volume;
        }
        if current_volume < segment_min {
            segment_min = current_volume;
        }
    }

    (count > 0).then_some(sum).and_then(finite_value)
}

fn minute_return(previous_close: Option<f64>, current_close: Option<f64>) -> Option<f64> {
    let (Some(previous), Some(current)) = (previous_close, current_close) else {
        return None;
    };
    if previous.abs() <= EPS {
        return None;
    }
    finite_value(current / previous - 1.0)
}

fn return_sign(value: f64) -> Option<i8> {
    if value > 0.0 {
        Some(1)
    } else if value < 0.0 {
        Some(-1)
    } else {
        None
    }
}

fn tags() -> Vec<String> {
    [
        "price_volume",
        "volume",
        "return",
        "volatility",
        "intraday",
        "minute_agg",
        "neutralize",
        "barra",
        "size",
        "sector",
        "daily",
        "MSZQ",
    ]
    .iter()
    .map(|value| value.to_string())
    .collect()
}

fn description(def: MszqMainForceVolatilityFactorDef) -> String {
    format!(
        "{} composites four 1-minute main-force volume trend return volatility components, then neutralizes by Barra SIZE and SW sector; it does not depend on derived intraday bars.",
        def.name
    )
}

fn dependencies() -> Vec<DataRequest> {
    vec![
        DataRequest::new(DatasetId::StockBarraDaily, &["SIZE"]),
        DataRequest::new(DatasetId::StockSwClassification, &["l1_code"]),
    ]
}

fn push_requested(
    values: &mut BTreeMap<&'static str, Vec<FactorValue>>,
    requested: &BTreeSet<&str>,
    raw_id: &'static str,
    key: &FactorRowKey,
    value: Option<f64>,
) {
    if requested.contains(raw_id) {
        values.entry(raw_id).or_default().push(FactorValue {
            key: key.clone(),
            value,
        });
    }
}

fn clean_positive(value: Option<f64>) -> Option<f64> {
    clean_intraday_value(value)
        .and_then(finite_value)
        .filter(|value| *value > 0.0)
}

fn clean_nonnegative(value: Option<f64>) -> Option<f64> {
    clean_intraday_value(value)
        .and_then(finite_value)
        .filter(|value| *value >= 0.0)
}

fn finite_option(value: Option<f64>) -> Option<f64> {
    value.and_then(finite_value)
}

fn finite_value(value: f64) -> Option<f64> {
    value.is_finite().then_some(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn point(close: f64, volume: f64) -> MinutePoint {
        MinutePoint {
            close: Some(close),
            volume: Some(volume),
        }
    }

    fn assert_close(actual: Option<f64>, expected: f64) {
        let actual = actual.expect("expected value");
        assert!(
            (actual - expected).abs() < 1e-10,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn mszq_minute_return_uses_simple_close_to_close_return() {
        assert_close(minute_return(Some(100.0), Some(101.5)), 0.015);
        assert_eq!(return_sign(0.015), Some(1));
        assert_eq!(return_sign(-0.015), Some(-1));
        assert_eq!(return_sign(0.0), None);
    }

    #[test]
    fn mszq_simple_raws_use_previous_minute_volume_without_segment_merge() {
        let points = vec![
            point(100.0, 10.0),
            point(101.0, 20.0),
            point(100.0, 30.0),
            point(102.0, 25.0),
            point(102.0, 40.0),
        ];

        let up = simple_volume_return(&points, 1);
        let down = simple_volume_return(&points, -1);

        assert_close(up, 0.01);
        assert_close(down, 100.0 / 101.0 - 1.0);
    }

    #[test]
    fn mszq_continuous_raws_merge_same_sign_segments_and_skip_segment_first_minute() {
        let points = vec![
            point(100.0, 10.0),
            point(101.0, 11.0),
            point(102.0, 15.0),
            point(101.0, 20.0),
            point(100.0, 25.0),
            point(99.0, 30.0),
            point(100.0, 31.0),
            point(101.0, 40.0),
        ];

        let up = continuous_volume_return(&points, 1);
        let down = continuous_volume_return(&points, -1);

        assert_close(up, (102.0 / 101.0 - 1.0) + (101.0 / 100.0 - 1.0));
        assert_close(down, (100.0 / 101.0 - 1.0) + (99.0 / 100.0 - 1.0));
    }

    #[test]
    fn mszq_continuous_raws_break_segments_on_zero_return() {
        let points = vec![
            point(100.0, 10.0),
            point(101.0, 11.0),
            point(101.0, 100.0),
            point(102.0, 12.0),
            point(103.0, 13.0),
        ];

        let up = continuous_volume_return(&points, 1);

        assert_close(up, 103.0 / 102.0 - 1.0);
    }

    #[test]
    fn mszq_adjusted_component_zscores_abs_then_uses_20d_std() {
        let panel = DailyPanel::from_index(
            vec![20260423, 20260424],
            vec!["a".to_string(), "b".to_string(), "c".to_string()],
            &[20260423, 20260424],
            vec![true, true, true, true, true, true],
        )
        .unwrap();
        let raw = panel
            .column_from_values(vec![
                Some(-1.0),
                Some(0.0),
                Some(1.0),
                Some(-2.0),
                Some(0.0),
                Some(2.0),
            ])
            .unwrap();

        let output = adjusted_volatility_component(&raw).unwrap();

        assert_close(output.values()[0], 0.0);
        assert_close(output.values()[1], 0.0);
        assert_close(output.values()[2], 0.0);
    }

    #[test]
    fn mszq_average_columns_uses_available_components() {
        let panel = DailyPanel::from_index(
            vec![20260424],
            vec!["a".to_string(), "b".to_string()],
            &[20260424],
            vec![true, true],
        )
        .unwrap();
        let left = panel
            .column_from_values(vec![Some(1.0), Some(2.0)])
            .unwrap();
        let right = panel.column_from_values(vec![Some(3.0), None]).unwrap();

        let output = average_columns(&panel, &[&left, &right]).unwrap();

        assert_close(output.values()[0], 2.0);
        assert_close(output.values()[1], 2.0);
    }

    #[test]
    fn mszq_main_force_volatility_factor_spec_has_mszq_tag_and_single_output() {
        let spec = factor_spec(MszqMainForceVolatilityFactorDef {
            id: "main_force_volatility",
            alias: "main_force_volatility",
            name: "Main Force Volatility",
        });

        assert_eq!(spec.id, "main_force_volatility");
        assert!(spec.tags.iter().any(|tag| tag == "MSZQ"));
        assert_eq!(spec.intraday_raw_dependencies.len(), 4);
        assert!(spec
            .description
            .contains("does not depend on derived intraday bars"));
    }
}

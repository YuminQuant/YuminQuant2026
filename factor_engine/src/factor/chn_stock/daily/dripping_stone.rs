use std::collections::{BTreeMap, BTreeSet};
use std::f64::consts::PI;
use std::sync::{Arc, Mutex, OnceLock};

use crate::core::{
    AssetClass, FactorContext, FactorRowKey, FactorSeries, FactorSpec, FactorValue, Frequency,
    IntradayDailyRawRequest, IntradayDailyRawSeries, IntradayDailyRawSpec, Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::{
    clean_intraday_value, intraday_time_in_range, quantile_linear, stock_minute_raw_spec,
};
use crate::factor::Factor;

pub const RAW_ID: &str = "daily_dripping_stone_spectral_ratio";

const RAW_VERSION: &str = "0.1.0";
const VERSION: &str = "0.1.0";
const EPS: f64 = 1.0e-12;

pub struct StockDailyDrippingStone;

struct SpectralCache {
    hann: Vec<f64>,
    bins: Vec<SpectralBin>,
}

struct SpectralBin {
    in_band: bool,
    cos: Vec<f64>,
    sin: Vec<f64>,
}

static SPECTRAL_CACHE: OnceLock<Mutex<BTreeMap<usize, Arc<SpectralCache>>>> = OnceLock::new();

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyDrippingStone)
}

pub fn raw_spec() -> IntradayDailyRawSpec {
    stock_minute_raw_spec(RAW_ID, RAW_VERSION, &["vol"], 1)
}

impl Factor for StockDailyDrippingStone {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "dripping_stone".to_string(),
            aliases: Vec::new(),
            name: "Dripping Stone".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: [
                "price_volume",
                "volume",
                "spectral",
                "intraday",
                "minute_agg",
                "daily",
                "FZZQ",
            ]
            .iter()
            .map(|value| value.to_string())
            .collect(),
            description: "Intraday volume spectral power ratio in the 2-5 minute period band after IQR clipping and Hann windowing.".to_string(),
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
            let volume = table.required_f64_cast("vol")?;
            let expected_times = expected_intraday_times(trade_times);
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
                    value: dripping_stone_ratio_from_rows(
                        &indices,
                        trade_times,
                        &volume,
                        &expected_times,
                    ),
                });
            }
        }

        Ok(Some(IntradayDailyRawSeries {
            spec: raw_spec(),
            values,
        }))
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let panel = data.intraday_daily_raw_panel(RAW_ID)?;
        let raw = panel.column(RAW_ID)?;
        Ok(raw.to_factor_series(self.spec()))
    }
}

fn expected_intraday_times(trade_times: &[Option<String>]) -> Vec<String> {
    trade_times
        .iter()
        .filter_map(|time| {
            let time = time.as_deref()?;
            intraday_time_in_range(time, "09:31:00", "14:57:00").then(|| time.to_string())
        })
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn dripping_stone_ratio_from_rows(
    indices: &[usize],
    trade_times: &[Option<String>],
    volume: &[Option<f64>],
    expected_times: &[String],
) -> Option<f64> {
    let values = selected_volume_series_from_rows(indices, trade_times, volume, expected_times)?;
    spectral_ratio_from_volume_series(&values)
}

fn selected_volume_series_from_rows(
    indices: &[usize],
    trade_times: &[Option<String>],
    volume: &[Option<f64>],
    expected_times: &[String],
) -> Option<Vec<f64>> {
    let mut selected = Vec::new();
    for idx in indices {
        let Some(time) = trade_times[*idx].as_deref() else {
            continue;
        };
        if !intraday_time_in_range(time, "09:31:00", "14:57:00") {
            continue;
        }
        let value = clean_intraday_value(volume[*idx])?;
        selected.push((time.to_string(), value));
    }
    if selected.len() != expected_times.len() || selected.is_empty() {
        return None;
    }
    if selected
        .iter()
        .zip(expected_times)
        .any(|((time, _), expected)| time != expected)
    {
        return None;
    }

    Some(selected.into_iter().map(|(_, value)| value).collect())
}

fn spectral_ratio_from_volume_series(values: &[f64]) -> Option<f64> {
    if values.len() < 6 || values.iter().any(|value| !value.is_finite()) {
        return None;
    }
    let clipped = iqr_clipped(values)?;
    let mean = clipped.iter().sum::<f64>() / clipped.len() as f64;
    let cache = spectral_cache(clipped.len());
    let windowed = clipped
        .iter()
        .zip(&cache.hann)
        .map(|(value, weight)| (value - mean) * weight)
        .collect::<Vec<_>>();

    let mut band_power = 0.0;
    let mut total_power = 0.0;
    for bin in &cache.bins {
        let mut real = 0.0;
        let mut imag = 0.0;
        for idx in 0..windowed.len() {
            real += windowed[idx] * bin.cos[idx];
            imag -= windowed[idx] * bin.sin[idx];
        }
        let power = real * real + imag * imag;
        total_power += power;
        if bin.in_band {
            band_power += power;
        }
    }

    (total_power > EPS).then_some(band_power / (total_power + EPS))
}

fn iqr_clipped(values: &[f64]) -> Option<Vec<f64>> {
    let q25 = quantile(values, 0.25)?;
    let q75 = quantile(values, 0.75)?;
    let median = quantile(values, 0.5)?;
    let iqr = q75 - q25;
    let lower = median - 3.0 * iqr;
    let upper = median + 3.0 * iqr;
    Some(
        values
            .iter()
            .map(|value| value.clamp(lower, upper))
            .collect(),
    )
}

fn quantile(values: &[f64], q: f64) -> Option<f64> {
    let mut values = values.to_vec();
    quantile_linear(&mut values, q)
}

fn spectral_cache(length: usize) -> Arc<SpectralCache> {
    let cache_map = SPECTRAL_CACHE.get_or_init(|| Mutex::new(BTreeMap::new()));
    let mut cache_map = cache_map.lock().expect("spectral cache mutex poisoned");
    if let Some(cache) = cache_map.get(&length) {
        return Arc::clone(cache);
    }
    let cache = Arc::new(build_spectral_cache(length));
    cache_map.insert(length, Arc::clone(&cache));
    cache
}

fn build_spectral_cache(length: usize) -> SpectralCache {
    let hann = if length == 1 {
        vec![1.0]
    } else {
        (0..length)
            .map(|idx| 0.5 * (1.0 - (2.0 * PI * idx as f64 / (length - 1) as f64).cos()))
            .collect()
    };
    let bins = (1..=length / 2)
        .map(|k| {
            let mut cos = Vec::with_capacity(length);
            let mut sin = Vec::with_capacity(length);
            for idx in 0..length {
                let angle = 2.0 * PI * k as f64 * idx as f64 / length as f64;
                cos.push(angle.cos());
                sin.push(angle.sin());
            }
            SpectralBin {
                in_band: in_target_period_band(length, k),
                cos,
                sin,
            }
        })
        .collect();
    SpectralCache { hann, bins }
}

fn in_target_period_band(length: usize, k: usize) -> bool {
    if k == 0 {
        return false;
    }
    let period = length as f64 / k as f64;
    (2.0..=5.0).contains(&period)
}

#[cfg(test)]
mod tests {
    use super::{
        dripping_stone_ratio_from_rows, in_target_period_band, iqr_clipped,
        selected_volume_series_from_rows, spectral_ratio_from_volume_series,
    };
    use std::f64::consts::PI;

    fn times(values: &[&str]) -> Vec<Option<String>> {
        values
            .iter()
            .map(|value| Some((*value).to_string()))
            .collect()
    }

    #[test]
    fn selected_volume_series_excludes_open_and_close_auction_minutes() {
        let indices = (0..5).collect::<Vec<_>>();
        let trade_times = times(&["09:30:00", "09:31:00", "09:32:00", "14:57:00", "14:58:00"]);
        let volume = vec![Some(9.0), Some(1.0), Some(2.0), Some(3.0), Some(8.0)];
        let expected_times = vec![
            "09:31:00".to_string(),
            "09:32:00".to_string(),
            "14:57:00".to_string(),
        ];

        assert_eq!(
            selected_volume_series_from_rows(&indices, &trade_times, &volume, &expected_times),
            Some(vec![1.0, 2.0, 3.0])
        );
    }

    #[test]
    fn selected_volume_series_requires_complete_expected_minutes() {
        let indices = (0..2).collect::<Vec<_>>();
        let trade_times = times(&["09:31:00", "09:33:00"]);
        let volume = vec![Some(1.0), Some(3.0)];
        let expected_times = vec![
            "09:31:00".to_string(),
            "09:32:00".to_string(),
            "09:33:00".to_string(),
        ];

        assert_eq!(
            selected_volume_series_from_rows(&indices, &trade_times, &volume, &expected_times),
            None
        );
    }

    #[test]
    fn iqr_clip_uses_median_plus_minus_three_iqr() {
        let clipped = iqr_clipped(&[1.0, 2.0, 3.0, 4.0, 100.0]).expect("clipped");

        assert_eq!(clipped, vec![1.0, 2.0, 3.0, 4.0, 9.0]);
    }

    #[test]
    fn constant_series_has_no_nonzero_frequency_power() {
        assert_eq!(spectral_ratio_from_volume_series(&vec![10.0; 32]), None);
    }

    #[test]
    fn three_minute_period_has_high_target_band_power() {
        let target = (0..240)
            .map(|idx| 100.0 + 10.0 * (2.0 * PI * idx as f64 / 3.0).sin())
            .collect::<Vec<_>>();
        let slow = (0..240)
            .map(|idx| 100.0 + 10.0 * (2.0 * PI * idx as f64 / 20.0).sin())
            .collect::<Vec<_>>();

        let target_ratio = spectral_ratio_from_volume_series(&target).expect("target ratio");
        let slow_ratio = spectral_ratio_from_volume_series(&slow).expect("slow ratio");

        assert!(target_ratio > 0.8, "target_ratio={target_ratio}");
        assert!(
            target_ratio > slow_ratio * 5.0,
            "target_ratio={target_ratio}, slow_ratio={slow_ratio}"
        );
    }

    #[test]
    fn target_band_uses_period_length_divided_by_frequency_bin() {
        assert!(!in_target_period_band(240, 47));
        assert!(in_target_period_band(240, 48));
        assert!(in_target_period_band(240, 120));
        assert!(!in_target_period_band(240, 40));
    }

    #[test]
    fn raw_returns_none_when_any_selected_volume_is_missing() {
        let indices = (0..6).collect::<Vec<_>>();
        let trade_times = times(&[
            "09:31:00", "09:32:00", "09:33:00", "09:34:00", "09:35:00", "09:36:00",
        ]);
        let volume = vec![Some(1.0), Some(2.0), None, Some(4.0), Some(5.0), Some(6.0)];
        let expected_times = trade_times
            .iter()
            .map(|value| value.clone().expect("time"))
            .collect::<Vec<_>>();

        assert_eq!(
            dripping_stone_ratio_from_rows(&indices, &trade_times, &volume, &expected_times),
            None
        );
    }
}

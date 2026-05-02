use std::collections::{BTreeSet, HashMap};

use rayon::prelude::*;

use crate::core::{
    AssetClass, DatasetId, FactorContext, FactorRowKey, FactorSeries, FactorSpec, FactorValue,
    Frequency, IntradayDailyRawRequest, IntradayDailyRawSeries, IntradayDailyRawSpec, Lookback,
};
use crate::data::{DataPool, Table};
use crate::error::Result;
use crate::factor::common::{
    clean_intraday_value, intraday_time_in_range, stock_minute_raw_spec, PanelColumn,
};
use crate::factor::Factor;
use crate::operators::{cs_demean_abs, cs_zscore, ts_mean, ts_std_dev};

pub const VOLUME_ENTROPY_RAW_ID: &str = "daily_volume_entropy";
pub const LIQUIDITY_ELASTICITY_RAW_ID: &str = "daily_liquidity_elasticity";

const RAW_VERSION: &str = "0.1.0";
const VERSION: &str = "0.1.0";
const WINDOW: usize = 20;
const ENTROPY_BIN_COUNT: usize = 48;
const ENTROPY_BIN_MINUTES: usize = 5;
const ELASTICITY_LOOKBACK: usize = 5;

pub struct StockDailyUndercurrentSurging;

#[derive(Clone, Debug)]
struct UndercurrentMinuteMatrix {
    times: Vec<String>,
    codes: Vec<String>,
    open: Vec<Option<f64>>,
    high: Vec<Option<f64>>,
    low: Vec<Option<f64>>,
    volume: Vec<Option<f64>>,
}

impl UndercurrentMinuteMatrix {
    fn time_count(&self) -> usize {
        self.times.len()
    }

    fn code_count(&self) -> usize {
        self.codes.len()
    }

    fn offset(&self, time_idx: usize, code_idx: usize) -> usize {
        time_idx * self.code_count() + code_idx
    }
}

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyUndercurrentSurging)
}

fn raw_spec(raw_id: &str) -> IntradayDailyRawSpec {
    stock_minute_raw_spec(raw_id, RAW_VERSION, &["open", "high", "low", "vol"], 1)
}

impl Factor for StockDailyUndercurrentSurging {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "undercurrent_surging".to_string(),
            aliases: Vec::new(),
            name: "Undercurrent Surging".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: [
                "price_volume",
                "volume",
                "liquidity",
                "entropy",
                "intraday",
                "minute_agg",
                "composite",
                "daily",
                "FZZQ",
            ]
            .iter()
            .map(|value| value.to_string())
            .collect(),
            description: "Composite intraday undercurrent factor from volume-distribution entropy and liquidity elasticity.".to_string(),
            dependencies: Vec::new(),
            intraday_raw_dependencies: vec![
                IntradayDailyRawRequest::new(VOLUME_ENTROPY_RAW_ID, WINDOW - 1),
                IntradayDailyRawRequest::new(LIQUIDITY_ELASTICITY_RAW_ID, WINDOW - 1),
            ],
            lookback: Lookback {
                trading_days: WINDOW - 1,
            },
        }
    }

    fn intraday_raw_specs(&self) -> Vec<IntradayDailyRawSpec> {
        vec![
            raw_spec(VOLUME_ENTROPY_RAW_ID),
            raw_spec(LIQUIDITY_ELASTICITY_RAW_ID),
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
        let wants_entropy = requested.contains(VOLUME_ENTROPY_RAW_ID);
        let wants_elasticity = requested.contains(LIQUIDITY_ELASTICITY_RAW_ID);
        if !wants_entropy && !wants_elasticity {
            return Ok(Vec::new());
        }

        let mut entropy_values = Vec::new();
        let mut elasticity_values = Vec::new();
        for trade_date in &context.target_dates {
            let Some(table) = data.minute(DatasetId::StockMinute1m, *trade_date) else {
                continue;
            };
            let matrix = UndercurrentMinuteMatrix::from_table(table)?;
            let entropy = wants_entropy.then(|| volume_entropy_from_matrix(&matrix));
            let elasticity = wants_elasticity.then(|| liquidity_elasticity_from_matrix(&matrix));

            for (code_idx, ts_code) in matrix.codes.iter().enumerate() {
                if let Some(values) = entropy.as_ref() {
                    entropy_values.push(FactorValue {
                        key: FactorRowKey::Daily {
                            trade_date: *trade_date,
                            ts_code: ts_code.clone(),
                        },
                        value: values[code_idx],
                    });
                }
                if let Some(values) = elasticity.as_ref() {
                    elasticity_values.push(FactorValue {
                        key: FactorRowKey::Daily {
                            trade_date: *trade_date,
                            ts_code: ts_code.clone(),
                        },
                        value: values[code_idx],
                    });
                }
            }
        }

        let mut output = Vec::new();
        if wants_entropy {
            output.push(IntradayDailyRawSeries {
                spec: raw_spec(VOLUME_ENTROPY_RAW_ID),
                values: entropy_values,
            });
        }
        if wants_elasticity {
            output.push(IntradayDailyRawSeries {
                spec: raw_spec(LIQUIDITY_ELASTICITY_RAW_ID),
                values: elasticity_values,
            });
        }
        Ok(output)
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let panel = data.intraday_daily_raw_panel(VOLUME_ENTROPY_RAW_ID)?;
        let entropy_distance = panel.column(VOLUME_ENTROPY_RAW_ID)?.cs(cs_demean_abs)?;
        let elasticity_distance = panel
            .column(LIQUIDITY_ELASTICITY_RAW_ID)?
            .cs(cs_demean_abs)?;

        let entropy_component = rolling_component(&entropy_distance)?;
        let elasticity_component = rolling_component(&elasticity_distance)?;
        let factor = average_pair(
            &entropy_component.cs(cs_zscore)?,
            &elasticity_component.cs(cs_zscore)?,
        )?;

        Ok(factor.to_factor_series(self.spec()))
    }
}

impl UndercurrentMinuteMatrix {
    fn from_table(table: &Table) -> Result<Self> {
        let ts_codes = table.required_utf8("ts_code")?;
        let trade_times = table.required_utf8("trade_time")?;
        let open = table.required_f64_cast("open")?;
        let high = table.required_f64_cast("high")?;
        let low = table.required_f64_cast("low")?;
        let volume = table.required_f64_cast("vol")?;

        let mut time_set = BTreeSet::new();
        let mut code_set = BTreeSet::new();
        for idx in 0..table.len {
            let Some(time) = trade_times[idx].as_deref() else {
                continue;
            };
            if !intraday_time_in_range(time, "09:31:00", "15:00:00") {
                continue;
            }
            if let Some(code) = ts_codes[idx].clone() {
                time_set.insert(time.to_string());
                code_set.insert(code);
            }
        }

        let times = time_set.into_iter().collect::<Vec<_>>();
        let codes = code_set.into_iter().collect::<Vec<_>>();
        let time_lookup = times
            .iter()
            .enumerate()
            .map(|(idx, value)| (value.clone(), idx))
            .collect::<HashMap<_, _>>();
        let code_lookup = codes
            .iter()
            .enumerate()
            .map(|(idx, value)| (value.clone(), idx))
            .collect::<HashMap<_, _>>();

        let shape_len = times.len() * codes.len();
        let mut open_values = vec![None; shape_len];
        let mut high_values = vec![None; shape_len];
        let mut low_values = vec![None; shape_len];
        let mut volume_values = vec![None; shape_len];
        for idx in 0..table.len {
            let (Some(time), Some(code)) = (trade_times[idx].clone(), ts_codes[idx].clone()) else {
                continue;
            };
            let (Some(time_idx), Some(code_idx)) = (
                time_lookup.get(&time).copied(),
                code_lookup.get(&code).copied(),
            ) else {
                continue;
            };
            let offset = time_idx * codes.len() + code_idx;
            open_values[offset] = clean_intraday_value(open[idx]);
            high_values[offset] = clean_intraday_value(high[idx]);
            low_values[offset] = clean_intraday_value(low[idx]);
            volume_values[offset] = clean_intraday_value(volume[idx]);
        }

        Ok(Self {
            times,
            codes,
            open: open_values,
            high: high_values,
            low: low_values,
            volume: volume_values,
        })
    }
}

fn rolling_component(values: &PanelColumn) -> Result<PanelColumn> {
    let mean20 = values.ts(|series| ts_mean(series, WINDOW, 1))?;
    let std20 = values.ts(|series| ts_std_dev(series, WINDOW, 1))?;
    average_pair(&mean20.cs(cs_zscore)?, &std20.cs(cs_zscore)?)
}

fn volume_entropy_from_matrix(matrix: &UndercurrentMinuteMatrix) -> Vec<Option<f64>> {
    let time_count = matrix.time_count();
    let code_count = matrix.code_count();
    let required_minutes = ENTROPY_BIN_COUNT * ENTROPY_BIN_MINUTES;
    if time_count < required_minutes || code_count == 0 {
        return vec![None; code_count];
    }

    let market_totals = market_volume_totals(matrix);
    (0..code_count)
        .into_par_iter()
        .map(|code_idx| volume_entropy_for_code(matrix, &market_totals, code_idx))
        .collect()
}

fn volume_entropy_for_code(
    matrix: &UndercurrentMinuteMatrix,
    market_totals: &[Option<f64>],
    code_idx: usize,
) -> Option<f64> {
    let mut bin_masses = [0.0; ENTROPY_BIN_COUNT];
    for (bin_idx, mass) in bin_masses.iter_mut().enumerate() {
        for offset in 0..ENTROPY_BIN_MINUTES {
            let time_idx = bin_idx * ENTROPY_BIN_MINUTES + offset;
            let Some(market_total) = market_totals[time_idx] else {
                continue;
            };
            if market_total.abs() <= f64::EPSILON {
                continue;
            }
            let Some(volume) = clean_nonnegative(matrix.volume[matrix.offset(time_idx, code_idx)])
            else {
                continue;
            };
            *mass += volume / market_total;
        }
    }

    let total_mass = bin_masses.iter().sum::<f64>();
    if total_mass.abs() <= f64::EPSILON {
        return None;
    }
    let entropy = bin_masses
        .iter()
        .filter_map(|mass| {
            let probability = *mass / total_mass;
            (probability > 0.0).then_some(-probability * probability.ln())
        })
        .sum::<f64>();
    Some(entropy)
}

fn market_volume_totals(matrix: &UndercurrentMinuteMatrix) -> Vec<Option<f64>> {
    (0..matrix.time_count())
        .map(|time_idx| {
            let total = (0..matrix.code_count())
                .filter_map(|code_idx| {
                    clean_nonnegative(matrix.volume[matrix.offset(time_idx, code_idx)])
                })
                .sum::<f64>();
            (total > 0.0).then_some(total)
        })
        .collect()
}

fn liquidity_elasticity_from_matrix(matrix: &UndercurrentMinuteMatrix) -> Vec<Option<f64>> {
    let code_count = matrix.code_count();
    if matrix.time_count() <= ELASTICITY_LOOKBACK || code_count == 0 {
        return vec![None; code_count];
    }
    let selected_times = matrix
        .times
        .iter()
        .enumerate()
        .filter_map(|(idx, time)| {
            intraday_time_in_range(time, "09:31:00", "14:57:00").then_some(idx)
        })
        .collect::<Vec<_>>();
    if selected_times.is_empty() {
        return vec![None; code_count];
    }

    (0..code_count)
        .into_par_iter()
        .map(|code_idx| liquidity_elasticity_for_code(matrix, &selected_times, code_idx))
        .collect()
}

fn liquidity_elasticity_for_code(
    matrix: &UndercurrentMinuteMatrix,
    selected_times: &[usize],
    code_idx: usize,
) -> Option<f64> {
    let mut spike_amplitudes = Vec::new();
    let mut normal_amplitudes = Vec::new();
    for (selected_pos, time_idx) in selected_times.iter().enumerate() {
        let offset = matrix.offset(*time_idx, code_idx);
        let Some(amplitude) =
            amplitude(matrix.open[offset], matrix.high[offset], matrix.low[offset])
        else {
            continue;
        };

        let spike = selected_pos >= ELASTICITY_LOOKBACK
            && volume_spike(matrix, selected_times, selected_pos, code_idx).unwrap_or(false);
        if spike {
            spike_amplitudes.push(amplitude);
        } else {
            normal_amplitudes.push(amplitude);
        }
    }

    let spike_mean = mean(&spike_amplitudes)?;
    let normal_mean = mean(&normal_amplitudes)?;
    if normal_mean.abs() <= f64::EPSILON {
        return None;
    }
    Some(1.0 - spike_mean / normal_mean)
}

fn volume_spike(
    matrix: &UndercurrentMinuteMatrix,
    selected_times: &[usize],
    selected_pos: usize,
    code_idx: usize,
) -> Option<bool> {
    if selected_pos < ELASTICITY_LOOKBACK {
        return None;
    }
    let current_time_idx = selected_times[selected_pos];
    let current = clean_nonnegative(matrix.volume[matrix.offset(current_time_idx, code_idx)])?;
    let mut sum = 0.0;
    for lag in 1..=ELASTICITY_LOOKBACK {
        let time_idx = selected_times[selected_pos - lag];
        let value = clean_nonnegative(matrix.volume[matrix.offset(time_idx, code_idx)])?;
        sum += value;
    }
    let previous_mean = sum / ELASTICITY_LOOKBACK as f64;
    Some(current > 2.0 * previous_mean)
}

fn amplitude(open: Option<f64>, high: Option<f64>, low: Option<f64>) -> Option<f64> {
    let (Some(open), Some(high), Some(low)) = (clean(open), clean(high), clean(low)) else {
        return None;
    };
    if open <= 0.0 {
        return None;
    }
    Some((high - low) / open)
}

fn average_pair(left: &PanelColumn, right: &PanelColumn) -> Result<PanelColumn> {
    left.zip_binary(right, |left, right| match (clean(left), clean(right)) {
        (Some(left), Some(right)) => Some((left + right) / 2.0),
        _ => None,
    })
}

fn mean(values: &[f64]) -> Option<f64> {
    (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64)
}

fn clean(value: Option<f64>) -> Option<f64> {
    value.filter(|value| !value.is_nan())
}

fn clean_nonnegative(value: Option<f64>) -> Option<f64> {
    clean(value).filter(|value| *value >= 0.0)
}

#[cfg(test)]
mod tests {
    use super::{
        liquidity_elasticity_from_matrix, market_volume_totals, volume_entropy_from_matrix,
        volume_spike, UndercurrentMinuteMatrix, ENTROPY_BIN_COUNT, ENTROPY_BIN_MINUTES,
    };

    fn matrix(time_count: usize, code_count: usize) -> UndercurrentMinuteMatrix {
        let times = (0..time_count)
            .map(|idx| format!("09:{:02}:00", 31 + idx))
            .collect::<Vec<_>>();
        UndercurrentMinuteMatrix {
            times,
            codes: (0..code_count).map(|idx| format!("S{idx}")).collect(),
            open: vec![Some(1.0); time_count * code_count],
            high: vec![Some(2.0); time_count * code_count],
            low: vec![Some(1.0); time_count * code_count],
            volume: vec![Some(1.0); time_count * code_count],
        }
    }

    #[test]
    fn volume_entropy_uses_forty_eight_equal_five_minute_bins() {
        let matrix = matrix(ENTROPY_BIN_COUNT * ENTROPY_BIN_MINUTES, 2);

        let entropy = volume_entropy_from_matrix(&matrix);

        assert!((entropy[0].expect("entropy") - (ENTROPY_BIN_COUNT as f64).ln()).abs() < 1e-12);
        assert!((entropy[1].expect("entropy") - (ENTROPY_BIN_COUNT as f64).ln()).abs() < 1e-12);
    }

    #[test]
    fn market_volume_totals_normalize_each_minute_cross_section() {
        let mut matrix = matrix(ENTROPY_BIN_COUNT * ENTROPY_BIN_MINUTES, 2);
        let first_code = matrix.offset(0, 0);
        let second_code = matrix.offset(0, 1);
        matrix.volume[first_code] = Some(3.0);
        matrix.volume[second_code] = Some(1.0);

        let totals = market_volume_totals(&matrix);

        assert_eq!(totals[0], Some(4.0));
    }

    #[test]
    fn liquidity_elasticity_uses_previous_five_volume_mean_and_two_times_threshold() {
        let mut matrix = matrix(7, 1);
        matrix.times = vec![
            "09:31:00".to_string(),
            "09:32:00".to_string(),
            "09:33:00".to_string(),
            "09:34:00".to_string(),
            "09:35:00".to_string(),
            "09:36:00".to_string(),
            "09:37:00".to_string(),
        ];
        matrix.volume = vec![
            Some(1.0),
            Some(1.0),
            Some(1.0),
            Some(1.0),
            Some(1.0),
            Some(3.0),
            Some(1.0),
        ];
        matrix.high[5] = Some(5.0);

        let selected_times = (0..7).collect::<Vec<_>>();
        assert_eq!(volume_spike(&matrix, &selected_times, 5, 0), Some(true));

        let elasticity = liquidity_elasticity_from_matrix(&matrix);
        assert_eq!(elasticity[0], Some(1.0 - 4.0 / 1.0));
    }

    #[test]
    fn liquidity_elasticity_outputs_none_without_spike_or_normal_side() {
        let mut matrix = matrix(7, 1);
        matrix.times = vec![
            "09:31:00".to_string(),
            "09:32:00".to_string(),
            "09:33:00".to_string(),
            "09:34:00".to_string(),
            "09:35:00".to_string(),
            "09:36:00".to_string(),
            "09:37:00".to_string(),
        ];

        let elasticity = liquidity_elasticity_from_matrix(&matrix);

        assert_eq!(elasticity[0], None);
    }
}

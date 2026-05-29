use std::any::Any;
use std::collections::{BTreeMap, BTreeSet, VecDeque};

use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorRowKey, FactorSeries, FactorSpec,
    FactorValue, Frequency, IntradayDailyRawAuxiliaryRequest, IntradayDailyRawRequest,
    IntradayDailyRawSeries, IntradayDailyRawSpec, Lookback,
};
use crate::data::{DataPool, Table};
use crate::error::{err, Result};
use crate::factor::common::stock_daily_ops::neutralize_size_sector;
use crate::factor::common::{clean_intraday_value, stock_minute_raw_spec};
use crate::factor::IntradayRawMaterializeMode;

pub const VERSION: &str = "0.3.0";
pub const RAW_VERSION: &str = "0.3.0";
pub const PROVIDER_KEY: &str = "kyzq_peak_valley_provider";

const WINDOW_DAYS: usize = 20;
const MINUTES_PER_DAY: usize = 240;
const EPS: f64 = f64::EPSILON;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum PeakValleyMetric {
    VolumePeakMinuteCount,
    VolumeValleyRelativeVwap,
    VolumeValleyVwapPercentile,
    VolumePeakIntervalKurtosis,
    VolumeValleyRidgeVwapRatio,
    VolumePeakRidgeSameTimeCountCorr,
    PricePeakMinuteCount,
    PriceRidgeMinuteReturn,
    PriceValleyRelativeVwap,
    PriceValleyVwapPercentile,
    PriceRidgeIntervalSkewness,
    PriceJumpAmountCorr,
}

#[derive(Clone, Copy, Debug)]
struct MetricInfo {
    id: &'static str,
    alias: &'static str,
    display_name: &'static str,
    raw_id: &'static str,
    group_tag: &'static str,
    description: &'static str,
}

impl PeakValleyMetric {
    pub fn id(self) -> &'static str {
        self.info().id
    }

    pub fn raw_id(self) -> &'static str {
        self.info().raw_id
    }

    fn info(self) -> MetricInfo {
        match self {
            Self::VolumePeakMinuteCount => MetricInfo {
                id: "volume_peak_minute_count",
                alias: "Volume Peak Minute Count",
                display_name: "volume_peak_minute_count",
                raw_id: "daily_kyzq_volume_peak_minute_count_raw",
                group_tag: "volume",
                description: "Count of volume-peak minutes over a strict 20-day by-time 1-minute matrix.",
            },
            Self::VolumeValleyRelativeVwap => MetricInfo {
                id: "volume_valley_relative_vwap",
                alias: "Volume Valley Relative VWAP",
                display_name: "volume_valley_relative_vwap",
                raw_id: "daily_kyzq_volume_valley_relative_vwap_raw",
                group_tag: "volume",
                description: "20-day mean of volume-valley VWAP divided by full-day VWAP.",
            },
            Self::VolumeValleyVwapPercentile => MetricInfo {
                id: "volume_valley_vwap_percentile",
                alias: "Volume Valley VWAP Percentile",
                display_name: "volume_valley_vwap_percentile",
                raw_id: "daily_kyzq_volume_valley_vwap_percentile_raw",
                group_tag: "volume",
                description: "20-day mean of volume-valley VWAP percentile in the daily price range.",
            },
            Self::VolumePeakIntervalKurtosis => MetricInfo {
                id: "volume_peak_interval_kurtosis",
                alias: "Volume Peak Interval Kurtosis",
                display_name: "volume_peak_interval_kurtosis",
                raw_id: "daily_kyzq_volume_peak_interval_kurtosis_raw",
                group_tag: "volume",
                description: "Kurtosis of flattened intervals between adjacent volume-peak minutes over 20 days.",
            },
            Self::VolumeValleyRidgeVwapRatio => MetricInfo {
                id: "volume_valley_ridge_vwap_ratio",
                alias: "Volume Valley Ridge VWAP Ratio",
                display_name: "volume_valley_ridge_vwap_ratio",
                raw_id: "daily_kyzq_volume_valley_ridge_vwap_ratio_raw",
                group_tag: "volume",
                description: "20-day mean of volume-valley VWAP divided by volume-ridge VWAP.",
            },
            Self::VolumePeakRidgeSameTimeCountCorr => MetricInfo {
                id: "volume_peak_ridge_same_time_count_corr",
                alias: "Volume Peak Ridge Same-Time Count Corr",
                display_name: "volume_peak_ridge_same_time_count_corr",
                raw_id: "daily_kyzq_volume_peak_ridge_same_time_count_corr_raw",
                group_tag: "volume",
                description: "Pearson correlation between same-time 20-day counts of volume peaks and ridges.",
            },
            Self::PricePeakMinuteCount => MetricInfo {
                id: "price_peak_minute_count",
                alias: "Price Peak Minute Count",
                display_name: "price_peak_minute_count",
                raw_id: "daily_kyzq_price_peak_minute_count_raw",
                group_tag: "price",
                description: "Count of price-peak minutes over a strict 20-day by-time 1-minute matrix.",
            },
            Self::PriceRidgeMinuteReturn => MetricInfo {
                id: "price_ridge_minute_return",
                alias: "Price Ridge Minute Return",
                display_name: "price_ridge_minute_return",
                raw_id: "daily_kyzq_price_ridge_minute_return_raw",
                group_tag: "price",
                description: "Sum of 1-minute returns at price-ridge minutes over 20 days.",
            },
            Self::PriceValleyRelativeVwap => MetricInfo {
                id: "price_valley_relative_vwap",
                alias: "Price Valley Relative VWAP",
                display_name: "price_valley_relative_vwap",
                raw_id: "daily_kyzq_price_valley_relative_vwap_raw",
                group_tag: "price",
                description: "20-day mean of price-valley VWAP divided by full-day VWAP.",
            },
            Self::PriceValleyVwapPercentile => MetricInfo {
                id: "price_valley_vwap_percentile",
                alias: "Price Valley VWAP Percentile",
                display_name: "price_valley_vwap_percentile",
                raw_id: "daily_kyzq_price_valley_vwap_percentile_raw",
                group_tag: "price",
                description: "20-day mean of price-valley VWAP percentile in the daily price range.",
            },
            Self::PriceRidgeIntervalSkewness => MetricInfo {
                id: "price_ridge_interval_skewness",
                alias: "Price Ridge Interval Skewness",
                display_name: "price_ridge_interval_skewness",
                raw_id: "daily_kyzq_price_ridge_interval_skewness_raw",
                group_tag: "price",
                description: "Skewness of flattened intervals between adjacent price-ridge minutes over 20 days.",
            },
            Self::PriceJumpAmountCorr => MetricInfo {
                id: "price_jump_amount_corr",
                alias: "Price Jump Amount Corr",
                display_name: "price_jump_amount_corr",
                raw_id: "daily_kyzq_price_jump_amount_corr_raw",
                group_tag: "price",
                description: "Pearson correlation between price-jump minute amount and next-minute amount over 20 days.",
            },
        }
    }
}

pub fn all_metrics() -> [PeakValleyMetric; 12] {
    [
        PeakValleyMetric::VolumePeakMinuteCount,
        PeakValleyMetric::VolumeValleyRelativeVwap,
        PeakValleyMetric::VolumeValleyVwapPercentile,
        PeakValleyMetric::VolumePeakIntervalKurtosis,
        PeakValleyMetric::VolumeValleyRidgeVwapRatio,
        PeakValleyMetric::VolumePeakRidgeSameTimeCountCorr,
        PeakValleyMetric::PricePeakMinuteCount,
        PeakValleyMetric::PriceRidgeMinuteReturn,
        PeakValleyMetric::PriceValleyRelativeVwap,
        PeakValleyMetric::PriceValleyVwapPercentile,
        PeakValleyMetric::PriceRidgeIntervalSkewness,
        PeakValleyMetric::PriceJumpAmountCorr,
    ]
}

pub fn factor_spec(metric: PeakValleyMetric) -> FactorSpec {
    let info = metric.info();
    FactorSpec {
        id: info.id.to_string(),
        aliases: vec![info.alias.to_string()],
        name: info.display_name.to_string(),
        asset_class: AssetClass::Stock,
        frequency: Frequency::Daily,
        version: VERSION.to_string(),
        tags: tags(info.group_tag),
        description: format!(
            "KYZQ peak/ridge/valley 1-minute matrix factor. {} Strictly requires a complete target-inclusive 20 trading day x 240 minute matrix and is neutralized by Barra SIZE and SW sector.",
            info.description
        ),
        dependencies: vec![
            DataRequest::new(DatasetId::StockDailyPv, &["pre_close"]),
            DataRequest::new(DatasetId::StockBarraDaily, &["SIZE"]),
            DataRequest::new(DatasetId::StockSwClassification, &["l1_code"]),
        ],
        intraday_raw_dependencies: vec![IntradayDailyRawRequest::new(
            info.raw_id,
            WINDOW_DAYS - 1,
        )],
        lookback: Lookback {
            trading_days: WINDOW_DAYS - 1,
        },
    }
}

pub fn raw_spec(metric: PeakValleyMetric) -> IntradayDailyRawSpec {
    stock_minute_raw_spec(
        metric.raw_id(),
        RAW_VERSION,
        &["high", "low", "close", "vol", "amount"],
        WINDOW_DAYS,
    )
}

pub fn intraday_raw_materialize_mode() -> IntradayRawMaterializeMode {
    IntradayRawMaterializeMode::Stateful
}

pub fn initial_intraday_raw_state() -> Box<dyn Any + Send> {
    Box::new(PeakValleyState::default())
}

pub fn intraday_raw_auxiliary_requirements(
    raw_ids: &[String],
) -> Vec<IntradayDailyRawAuxiliaryRequest> {
    let requested = requested_metrics(raw_ids);
    if requested.is_empty() {
        Vec::new()
    } else {
        vec![IntradayDailyRawAuxiliaryRequest::new(
            DataRequest::new(DatasetId::StockDailyPv, &["pre_close"]),
            0,
        )]
    }
}

pub fn minute_compute_stateful_many(
    raw_ids: &[String],
    context: &FactorContext,
    data: &DataPool,
    state: &mut dyn Any,
) -> Result<Vec<IntradayDailyRawSeries>> {
    let requested = requested_metrics(raw_ids);
    if requested.is_empty() {
        return Ok(Vec::new());
    }

    let state = state
        .downcast_mut::<PeakValleyState>()
        .ok_or_else(|| err("KYZQ peak/valley raw received incompatible state"))?;
    let trade_date = *context
        .target_dates
        .first()
        .ok_or_else(|| err("KYZQ peak/valley raw requires one target date"))?;

    let (day, current_stocks) = match data.minute(DatasetId::StockMinute1m, trade_date) {
        Some(table) => strict_day_from_table(table, pre_close_map(data, trade_date)?)?,
        None => (StrictMinuteDay::default(), BTreeSet::new()),
    };
    state.push_day(day);
    let raw_values = state.values_for_current_stocks(&current_stocks, &requested);

    let mut output = Vec::new();
    for metric in all_metrics() {
        if !requested.contains(&metric) {
            continue;
        }
        let values = raw_values
            .iter()
            .map(|(ts_code, values)| FactorValue {
                key: FactorRowKey::Daily {
                    trade_date,
                    ts_code: ts_code.clone(),
                },
                value: values.value(metric),
            })
            .collect();
        output.push(IntradayDailyRawSeries {
            spec: raw_spec(metric),
            values,
        });
    }
    Ok(output)
}

pub fn compute_factor(metric: PeakValleyMetric, data: &DataPool) -> Result<FactorSeries> {
    let panel = data.intraday_daily_raw_panel(metric.raw_id())?;
    let raw = panel.column(metric.raw_id())?;
    let factor = neutralize_size_sector(&raw, panel, data)?;
    Ok(factor.to_factor_series(factor_spec(metric)))
}

#[macro_export]
macro_rules! define_kyzq_peak_valley_factor {
    ($struct_name:ident, $metric:expr) => {
        pub struct $struct_name;

        pub fn create() -> Box<dyn $crate::factor::Factor> {
            Box::new($struct_name)
        }

        impl $crate::factor::Factor for $struct_name {
            fn spec(&self) -> $crate::core::FactorSpec {
                $crate::factor::common::kyzq_peak_valley::factor_spec($metric)
            }

            fn intraday_raw_specs(&self) -> Vec<$crate::core::IntradayDailyRawSpec> {
                vec![$crate::factor::common::kyzq_peak_valley::raw_spec($metric)]
            }

            fn intraday_raw_provider_key(&self, _raw_id: &str) -> String {
                $crate::factor::common::kyzq_peak_valley::PROVIDER_KEY.to_string()
            }

            fn intraday_raw_materialize_mode(
                &self,
                _raw_ids: &[String],
            ) -> $crate::factor::IntradayRawMaterializeMode {
                $crate::factor::common::kyzq_peak_valley::intraday_raw_materialize_mode()
            }

            fn initial_intraday_raw_state(
                &self,
                _raw_ids: &[String],
            ) -> Box<dyn std::any::Any + Send> {
                $crate::factor::common::kyzq_peak_valley::initial_intraday_raw_state()
            }

            fn intraday_raw_auxiliary_requirements(
                &self,
                raw_ids: &[String],
            ) -> Vec<$crate::core::IntradayDailyRawAuxiliaryRequest> {
                $crate::factor::common::kyzq_peak_valley::intraday_raw_auxiliary_requirements(
                    raw_ids,
                )
            }

            fn minute_compute_stateful_many(
                &self,
                raw_ids: &[String],
                context: &$crate::core::FactorContext,
                data: &$crate::data::DataPool,
                state: &mut dyn std::any::Any,
            ) -> $crate::error::Result<Vec<$crate::core::IntradayDailyRawSeries>> {
                $crate::factor::common::kyzq_peak_valley::minute_compute_stateful_many(
                    raw_ids, context, data, state,
                )
            }

            fn compute(
                &self,
                _context: &$crate::core::FactorContext,
                data: &$crate::data::DataPool,
            ) -> $crate::error::Result<$crate::core::FactorSeries> {
                $crate::factor::common::kyzq_peak_valley::compute_factor($metric, data)
            }
        }
    };
}

#[derive(Debug, Default)]
pub struct PeakValleyState {
    days: VecDeque<StrictMinuteDay>,
}

#[derive(Clone, Debug, Default)]
struct StrictMinuteDay {
    by_stock: BTreeMap<String, StrictStockDay>,
}

#[derive(Clone, Debug)]
struct StrictStockDay {
    points: [MinutePoint; MINUTES_PER_DAY],
    pre_close: f64,
}

#[derive(Clone, Copy, Debug, Default)]
struct MinutePoint {
    high: f64,
    low: f64,
    close: f64,
    vol: f64,
    amount: f64,
}

#[derive(Clone, Copy, Debug, Default)]
struct MetricValues {
    volume_peak_minute_count: Option<f64>,
    volume_valley_relative_vwap: Option<f64>,
    volume_valley_vwap_percentile: Option<f64>,
    volume_peak_interval_kurtosis: Option<f64>,
    volume_valley_ridge_vwap_ratio: Option<f64>,
    volume_peak_ridge_same_time_count_corr: Option<f64>,
    price_peak_minute_count: Option<f64>,
    price_ridge_minute_return: Option<f64>,
    price_valley_relative_vwap: Option<f64>,
    price_valley_vwap_percentile: Option<f64>,
    price_ridge_interval_skewness: Option<f64>,
    price_jump_amount_corr: Option<f64>,
}

impl MetricValues {
    fn value(self, metric: PeakValleyMetric) -> Option<f64> {
        match metric {
            PeakValleyMetric::VolumePeakMinuteCount => self.volume_peak_minute_count,
            PeakValleyMetric::VolumeValleyRelativeVwap => self.volume_valley_relative_vwap,
            PeakValleyMetric::VolumeValleyVwapPercentile => self.volume_valley_vwap_percentile,
            PeakValleyMetric::VolumePeakIntervalKurtosis => self.volume_peak_interval_kurtosis,
            PeakValleyMetric::VolumeValleyRidgeVwapRatio => self.volume_valley_ridge_vwap_ratio,
            PeakValleyMetric::VolumePeakRidgeSameTimeCountCorr => {
                self.volume_peak_ridge_same_time_count_corr
            }
            PeakValleyMetric::PricePeakMinuteCount => self.price_peak_minute_count,
            PeakValleyMetric::PriceRidgeMinuteReturn => self.price_ridge_minute_return,
            PeakValleyMetric::PriceValleyRelativeVwap => self.price_valley_relative_vwap,
            PeakValleyMetric::PriceValleyVwapPercentile => self.price_valley_vwap_percentile,
            PeakValleyMetric::PriceRidgeIntervalSkewness => self.price_ridge_interval_skewness,
            PeakValleyMetric::PriceJumpAmountCorr => self.price_jump_amount_corr,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct MatrixStates {
    volume_peak: [[bool; MINUTES_PER_DAY]; WINDOW_DAYS],
    volume_ridge: [[bool; MINUTES_PER_DAY]; WINDOW_DAYS],
    volume_valley: [[bool; MINUTES_PER_DAY]; WINDOW_DAYS],
    price_jump: [[bool; MINUTES_PER_DAY]; WINDOW_DAYS],
    price_peak: [[bool; MINUTES_PER_DAY]; WINDOW_DAYS],
    price_ridge: [[bool; MINUTES_PER_DAY]; WINDOW_DAYS],
    price_valley: [[bool; MINUTES_PER_DAY]; WINDOW_DAYS],
}

impl Default for MatrixStates {
    fn default() -> Self {
        Self {
            volume_peak: [[false; MINUTES_PER_DAY]; WINDOW_DAYS],
            volume_ridge: [[false; MINUTES_PER_DAY]; WINDOW_DAYS],
            volume_valley: [[false; MINUTES_PER_DAY]; WINDOW_DAYS],
            price_jump: [[false; MINUTES_PER_DAY]; WINDOW_DAYS],
            price_peak: [[false; MINUTES_PER_DAY]; WINDOW_DAYS],
            price_ridge: [[false; MINUTES_PER_DAY]; WINDOW_DAYS],
            price_valley: [[false; MINUTES_PER_DAY]; WINDOW_DAYS],
        }
    }
}

impl PeakValleyState {
    fn push_day(&mut self, day: StrictMinuteDay) {
        self.days.push_back(day);
        while self.days.len() > WINDOW_DAYS {
            self.days.pop_front();
        }
    }

    fn values_for_current_stocks(
        &self,
        current_stocks: &BTreeSet<String>,
        requested: &BTreeSet<PeakValleyMetric>,
    ) -> BTreeMap<String, MetricValues> {
        current_stocks
            .iter()
            .map(|ts_code| {
                let values = self
                    .strict_window_for(ts_code)
                    .map(|days| compute_window_metrics(&days, requested))
                    .unwrap_or_default();
                (ts_code.clone(), values)
            })
            .collect()
    }

    fn strict_window_for(&self, ts_code: &str) -> Option<Vec<&StrictStockDay>> {
        if self.days.len() != WINDOW_DAYS {
            return None;
        }
        let mut output = Vec::with_capacity(WINDOW_DAYS);
        for day in &self.days {
            output.push(day.by_stock.get(ts_code)?);
        }
        Some(output)
    }
}

fn compute_window_metrics(
    days: &[&StrictStockDay],
    requested: &BTreeSet<PeakValleyMetric>,
) -> MetricValues {
    if days.len() != WINDOW_DAYS {
        return MetricValues::default();
    }
    let states = classify_states(days);
    let mut values = MetricValues::default();

    if requested.contains(&PeakValleyMetric::VolumePeakMinuteCount) {
        values.volume_peak_minute_count = Some(count_flags(&states.volume_peak) as f64);
    }
    if requested.contains(&PeakValleyMetric::VolumeValleyRelativeVwap) {
        values.volume_valley_relative_vwap = mean_daily_state_ratio(days, &states.volume_valley);
    }
    if requested.contains(&PeakValleyMetric::VolumeValleyVwapPercentile) {
        values.volume_valley_vwap_percentile =
            mean_daily_state_percentile(days, &states.volume_valley);
    }
    if requested.contains(&PeakValleyMetric::VolumePeakIntervalKurtosis) {
        values.volume_peak_interval_kurtosis =
            kurtosis(&daily_flag_intervals(&states.volume_peak), 4);
    }
    if requested.contains(&PeakValleyMetric::VolumeValleyRidgeVwapRatio) {
        values.volume_valley_ridge_vwap_ratio =
            mean_daily_state_pair_vwap_ratio(days, &states.volume_valley, &states.volume_ridge);
    }
    if requested.contains(&PeakValleyMetric::VolumePeakRidgeSameTimeCountCorr) {
        values.volume_peak_ridge_same_time_count_corr =
            same_time_count_corr(&states.volume_peak, &states.volume_ridge);
    }
    if requested.contains(&PeakValleyMetric::PricePeakMinuteCount) {
        values.price_peak_minute_count = Some(count_flags(&states.price_peak) as f64);
    }
    if requested.contains(&PeakValleyMetric::PriceRidgeMinuteReturn) {
        values.price_ridge_minute_return = sum_state_returns(days, &states.price_ridge);
    }
    if requested.contains(&PeakValleyMetric::PriceValleyRelativeVwap) {
        values.price_valley_relative_vwap = mean_daily_state_ratio(days, &states.price_valley);
    }
    if requested.contains(&PeakValleyMetric::PriceValleyVwapPercentile) {
        values.price_valley_vwap_percentile =
            mean_daily_state_percentile(days, &states.price_valley);
    }
    if requested.contains(&PeakValleyMetric::PriceRidgeIntervalSkewness) {
        values.price_ridge_interval_skewness =
            skewness(&daily_flag_intervals(&states.price_ridge), 3);
    }
    if requested.contains(&PeakValleyMetric::PriceJumpAmountCorr) {
        values.price_jump_amount_corr = price_jump_amount_corr(days, &states.price_jump);
    }

    values
}

fn classify_states(days: &[&StrictStockDay]) -> MatrixStates {
    let mut states = MatrixStates::default();
    let mut volume_erupt = [[false; MINUTES_PER_DAY]; WINDOW_DAYS];
    let mut volume_mild = [[false; MINUTES_PER_DAY]; WINDOW_DAYS];
    let mut price_jump = [[false; MINUTES_PER_DAY]; WINDOW_DAYS];
    let mut price_non_jump = [[false; MINUTES_PER_DAY]; WINDOW_DAYS];

    for minute_idx in 0..MINUTES_PER_DAY {
        let volume_values = days
            .iter()
            .map(|day| day.points[minute_idx].vol)
            .collect::<Vec<_>>();
        let amplitude_values = days
            .iter()
            .map(|day| amplitude(day.points[minute_idx]))
            .collect::<Vec<_>>();
        let (volume_mean, volume_std) = mean_std(&volume_values);
        let (amplitude_mean, amplitude_std) = mean_std(&amplitude_values);
        let volume_threshold = volume_mean + volume_std;
        let amplitude_threshold = amplitude_mean + amplitude_std;
        for day_idx in 0..WINDOW_DAYS {
            let volume = volume_values[day_idx];
            volume_erupt[day_idx][minute_idx] = volume > volume_threshold;
            volume_mild[day_idx][minute_idx] = volume < volume_threshold;
            let amp = amplitude_values[day_idx];
            price_jump[day_idx][minute_idx] = amp > amplitude_threshold;
            price_non_jump[day_idx][minute_idx] = amp < amplitude_threshold;
            states.volume_valley[day_idx][minute_idx] = volume_mild[day_idx][minute_idx];
            states.price_jump[day_idx][minute_idx] = price_jump[day_idx][minute_idx];
            states.price_valley[day_idx][minute_idx] = price_non_jump[day_idx][minute_idx];
        }
    }

    for day_idx in 0..WINDOW_DAYS {
        for minute_idx in 0..MINUTES_PER_DAY {
            if volume_erupt[day_idx][minute_idx] {
                let left = minute_idx.checked_sub(1);
                let right = if minute_idx + 1 < MINUTES_PER_DAY {
                    Some(minute_idx + 1)
                } else {
                    None
                };
                let neighbors = [left, right].into_iter().flatten().collect::<Vec<_>>();
                states.volume_peak[day_idx][minute_idx] =
                    !neighbors.is_empty() && neighbors.iter().all(|idx| volume_mild[day_idx][*idx]);
                states.volume_ridge[day_idx][minute_idx] =
                    neighbors.iter().any(|idx| volume_erupt[day_idx][*idx]);
            }

            if price_jump[day_idx][minute_idx] && minute_idx > 0 && minute_idx + 1 < MINUTES_PER_DAY
            {
                let local_high =
                    price_jump[day_idx][minute_idx - 1] && price_jump[day_idx][minute_idx + 1];
                let local_low = price_non_jump[day_idx][minute_idx - 1]
                    && price_non_jump[day_idx][minute_idx + 1];
                let gap = neighbor_price_gap(
                    days[day_idx].points[minute_idx - 1],
                    days[day_idx].points[minute_idx + 1],
                );
                states.price_peak[day_idx][minute_idx] = !local_high && !gap;
                states.price_ridge[day_idx][minute_idx] = !local_low && gap;
            }
        }
    }

    states
}

fn strict_day_from_table(
    table: &Table,
    pre_close: BTreeMap<String, f64>,
) -> Result<(StrictMinuteDay, BTreeSet<String>)> {
    let ts_codes = table.required_utf8("ts_code")?;
    let trade_times = table.required_utf8("trade_time")?;
    let high = table.required_f64_cast("high")?;
    let low = table.required_f64_cast("low")?;
    let close = table.required_f64_cast("close")?;
    let volume = table.required_f64_cast("vol")?;
    let amount = table.required_f64_cast("amount")?;

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

    let mut current_stocks = BTreeSet::new();
    let mut by_stock = BTreeMap::new();
    for (ts_code, indices) in grouped {
        current_stocks.insert(ts_code.clone());
        let Some(pre_close) = pre_close.get(&ts_code).copied() else {
            continue;
        };
        if pre_close <= EPS {
            continue;
        }
        if let Some(day) = strict_stock_day_from_indices(
            &indices,
            &trade_times,
            &high,
            &low,
            &close,
            &volume,
            &amount,
            pre_close,
        ) {
            by_stock.insert(ts_code, day);
        }
    }

    Ok((StrictMinuteDay { by_stock }, current_stocks))
}

fn strict_stock_day_from_indices(
    indices: &[usize],
    trade_times: &[Option<String>],
    high: &[Option<f64>],
    low: &[Option<f64>],
    close: &[Option<f64>],
    volume: &[Option<f64>],
    amount: &[Option<f64>],
    pre_close: f64,
) -> Option<StrictStockDay> {
    let mut points = [None; MINUTES_PER_DAY];
    for idx in indices {
        let Some(trade_time) = trade_times[*idx].as_deref() else {
            continue;
        };
        let Some(minute_idx) = minute_index(trade_time) else {
            continue;
        };
        if points[minute_idx].is_some() {
            return None;
        }
        points[minute_idx] = Some(MinutePoint {
            high: clean_positive(high[*idx])?,
            low: clean_positive(low[*idx])?,
            close: clean_positive(close[*idx])?,
            vol: clean_nonnegative(volume[*idx])?,
            amount: clean_nonnegative(amount[*idx])?,
        });
    }
    if points.iter().any(Option::is_none) {
        return None;
    }
    let points = points.map(|point| point.expect("checked complete"));
    Some(StrictStockDay { points, pre_close })
}

fn pre_close_map(data: &DataPool, trade_date: i32) -> Result<BTreeMap<String, f64>> {
    let table = data.daily(DatasetId::StockDailyPv)?;
    let trade_dates = table.required_i32("trade_date")?;
    let ts_codes = table.required_utf8("ts_code")?;
    let pre_close = table.required_f64_cast("pre_close")?;
    let mut output = BTreeMap::new();
    for idx in 0..table.len {
        if trade_dates[idx] != Some(trade_date) {
            continue;
        }
        let (Some(ts_code), Some(value)) = (ts_codes[idx].clone(), clean_positive(pre_close[idx]))
        else {
            continue;
        };
        output.insert(ts_code, value);
    }
    Ok(output)
}

fn requested_metrics(raw_ids: &[String]) -> BTreeSet<PeakValleyMetric> {
    let requested = raw_ids.iter().map(String::as_str).collect::<BTreeSet<_>>();
    all_metrics()
        .into_iter()
        .filter(|metric| requested.contains(metric.raw_id()))
        .collect()
}

fn tags(group_tag: &str) -> Vec<String> {
    [
        "KYZQ",
        "price_volume",
        group_tag,
        "intraday",
        "minute_agg",
        "peak_valley",
        "strict_20d",
        "neutralize",
        "barra",
        "size",
        "sector",
        "daily",
    ]
    .iter()
    .map(|value| value.to_string())
    .collect()
}

fn count_flags(flags: &[[bool; MINUTES_PER_DAY]; WINDOW_DAYS]) -> usize {
    flags.iter().flatten().filter(|flag| **flag).count()
}

fn mean_daily_state_ratio(
    days: &[&StrictStockDay],
    flags: &[[bool; MINUTES_PER_DAY]; WINDOW_DAYS],
) -> Option<f64> {
    let mut values = Vec::new();
    for day_idx in 0..WINDOW_DAYS {
        let Some(state_vwap) = state_vwap(days[day_idx], &flags[day_idx]) else {
            continue;
        };
        let Some(day_vwap) = full_day_vwap(days[day_idx]) else {
            continue;
        };
        if day_vwap.abs() > EPS {
            values.push(state_vwap / day_vwap);
        }
    }
    mean(&values)
}

fn mean_daily_state_percentile(
    days: &[&StrictStockDay],
    flags: &[[bool; MINUTES_PER_DAY]; WINDOW_DAYS],
) -> Option<f64> {
    let mut values = Vec::new();
    for day_idx in 0..WINDOW_DAYS {
        let Some(vwap) = state_vwap(days[day_idx], &flags[day_idx]) else {
            continue;
        };
        let (lower, upper) = daily_price_bounds(days[day_idx]);
        if upper - lower <= EPS {
            continue;
        }
        values.push((vwap - lower) / (upper - lower));
    }
    mean(&values)
}

fn mean_daily_state_pair_vwap_ratio(
    days: &[&StrictStockDay],
    left_flags: &[[bool; MINUTES_PER_DAY]; WINDOW_DAYS],
    right_flags: &[[bool; MINUTES_PER_DAY]; WINDOW_DAYS],
) -> Option<f64> {
    let mut values = Vec::new();
    for day_idx in 0..WINDOW_DAYS {
        let Some(left) = state_vwap(days[day_idx], &left_flags[day_idx]) else {
            continue;
        };
        let Some(right) = state_vwap(days[day_idx], &right_flags[day_idx]) else {
            continue;
        };
        if right.abs() > EPS {
            values.push(left / right);
        }
    }
    mean(&values)
}

fn state_vwap(day: &StrictStockDay, flags: &[bool; MINUTES_PER_DAY]) -> Option<f64> {
    let mut amount_sum = 0.0;
    let mut volume_sum = 0.0;
    let mut count = 0usize;
    for (idx, flag) in flags.iter().enumerate() {
        if !*flag {
            continue;
        }
        amount_sum += day.points[idx].amount;
        volume_sum += day.points[idx].vol;
        count += 1;
    }
    if count == 0 || volume_sum <= EPS {
        return None;
    }
    finite_option(Some(amount_sum / volume_sum))
}

fn full_day_vwap(day: &StrictStockDay) -> Option<f64> {
    let amount_sum = day.points.iter().map(|point| point.amount).sum::<f64>();
    let volume_sum = day.points.iter().map(|point| point.vol).sum::<f64>();
    if volume_sum <= EPS {
        return None;
    }
    finite_option(Some(amount_sum / volume_sum))
}

fn daily_price_bounds(day: &StrictStockDay) -> (f64, f64) {
    let high = day
        .points
        .iter()
        .map(|point| point.high)
        .fold(f64::NEG_INFINITY, f64::max);
    let low = day
        .points
        .iter()
        .map(|point| point.low)
        .fold(f64::INFINITY, f64::min);
    (low.min(day.pre_close), high.max(day.pre_close))
}

fn daily_flag_intervals(flags: &[[bool; MINUTES_PER_DAY]; WINDOW_DAYS]) -> Vec<f64> {
    let mut intervals = Vec::new();
    for day_flags in flags {
        let mut positions = Vec::new();
        for (minute_idx, flag) in day_flags.iter().enumerate() {
            if *flag {
                positions.push(minute_idx as f64);
            }
        }
        intervals.extend(
            positions
                .windows(2)
                .filter_map(|pair| finite_option(Some(pair[1] - pair[0]))),
        );
    }
    intervals
}

fn same_time_count_corr(
    left: &[[bool; MINUTES_PER_DAY]; WINDOW_DAYS],
    right: &[[bool; MINUTES_PER_DAY]; WINDOW_DAYS],
) -> Option<f64> {
    let mut left_counts = Vec::with_capacity(MINUTES_PER_DAY);
    let mut right_counts = Vec::with_capacity(MINUTES_PER_DAY);
    for minute_idx in 0..MINUTES_PER_DAY {
        left_counts.push(
            (0..WINDOW_DAYS)
                .filter(|day_idx| left[*day_idx][minute_idx])
                .count() as f64,
        );
        right_counts.push(
            (0..WINDOW_DAYS)
                .filter(|day_idx| right[*day_idx][minute_idx])
                .count() as f64,
        );
    }
    pearson_corr(&left_counts, &right_counts)
}

fn sum_state_returns(
    days: &[&StrictStockDay],
    flags: &[[bool; MINUTES_PER_DAY]; WINDOW_DAYS],
) -> Option<f64> {
    let mut sum = 0.0;
    let mut count = 0usize;
    for day_idx in 0..WINDOW_DAYS {
        for minute_idx in 0..MINUTES_PER_DAY {
            if !flags[day_idx][minute_idx] {
                continue;
            }
            if let Some(value) = minute_return(days[day_idx], minute_idx) {
                sum += value;
                count += 1;
            }
        }
    }
    finite_option((count > 0).then_some(sum))
}

fn price_jump_amount_corr(
    days: &[&StrictStockDay],
    flags: &[[bool; MINUTES_PER_DAY]; WINDOW_DAYS],
) -> Option<f64> {
    let mut current = Vec::new();
    let mut next = Vec::new();
    for day_idx in 0..WINDOW_DAYS {
        for minute_idx in 0..(MINUTES_PER_DAY - 1) {
            if flags[day_idx][minute_idx] {
                current.push(days[day_idx].points[minute_idx].amount);
                next.push(days[day_idx].points[minute_idx + 1].amount);
            }
        }
    }
    pearson_corr(&current, &next)
}

fn amplitude(point: MinutePoint) -> f64 {
    point.high / point.low - 1.0
}

fn minute_return(day: &StrictStockDay, minute_idx: usize) -> Option<f64> {
    let previous = if minute_idx == 0 {
        day.pre_close
    } else {
        day.points[minute_idx - 1].close
    };
    if previous.abs() <= EPS {
        return None;
    }
    finite_option(Some(day.points[minute_idx].close / previous - 1.0))
}

fn neighbor_price_gap(left: MinutePoint, right: MinutePoint) -> bool {
    left.high.min(right.high) < left.low.max(right.low)
}

fn mean_std(values: &[f64]) -> (f64, f64) {
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|value| {
            let diff = value - mean;
            diff * diff
        })
        .sum::<f64>()
        / values.len() as f64;
    (mean, variance.sqrt())
}

fn mean(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    finite_option(Some(values.iter().sum::<f64>() / values.len() as f64))
}

fn pearson_corr(left: &[f64], right: &[f64]) -> Option<f64> {
    if left.len() != right.len() || left.len() < 2 {
        return None;
    }
    let left_mean = left.iter().sum::<f64>() / left.len() as f64;
    let right_mean = right.iter().sum::<f64>() / right.len() as f64;
    let mut cov = 0.0;
    let mut left_var = 0.0;
    let mut right_var = 0.0;
    for idx in 0..left.len() {
        let left_diff = left[idx] - left_mean;
        let right_diff = right[idx] - right_mean;
        cov += left_diff * right_diff;
        left_var += left_diff * left_diff;
        right_var += right_diff * right_diff;
    }
    let denom = (left_var * right_var).sqrt();
    if denom <= EPS {
        return None;
    }
    finite_option(Some(cov / denom))
}

fn skewness(values: &[f64], min_count: usize) -> Option<f64> {
    if values.len() < min_count {
        return None;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let mut m2 = 0.0;
    let mut m3 = 0.0;
    for value in values {
        let diff = value - mean;
        let diff2 = diff * diff;
        m2 += diff2;
        m3 += diff2 * diff;
    }
    m2 /= values.len() as f64;
    m3 /= values.len() as f64;
    if m2 <= EPS {
        return None;
    }
    finite_option(Some(m3 / m2.sqrt().powi(3)))
}

fn kurtosis(values: &[f64], min_count: usize) -> Option<f64> {
    if values.len() < min_count {
        return None;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let mut m2 = 0.0;
    let mut m4 = 0.0;
    for value in values {
        let diff = value - mean;
        let diff2 = diff * diff;
        m2 += diff2;
        m4 += diff2 * diff2;
    }
    m2 /= values.len() as f64;
    m4 /= values.len() as f64;
    if m2 <= EPS {
        return None;
    }
    finite_option(Some(m4 / (m2 * m2)))
}

fn minute_index(trade_time: &str) -> Option<usize> {
    let minutes = time_to_minutes(trade_time)?;
    let morning_start = 9 * 60 + 31;
    let morning_end = 11 * 60 + 30;
    let afternoon_start = 13 * 60 + 1;
    let afternoon_end = 15 * 60;
    if (morning_start..=morning_end).contains(&minutes) {
        return Some((minutes - morning_start) as usize);
    }
    if (afternoon_start..=afternoon_end).contains(&minutes) {
        return Some(120 + (minutes - afternoon_start) as usize);
    }
    None
}

fn time_to_minutes(value: &str) -> Option<i32> {
    let value = value.trim();
    if value.is_empty() {
        return None;
    }
    let time = value
        .rsplit_once(' ')
        .map(|(_, right)| right)
        .or_else(|| value.rsplit_once('T').map(|(_, right)| right))
        .unwrap_or(value)
        .trim();
    if time.len() < 5 {
        return None;
    }
    let hour = time.get(0..2)?.parse::<i32>().ok()?;
    let minute = time.get(3..5)?.parse::<i32>().ok()?;
    Some(hour * 60 + minute)
}

fn clean_positive(value: Option<f64>) -> Option<f64> {
    clean_intraday_value(value).filter(|value| *value > 0.0)
}

fn clean_nonnegative(value: Option<f64>) -> Option<f64> {
    clean_intraday_value(value).filter(|value| *value >= 0.0)
}

fn finite_option(value: Option<f64>) -> Option<f64> {
    value.filter(|value| value.is_finite())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: Option<f64>, expected: f64) {
        let actual = actual.expect("value");
        assert!(
            (actual - expected).abs() < 1e-10,
            "expected {expected}, got {actual}"
        );
    }

    fn point(close: f64, vol: f64, amount: f64) -> MinutePoint {
        MinutePoint {
            high: close * 1.01,
            low: close * 0.99,
            close,
            vol,
            amount,
        }
    }

    fn strict_day(pre_close: f64, vol_overrides: &[(usize, f64)]) -> StrictStockDay {
        let mut points = [point(10.0, 10.0, 100.0); MINUTES_PER_DAY];
        for (idx, value) in vol_overrides {
            points[*idx].vol = *value;
            points[*idx].amount = *value * 10.0;
        }
        StrictStockDay { points, pre_close }
    }

    fn strict_day_with_base(
        pre_close: f64,
        base_vol: f64,
        vol_overrides: &[(usize, f64)],
    ) -> StrictStockDay {
        let mut points = [point(10.0, base_vol, base_vol * 10.0); MINUTES_PER_DAY];
        for (idx, value) in vol_overrides {
            points[*idx].vol = *value;
            points[*idx].amount = *value * 10.0;
        }
        StrictStockDay { points, pre_close }
    }

    #[test]
    fn kyzq_peak_valley_minute_index_uses_regular_session() {
        assert_eq!(minute_index("09:31:00"), Some(0));
        assert_eq!(minute_index("11:30:00"), Some(119));
        assert_eq!(minute_index("13:01:00"), Some(120));
        assert_eq!(minute_index("15:00:00"), Some(239));
        assert_eq!(minute_index("09:30:00"), None);
    }

    #[test]
    fn kyzq_peak_valley_strict_window_requires_twenty_days() {
        let mut state = PeakValleyState::default();
        let day = StrictMinuteDay {
            by_stock: BTreeMap::from([("000001.SZ".to_string(), strict_day(9.9, &[]))]),
        };
        for _ in 0..19 {
            state.push_day(day.clone());
        }
        assert!(state.strict_window_for("000001.SZ").is_none());
        state.push_day(day);
        assert_eq!(state.strict_window_for("000001.SZ").unwrap().len(), 20);
    }

    #[test]
    fn kyzq_peak_valley_volume_peak_requires_eruption_with_mild_neighbors() {
        let mut owned = (0..19)
            .map(|_| {
                strict_day_with_base(
                    9.9,
                    0.0,
                    &[(98, 100.0), (99, 100.0), (101, 100.0), (102, 100.0)],
                )
            })
            .collect::<Vec<_>>();
        owned.push(strict_day_with_base(9.9, 0.0, &[(100, 100.0)]));
        let days = owned.iter().collect::<Vec<_>>();
        let requested = BTreeSet::from([PeakValleyMetric::VolumePeakMinuteCount]);

        let values = compute_window_metrics(&days, &requested);

        assert_close(values.volume_peak_minute_count, 1.0);
    }

    #[test]
    fn kyzq_peak_valley_volume_mild_uses_mean_plus_std_threshold() {
        let mut owned = (0..19)
            .map(|_| strict_day_with_base(9.9, 0.0, &[(0, 10.0)]))
            .collect::<Vec<_>>();
        owned.push(strict_day_with_base(9.9, 0.0, &[(0, 5.0)]));
        let days = owned.iter().collect::<Vec<_>>();
        let states = classify_states(&days);

        assert!(states.volume_valley[WINDOW_DAYS - 1][0]);
    }

    #[test]
    fn kyzq_peak_valley_intervals_do_not_cross_days() {
        let mut flags = [[false; MINUTES_PER_DAY]; WINDOW_DAYS];
        flags[0][10] = true;
        flags[0][15] = true;
        flags[1][239] = true;
        flags[2][0] = true;

        assert_eq!(daily_flag_intervals(&flags), vec![5.0]);
    }

    #[test]
    fn kyzq_peak_valley_vwap_percentile_uses_pre_close_in_bounds() {
        let day = strict_day(9.0, &[]);
        let flags = {
            let mut flags = [false; MINUTES_PER_DAY];
            flags[0] = true;
            flags
        };
        let percentile =
            mean_daily_state_percentile(&vec![&day; WINDOW_DAYS], &[flags; WINDOW_DAYS]);
        assert!(percentile.is_some());
    }

    #[test]
    fn kyzq_peak_valley_factor_spec_registers_only_selected_formal_factor() {
        let spec = factor_spec(PeakValleyMetric::PriceJumpAmountCorr);
        assert_eq!(spec.id, "price_jump_amount_corr");
        assert!(spec.tags.iter().any(|tag| tag == "KYZQ"));
        assert!(spec.tags.iter().any(|tag| tag == "peak_valley"));
    }
}

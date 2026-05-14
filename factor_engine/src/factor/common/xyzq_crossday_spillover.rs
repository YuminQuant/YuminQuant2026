use std::any::Any;
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};

use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorRowKey, FactorSeries, FactorSpec,
    FactorValue, Frequency, IntradayDailyRawAuxiliaryRequest, IntradayDailyRawRequest,
    IntradayDailyRawSeries, IntradayDailyRawSpec, Lookback,
};
use crate::data::DataPool;
use crate::error::{err, Result};
use crate::factor::common::stock_daily_ops::neutralize_size_sector;
use crate::factor::common::stock_daily_raw_ids::{
    CROSSDAY_CLOSEVOLCORR_SPILLOVER_RAW_ID, CROSSDAY_RETSHARP_SPILLOVER_RAW_ID,
    CROSSDAY_RETVOLCORR_SPILLOVER_RAW_ID, CROSSDAY_TAYLORRET_SPILLOVER_RAW_ID,
    CROSSDAY_VOLRATIO_SPILLOVER_RAW_ID,
};
use crate::factor::common::{
    intraday_time_in_range, stock_minute_raw_spec, ClassificationLevel, ClassificationMap,
    DailyPanel, PanelColumn,
};
use crate::factor::IntradayRawMaterializeMode;
use crate::operators::cs_pctrank;

pub const RAW_VERSION: &str = "0.1.0";
pub const VERSION: &str = "0.1.0";
pub const PROVIDER_KEY: &str = "xyzq_crossday_spillover_provider";

const RAW_WINDOW_DAYS: usize = 20;
const HOURLY_ROLLING_WINDOW: usize = 8;
const TAYLOR_MINUTE_WINDOW: usize = 480;
const RETSHARP_SPILLOVER_WINDOW: usize = 28;
const MEDIUM_SPILLOVER_WINDOW: usize = 34;
const CORR_SPILLOVER_WINDOW: usize = 4;
const EPS: f64 = f64::EPSILON;

#[derive(Clone, Copy, Debug)]
pub struct XyzqCrossdaySpilloverFactorDef {
    pub id: &'static str,
    pub alias: &'static str,
    pub name: &'static str,
    pub raw_id: &'static str,
}

#[derive(Clone, Copy, Debug, Default)]
struct HourFeatureValues {
    retsharp: Option<f64>,
    volratio: Option<f64>,
    taylorret: Option<f64>,
    retvolcorr: Option<f64>,
    closevolcorr: Option<f64>,
}

#[derive(Clone, Copy, Debug, Default)]
struct FinalValues {
    retsharp: Option<f64>,
    volratio: Option<f64>,
    taylorret: Option<f64>,
    retvolcorr: Option<f64>,
    closevolcorr: Option<f64>,
}

#[derive(Clone, Copy, Debug, Default)]
struct TaylorTerm {
    term: f64,
    abs_simple: f64,
}

#[derive(Clone, Debug, Default)]
struct HourBlock {
    hourly_return: Option<f64>,
    hourly_volume: Option<f64>,
    hourly_close: Option<f64>,
    taylor_terms: Vec<TaylorTerm>,
}

#[derive(Clone, Debug)]
struct MinutePoint {
    time: String,
    close: Option<f64>,
    vol: Option<f64>,
}

#[derive(Debug, Default)]
pub struct CrossdaySpilloverState {
    stocks: BTreeMap<String, StockState>,
}

#[derive(Debug, Default)]
struct StockState {
    hourly_returns: VecDeque<Option<f64>>,
    hourly_volumes: VecDeque<Option<f64>>,
    hourly_closes: VecDeque<Option<f64>>,
    taylor_terms: VecDeque<TaylorTerm>,
    taylor_sum: f64,
    taylor_abs_sum: f64,
    retsharp_spillovers: VecDeque<Option<f64>>,
    volratio_spillovers: VecDeque<Option<f64>>,
    taylorret_spillovers: VecDeque<Option<f64>>,
    retvolcorr_spillovers: VecDeque<Option<f64>>,
    closevolcorr_spillovers: VecDeque<Option<f64>>,
}

pub fn all_raw_ids() -> [&'static str; 5] {
    [
        CROSSDAY_RETSHARP_SPILLOVER_RAW_ID,
        CROSSDAY_VOLRATIO_SPILLOVER_RAW_ID,
        CROSSDAY_TAYLORRET_SPILLOVER_RAW_ID,
        CROSSDAY_RETVOLCORR_SPILLOVER_RAW_ID,
        CROSSDAY_CLOSEVOLCORR_SPILLOVER_RAW_ID,
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

pub fn factor_spec(def: XyzqCrossdaySpilloverFactorDef) -> FactorSpec {
    FactorSpec {
        id: def.id.to_string(),
        aliases: vec![def.alias.to_string()],
        name: def.name.to_string(),
        asset_class: AssetClass::Stock,
        frequency: Frequency::Daily,
        version: VERSION.to_string(),
        tags: tags(),
        description: format!(
            "{} cross-day hourly industry spillover factor, neutralized by Barra SIZE and SW sector.",
            def.name
        ),
        dependencies: vec![
            DataRequest::new(DatasetId::StockBarraDaily, &["SIZE"]),
            DataRequest::new(DatasetId::StockSwClassification, &["l1_code"]),
        ],
        intraday_raw_dependencies: vec![IntradayDailyRawRequest::new(def.raw_id, 0)],
        lookback: Lookback { trading_days: 0 },
    }
}

pub fn compute_factor(
    def: XyzqCrossdaySpilloverFactorDef,
    data: &DataPool,
) -> Result<FactorSeries> {
    let panel = data.intraday_daily_raw_panel(def.raw_id)?;
    let raw = panel.column(def.raw_id)?;
    let factor = neutralize_size_sector(&raw, &panel, data)?;
    Ok(factor.to_factor_series(factor_spec(def)))
}

pub fn composite_factor_spec() -> FactorSpec {
    FactorSpec {
        id: "crossday_intraday_spillover".to_string(),
        aliases: vec!["crossday_intraday_spillover".to_string()],
        name: "crossday_intraday_spillover".to_string(),
        asset_class: AssetClass::Stock,
        frequency: Frequency::Daily,
        version: VERSION.to_string(),
        tags: tags(),
        description: "Composite cross-day hourly intraday industry spillover factor from retsharp, volratio, taylorret, retvolcorr, and closevolcorr components, neutralized by Barra SIZE and SW sector.".to_string(),
        dependencies: vec![
            DataRequest::new(DatasetId::StockBarraDaily, &["SIZE"]),
            DataRequest::new(DatasetId::StockSwClassification, &["l1_code"]),
        ],
        intraday_raw_dependencies: all_raw_ids()
            .iter()
            .map(|raw_id| IntradayDailyRawRequest::new(raw_id, 0))
            .collect(),
        lookback: Lookback { trading_days: 0 },
    }
}

pub fn compute_composite_factor(data: &DataPool) -> Result<FactorSeries> {
    let panel = data.intraday_daily_raw_panel(CROSSDAY_RETSHARP_SPILLOVER_RAW_ID)?;
    let mut scored = Vec::with_capacity(all_raw_ids().len());
    for raw_id in all_raw_ids() {
        scored.push(rank_score_component(&panel.column(raw_id)?)?);
    }
    let composite = average_columns(panel, &scored)?;
    let filled = fill_missing_with_cs_mean(&composite)?;
    let factor = neutralize_size_sector(&filled, panel, data)?;
    Ok(factor.to_factor_series(composite_factor_spec()))
}

#[macro_export]
macro_rules! define_xyzq_crossday_spillover_factor {
    ($struct_name:ident, $id:expr, $alias:expr, $name:expr, $raw_id:expr) => {
        const DEF: $crate::factor::common::xyzq_crossday_spillover::XyzqCrossdaySpilloverFactorDef =
            $crate::factor::common::xyzq_crossday_spillover::XyzqCrossdaySpilloverFactorDef {
                id: $id,
                alias: $alias,
                name: $name,
                raw_id: $raw_id,
            };

        pub struct $struct_name;

        pub fn create() -> Box<dyn $crate::factor::Factor> {
            Box::new($struct_name)
        }

        impl $crate::factor::Factor for $struct_name {
            fn spec(&self) -> $crate::core::FactorSpec {
                $crate::factor::common::xyzq_crossday_spillover::factor_spec(DEF)
            }

            fn intraday_raw_specs(&self) -> Vec<$crate::core::IntradayDailyRawSpec> {
                vec![$crate::factor::common::xyzq_crossday_spillover::raw_spec(DEF.raw_id)]
            }

            fn intraday_raw_provider_key(&self, _raw_id: &str) -> String {
                $crate::factor::common::xyzq_crossday_spillover::PROVIDER_KEY.to_string()
            }

            fn intraday_raw_materialize_mode(
                &self,
                _raw_ids: &[String],
            ) -> $crate::factor::IntradayRawMaterializeMode {
                $crate::factor::common::xyzq_crossday_spillover::intraday_raw_materialize_mode()
            }

            fn initial_intraday_raw_state(&self, _raw_ids: &[String]) -> Box<dyn std::any::Any + Send> {
                $crate::factor::common::xyzq_crossday_spillover::initial_intraday_raw_state()
            }

            fn intraday_raw_auxiliary_requirements(
                &self,
                raw_ids: &[String],
            ) -> Vec<$crate::core::IntradayDailyRawAuxiliaryRequest> {
                $crate::factor::common::xyzq_crossday_spillover::intraday_raw_auxiliary_requirements(raw_ids)
            }

            fn minute_compute_stateful_many(
                &self,
                raw_ids: &[String],
                context: &$crate::core::FactorContext,
                data: &$crate::data::DataPool,
                state: &mut dyn std::any::Any,
            ) -> $crate::error::Result<Vec<$crate::core::IntradayDailyRawSeries>> {
                $crate::factor::common::xyzq_crossday_spillover::minute_compute_stateful_many(
                    raw_ids, context, data, state,
                )
            }

            fn compute(
                &self,
                _context: &$crate::core::FactorContext,
                data: &$crate::data::DataPool,
            ) -> $crate::error::Result<$crate::core::FactorSeries> {
                $crate::factor::common::xyzq_crossday_spillover::compute_factor(DEF, data)
            }
        }
    };
}

fn rank_score_component(values: &PanelColumn) -> Result<PanelColumn> {
    values.cs(|cross_section| {
        cs_pctrank(cross_section, true)
            .into_iter()
            .map(|rank| match finite_option(rank) {
                Some(rank) => {
                    let score = 2.0 * rank - 1.0;
                    (rank < 0.9).then_some(score).and_then(finite_value)
                }
                None => None,
            })
            .collect()
    })
}

fn average_columns(panel: &DailyPanel, columns: &[PanelColumn]) -> Result<PanelColumn> {
    let mut values = vec![None; panel.shape_len()];
    for idx in 0..panel.shape_len() {
        let mut sum = 0.0;
        let mut count = 0usize;
        for column in columns {
            if let Some(value) = finite_option(column.values()[idx]) {
                sum += value;
                count += 1;
            }
        }
        if count > 0 {
            values[idx] = finite_value(sum / count as f64);
        }
    }
    panel.column_from_values(values)
}

fn fill_missing_with_cs_mean(values: &PanelColumn) -> Result<PanelColumn> {
    values.cs(|cross_section| {
        let finite = cross_section
            .iter()
            .filter_map(|value| finite_option(*value))
            .collect::<Vec<_>>();
        let mean = if finite.is_empty() {
            None
        } else {
            finite_value(finite.iter().sum::<f64>() / finite.len() as f64)
        };
        cross_section
            .iter()
            .map(|value| finite_option(*value).or(mean))
            .collect()
    })
}

pub fn intraday_raw_materialize_mode() -> IntradayRawMaterializeMode {
    IntradayRawMaterializeMode::Stateful
}

pub fn initial_intraday_raw_state() -> Box<dyn Any + Send> {
    Box::new(CrossdaySpilloverState::default())
}

pub fn intraday_raw_auxiliary_requirements(
    raw_ids: &[String],
) -> Vec<IntradayDailyRawAuxiliaryRequest> {
    let requested = raw_ids.iter().map(String::as_str).collect::<BTreeSet<_>>();
    if requested.is_empty()
        || requested
            .iter()
            .all(|raw_id| !all_raw_ids().contains(raw_id))
    {
        Vec::new()
    } else {
        vec![IntradayDailyRawAuxiliaryRequest::new(
            DataRequest::new(DatasetId::StockSwClassification, &["l1_code"]),
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
    let requested = raw_ids
        .iter()
        .map(String::as_str)
        .filter(|raw_id| all_raw_ids().contains(raw_id))
        .collect::<BTreeSet<_>>();
    if requested.is_empty() {
        return Ok(Vec::new());
    }

    let state = state
        .downcast_mut::<CrossdaySpilloverState>()
        .ok_or_else(|| err("crossday spillover stateful raw received incompatible state"))?;
    let trade_date = *context
        .target_dates
        .first()
        .ok_or_else(|| err("crossday spillover stateful raw requires one target date"))?;

    let Some(table) = data.minute(DatasetId::StockMinute1m, trade_date) else {
        return Ok(series_from_values(trade_date, requested, BTreeMap::new()));
    };
    let sector_map = ClassificationMap::from_table(
        data.daily(DatasetId::StockSwClassification)?,
        ClassificationLevel::Sector,
    )?;
    let day_blocks = day_blocks_by_stock(table)?;

    let mut final_values = BTreeMap::<String, FinalValues>::new();
    for hour_idx in 0..HOUR_BLOCKS.len() {
        let mut hourly_features = BTreeMap::<String, HourFeatureValues>::new();
        for (ts_code, blocks) in &day_blocks {
            let stock_state = state.stocks.entry(ts_code.clone()).or_default();
            let features = stock_state.apply_hour(&blocks[hour_idx]);
            hourly_features.insert(ts_code.clone(), features);
        }
        let sectors = hourly_features
            .keys()
            .map(|ts_code| {
                (
                    ts_code.clone(),
                    sector_map
                        .group_for(trade_date, ts_code)
                        .map(str::to_string),
                )
            })
            .collect::<BTreeMap<_, _>>();
        let peers = industry_peer_spillovers(&hourly_features, &sectors);
        for (ts_code, peer_values) in peers {
            state
                .stocks
                .entry(ts_code.clone())
                .or_default()
                .push_peer_spillovers(peer_values);
        }
    }

    for ts_code in day_blocks.keys() {
        if let Some(stock_state) = state.stocks.get(ts_code) {
            final_values.insert(ts_code.clone(), stock_state.final_values());
        }
    }

    Ok(series_from_values(trade_date, requested, final_values))
}

fn series_from_values(
    trade_date: i32,
    requested: BTreeSet<&str>,
    values: BTreeMap<String, FinalValues>,
) -> Vec<IntradayDailyRawSeries> {
    let mut by_raw_id = all_raw_ids()
        .iter()
        .map(|raw_id| (*raw_id, Vec::<FactorValue>::new()))
        .collect::<BTreeMap<_, _>>();
    for (ts_code, final_values) in values {
        let key = FactorRowKey::Daily {
            trade_date,
            ts_code,
        };
        push_value(
            &mut by_raw_id,
            &requested,
            CROSSDAY_RETSHARP_SPILLOVER_RAW_ID,
            &key,
            final_values.retsharp,
        );
        push_value(
            &mut by_raw_id,
            &requested,
            CROSSDAY_VOLRATIO_SPILLOVER_RAW_ID,
            &key,
            final_values.volratio,
        );
        push_value(
            &mut by_raw_id,
            &requested,
            CROSSDAY_TAYLORRET_SPILLOVER_RAW_ID,
            &key,
            final_values.taylorret,
        );
        push_value(
            &mut by_raw_id,
            &requested,
            CROSSDAY_RETVOLCORR_SPILLOVER_RAW_ID,
            &key,
            final_values.retvolcorr,
        );
        push_value(
            &mut by_raw_id,
            &requested,
            CROSSDAY_CLOSEVOLCORR_SPILLOVER_RAW_ID,
            &key,
            final_values.closevolcorr,
        );
    }

    let mut output = Vec::new();
    for raw_id in all_raw_ids() {
        if requested.contains(raw_id) {
            output.push(IntradayDailyRawSeries {
                spec: raw_spec(raw_id),
                values: by_raw_id.remove(raw_id).unwrap_or_default(),
            });
        }
    }
    output
}

fn push_value(
    by_raw_id: &mut BTreeMap<&'static str, Vec<FactorValue>>,
    requested: &BTreeSet<&str>,
    raw_id: &'static str,
    key: &FactorRowKey,
    value: Option<f64>,
) {
    if requested.contains(raw_id) {
        by_raw_id.entry(raw_id).or_default().push(FactorValue {
            key: key.clone(),
            value,
        });
    }
}

fn day_blocks_by_stock(table: &crate::data::Table) -> Result<BTreeMap<String, [HourBlock; 4]>> {
    let ts_codes = table.required_utf8("ts_code")?;
    let trade_times = table.required_utf8("trade_time")?;
    let close = table.required_f64_cast("close")?;
    let vol = table.required_f64_cast("vol")?;

    let mut grouped = BTreeMap::<String, Vec<usize>>::new();
    for idx in 0..table.len {
        let (Some(ts_code), Some(_trade_time)) = (ts_codes[idx].clone(), trade_times[idx].clone())
        else {
            continue;
        };
        grouped.entry(ts_code).or_default().push(idx);
    }

    let mut output = BTreeMap::new();
    for (ts_code, mut indices) in grouped {
        indices.sort_by(|left, right| trade_times[*left].cmp(&trade_times[*right]));
        let points = indices
            .iter()
            .filter_map(|idx| {
                let time = trade_times[*idx].clone()?;
                Some(MinutePoint {
                    time,
                    close: finite_option(close[*idx]),
                    vol: finite_option(vol[*idx]),
                })
            })
            .collect::<Vec<_>>();
        output.insert(ts_code, hour_blocks_for_stock(&points));
    }
    Ok(output)
}

fn hour_blocks_for_stock(points: &[MinutePoint]) -> [HourBlock; 4] {
    std::array::from_fn(|idx| hour_block(points, HOUR_BLOCKS[idx]))
}

fn hour_block(points: &[MinutePoint], def: HourBlockDef) -> HourBlock {
    let anchor_close = close_at(points, def.anchor);
    let end_close = close_at(points, def.end);
    let hourly_return = match (end_close, anchor_close) {
        (Some(end_close), Some(anchor_close)) if anchor_close.abs() > EPS => {
            finite_value(end_close / anchor_close - 1.0)
        }
        _ => None,
    };

    let mut volume_sum = 0.0;
    let mut volume_count = 0usize;
    let mut prev_close = anchor_close;
    let mut taylor_terms = Vec::new();
    for point in points
        .iter()
        .filter(|point| intraday_time_in_range(&point.time, def.start, def.end))
    {
        if let Some(vol) = point.vol {
            volume_sum += vol;
            volume_count += 1;
        }
        if let (Some(current), Some(previous)) = (point.close, prev_close) {
            if current > EPS && previous > EPS {
                let simple = current / previous - 1.0;
                let log_ret = (current / previous).ln();
                let term = 2.0 * (simple - log_ret) - log_ret * log_ret;
                if simple.is_finite() && log_ret.is_finite() && term.is_finite() {
                    taylor_terms.push(TaylorTerm {
                        term,
                        abs_simple: simple.abs(),
                    });
                }
            }
        }
        prev_close = point.close;
    }

    HourBlock {
        hourly_return,
        hourly_volume: (volume_count > 0)
            .then_some(volume_sum)
            .and_then(finite_value),
        hourly_close: end_close.and_then(finite_value),
        taylor_terms,
    }
}

fn close_at(points: &[MinutePoint], target: &str) -> Option<f64> {
    points
        .iter()
        .find(|point| intraday_time_in_range(&point.time, target, target))
        .and_then(|point| point.close)
        .filter(|value| *value > EPS)
}

impl StockState {
    fn apply_hour(&mut self, block: &HourBlock) -> HourFeatureValues {
        push_capped(
            &mut self.hourly_returns,
            block.hourly_return,
            HOURLY_ROLLING_WINDOW,
        );
        push_capped(
            &mut self.hourly_volumes,
            block.hourly_volume,
            HOURLY_ROLLING_WINDOW,
        );
        push_capped(
            &mut self.hourly_closes,
            block.hourly_close,
            HOURLY_ROLLING_WINDOW,
        );
        for term in &block.taylor_terms {
            self.push_taylor_term(*term);
        }

        let retsharp = mean_std_ratio(&self.hourly_returns);
        let volratio = match (block.hourly_volume, sum_options(&self.hourly_volumes)) {
            (Some(latest), Some(total)) if total.abs() > EPS => finite_value(latest / total),
            _ => None,
        };
        let taylorret = if self.taylor_abs_sum.abs() > EPS {
            finite_value(self.taylor_sum / self.taylor_abs_sum)
        } else {
            None
        };
        let retvolcorr = corr_options(&self.hourly_returns, &self.hourly_volumes);
        let closevolcorr = corr_options(&self.hourly_closes, &self.hourly_volumes);

        HourFeatureValues {
            retsharp,
            volratio,
            taylorret,
            retvolcorr,
            closevolcorr,
        }
    }

    fn push_taylor_term(&mut self, term: TaylorTerm) {
        self.taylor_sum += term.term;
        self.taylor_abs_sum += term.abs_simple;
        self.taylor_terms.push_back(term);
        while self.taylor_terms.len() > TAYLOR_MINUTE_WINDOW {
            if let Some(removed) = self.taylor_terms.pop_front() {
                self.taylor_sum -= removed.term;
                self.taylor_abs_sum -= removed.abs_simple;
            }
        }
    }

    fn push_peer_spillovers(&mut self, values: HourFeatureValues) {
        push_capped(
            &mut self.retsharp_spillovers,
            values.retsharp,
            RETSHARP_SPILLOVER_WINDOW,
        );
        push_capped(
            &mut self.volratio_spillovers,
            values.volratio,
            MEDIUM_SPILLOVER_WINDOW,
        );
        push_capped(
            &mut self.taylorret_spillovers,
            values.taylorret,
            MEDIUM_SPILLOVER_WINDOW,
        );
        push_capped(
            &mut self.retvolcorr_spillovers,
            values.retvolcorr,
            CORR_SPILLOVER_WINDOW,
        );
        push_capped(
            &mut self.closevolcorr_spillovers,
            values.closevolcorr,
            CORR_SPILLOVER_WINDOW,
        );
    }

    fn final_values(&self) -> FinalValues {
        FinalValues {
            retsharp: mean_options(&self.retsharp_spillovers),
            volratio: mean_options(&self.volratio_spillovers),
            taylorret: mean_options(&self.taylorret_spillovers),
            retvolcorr: mean_options(&self.retvolcorr_spillovers),
            closevolcorr: mean_options(&self.closevolcorr_spillovers),
        }
    }
}

fn industry_peer_spillovers(
    features: &BTreeMap<String, HourFeatureValues>,
    sectors: &BTreeMap<String, Option<String>>,
) -> BTreeMap<String, HourFeatureValues> {
    let retsharp = peer_values(features, sectors, |values| values.retsharp);
    let volratio = peer_values(features, sectors, |values| values.volratio);
    let taylorret = peer_values(features, sectors, |values| values.taylorret);
    let retvolcorr = peer_values(features, sectors, |values| values.retvolcorr);
    let closevolcorr = peer_values(features, sectors, |values| values.closevolcorr);

    features
        .keys()
        .map(|ts_code| {
            (
                ts_code.clone(),
                HourFeatureValues {
                    retsharp: retsharp.get(ts_code).copied().flatten(),
                    volratio: volratio.get(ts_code).copied().flatten(),
                    taylorret: taylorret.get(ts_code).copied().flatten(),
                    retvolcorr: retvolcorr.get(ts_code).copied().flatten(),
                    closevolcorr: closevolcorr.get(ts_code).copied().flatten(),
                },
            )
        })
        .collect()
}

fn peer_values<F>(
    features: &BTreeMap<String, HourFeatureValues>,
    sectors: &BTreeMap<String, Option<String>>,
    pick: F,
) -> BTreeMap<String, Option<f64>>
where
    F: Fn(&HourFeatureValues) -> Option<f64>,
{
    let mut sums = HashMap::<&str, (f64, usize)>::new();
    for (ts_code, feature) in features {
        let (Some(sector), Some(value)) = (
            sectors.get(ts_code).and_then(|value| value.as_deref()),
            pick(feature),
        ) else {
            continue;
        };
        let entry = sums.entry(sector).or_insert((0.0, 0));
        entry.0 += value;
        entry.1 += 1;
    }

    features
        .iter()
        .map(|(ts_code, feature)| {
            let value = sectors
                .get(ts_code)
                .and_then(|sector| sector.as_deref())
                .and_then(|sector| {
                    let (sum, count) = *sums.get(sector)?;
                    match pick(feature) {
                        Some(own) => {
                            if count > 1 {
                                finite_value((sum - own) / (count - 1) as f64)
                            } else {
                                None
                            }
                        }
                        None => (count > 0)
                            .then(|| sum / count as f64)
                            .and_then(finite_value),
                    }
                });
            (ts_code.clone(), value)
        })
        .collect()
}

fn push_capped(values: &mut VecDeque<Option<f64>>, value: Option<f64>, cap: usize) {
    values.push_back(value.and_then(finite_value));
    while values.len() > cap {
        values.pop_front();
    }
}

fn mean_options(values: &VecDeque<Option<f64>>) -> Option<f64> {
    let mut sum = 0.0;
    let mut count = 0usize;
    for value in values {
        if let Some(value) = (*value).and_then(finite_value) {
            sum += value;
            count += 1;
        }
    }
    (count > 0)
        .then(|| sum / count as f64)
        .and_then(finite_value)
}

fn sum_options(values: &VecDeque<Option<f64>>) -> Option<f64> {
    let mut sum = 0.0;
    let mut count = 0usize;
    for value in values {
        if let Some(value) = (*value).and_then(finite_value) {
            sum += value;
            count += 1;
        }
    }
    (count > 0).then_some(sum).and_then(finite_value)
}

fn mean_std_ratio(values: &VecDeque<Option<f64>>) -> Option<f64> {
    let finite = values
        .iter()
        .filter_map(|value| (*value).and_then(finite_value))
        .collect::<Vec<_>>();
    if finite.len() < 2 {
        return None;
    }
    let mean = finite.iter().sum::<f64>() / finite.len() as f64;
    let variance = finite
        .iter()
        .map(|value| {
            let diff = value - mean;
            diff * diff
        })
        .sum::<f64>()
        / finite.len() as f64;
    if variance <= EPS {
        return None;
    }
    finite_value(mean / variance.sqrt())
}

fn corr_options(left: &VecDeque<Option<f64>>, right: &VecDeque<Option<f64>>) -> Option<f64> {
    let pairs = left
        .iter()
        .zip(right.iter())
        .filter_map(|(left, right)| {
            Some((
                (*left).and_then(finite_value)?,
                (*right).and_then(finite_value)?,
            ))
        })
        .collect::<Vec<_>>();
    if pairs.len() < 2 {
        return None;
    }
    let mean_left = pairs.iter().map(|(left, _)| *left).sum::<f64>() / pairs.len() as f64;
    let mean_right = pairs.iter().map(|(_, right)| *right).sum::<f64>() / pairs.len() as f64;
    let mut cov = 0.0;
    let mut var_left = 0.0;
    let mut var_right = 0.0;
    for (left, right) in pairs {
        let left_diff = left - mean_left;
        let right_diff = right - mean_right;
        cov += left_diff * right_diff;
        var_left += left_diff * left_diff;
        var_right += right_diff * right_diff;
    }
    let denominator = (var_left * var_right).sqrt();
    if denominator <= EPS {
        return None;
    }
    finite_value(cov / denominator)
}

fn finite_option(value: Option<f64>) -> Option<f64> {
    value.and_then(finite_value)
}

fn finite_value(value: f64) -> Option<f64> {
    value.is_finite().then_some(value)
}

fn tags() -> Vec<String> {
    [
        "price_volume",
        "industry_spillover",
        "crossday",
        "hourly",
        "intraday",
        "minute_agg",
        "neutralize",
        "barra",
        "size",
        "sector",
        "daily",
        "XYZQ",
    ]
    .iter()
    .map(|value| value.to_string())
    .collect()
}

#[derive(Clone, Copy, Debug)]
struct HourBlockDef {
    start: &'static str,
    end: &'static str,
    anchor: &'static str,
}

const HOUR_BLOCKS: [HourBlockDef; 4] = [
    HourBlockDef {
        start: "09:31:00",
        end: "10:30:00",
        anchor: "09:30:00",
    },
    HourBlockDef {
        start: "10:31:00",
        end: "11:30:00",
        anchor: "10:30:00",
    },
    HourBlockDef {
        start: "13:01:00",
        end: "14:00:00",
        anchor: "11:30:00",
    },
    HourBlockDef {
        start: "14:01:00",
        end: "15:00:00",
        anchor: "14:00:00",
    },
];

#[cfg(test)]
mod tests {
    use super::*;

    fn point(time: &str, close: f64, vol: f64) -> MinutePoint {
        MinutePoint {
            time: time.to_string(),
            close: Some(close),
            vol: Some(vol),
        }
    }

    fn assert_close(actual: Option<f64>, expected: f64) {
        let actual = actual.expect("value exists");
        assert!(
            (actual - expected).abs() < 1e-10,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn hour_blocks_use_xyzq_anchors_and_lunch_bridge() {
        let points = vec![
            point("09:30:00", 100.0, 0.0),
            point("09:31:00", 101.0, 1.0),
            point("10:30:00", 110.0, 2.0),
            point("10:31:00", 111.0, 3.0),
            point("11:30:00", 121.0, 4.0),
            point("13:01:00", 122.0, 5.0),
            point("14:00:00", 133.1, 6.0),
            point("14:01:00", 134.0, 7.0),
            point("15:00:00", 146.41, 8.0),
        ];

        let blocks = hour_blocks_for_stock(&points);

        assert_close(blocks[0].hourly_return, 0.10);
        assert_close(blocks[1].hourly_return, 0.10);
        assert_close(blocks[2].hourly_return, 0.10);
        assert_close(blocks[3].hourly_return, 0.10);
        assert_close(blocks[2].hourly_volume, 11.0);
    }

    #[test]
    fn peer_values_leave_out_self_and_use_sector_mean_when_self_missing() {
        let features = BTreeMap::from([
            (
                "a".to_string(),
                HourFeatureValues {
                    retsharp: Some(1.0),
                    ..HourFeatureValues::default()
                },
            ),
            (
                "b".to_string(),
                HourFeatureValues {
                    retsharp: Some(3.0),
                    ..HourFeatureValues::default()
                },
            ),
            (
                "c".to_string(),
                HourFeatureValues {
                    retsharp: None,
                    ..HourFeatureValues::default()
                },
            ),
        ]);
        let sectors = BTreeMap::from([
            ("a".to_string(), Some("bank".to_string())),
            ("b".to_string(), Some("bank".to_string())),
            ("c".to_string(), Some("bank".to_string())),
        ]);

        let peers = peer_values(&features, &sectors, |values| values.retsharp);

        assert_close(peers["a"], 3.0);
        assert_close(peers["b"], 1.0);
        assert_close(peers["c"], 2.0);
    }

    #[test]
    fn stock_state_rolls_hourly_features_and_spillovers() {
        let mut state = StockState::default();
        for idx in 0..8 {
            let block = HourBlock {
                hourly_return: Some(idx as f64 + 1.0),
                hourly_volume: Some(10.0 + idx as f64),
                hourly_close: Some(100.0 + idx as f64),
                taylor_terms: vec![TaylorTerm {
                    term: 1.0,
                    abs_simple: 2.0,
                }],
            };
            let features = state.apply_hour(&block);
            state.push_peer_spillovers(features);
        }

        let final_values = state.final_values();
        assert!(final_values.retsharp.is_some());
        assert_close(final_values.taylorret, 0.5);
        assert!(final_values.retvolcorr.is_some());
        assert!(final_values.closevolcorr.is_some());
    }
}

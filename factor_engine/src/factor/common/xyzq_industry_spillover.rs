use std::collections::{BTreeMap, BTreeSet, HashMap};

use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorRowKey, FactorSeries, FactorSpec,
    FactorValue, Frequency, IntradayDailyRawRequest, IntradayDailyRawSeries, IntradayDailyRawSpec,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::stock_daily_ops::neutralize_size_sector;
use crate::factor::common::stock_daily_raw_ids::{
    XYZQ_SPILL_AFTVOLRATIO_RAW_ID, XYZQ_SPILL_DOLVOLSUB_RAW_ID, XYZQ_SPILL_MORDOLVOL_RAW_ID,
    XYZQ_SPILL_MORNINGRET_RAW_ID, XYZQ_SPILL_MORVOLMINUSAFTVOL_RAW_ID,
    XYZQ_SPILL_MORVOLRATIO_RAW_ID, XYZQ_SPILL_OUTBOUNDRET_RAW_ID, XYZQ_SPILL_RETSHARP_RAW_ID,
    XYZQ_SPILL_RETSKEW_RAW_ID, XYZQ_SPILL_RETVOLCORR_RAW_ID, XYZQ_SPILL_RVDIFF_RAW_ID,
    XYZQ_SPILL_RV_RAW_ID, XYZQ_SPILL_TAYLORRET_RAW_ID, XYZQ_SPILL_VARVARSIGNSUB_RAW_ID,
    XYZQ_SPILL_VARVAR_RAW_ID,
};
use crate::factor::common::{
    clean_intraday_value, intraday_time_in_range, quantile_linear, stock_minute_raw_spec,
    ClassificationLevel, ClassificationMap, DailyPanel, PanelColumn,
};
use crate::operators::{cs_pctrank, ts_mean};

pub const RAW_VERSION: &str = "0.1.0";
pub const VERSION: &str = "0.1.0";
pub const PROVIDER_KEY: &str = "xyzq_industry_spillover_intraday_provider";

const RAW_WINDOW_DAYS: usize = 1;
const RAW_LOOKBACK: usize = 24;
const MIN_PERIODS: usize = 1;
const INDUSTRY_MOM_LOOKBACK: usize = 39;
const INDUSTRY_MOM_RETURN_WINDOW: usize = 20;
const INDUSTRY_MOM_SMOOTH_WINDOW: usize = 20;
const SAMPLE_START: &str = "09:31:00";
const SAMPLE_END: &str = "15:00:00";
const OPEN_ANCHOR: &str = "09:30:00";
const MORNING_END: &str = "10:00:00";
const AFTERNOON_START: &str = "14:31:00";
const AFTERNOON_END: &str = "15:00:00";
const EPS: f64 = f64::EPSILON;

#[derive(Clone, Copy, Debug)]
pub struct XyzqIndustrySpilloverFactorDef {
    pub id: &'static str,
    pub alias: &'static str,
    pub name: &'static str,
    pub mode: XyzqIndustrySpilloverMode,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum XyzqIndustrySpilloverMode {
    IndustryMomentum,
    HighCorr1,
    HighCorr2,
    HighCorr3,
    RetVolCorr,
    RetSkew,
    TaylorRet,
}

#[derive(Clone, Copy, Debug)]
enum RollingSpec {
    Mean(usize),
    MeanStd(usize),
}

#[derive(Clone, Copy, Debug)]
struct ComponentSpec {
    raw_id: &'static str,
    rolling: RollingSpec,
    sign: f64,
}

#[derive(Clone, Copy, Debug, Default)]
struct SpilloverMinuteStats {
    rv: Option<f64>,
    retskew: Option<f64>,
    aftvolratio: Option<f64>,
    varvar: Option<f64>,
    retvolcorr: Option<f64>,
    morvolratio: Option<f64>,
    taylorret: Option<f64>,
    morningret: Option<f64>,
    outboundret: Option<f64>,
    rvdiff: Option<f64>,
    morvolminusaftvol: Option<f64>,
    varvarsignsub: Option<f64>,
    retsharp: Option<f64>,
    mordolvol: Option<f64>,
    dolvolsub: Option<f64>,
}

#[derive(Clone, Debug)]
struct MinutePoint {
    time: String,
    in_sample: bool,
    close: Option<f64>,
    vol: Option<f64>,
    amount: Option<f64>,
}

#[derive(Clone, Copy, Debug)]
struct ReturnPoint {
    time_in_sample: bool,
    simple_ret: f64,
    log_ret: f64,
    vol: Option<f64>,
}

pub fn all_raw_ids() -> [&'static str; 15] {
    [
        XYZQ_SPILL_RV_RAW_ID,
        XYZQ_SPILL_RETSKEW_RAW_ID,
        XYZQ_SPILL_AFTVOLRATIO_RAW_ID,
        XYZQ_SPILL_VARVAR_RAW_ID,
        XYZQ_SPILL_RETVOLCORR_RAW_ID,
        XYZQ_SPILL_MORVOLRATIO_RAW_ID,
        XYZQ_SPILL_TAYLORRET_RAW_ID,
        XYZQ_SPILL_MORNINGRET_RAW_ID,
        XYZQ_SPILL_OUTBOUNDRET_RAW_ID,
        XYZQ_SPILL_RVDIFF_RAW_ID,
        XYZQ_SPILL_MORVOLMINUSAFTVOL_RAW_ID,
        XYZQ_SPILL_VARVARSIGNSUB_RAW_ID,
        XYZQ_SPILL_RETSHARP_RAW_ID,
        XYZQ_SPILL_MORDOLVOL_RAW_ID,
        XYZQ_SPILL_DOLVOLSUB_RAW_ID,
    ]
}

pub fn raw_spec(raw_id: &str) -> IntradayDailyRawSpec {
    stock_minute_raw_spec(
        raw_id,
        RAW_VERSION,
        &["close", "vol", "amount"],
        RAW_WINDOW_DAYS,
    )
}

pub fn raw_specs() -> Vec<IntradayDailyRawSpec> {
    all_raw_ids()
        .iter()
        .map(|raw_id| raw_spec(raw_id))
        .collect()
}

pub fn raw_ids_for_mode(mode: XyzqIndustrySpilloverMode) -> Vec<&'static str> {
    match mode {
        XyzqIndustrySpilloverMode::IndustryMomentum => Vec::new(),
        XyzqIndustrySpilloverMode::HighCorr1 => highcorr_1_components()
            .iter()
            .map(|component| component.raw_id)
            .collect(),
        XyzqIndustrySpilloverMode::HighCorr2 => highcorr_2_components()
            .iter()
            .map(|component| component.raw_id)
            .collect(),
        XyzqIndustrySpilloverMode::HighCorr3 => highcorr_3_components()
            .iter()
            .map(|component| component.raw_id)
            .collect(),
        XyzqIndustrySpilloverMode::RetVolCorr => vec![XYZQ_SPILL_RETVOLCORR_RAW_ID],
        XyzqIndustrySpilloverMode::RetSkew => vec![XYZQ_SPILL_RETSKEW_RAW_ID],
        XyzqIndustrySpilloverMode::TaylorRet => vec![XYZQ_SPILL_TAYLORRET_RAW_ID],
    }
}

pub fn raw_specs_for_mode(mode: XyzqIndustrySpilloverMode) -> Vec<IntradayDailyRawSpec> {
    raw_ids_for_mode(mode)
        .iter()
        .map(|raw_id| raw_spec(raw_id))
        .collect()
}

pub fn factor_spec(def: XyzqIndustrySpilloverFactorDef) -> FactorSpec {
    let (dependencies, raw_dependencies, lookback) = match def.mode {
        XyzqIndustrySpilloverMode::IndustryMomentum => (
            vec![
                DataRequest::new(DatasetId::StockDailyPv, &["close"]),
                DataRequest::new(DatasetId::StockAdjFactor, &["adj_factor"]),
                DataRequest::new(DatasetId::StockBarraDaily, &["SIZE"]),
                DataRequest::new(DatasetId::StockSwClassification, &["l1_code"]),
            ],
            Vec::new(),
            INDUSTRY_MOM_LOOKBACK,
        ),
        _ => (
            vec![
                DataRequest::new(DatasetId::StockBarraDaily, &["SIZE"]),
                DataRequest::new(DatasetId::StockSwClassification, &["l1_code"]),
            ],
            raw_ids_for_mode(def.mode)
                .iter()
                .map(|raw_id| IntradayDailyRawRequest::new(*raw_id, RAW_LOOKBACK))
                .collect(),
            RAW_LOOKBACK,
        ),
    };

    FactorSpec {
        id: def.id.to_string(),
        aliases: vec![def.alias.to_string()],
        name: def.name.to_string(),
        asset_class: AssetClass::Stock,
        frequency: Frequency::Daily,
        version: VERSION.to_string(),
        tags: tags(),
        description: format!(
            "{} industry peer spillover factor with final SIZE and SW L1 neutralization.",
            def.name
        ),
        dependencies,
        intraday_raw_dependencies: raw_dependencies,
        lookback: Lookback {
            trading_days: lookback,
        },
    }
}

pub fn compute_factor(
    def: XyzqIndustrySpilloverFactorDef,
    data: &DataPool,
) -> Result<FactorSeries> {
    let factor = match def.mode {
        XyzqIndustrySpilloverMode::IndustryMomentum => compute_industry_momentum(data)?,
        XyzqIndustrySpilloverMode::HighCorr1 => compute_composite(
            data,
            highcorr_1_components().as_slice(),
            raw_ids_for_mode(def.mode)[0],
        )?,
        XyzqIndustrySpilloverMode::HighCorr2 => compute_composite(
            data,
            highcorr_2_components().as_slice(),
            raw_ids_for_mode(def.mode)[0],
        )?,
        XyzqIndustrySpilloverMode::HighCorr3 => compute_composite(
            data,
            highcorr_3_components().as_slice(),
            raw_ids_for_mode(def.mode)[0],
        )?,
        XyzqIndustrySpilloverMode::RetVolCorr => compute_independent(
            data,
            ComponentSpec {
                raw_id: XYZQ_SPILL_RETVOLCORR_RAW_ID,
                rolling: RollingSpec::MeanStd(5),
                sign: 1.0,
            },
        )?,
        XyzqIndustrySpilloverMode::RetSkew => compute_independent(
            data,
            ComponentSpec {
                raw_id: XYZQ_SPILL_RETSKEW_RAW_ID,
                rolling: RollingSpec::Mean(1),
                sign: 1.0,
            },
        )?,
        XyzqIndustrySpilloverMode::TaylorRet => compute_independent(
            data,
            ComponentSpec {
                raw_id: XYZQ_SPILL_TAYLORRET_RAW_ID,
                rolling: RollingSpec::Mean(10),
                sign: 1.0,
            },
        )?,
    };
    Ok(factor.to_factor_series(factor_spec(def)))
}

#[macro_export]
macro_rules! define_xyzq_industry_spillover_factor {
    ($struct_name:ident, $id:expr, $alias:expr, $name:expr, $mode:expr) => {
        const DEF: $crate::factor::common::xyzq_industry_spillover::XyzqIndustrySpilloverFactorDef =
            $crate::factor::common::xyzq_industry_spillover::XyzqIndustrySpilloverFactorDef {
                id: $id,
                alias: $alias,
                name: $name,
                mode: $mode,
            };

        pub struct $struct_name;

        pub fn create() -> Box<dyn $crate::factor::Factor> {
            Box::new($struct_name)
        }

        impl $crate::factor::Factor for $struct_name {
            fn spec(&self) -> $crate::core::FactorSpec {
                $crate::factor::common::xyzq_industry_spillover::factor_spec(DEF)
            }

            fn intraday_raw_specs(&self) -> Vec<$crate::core::IntradayDailyRawSpec> {
                $crate::factor::common::xyzq_industry_spillover::raw_specs_for_mode(DEF.mode)
            }

            fn intraday_raw_provider_key(&self, _raw_id: &str) -> String {
                $crate::factor::common::xyzq_industry_spillover::PROVIDER_KEY.to_string()
            }

            fn minute_compute(
                &self,
                raw_id: &str,
                context: &$crate::core::FactorContext,
                data: &$crate::data::DataPool,
            ) -> $crate::error::Result<Option<$crate::core::IntradayDailyRawSeries>> {
                let raw_ids = vec![raw_id.to_string()];
                Ok(
                    $crate::factor::common::xyzq_industry_spillover::minute_compute_many(
                        &raw_ids, context, data,
                    )?
                    .into_iter()
                    .next(),
                )
            }

            fn minute_compute_many(
                &self,
                raw_ids: &[String],
                context: &$crate::core::FactorContext,
                data: &$crate::data::DataPool,
            ) -> $crate::error::Result<Vec<$crate::core::IntradayDailyRawSeries>> {
                $crate::factor::common::xyzq_industry_spillover::minute_compute_many(
                    raw_ids, context, data,
                )
            }

            fn compute(
                &self,
                _context: &$crate::core::FactorContext,
                data: &$crate::data::DataPool,
            ) -> $crate::error::Result<$crate::core::FactorSeries> {
                $crate::factor::common::xyzq_industry_spillover::compute_factor(DEF, data)
            }
        }
    };
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
        let vol = table.required_f64_cast("vol")?;
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

        for (ts_code, mut indices) in grouped {
            indices.sort_by(|left, right| trade_times[*left].cmp(&trade_times[*right]));
            let points = minute_points_from_indices(&indices, &trade_times, &close, &vol, &amount);
            let stats = spillover_stats_for(&points);
            let key = FactorRowKey::Daily {
                trade_date: *trade_date,
                ts_code,
            };

            push_requested(
                &mut values,
                &requested,
                XYZQ_SPILL_RV_RAW_ID,
                &key,
                stats.rv,
            );
            push_requested(
                &mut values,
                &requested,
                XYZQ_SPILL_RETSKEW_RAW_ID,
                &key,
                stats.retskew,
            );
            push_requested(
                &mut values,
                &requested,
                XYZQ_SPILL_AFTVOLRATIO_RAW_ID,
                &key,
                stats.aftvolratio,
            );
            push_requested(
                &mut values,
                &requested,
                XYZQ_SPILL_VARVAR_RAW_ID,
                &key,
                stats.varvar,
            );
            push_requested(
                &mut values,
                &requested,
                XYZQ_SPILL_RETVOLCORR_RAW_ID,
                &key,
                stats.retvolcorr,
            );
            push_requested(
                &mut values,
                &requested,
                XYZQ_SPILL_MORVOLRATIO_RAW_ID,
                &key,
                stats.morvolratio,
            );
            push_requested(
                &mut values,
                &requested,
                XYZQ_SPILL_TAYLORRET_RAW_ID,
                &key,
                stats.taylorret,
            );
            push_requested(
                &mut values,
                &requested,
                XYZQ_SPILL_MORNINGRET_RAW_ID,
                &key,
                stats.morningret,
            );
            push_requested(
                &mut values,
                &requested,
                XYZQ_SPILL_OUTBOUNDRET_RAW_ID,
                &key,
                stats.outboundret,
            );
            push_requested(
                &mut values,
                &requested,
                XYZQ_SPILL_RVDIFF_RAW_ID,
                &key,
                stats.rvdiff,
            );
            push_requested(
                &mut values,
                &requested,
                XYZQ_SPILL_MORVOLMINUSAFTVOL_RAW_ID,
                &key,
                stats.morvolminusaftvol,
            );
            push_requested(
                &mut values,
                &requested,
                XYZQ_SPILL_VARVARSIGNSUB_RAW_ID,
                &key,
                stats.varvarsignsub,
            );
            push_requested(
                &mut values,
                &requested,
                XYZQ_SPILL_RETSHARP_RAW_ID,
                &key,
                stats.retsharp,
            );
            push_requested(
                &mut values,
                &requested,
                XYZQ_SPILL_MORDOLVOL_RAW_ID,
                &key,
                stats.mordolvol,
            );
            push_requested(
                &mut values,
                &requested,
                XYZQ_SPILL_DOLVOLSUB_RAW_ID,
                &key,
                stats.dolvolsub,
            );
        }
    }

    let mut output = Vec::new();
    for raw_id in all_raw_ids() {
        if requested.contains(raw_id) {
            output.push(IntradayDailyRawSeries {
                spec: raw_spec(raw_id),
                values: values.remove(raw_id).unwrap_or_default(),
            });
        }
    }
    Ok(output)
}

fn compute_industry_momentum(data: &DataPool) -> Result<PanelColumn> {
    let panel = data.daily_panel(DatasetId::StockDailyPv)?;
    let close = panel.column("close")?;
    let adj_factor =
        panel.column_from_table(data.daily(DatasetId::StockAdjFactor)?, "adj_factor")?;
    let adj_close =
        close.zip_binary(&adj_factor, |close, adj| match (clean(close), clean(adj)) {
            (Some(close), Some(adj)) => finite_value(close * adj),
            _ => None,
        })?;
    let period_ret = adj_close.ts(|values| period_return(values, INDUSTRY_MOM_RETURN_WINDOW))?;
    let peer = industry_peer_mean(&period_ret, data)?;
    let smoothed = peer.ts(|values| ts_mean(values, INDUSTRY_MOM_SMOOTH_WINDOW, MIN_PERIODS))?;
    neutralize_size_sector(&smoothed, panel, data)
}

fn compute_composite(
    data: &DataPool,
    components: &[ComponentSpec],
    panel_raw_id: &'static str,
) -> Result<PanelColumn> {
    let panel = data.intraday_daily_raw_panel(panel_raw_id)?;
    let mut ranked_components = Vec::with_capacity(components.len());
    for component in components {
        let values = component_series(data, &panel, *component)?;
        ranked_components.push(rank_score_component(&values)?);
    }
    let composite = average_columns(&panel, &ranked_components)?;
    let filled = fill_missing_with_cs_mean(&composite)?;
    neutralize_size_sector(&filled, &panel, data)
}

fn compute_independent(data: &DataPool, component: ComponentSpec) -> Result<PanelColumn> {
    let panel = data.intraday_daily_raw_panel(component.raw_id)?;
    let values = component_series(data, &panel, component)?;
    neutralize_size_sector(&values, &panel, data)
}

fn component_series(
    data: &DataPool,
    panel: &DailyPanel,
    component: ComponentSpec,
) -> Result<PanelColumn> {
    let raw = panel.column(component.raw_id)?;
    let peer = industry_peer_mean(&raw, data)?;
    let rolled = match component.rolling {
        RollingSpec::Mean(window) => peer.ts(|values| ts_mean(values, window, MIN_PERIODS))?,
        RollingSpec::MeanStd(window) => peer.ts(|values| ts_mean_std_ratio(values, window))?,
    };
    Ok(if (component.sign - 1.0).abs() <= EPS {
        rolled
    } else {
        rolled.map_values(|value| clean(value).map(|value| value * component.sign))
    })
}

fn industry_peer_mean(values: &PanelColumn, data: &DataPool) -> Result<PanelColumn> {
    let sector_map = ClassificationMap::from_table(
        data.daily(DatasetId::StockSwClassification)?,
        ClassificationLevel::Sector,
    )?;
    values.cs_by_group(
        |trade_date, ts_codes| sector_map.groups_for(trade_date, ts_codes),
        industry_peer_mean_cs,
    )
}

fn industry_peer_mean_cs(values: &[Option<f64>], groups: &[Option<String>]) -> Vec<Option<f64>> {
    let mut sums = HashMap::<&str, (f64, usize)>::new();
    for (value, group) in values.iter().zip(groups) {
        let (Some(value), Some(group)) = (clean(*value), group.as_deref()) else {
            continue;
        };
        let entry = sums.entry(group).or_insert((0.0, 0));
        entry.0 += value;
        entry.1 += 1;
    }

    values
        .iter()
        .zip(groups)
        .map(|(value, group)| {
            let group = group.as_deref()?;
            let (sum, count) = *sums.get(group)?;
            match clean(*value) {
                Some(value) => {
                    if count > 1 {
                        finite_value((sum - value) / (count - 1) as f64)
                    } else {
                        None
                    }
                }
                None => (count > 0)
                    .then(|| sum / count as f64)
                    .and_then(finite_value),
            }
        })
        .collect()
}

fn rank_score_component(values: &PanelColumn) -> Result<PanelColumn> {
    values.cs(|cross_section| {
        cs_pctrank(cross_section, true)
            .into_iter()
            .map(|rank| match clean(rank) {
                Some(rank) if rank < 0.9 => Some(2.0 * rank - 1.0),
                _ => None,
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
            if let Some(value) = clean(column.values()[idx]) {
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
            .filter_map(|value| clean(*value))
            .collect::<Vec<_>>();
        let Some(mean) = mean(&finite) else {
            return vec![None; cross_section.len()];
        };
        cross_section
            .iter()
            .map(|value| clean(*value).or(Some(mean)))
            .collect()
    })
}

fn highcorr_1_components() -> Vec<ComponentSpec> {
    vec![
        ComponentSpec {
            raw_id: XYZQ_SPILL_MORDOLVOL_RAW_ID,
            rolling: RollingSpec::Mean(25),
            sign: -1.0,
        },
        ComponentSpec {
            raw_id: XYZQ_SPILL_DOLVOLSUB_RAW_ID,
            rolling: RollingSpec::Mean(25),
            sign: -1.0,
        },
        ComponentSpec {
            raw_id: XYZQ_SPILL_RETSHARP_RAW_ID,
            rolling: RollingSpec::MeanStd(25),
            sign: 1.0,
        },
        ComponentSpec {
            raw_id: XYZQ_SPILL_MORNINGRET_RAW_ID,
            rolling: RollingSpec::Mean(15),
            sign: 1.0,
        },
        ComponentSpec {
            raw_id: XYZQ_SPILL_RVDIFF_RAW_ID,
            rolling: RollingSpec::Mean(15),
            sign: 1.0,
        },
        ComponentSpec {
            raw_id: XYZQ_SPILL_OUTBOUNDRET_RAW_ID,
            rolling: RollingSpec::Mean(15),
            sign: 1.0,
        },
        ComponentSpec {
            raw_id: XYZQ_SPILL_VARVARSIGNSUB_RAW_ID,
            rolling: RollingSpec::Mean(25),
            sign: 1.0,
        },
    ]
}

fn highcorr_2_components() -> Vec<ComponentSpec> {
    vec![
        ComponentSpec {
            raw_id: XYZQ_SPILL_VARVAR_RAW_ID,
            rolling: RollingSpec::Mean(3),
            sign: 1.0,
        },
        ComponentSpec {
            raw_id: XYZQ_SPILL_RV_RAW_ID,
            rolling: RollingSpec::Mean(1),
            sign: 1.0,
        },
    ]
}

fn highcorr_3_components() -> Vec<ComponentSpec> {
    vec![
        ComponentSpec {
            raw_id: XYZQ_SPILL_MORVOLRATIO_RAW_ID,
            rolling: RollingSpec::Mean(10),
            sign: 1.0,
        },
        ComponentSpec {
            raw_id: XYZQ_SPILL_AFTVOLRATIO_RAW_ID,
            rolling: RollingSpec::Mean(1),
            sign: -1.0,
        },
        ComponentSpec {
            raw_id: XYZQ_SPILL_MORVOLMINUSAFTVOL_RAW_ID,
            rolling: RollingSpec::Mean(15),
            sign: 1.0,
        },
    ]
}

fn tags() -> Vec<String> {
    [
        "price_volume",
        "industry_spillover",
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

fn minute_points_from_indices(
    indices: &[usize],
    trade_times: &[Option<String>],
    close: &[Option<f64>],
    vol: &[Option<f64>],
    amount: &[Option<f64>],
) -> Vec<MinutePoint> {
    indices
        .iter()
        .filter_map(|idx| {
            let time = trade_times[*idx].clone()?;
            Some(MinutePoint {
                in_sample: intraday_time_in_range(&time, SAMPLE_START, SAMPLE_END),
                time,
                close: clean_intraday_value(close[*idx]).filter(|value| *value > 0.0),
                vol: clean_intraday_value(vol[*idx]).filter(|value| *value >= 0.0),
                amount: clean_intraday_value(amount[*idx]).filter(|value| *value >= 0.0),
            })
        })
        .collect()
}

fn spillover_stats_for(points: &[MinutePoint]) -> SpilloverMinuteStats {
    let returns = return_points(points);
    let simple_returns = returns
        .iter()
        .filter(|point| point.time_in_sample)
        .map(|point| point.simple_ret)
        .collect::<Vec<_>>();
    let log_returns = returns
        .iter()
        .filter(|point| point.time_in_sample)
        .map(|point| point.log_ret)
        .collect::<Vec<_>>();
    let volumes = returns
        .iter()
        .filter(|point| point.time_in_sample)
        .filter_map(|point| point.vol)
        .collect::<Vec<_>>();

    let rv = (!simple_returns.is_empty())
        .then(|| {
            simple_returns
                .iter()
                .map(|value| value * value)
                .sum::<f64>()
        })
        .and_then(finite_value);
    let retskew = skewness(&simple_returns);
    let varvar_value = varvar(&simple_returns);
    let retvolcorr = pearson_pairs(
        &returns
            .iter()
            .filter(|point| point.time_in_sample)
            .filter_map(|point| point.vol.map(|vol| (point.simple_ret, vol)))
            .collect::<Vec<_>>(),
    );
    let retsharp = match (mean(&simple_returns), std_pop(&simple_returns)) {
        (Some(mean), Some(std)) if std.abs() > EPS => finite_value(mean / std),
        _ => None,
    };
    let taylorret = taylorret(&returns);
    let outboundret = outboundret(&simple_returns);
    let rvdiff = rv_diff(&simple_returns);
    let varvar_pos = varvar(
        &simple_returns
            .iter()
            .copied()
            .filter(|value| *value > 0.0)
            .collect::<Vec<_>>(),
    );
    let varvar_neg = varvar(
        &simple_returns
            .iter()
            .copied()
            .filter(|value| *value < 0.0)
            .collect::<Vec<_>>(),
    );
    let varvarsignsub = match (varvar_pos, varvar_neg) {
        (Some(pos), Some(neg)) => finite_value(pos - neg),
        _ => None,
    };
    let morvolratio = window_share(
        points,
        |point| point.in_sample && intraday_time_in_range(&point.time, SAMPLE_START, MORNING_END),
        |point| point.vol,
    );
    let aftvolratio = window_share(
        points,
        |point| {
            point.in_sample && intraday_time_in_range(&point.time, AFTERNOON_START, AFTERNOON_END)
        },
        |point| point.vol,
    );
    let morningret = close_at(points, MORNING_END).and_then(|end| {
        close_at(points, OPEN_ANCHOR).and_then(|start| safe_div(Some(end - start), Some(start)))
    });
    let mor_amount_share = window_share(
        points,
        |point| point.in_sample && intraday_time_in_range(&point.time, SAMPLE_START, MORNING_END),
        |point| point.amount,
    );
    let aft_amount_share = window_share(
        points,
        |point| {
            point.in_sample && intraday_time_in_range(&point.time, AFTERNOON_START, AFTERNOON_END)
        },
        |point| point.amount,
    );
    let mordolvol = match (mor_amount_share, morvolratio) {
        (Some(amount), Some(volume)) => finite_value(amount - volume),
        _ => None,
    };
    let aftdolvol = match (aft_amount_share, aftvolratio) {
        (Some(amount), Some(volume)) => finite_value(amount - volume),
        _ => None,
    };
    let dolvolsub = match (mordolvol, aftdolvol) {
        (Some(mor), Some(aft)) => finite_value(mor - aft),
        _ => None,
    };
    let morvolminusaftvol = match (morvolratio, aftvolratio) {
        (Some(mor), Some(aft)) => finite_value(mor - aft),
        _ => None,
    };

    let _ = log_returns;
    let _ = volumes;

    SpilloverMinuteStats {
        rv,
        retskew,
        aftvolratio,
        varvar: varvar_value,
        retvolcorr,
        morvolratio,
        taylorret,
        morningret,
        outboundret,
        rvdiff,
        morvolminusaftvol,
        varvarsignsub,
        retsharp,
        mordolvol,
        dolvolsub,
    }
}

fn return_points(points: &[MinutePoint]) -> Vec<ReturnPoint> {
    let mut output = Vec::new();
    let mut prev_close: Option<f64> = None;
    for point in points {
        if let (Some(previous), Some(current)) = (prev_close, point.close) {
            if previous > 0.0 && current > 0.0 && point.in_sample {
                let simple_ret: f64 = current / previous - 1.0;
                let log_ret: f64 = (current / previous).ln();
                if simple_ret.is_finite() && log_ret.is_finite() {
                    output.push(ReturnPoint {
                        time_in_sample: true,
                        simple_ret,
                        log_ret,
                        vol: point.vol,
                    });
                }
            }
        }
        if point.close.is_some() {
            prev_close = point.close;
        }
    }
    output
}

fn window_share<F, G>(points: &[MinutePoint], mut in_window: F, mut value: G) -> Option<f64>
where
    F: FnMut(&MinutePoint) -> bool,
    G: FnMut(&MinutePoint) -> Option<f64>,
{
    let mut total = 0.0;
    let mut part = 0.0;
    for point in points {
        if !point.in_sample {
            continue;
        }
        let Some(value) = value(point) else {
            continue;
        };
        total += value;
        if in_window(point) {
            part += value;
        }
    }
    if total.abs() <= EPS {
        return None;
    }
    finite_value(part / total)
}

fn close_at(points: &[MinutePoint], target: &str) -> Option<f64> {
    points
        .iter()
        .find(|point| intraday_time_in_range(&point.time, target, target))
        .and_then(|point| point.close)
}

fn period_return(values: &[Option<f64>], window: usize) -> Vec<Option<f64>> {
    let mut output = vec![None; values.len()];
    for idx in window..values.len() {
        let Some(current) = clean(values[idx]) else {
            continue;
        };
        let Some(previous) = clean(values[idx - window]) else {
            continue;
        };
        if previous.abs() > EPS {
            output[idx] = finite_value(current / previous - 1.0);
        }
    }
    output
}

fn ts_mean_std_ratio(values: &[Option<f64>], window: usize) -> Vec<Option<f64>> {
    let mut output = vec![None; values.len()];
    for idx in 0..values.len() {
        let start = idx + 1 - (idx + 1).min(window);
        let window_values = values[start..=idx]
            .iter()
            .filter_map(|value| clean(*value))
            .collect::<Vec<_>>();
        if window_values.len() < 2 {
            continue;
        }
        let Some(mean) = mean(&window_values) else {
            continue;
        };
        let Some(std) = std_pop(&window_values) else {
            continue;
        };
        if std.abs() > EPS {
            output[idx] = finite_value(mean / std);
        }
    }
    output
}

fn varvar(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let rolling = rolling_std(values, 5);
    let rolling2 = rolling_std(&rolling, 5);
    mean(&rolling2)
}

fn rolling_std(values: &[f64], window: usize) -> Vec<f64> {
    let mut output = Vec::with_capacity(values.len());
    for idx in 0..values.len() {
        let start = idx + 1 - (idx + 1).min(window);
        output.push(std_pop(&values[start..=idx]).unwrap_or(0.0));
    }
    output
}

fn outboundret(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let q20 = quantile(values, 0.2)?;
    let q80 = quantile(values, 0.8)?;
    finite_value(
        values
            .iter()
            .filter(|value| **value <= q20 || **value >= q80)
            .sum(),
    )
}

fn rv_diff(values: &[f64]) -> Option<f64> {
    let total = values.iter().map(|value| value * value).sum::<f64>();
    if total.abs() <= EPS {
        return Some(0.0);
    }
    let up = values
        .iter()
        .filter(|value| **value > 0.0)
        .map(|value| value * value)
        .sum::<f64>();
    let down = values
        .iter()
        .filter(|value| **value < 0.0)
        .map(|value| value * value)
        .sum::<f64>();
    finite_value((up - down) / total)
}

fn taylorret(points: &[ReturnPoint]) -> Option<f64> {
    let simple = points
        .iter()
        .filter(|point| point.time_in_sample)
        .map(|point| point.simple_ret)
        .collect::<Vec<_>>();
    let denom = simple.iter().map(|value| value.abs()).sum::<f64>() / simple.len().max(1) as f64;
    if denom.abs() <= EPS {
        return None;
    }
    let terms = points
        .iter()
        .filter(|point| point.time_in_sample)
        .map(|point| {
            (2.0 * (point.simple_ret - point.log_ret) - point.log_ret * point.log_ret) / denom
        })
        .collect::<Vec<_>>();
    mean(&terms)
}

fn skewness(values: &[f64]) -> Option<f64> {
    if values.len() < 2 {
        return None;
    }
    let mean = mean(values)?;
    let std = std_pop(values)?;
    if std.abs() <= EPS {
        return None;
    }
    finite_value(
        values
            .iter()
            .map(|value| ((value - mean) / std).powi(3))
            .sum::<f64>()
            / values.len() as f64,
    )
}

fn pearson_pairs(pairs: &[(f64, f64)]) -> Option<f64> {
    if pairs.len() < 2 {
        return None;
    }
    let mean_x = pairs.iter().map(|(x, _)| *x).sum::<f64>() / pairs.len() as f64;
    let mean_y = pairs.iter().map(|(_, y)| *y).sum::<f64>() / pairs.len() as f64;
    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;
    for (x, y) in pairs {
        let dx = x - mean_x;
        let dy = y - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }
    if var_x <= EPS || var_y <= EPS {
        return None;
    }
    finite_value(cov / (var_x.sqrt() * var_y.sqrt()))
}

fn mean(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    finite_value(values.iter().sum::<f64>() / values.len() as f64)
}

fn std_pop(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mean = mean(values)?;
    finite_value(
        (values
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            / values.len() as f64)
            .sqrt(),
    )
}

fn quantile(values: &[f64], q: f64) -> Option<f64> {
    let mut values = values.to_vec();
    quantile_linear(&mut values, q)
}

fn safe_div(numerator: Option<f64>, denominator: Option<f64>) -> Option<f64> {
    match (clean(numerator), clean(denominator)) {
        (Some(numerator), Some(denominator)) if denominator.abs() > EPS => {
            finite_value(numerator / denominator)
        }
        _ => None,
    }
}

fn clean(value: Option<f64>) -> Option<f64> {
    value.filter(|value| value.is_finite())
}

fn finite_value(value: f64) -> Option<f64> {
    value.is_finite().then_some(value)
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

    fn point(time: &str, close: f64, vol: f64, amount: f64) -> MinutePoint {
        MinutePoint {
            time: time.to_string(),
            in_sample: intraday_time_in_range(time, SAMPLE_START, SAMPLE_END),
            close: Some(close),
            vol: Some(vol),
            amount: Some(amount),
        }
    }

    #[test]
    fn xyzq_spillover_uses_0930_as_return_anchor() {
        let points = vec![
            point("09:30:00", 100.0, 0.0, 0.0),
            point("09:31:00", 101.0, 1.0, 1.0),
        ];
        let returns = return_points(&points);

        assert_eq!(returns.len(), 1);
        assert!((returns[0].simple_ret - 0.01).abs() < 1e-12);
    }

    #[test]
    fn xyzq_spillover_industry_peer_mean_excludes_self() {
        let values = vec![Some(1.0), Some(3.0), None, Some(10.0)];
        let groups = vec![
            Some("a".to_string()),
            Some("a".to_string()),
            Some("a".to_string()),
            Some("b".to_string()),
        ];
        let peer = industry_peer_mean_cs(&values, &groups);

        assert_close(peer[0], Some(3.0));
        assert_close(peer[1], Some(1.0));
        assert_close(peer[2], Some(2.0));
        assert_eq!(peer[3], None);
    }

    #[test]
    fn xyzq_spillover_mean_std_requires_two_values() {
        let values = vec![Some(1.0), None, Some(3.0)];
        let ratio = ts_mean_std_ratio(&values, 3);

        assert_eq!(ratio[0], None);
        assert_close(ratio[2], Some(2.0));
    }

    #[test]
    fn xyzq_spillover_raw_stats_emit_core_values() {
        let points = vec![
            point("09:30:00", 100.0, 0.0, 0.0),
            point("09:31:00", 101.0, 1.0, 2.0),
            point("09:32:00", 102.0, 2.0, 4.0),
            point("14:31:00", 101.0, 3.0, 3.0),
            point("15:00:00", 103.0, 4.0, 8.0),
        ];
        let stats = spillover_stats_for(&points);

        assert!(stats.rv.is_some());
        assert!(stats.morvolratio.is_some());
        assert!(stats.aftvolratio.is_some());
        assert!(stats.mordolvol.is_some());
    }
}

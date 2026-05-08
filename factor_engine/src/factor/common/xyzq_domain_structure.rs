use std::collections::{BTreeMap, BTreeSet};

use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorRowKey, FactorSeries, FactorSpec,
    FactorValue, Frequency, IntradayDailyRawRequest, IntradayDailyRawSeries, IntradayDailyRawSpec,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::stock_daily_ops::neutralize_size_sector;
use crate::factor::common::{clean_intraday_value, intraday_time_in_range, stock_minute_raw_spec};
use crate::factor::common::{DailyPanel, PanelColumn};
use crate::operators::{cs_pctrank, cs_zscore, ts_mean};

pub const RAW_VERSION: &str = "0.1.0";
pub const VERSION: &str = "0.1.0";

const RAW_WINDOW_DAYS: usize = 1;
const DS_LOOKBACK: usize = 19;
const SIM_LOOKBACK: usize = 28;
const ROLLING_WINDOW: usize = 20;
const SIM_WINDOW: usize = 10;
const MIN_PERIODS: usize = 1;
const EPSILON: f64 = 0.01;
const TIME_DOMAIN_COUNT: usize = 8;
const Q_DOMAIN_COUNT: usize = 5;
const BLOCK_COUNT: usize = 16;

pub const TIME_RTN_RAW_IDS: [&str; TIME_DOMAIN_COUNT] = [
    "daily_qchr_t_rtn_0",
    "daily_qchr_t_rtn_1",
    "daily_qchr_t_rtn_2",
    "daily_qchr_t_rtn_3",
    "daily_qchr_t_rtn_4",
    "daily_qchr_t_rtn_5",
    "daily_qchr_t_rtn_6",
    "daily_qchr_t_rtn_7",
];

pub const TIME_VOL_RAW_IDS: [&str; TIME_DOMAIN_COUNT] = [
    "daily_qchr_t_vol_0",
    "daily_qchr_t_vol_1",
    "daily_qchr_t_vol_2",
    "daily_qchr_t_vol_3",
    "daily_qchr_t_vol_4",
    "daily_qchr_t_vol_5",
    "daily_qchr_t_vol_6",
    "daily_qchr_t_vol_7",
];

pub const PRICE_STD_RAW_IDS: [&str; Q_DOMAIN_COUNT] = [
    "daily_qchr_p_std_0",
    "daily_qchr_p_std_1",
    "daily_qchr_p_std_2",
    "daily_qchr_p_std_3",
    "daily_qchr_p_std_4",
];

pub const PRICE_VOL_RAW_IDS: [&str; Q_DOMAIN_COUNT] = [
    "daily_qchr_p_vol_0",
    "daily_qchr_p_vol_1",
    "daily_qchr_p_vol_2",
    "daily_qchr_p_vol_3",
    "daily_qchr_p_vol_4",
];

pub const PRICE_RTN_RAW_IDS: [&str; Q_DOMAIN_COUNT] = [
    "daily_qchr_p_rtn_0",
    "daily_qchr_p_rtn_1",
    "daily_qchr_p_rtn_2",
    "daily_qchr_p_rtn_3",
    "daily_qchr_p_rtn_4",
];

pub const VOLUME_STD_RAW_IDS: [&str; Q_DOMAIN_COUNT] = [
    "daily_qchr_v_std_0",
    "daily_qchr_v_std_1",
    "daily_qchr_v_std_2",
    "daily_qchr_v_std_3",
    "daily_qchr_v_std_4",
];

pub const VOLUME_RTN_RAW_IDS: [&str; Q_DOMAIN_COUNT] = [
    "daily_qchr_v_rtn_0",
    "daily_qchr_v_rtn_1",
    "daily_qchr_v_rtn_2",
    "daily_qchr_v_rtn_3",
    "daily_qchr_v_rtn_4",
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum XyzqDomainRawFamily {
    Time,
    Price,
    Volume,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum XyzqDomainFeature {
    TimeRtn,
    TimeVol,
    PriceStd,
    PriceVol,
    PriceRtn,
    VolumeStd,
    VolumeRtn,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum XyzqDomainCorrStatistic {
    Mean,
    Std,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum XyzqDomainKind {
    Time,
    Price,
    Volume,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum XyzqDomainFactorKind {
    IntraDs {
        feature: XyzqDomainFeature,
    },
    IntraDs2 {
        feature: XyzqDomainFeature,
    },
    Corr {
        feature: XyzqDomainFeature,
        statistic: XyzqDomainCorrStatistic,
    },
    PeerDs {
        domain: XyzqDomainKind,
    },
}

#[derive(Clone, Copy, Debug)]
pub struct XyzqDomainFactorDef {
    pub id: &'static str,
    pub alias: &'static str,
    pub name: &'static str,
    pub kind: XyzqDomainFactorKind,
}

#[derive(Clone, Copy, Debug, Default)]
struct DomainStats {
    time_rtn: [Option<f64>; TIME_DOMAIN_COUNT],
    time_vol: [Option<f64>; TIME_DOMAIN_COUNT],
    price_std: [Option<f64>; Q_DOMAIN_COUNT],
    price_vol: [Option<f64>; Q_DOMAIN_COUNT],
    price_rtn: [Option<f64>; Q_DOMAIN_COUNT],
    volume_std: [Option<f64>; Q_DOMAIN_COUNT],
    volume_rtn: [Option<f64>; Q_DOMAIN_COUNT],
}

#[derive(Clone, Debug)]
struct MinutePoint {
    time: String,
    close: Option<f64>,
    high: Option<f64>,
    low: Option<f64>,
    vol: Option<f64>,
}

#[derive(Clone, Copy, Debug)]
struct BlockDef {
    start: &'static str,
    end: &'static str,
    anchor: &'static str,
}

#[derive(Clone, Copy, Debug)]
struct BlockStats {
    rtn: Option<f64>,
    std: Option<f64>,
    vol_share: Option<f64>,
    price_split: Option<f64>,
}

#[derive(Clone, Debug)]
struct NormalizedWindowColumn {
    valid: Vec<bool>,
    values: Vec<[f64; SIM_WINDOW]>,
}

pub fn time_raw_ids() -> Vec<&'static str> {
    TIME_RTN_RAW_IDS
        .iter()
        .chain(TIME_VOL_RAW_IDS.iter())
        .copied()
        .collect()
}

pub fn price_raw_ids() -> Vec<&'static str> {
    PRICE_STD_RAW_IDS
        .iter()
        .chain(PRICE_VOL_RAW_IDS.iter())
        .chain(PRICE_RTN_RAW_IDS.iter())
        .copied()
        .collect()
}

pub fn volume_raw_ids() -> Vec<&'static str> {
    VOLUME_STD_RAW_IDS
        .iter()
        .chain(VOLUME_RTN_RAW_IDS.iter())
        .copied()
        .collect()
}

pub fn raw_ids_for_family(family: XyzqDomainRawFamily) -> Vec<&'static str> {
    match family {
        XyzqDomainRawFamily::Time => time_raw_ids(),
        XyzqDomainRawFamily::Price => price_raw_ids(),
        XyzqDomainRawFamily::Volume => volume_raw_ids(),
    }
}

pub fn raw_specs_for_family(family: XyzqDomainRawFamily) -> Vec<IntradayDailyRawSpec> {
    raw_ids_for_family(family)
        .into_iter()
        .map(|raw_id| raw_spec_for_family(raw_id, family))
        .collect()
}

fn raw_spec_for_family(raw_id: &str, family: XyzqDomainRawFamily) -> IntradayDailyRawSpec {
    let columns = match family {
        XyzqDomainRawFamily::Time | XyzqDomainRawFamily::Volume => vec!["close", "vol"],
        XyzqDomainRawFamily::Price => vec!["close", "high", "low", "vol"],
    };
    stock_minute_raw_spec(raw_id, RAW_VERSION, &columns, RAW_WINDOW_DAYS)
}

fn raw_spec(raw_id: &str) -> IntradayDailyRawSpec {
    if TIME_RTN_RAW_IDS.contains(&raw_id) || TIME_VOL_RAW_IDS.contains(&raw_id) {
        raw_spec_for_family(raw_id, XyzqDomainRawFamily::Time)
    } else if PRICE_STD_RAW_IDS.contains(&raw_id)
        || PRICE_VOL_RAW_IDS.contains(&raw_id)
        || PRICE_RTN_RAW_IDS.contains(&raw_id)
    {
        raw_spec_for_family(raw_id, XyzqDomainRawFamily::Price)
    } else {
        raw_spec_for_family(raw_id, XyzqDomainRawFamily::Volume)
    }
}

pub fn factor_spec(def: XyzqDomainFactorDef) -> FactorSpec {
    let lookback = lookback_for_kind(def.kind);
    FactorSpec {
        id: def.id.to_string(),
        aliases: vec![def.alias.to_string()],
        name: def.name.to_string(),
        asset_class: AssetClass::Stock,
        frequency: Frequency::Daily,
        version: VERSION.to_string(),
        tags: tags(),
        description: format!(
            "{} from XYZQ intraday domain features, rolling mean, cross-sectional percentile rank, and SIZE/SW-sector neutralization.",
            def.name
        ),
        dependencies: dependencies(def.kind),
        intraday_raw_dependencies: raw_dependencies_for_kind(def.kind)
            .into_iter()
            .map(|raw_id| IntradayDailyRawRequest::new(raw_id, lookback))
            .collect(),
        lookback: Lookback {
            trading_days: lookback,
        },
    }
}

pub fn compute_factor(def: XyzqDomainFactorDef, data: &DataPool) -> Result<FactorSeries> {
    let spec = factor_spec(def);
    let raw_ids = raw_dependencies_for_kind(def.kind);
    let panel = data.intraday_daily_raw_panel(raw_ids[0])?;
    let raw = match def.kind {
        XyzqDomainFactorKind::IntraDs { feature } => {
            let columns = raw_columns(panel, feature_raw_ids(feature))?;
            derive_by_row(panel, &columns, |values| intra_ds_value(values))
        }
        XyzqDomainFactorKind::IntraDs2 { feature } => {
            let feature_columns = raw_columns(panel, feature_raw_ids(feature))?;
            let rtn_columns = raw_columns(panel, domain_return_raw_ids(feature.domain()))?;
            derive_ds2_by_row(panel, &feature_columns, &rtn_columns)
        }
        XyzqDomainFactorKind::Corr { feature, statistic } => {
            let columns = raw_columns(panel, feature_raw_ids(feature))?;
            compute_similarity_stat(panel, &columns, statistic)
        }
        XyzqDomainFactorKind::PeerDs { domain } => compute_peer_ds(panel, data, domain),
    }?;
    let smoothed = raw.ts(|values| ts_mean(values, ROLLING_WINDOW, MIN_PERIODS))?;
    let ranked = smoothed.cs(|values| cs_pctrank(values, true))?;
    let factor = neutralize_size_sector(&ranked, panel, data)?;
    Ok(factor.to_factor_series(spec))
}

#[macro_export]
macro_rules! define_xyzq_domain_factor {
    ($struct_name:ident, $id:expr, $alias:expr, $name:expr, $kind:expr) => {
        const DEF: $crate::factor::common::xyzq_domain_structure::XyzqDomainFactorDef =
            $crate::factor::common::xyzq_domain_structure::XyzqDomainFactorDef {
                id: $id,
                alias: $alias,
                name: $name,
                kind: $kind,
            };

        pub struct $struct_name;

        pub fn create() -> Box<dyn $crate::factor::Factor> {
            Box::new($struct_name)
        }

        impl $crate::factor::Factor for $struct_name {
            fn spec(&self) -> $crate::core::FactorSpec {
                $crate::factor::common::xyzq_domain_structure::factor_spec(DEF)
            }

            fn compute(
                &self,
                _context: &$crate::core::FactorContext,
                data: &$crate::data::DataPool,
            ) -> $crate::error::Result<$crate::core::FactorSeries> {
                $crate::factor::common::xyzq_domain_structure::compute_factor(DEF, data)
            }
        }
    };
}

pub fn minute_compute_many_for(
    raw_ids: &[String],
    context: &FactorContext,
    data: &DataPool,
    family: XyzqDomainRawFamily,
) -> Result<Vec<IntradayDailyRawSeries>> {
    let family_raw_ids = raw_ids_for_family(family);
    let requested = raw_ids
        .iter()
        .map(String::as_str)
        .filter(|raw_id| family_raw_ids.contains(raw_id))
        .collect::<BTreeSet<_>>();
    if requested.is_empty() {
        return Ok(Vec::new());
    }

    let mut values = family_raw_ids
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
        let high = if family == XyzqDomainRawFamily::Price {
            Some(table.required_f64_cast("high")?)
        } else {
            None
        };
        let low = if family == XyzqDomainRawFamily::Price {
            Some(table.required_f64_cast("low")?)
        } else {
            None
        };

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
            let points = minute_points_from_indices(
                &indices,
                &trade_times,
                &close,
                high.as_deref(),
                low.as_deref(),
                &vol,
            );
            let stats = domain_stats_for(&points, family);
            let key = FactorRowKey::Daily {
                trade_date: *trade_date,
                ts_code,
            };
            push_family_values(&mut values, &requested, &key, &stats, family);
        }
    }

    let mut output = Vec::new();
    for raw_id in family_raw_ids {
        if requested.contains(raw_id) {
            output.push(IntradayDailyRawSeries {
                spec: raw_spec(raw_id),
                values: values.remove(raw_id).unwrap_or_default(),
            });
        }
    }
    Ok(output)
}

fn lookback_for_kind(kind: XyzqDomainFactorKind) -> usize {
    match kind {
        XyzqDomainFactorKind::IntraDs { .. } | XyzqDomainFactorKind::IntraDs2 { .. } => DS_LOOKBACK,
        XyzqDomainFactorKind::Corr { .. } | XyzqDomainFactorKind::PeerDs { .. } => SIM_LOOKBACK,
    }
}

fn dependencies(kind: XyzqDomainFactorKind) -> Vec<DataRequest> {
    let mut dependencies = vec![
        DataRequest::new(DatasetId::StockBarraDaily, &["SIZE"]),
        DataRequest::new(DatasetId::StockSwClassification, &["l1_code"]),
    ];
    if matches!(kind, XyzqDomainFactorKind::PeerDs { .. }) {
        dependencies.push(DataRequest::new(DatasetId::StockDailyPv, &["close"]));
    }
    dependencies
}

fn raw_dependencies_for_kind(kind: XyzqDomainFactorKind) -> Vec<&'static str> {
    let mut raw_ids = BTreeSet::<&'static str>::new();
    match kind {
        XyzqDomainFactorKind::IntraDs { feature } | XyzqDomainFactorKind::Corr { feature, .. } => {
            raw_ids.extend(feature_raw_ids(feature));
        }
        XyzqDomainFactorKind::IntraDs2 { feature } => {
            raw_ids.extend(feature_raw_ids(feature));
            raw_ids.extend(domain_return_raw_ids(feature.domain()));
        }
        XyzqDomainFactorKind::PeerDs { domain } => {
            for feature in domain.peer_features() {
                raw_ids.extend(feature_raw_ids(feature));
            }
        }
    }
    raw_ids.into_iter().collect()
}

fn feature_raw_ids(feature: XyzqDomainFeature) -> Vec<&'static str> {
    match feature {
        XyzqDomainFeature::TimeRtn => TIME_RTN_RAW_IDS.to_vec(),
        XyzqDomainFeature::TimeVol => TIME_VOL_RAW_IDS.to_vec(),
        XyzqDomainFeature::PriceStd => PRICE_STD_RAW_IDS.to_vec(),
        XyzqDomainFeature::PriceVol => PRICE_VOL_RAW_IDS.to_vec(),
        XyzqDomainFeature::PriceRtn => PRICE_RTN_RAW_IDS.to_vec(),
        XyzqDomainFeature::VolumeStd => VOLUME_STD_RAW_IDS.to_vec(),
        XyzqDomainFeature::VolumeRtn => VOLUME_RTN_RAW_IDS.to_vec(),
    }
}

fn domain_return_raw_ids(domain: XyzqDomainKind) -> Vec<&'static str> {
    match domain {
        XyzqDomainKind::Time => TIME_RTN_RAW_IDS.to_vec(),
        XyzqDomainKind::Price => PRICE_RTN_RAW_IDS.to_vec(),
        XyzqDomainKind::Volume => VOLUME_RTN_RAW_IDS.to_vec(),
    }
}

impl XyzqDomainFeature {
    fn domain(self) -> XyzqDomainKind {
        match self {
            XyzqDomainFeature::TimeRtn | XyzqDomainFeature::TimeVol => XyzqDomainKind::Time,
            XyzqDomainFeature::PriceStd
            | XyzqDomainFeature::PriceVol
            | XyzqDomainFeature::PriceRtn => XyzqDomainKind::Price,
            XyzqDomainFeature::VolumeStd | XyzqDomainFeature::VolumeRtn => XyzqDomainKind::Volume,
        }
    }
}

impl XyzqDomainKind {
    fn peer_features(self) -> Vec<XyzqDomainFeature> {
        match self {
            XyzqDomainKind::Time => vec![XyzqDomainFeature::TimeRtn, XyzqDomainFeature::TimeVol],
            XyzqDomainKind::Price => {
                vec![XyzqDomainFeature::PriceStd, XyzqDomainFeature::PriceVol]
            }
            XyzqDomainKind::Volume => {
                vec![XyzqDomainFeature::VolumeRtn, XyzqDomainFeature::VolumeStd]
            }
        }
    }
}

fn tags() -> Vec<String> {
    [
        "price_volume",
        "return",
        "volume",
        "intraday",
        "minute_agg",
        "domain",
        "similarity",
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

fn raw_columns(panel: &DailyPanel, raw_ids: Vec<&str>) -> Result<Vec<PanelColumn>> {
    raw_ids
        .into_iter()
        .map(|raw_id| panel.column(raw_id))
        .collect()
}

fn derive_by_row<F>(panel: &DailyPanel, columns: &[PanelColumn], mut f: F) -> Result<PanelColumn>
where
    F: FnMut(&[Option<f64>]) -> Option<f64>,
{
    let mut output = Vec::with_capacity(panel.shape_len());
    let mut scratch = vec![None; columns.len()];
    for offset in 0..panel.shape_len() {
        for (idx, column) in columns.iter().enumerate() {
            scratch[idx] = column.values()[offset];
        }
        output.push(f(&scratch));
    }
    panel.column_from_values(output)
}

fn derive_ds2_by_row(
    panel: &DailyPanel,
    feature_columns: &[PanelColumn],
    rtn_columns: &[PanelColumn],
) -> Result<PanelColumn> {
    let mut output = Vec::with_capacity(panel.shape_len());
    let mut feature_values = vec![None; feature_columns.len()];
    let mut rtn_values = vec![None; rtn_columns.len()];
    for offset in 0..panel.shape_len() {
        for (idx, column) in feature_columns.iter().enumerate() {
            feature_values[idx] = column.values()[offset];
        }
        for (idx, column) in rtn_columns.iter().enumerate() {
            rtn_values[idx] = column.values()[offset];
        }
        output.push(intra_ds2_value(&feature_values, &rtn_values));
    }
    panel.column_from_values(output)
}

fn minute_points_from_indices(
    indices: &[usize],
    trade_times: &[Option<String>],
    close: &[Option<f64>],
    high: Option<&[Option<f64>]>,
    low: Option<&[Option<f64>]>,
    vol: &[Option<f64>],
) -> Vec<MinutePoint> {
    indices
        .iter()
        .filter_map(|idx| {
            let time = trade_times[*idx].clone()?;
            Some(MinutePoint {
                time,
                close: clean_intraday_value(close[*idx]).filter(|value| *value > 0.0),
                high: high
                    .and_then(|values| clean_intraday_value(values[*idx]))
                    .filter(|value| *value > 0.0),
                low: low
                    .and_then(|values| clean_intraday_value(values[*idx]))
                    .filter(|value| *value > 0.0),
                vol: clean_intraday_value(vol[*idx]).filter(|value| *value >= 0.0),
            })
        })
        .collect()
}

fn domain_stats_for(points: &[MinutePoint], family: XyzqDomainRawFamily) -> DomainStats {
    match family {
        XyzqDomainRawFamily::Time => time_domain_stats(points),
        XyzqDomainRawFamily::Price => price_domain_stats(points),
        XyzqDomainRawFamily::Volume => volume_domain_stats(points),
    }
}

fn push_family_values(
    values: &mut BTreeMap<&'static str, Vec<FactorValue>>,
    requested: &BTreeSet<&str>,
    key: &FactorRowKey,
    stats: &DomainStats,
    family: XyzqDomainRawFamily,
) {
    match family {
        XyzqDomainRawFamily::Time => {
            push_array(values, requested, key, &TIME_RTN_RAW_IDS, &stats.time_rtn);
            push_array(values, requested, key, &TIME_VOL_RAW_IDS, &stats.time_vol);
        }
        XyzqDomainRawFamily::Price => {
            push_array(values, requested, key, &PRICE_STD_RAW_IDS, &stats.price_std);
            push_array(values, requested, key, &PRICE_VOL_RAW_IDS, &stats.price_vol);
            push_array(values, requested, key, &PRICE_RTN_RAW_IDS, &stats.price_rtn);
        }
        XyzqDomainRawFamily::Volume => {
            push_array(
                values,
                requested,
                key,
                &VOLUME_STD_RAW_IDS,
                &stats.volume_std,
            );
            push_array(
                values,
                requested,
                key,
                &VOLUME_RTN_RAW_IDS,
                &stats.volume_rtn,
            );
        }
    }
}

fn push_array<const N: usize>(
    values: &mut BTreeMap<&'static str, Vec<FactorValue>>,
    requested: &BTreeSet<&str>,
    key: &FactorRowKey,
    raw_ids: &[&'static str; N],
    raw_values: &[Option<f64>; N],
) {
    for (raw_id, value) in raw_ids.iter().zip(raw_values.iter()) {
        if requested.contains(raw_id) {
            values.entry(*raw_id).or_default().push(FactorValue {
                key: key.clone(),
                value: *value,
            });
        }
    }
}

fn time_domain_stats(points: &[MinutePoint]) -> DomainStats {
    let total_vol = selected_volume_total(points);
    let mut stats = DomainStats::default();
    let buckets = [
        ("09:31:00", "10:00:00", "09:30:00"),
        ("10:01:00", "10:30:00", "10:00:00"),
        ("10:31:00", "11:00:00", "10:30:00"),
        ("11:01:00", "11:30:00", "11:00:00"),
        ("13:01:00", "13:30:00", "11:30:00"),
        ("13:31:00", "14:00:00", "13:30:00"),
        ("14:01:00", "14:30:00", "14:00:00"),
        ("14:31:00", "15:00:00", "14:30:00"),
    ];
    for (idx, (start, end, anchor)) in buckets.iter().enumerate() {
        let end_close = close_at(points, end);
        let anchor_close = close_at(points, anchor);
        stats.time_rtn[idx] = match (end_close, anchor_close) {
            (Some(end_close), Some(anchor_close)) if anchor_close > f64::EPSILON => {
                finite_value(end_close / anchor_close - 1.0)
            }
            _ => None,
        };
        let bucket_vol = volume_sum(points, start, end);
        stats.time_vol[idx] = match (bucket_vol, total_vol) {
            (Some(bucket_vol), Some(total_vol)) if total_vol > f64::EPSILON => {
                finite_value(bucket_vol / total_vol)
            }
            _ => None,
        };
    }
    stats
}

fn price_domain_stats(points: &[MinutePoint]) -> DomainStats {
    let Some(blocks) = block_stats(points) else {
        return DomainStats::default();
    };
    let split_keys = blocks.map(|block| block.price_split);
    let Some(groups) = qcut_groups_16(&split_keys) else {
        return DomainStats::default();
    };
    let mut stats = DomainStats::default();
    stats.price_std = mean_by_group(&blocks.map(|block| block.std), &groups);
    stats.price_vol = mean_by_group(&blocks.map(|block| block.vol_share), &groups);
    stats.price_rtn = mean_by_group(&blocks.map(|block| block.rtn), &groups);
    stats
}

fn volume_domain_stats(points: &[MinutePoint]) -> DomainStats {
    let Some(blocks) = block_stats(points) else {
        return DomainStats::default();
    };
    let split_keys = blocks.map(|block| block.vol_share);
    let Some(groups) = qcut_groups_16(&split_keys) else {
        return DomainStats::default();
    };
    let mut stats = DomainStats::default();
    stats.volume_std = mean_by_group(&blocks.map(|block| block.std), &groups);
    stats.volume_rtn = mean_by_group(&blocks.map(|block| block.rtn), &groups);
    stats
}

fn block_stats(points: &[MinutePoint]) -> Option<[BlockStats; BLOCK_COUNT]> {
    let total_vol = selected_volume_total(points)?;
    if total_vol <= f64::EPSILON {
        return None;
    }
    let selected_high = selected_values(points, |point| point.high);
    let selected_low = selected_values(points, |point| point.low);
    let high_ranks = intraday_pct_ranks(&selected_high);
    let low_ranks = intraday_pct_ranks(&selected_low);
    let selected_points = points
        .iter()
        .filter(|point| intraday_time_in_range(&point.time, "09:31:00", "15:00:00"))
        .collect::<Vec<_>>();

    let block_defs = block_defs();
    let mut output = [BlockStats {
        rtn: None,
        std: None,
        vol_share: None,
        price_split: None,
    }; BLOCK_COUNT];
    for (block_idx, block) in block_defs.iter().enumerate() {
        let anchor_close = close_at(points, block.anchor);
        let end_close = close_at(points, block.end);
        output[block_idx].rtn = match (end_close, anchor_close) {
            (Some(end_close), Some(anchor_close)) if anchor_close > f64::EPSILON => {
                finite_value(end_close / anchor_close - 1.0)
            }
            _ => None,
        };
        let mut prev_close = anchor_close;
        let mut returns = Vec::new();
        let mut volume = 0.0;
        let mut volume_count = 0usize;
        let mut rank_sum = 0.0;
        let mut rank_count = 0usize;
        for (selected_idx, point) in selected_points.iter().enumerate() {
            if !intraday_time_in_range(&point.time, block.start, block.end) {
                continue;
            }
            if let (Some(current), Some(previous)) = (point.close, prev_close) {
                if previous > f64::EPSILON {
                    if let Some(ret) = finite_value(current / previous - 1.0) {
                        returns.push(ret);
                    }
                }
            }
            prev_close = point.close;
            if let Some(vol) = point.vol {
                volume += vol;
                volume_count += 1;
            }
            if let (Some(high), Some(low)) = (high_ranks[selected_idx], low_ranks[selected_idx]) {
                rank_sum += high + low;
                rank_count += 1;
            }
        }
        output[block_idx].std = std_dev(&returns);
        output[block_idx].vol_share = (volume_count > 0)
            .then(|| finite_value(volume / total_vol))
            .flatten();
        output[block_idx].price_split = (rank_count > 0).then_some(rank_sum);
    }
    Some(output)
}

fn block_defs() -> [BlockDef; BLOCK_COUNT] {
    [
        BlockDef {
            start: "09:31:00",
            end: "09:45:00",
            anchor: "09:30:00",
        },
        BlockDef {
            start: "09:46:00",
            end: "10:00:00",
            anchor: "09:45:00",
        },
        BlockDef {
            start: "10:01:00",
            end: "10:15:00",
            anchor: "10:00:00",
        },
        BlockDef {
            start: "10:16:00",
            end: "10:30:00",
            anchor: "10:15:00",
        },
        BlockDef {
            start: "10:31:00",
            end: "10:45:00",
            anchor: "10:30:00",
        },
        BlockDef {
            start: "10:46:00",
            end: "11:00:00",
            anchor: "10:45:00",
        },
        BlockDef {
            start: "11:01:00",
            end: "11:15:00",
            anchor: "11:00:00",
        },
        BlockDef {
            start: "11:16:00",
            end: "11:30:00",
            anchor: "11:15:00",
        },
        BlockDef {
            start: "13:01:00",
            end: "13:15:00",
            anchor: "11:30:00",
        },
        BlockDef {
            start: "13:16:00",
            end: "13:30:00",
            anchor: "13:15:00",
        },
        BlockDef {
            start: "13:31:00",
            end: "13:45:00",
            anchor: "13:30:00",
        },
        BlockDef {
            start: "13:46:00",
            end: "14:00:00",
            anchor: "13:45:00",
        },
        BlockDef {
            start: "14:01:00",
            end: "14:15:00",
            anchor: "14:00:00",
        },
        BlockDef {
            start: "14:16:00",
            end: "14:30:00",
            anchor: "14:15:00",
        },
        BlockDef {
            start: "14:31:00",
            end: "14:45:00",
            anchor: "14:30:00",
        },
        BlockDef {
            start: "14:46:00",
            end: "15:00:00",
            anchor: "14:45:00",
        },
    ]
}

fn selected_volume_total(points: &[MinutePoint]) -> Option<f64> {
    let mut sum = 0.0;
    let mut count = 0usize;
    for point in points
        .iter()
        .filter(|point| intraday_time_in_range(&point.time, "09:31:00", "15:00:00"))
    {
        let vol = point.vol?;
        sum += vol;
        count += 1;
    }
    (count > 0).then_some(sum)
}

fn volume_sum(points: &[MinutePoint], start: &str, end: &str) -> Option<f64> {
    let mut sum = 0.0;
    let mut count = 0usize;
    for point in points
        .iter()
        .filter(|point| intraday_time_in_range(&point.time, start, end))
    {
        let vol = point.vol?;
        sum += vol;
        count += 1;
    }
    (count > 0).then_some(sum)
}

fn close_at(points: &[MinutePoint], target: &str) -> Option<f64> {
    points
        .iter()
        .find(|point| intraday_time_in_range(&point.time, target, target))
        .and_then(|point| point.close)
}

fn selected_values<F>(points: &[MinutePoint], mut f: F) -> Vec<Option<f64>>
where
    F: FnMut(&MinutePoint) -> Option<f64>,
{
    points
        .iter()
        .filter(|point| intraday_time_in_range(&point.time, "09:31:00", "15:00:00"))
        .map(|point| f(point))
        .collect()
}

fn intraday_pct_ranks(values: &[Option<f64>]) -> Vec<Option<f64>> {
    let mut pairs = values
        .iter()
        .enumerate()
        .filter_map(|(idx, value)| {
            (*value)
                .filter(|value| value.is_finite())
                .map(|value| (idx, value))
        })
        .collect::<Vec<_>>();
    if pairs.len() < 2 {
        return vec![None; values.len()];
    }
    pairs.sort_by(|left, right| {
        left.1
            .total_cmp(&right.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    let denominator = pairs.len() as f64 - 1.0;
    let mut output = vec![None; values.len()];
    for (rank_idx, (idx, _)) in pairs.into_iter().enumerate() {
        output[idx] = Some(rank_idx as f64 / denominator);
    }
    output
}

fn qcut_groups_16(keys: &[Option<f64>; BLOCK_COUNT]) -> Option<[usize; BLOCK_COUNT]> {
    let mut pairs = keys
        .iter()
        .enumerate()
        .map(|(idx, value)| {
            value
                .filter(|value| value.is_finite())
                .map(|value| (idx, value))
        })
        .collect::<Option<Vec<_>>>()?;
    pairs.sort_by(|left, right| {
        left.1
            .total_cmp(&right.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    let mut groups = [0usize; BLOCK_COUNT];
    for (rank, (idx, _)) in pairs.into_iter().enumerate() {
        groups[idx] = (rank * Q_DOMAIN_COUNT / BLOCK_COUNT).min(Q_DOMAIN_COUNT - 1);
    }
    Some(groups)
}

fn mean_by_group(
    values: &[Option<f64>; BLOCK_COUNT],
    groups: &[usize; BLOCK_COUNT],
) -> [Option<f64>; Q_DOMAIN_COUNT] {
    let mut sums = [0.0; Q_DOMAIN_COUNT];
    let mut counts = [0usize; Q_DOMAIN_COUNT];
    for (value, group) in values.iter().zip(groups.iter()) {
        if let Some(value) = value.filter(|value| value.is_finite()) {
            sums[*group] += value;
            counts[*group] += 1;
        }
    }
    let mut output = [None; Q_DOMAIN_COUNT];
    for idx in 0..Q_DOMAIN_COUNT {
        if counts[idx] > 0 {
            output[idx] = finite_value(sums[idx] / counts[idx] as f64);
        }
    }
    output
}

fn intra_ds_value(values: &[Option<f64>]) -> Option<f64> {
    let valid = values
        .iter()
        .filter_map(|value| value.filter(|value| value.is_finite()))
        .collect::<Vec<_>>();
    if valid.len() < 2 {
        return None;
    }
    let avg = valid.iter().sum::<f64>() / valid.len() as f64;
    valid
        .into_iter()
        .filter_map(|value| finite_value((value - avg).abs() / (value.abs() + avg.abs() + EPSILON)))
        .reduce(f64::max)
}

fn intra_ds2_value(feature_values: &[Option<f64>], rtn_values: &[Option<f64>]) -> Option<f64> {
    let valid_features = feature_values
        .iter()
        .filter_map(|value| value.filter(|value| value.is_finite()))
        .collect::<Vec<_>>();
    if valid_features.len() < 2 || feature_values.len() != rtn_values.len() {
        return None;
    }
    let avg = valid_features.iter().sum::<f64>() / valid_features.len() as f64;
    let mut best_idx = None;
    let mut best_ds = f64::NEG_INFINITY;
    for (idx, value) in feature_values.iter().enumerate() {
        let Some(value) = value.filter(|value| value.is_finite()) else {
            continue;
        };
        let Some(ds) = finite_value((value - avg).abs() / (value.abs() + avg.abs() + EPSILON))
        else {
            continue;
        };
        if ds > best_ds {
            best_ds = ds;
            best_idx = Some(idx);
        }
    }
    let idx = best_idx?;
    let rtn = rtn_values[idx].filter(|value| value.is_finite())?;
    finite_value(best_ds * rtn)
}

fn compute_similarity_stat(
    panel: &DailyPanel,
    columns: &[PanelColumn],
    statistic: XyzqDomainCorrStatistic,
) -> Result<PanelColumn> {
    let standardized = columns
        .iter()
        .map(|column| column.cs(|values| cs_zscore(values)))
        .collect::<Result<Vec<_>>>()?;
    let date_count = panel.dates().len();
    let instrument_count = panel.instruments().len();
    let mut output = vec![None; panel.shape_len()];
    for date_idx in 0..date_count {
        if date_idx + 1 < SIM_WINDOW {
            continue;
        }
        let windows = normalized_window_columns(&standardized, date_idx, instrument_count);
        let mut sums = vec![0.0; instrument_count];
        let mut sumsq = vec![0.0; instrument_count];
        let mut counts = vec![0usize; instrument_count];
        for right in 0..instrument_count {
            for left in 0..right {
                let pair = pair_similarity_from_windows(&windows, left, right);
                if let Some(value) = pair {
                    sums[left] += value;
                    sums[right] += value;
                    sumsq[left] += value * value;
                    sumsq[right] += value * value;
                    counts[left] += 1;
                    counts[right] += 1;
                }
            }
        }
        for instrument_idx in 0..instrument_count {
            let offset = date_idx * instrument_count + instrument_idx;
            output[offset] = match statistic {
                XyzqDomainCorrStatistic::Mean if counts[instrument_idx] > 0 => {
                    finite_value(sums[instrument_idx] / counts[instrument_idx] as f64)
                }
                XyzqDomainCorrStatistic::Std if counts[instrument_idx] >= 2 => {
                    let mean = sums[instrument_idx] / counts[instrument_idx] as f64;
                    let variance =
                        sumsq[instrument_idx] / counts[instrument_idx] as f64 - mean * mean;
                    finite_value(variance.max(0.0).sqrt())
                }
                _ => None,
            };
        }
    }
    panel.column_from_values(output)
}

fn compute_peer_ds(
    panel: &DailyPanel,
    data: &DataPool,
    domain: XyzqDomainKind,
) -> Result<PanelColumn> {
    let feature_columns = domain
        .peer_features()
        .into_iter()
        .map(|feature| {
            raw_columns(panel, feature_raw_ids(feature)).and_then(|columns| {
                columns
                    .into_iter()
                    .map(|column| column.cs(|values| cs_zscore(values)))
                    .collect::<Result<Vec<_>>>()
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let close = panel.column_from_table(data.daily(DatasetId::StockDailyPv)?, "close")?;
    let five_day_return = close.ts(|values| five_day_returns(values))?;
    let date_count = panel.dates().len();
    let instrument_count = panel.instruments().len();
    let mut output = vec![None; panel.shape_len()];
    for date_idx in 0..date_count {
        if date_idx + 1 < SIM_WINDOW {
            continue;
        }
        let feature_windows = feature_columns
            .iter()
            .map(|columns| normalized_window_columns(columns, date_idx, instrument_count))
            .collect::<Vec<_>>();
        let mut peers = vec![Vec::<(f64, usize)>::new(); instrument_count];
        for right in 0..instrument_count {
            for left in 0..right {
                let pair = pair_domain_similarity_from_windows(&feature_windows, left, right);
                if let Some(value) = pair {
                    peers[left].push((value, right));
                    peers[right].push((value, left));
                }
            }
        }
        for instrument_idx in 0..instrument_count {
            let self_offset = date_idx * instrument_count + instrument_idx;
            let Some(self_ret) =
                five_day_return.values()[self_offset].filter(|value| value.is_finite())
            else {
                continue;
            };
            let mut peer_list = std::mem::take(&mut peers[instrument_idx]);
            if peer_list.is_empty() {
                continue;
            }
            let k = ((peer_list.len() as f64) * 0.1).ceil().max(1.0) as usize;
            if peer_list.len() > k {
                peer_list.select_nth_unstable_by(k, |left, right| {
                    right
                        .0
                        .total_cmp(&left.0)
                        .then_with(|| left.1.cmp(&right.1))
                });
            }
            let mut peer_sum = 0.0;
            let mut peer_count = 0usize;
            for (_, peer_idx) in peer_list.into_iter().take(k) {
                let offset = date_idx * instrument_count + peer_idx;
                if let Some(value) =
                    five_day_return.values()[offset].filter(|value| value.is_finite())
                {
                    peer_sum += value;
                    peer_count += 1;
                }
            }
            if peer_count == 0 {
                continue;
            }
            let peer_mean = peer_sum / peer_count as f64;
            output[self_offset] = finite_value(
                (self_ret - peer_mean).abs() / (self_ret.abs() + peer_mean.abs() + EPSILON),
            );
        }
    }
    panel.column_from_values(output)
}

fn normalized_window_columns(
    columns: &[PanelColumn],
    date_idx: usize,
    instrument_count: usize,
) -> Vec<NormalizedWindowColumn> {
    columns
        .iter()
        .map(|column| normalized_window_column(column, date_idx, instrument_count))
        .collect()
}

fn normalized_window_column(
    column: &PanelColumn,
    date_idx: usize,
    instrument_count: usize,
) -> NormalizedWindowColumn {
    let start = date_idx + 1 - SIM_WINDOW;
    let mut valid = vec![false; instrument_count];
    let mut values = vec![[0.0; SIM_WINDOW]; instrument_count];
    for instrument_idx in 0..instrument_count {
        let mut window = [0.0; SIM_WINDOW];
        let mut ok = true;
        for (pos, day) in (start..=date_idx).enumerate() {
            let Some(value) = column.values()[day * instrument_count + instrument_idx]
                .filter(|value| value.is_finite())
            else {
                ok = false;
                break;
            };
            window[pos] = value;
        }
        if !ok {
            continue;
        }
        let mean = window.iter().sum::<f64>() / SIM_WINDOW as f64;
        let mut norm_sq = 0.0;
        for value in &mut window {
            *value -= mean;
            norm_sq += *value * *value;
        }
        if norm_sq <= f64::EPSILON {
            continue;
        }
        let norm = norm_sq.sqrt();
        for value in &mut window {
            *value /= norm;
        }
        valid[instrument_idx] = true;
        values[instrument_idx] = window;
    }
    NormalizedWindowColumn { valid, values }
}

fn pair_domain_similarity_from_windows(
    feature_columns: &[Vec<NormalizedWindowColumn>],
    left: usize,
    right: usize,
) -> Option<f64> {
    let mut sum = 0.0;
    let mut count = 0usize;
    for columns in feature_columns {
        if let Some(value) = pair_similarity_from_windows(columns, left, right) {
            sum += value;
            count += 1;
        }
    }
    (count > 0)
        .then(|| finite_value(sum / count as f64))
        .flatten()
}

fn pair_similarity_from_windows(
    columns: &[NormalizedWindowColumn],
    left: usize,
    right: usize,
) -> Option<f64> {
    let mut sum = 0.0;
    let mut count = 0usize;
    for column in columns {
        if let Some(value) = pair_domain_corr_from_window(column, left, right) {
            sum += value;
            count += 1;
        }
    }
    (count > 0)
        .then(|| finite_value(sum / count as f64))
        .flatten()
}

fn pair_domain_corr_from_window(
    column: &NormalizedWindowColumn,
    left: usize,
    right: usize,
) -> Option<f64> {
    if !column.valid[left] || !column.valid[right] {
        return None;
    }
    let left_values = &column.values[left];
    let right_values = &column.values[right];
    let mut dot = 0.0;
    for idx in 0..SIM_WINDOW {
        dot += left_values[idx] * right_values[idx];
    }
    finite_value(dot)
}

fn five_day_returns(values: &[Option<f64>]) -> Vec<Option<f64>> {
    let mut output = vec![None; values.len()];
    for idx in 5..values.len() {
        output[idx] = match (values[idx], values[idx - 5]) {
            (Some(current), Some(previous)) if previous > f64::EPSILON => {
                finite_value(current / previous - 1.0)
            }
            _ => None,
        };
    }
    output
}

#[cfg(test)]
fn pearson_corr(x: &[f64], y: &[f64]) -> Option<f64> {
    if x.len() != y.len() || x.len() < 2 {
        return None;
    }
    let mean_x = x.iter().sum::<f64>() / x.len() as f64;
    let mean_y = y.iter().sum::<f64>() / y.len() as f64;
    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;
    for (x, y) in x.iter().zip(y.iter()) {
        let dx = x - mean_x;
        let dy = y - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }
    if var_x <= f64::EPSILON || var_y <= f64::EPSILON {
        return None;
    }
    finite_value(cov / (var_x.sqrt() * var_y.sqrt()))
}

fn std_dev(values: &[f64]) -> Option<f64> {
    if values.len() < 2 {
        return None;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|value| {
            let diff = value - mean;
            diff * diff
        })
        .sum::<f64>()
        / values.len() as f64;
    finite_value(variance.sqrt())
}

fn finite_value(value: f64) -> Option<f64> {
    value.is_finite().then_some(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close_point(time: &str, close: f64, vol: f64) -> MinutePoint {
        MinutePoint {
            time: time.to_string(),
            close: Some(close),
            high: Some(close),
            low: Some(close),
            vol: Some(vol),
        }
    }

    #[test]
    fn xyzq_domain_time_uses_0930_and_lunch_anchors() {
        let points = vec![
            close_point("09:30:00", 100.0, 0.0),
            close_point("10:00:00", 110.0, 10.0),
            close_point("11:30:00", 120.0, 10.0),
            close_point("13:30:00", 132.0, 10.0),
            close_point("15:00:00", 150.0, 10.0),
        ];
        let stats = time_domain_stats(&points);
        assert!((stats.time_rtn[0].unwrap() - 0.1).abs() < 1e-12);
        assert!((stats.time_rtn[4].unwrap() - 0.1).abs() < 1e-12);
    }

    #[test]
    fn xyzq_domain_qcut_groups_use_front_loaded_group_sizes() {
        let keys = [
            Some(0.0),
            Some(1.0),
            Some(2.0),
            Some(3.0),
            Some(4.0),
            Some(5.0),
            Some(6.0),
            Some(7.0),
            Some(8.0),
            Some(9.0),
            Some(10.0),
            Some(11.0),
            Some(12.0),
            Some(13.0),
            Some(14.0),
            Some(15.0),
        ];
        let groups = qcut_groups_16(&keys).expect("groups");
        assert_eq!(groups, [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]);
    }

    #[test]
    fn xyzq_domain_ds_and_ds2_use_max_significance_domain_return() {
        let values = [Some(1.0), Some(2.0), Some(5.0)];
        let returns = [Some(-0.01), Some(0.02), Some(0.03)];
        let ds = intra_ds_value(&values).expect("ds");
        let ds2 = intra_ds2_value(&values, &returns).expect("ds2");
        assert!(ds > 0.0);
        assert!(ds2 < 0.0);
    }

    #[test]
    fn xyzq_domain_pair_corr_requires_complete_ten_days() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let y = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0];
        assert_eq!(pearson_corr(&x, &y), Some(1.0));
    }
}

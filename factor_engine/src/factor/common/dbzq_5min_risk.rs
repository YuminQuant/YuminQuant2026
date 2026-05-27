use std::collections::BTreeMap;

use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorRowKey, FactorSeries, FactorSpec,
    FactorValue, Frequency, IntradayDailyRawRequest, IntradayDailyRawSeries, IntradayDailyRawSpec,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::stock_daily_ops::neutralize_size_sector;
use crate::factor::common::stock_daily_raw_ids::{
    CVAR90_5MIN_RAW_ID, CVAR90_RT_5MIN_RAW_ID, CVAR95_5MIN_RAW_ID, CVAR95_RT_5MIN_RAW_ID,
    ID_CVAR90_5MIN_RAW_ID, ID_CVAR90_RT_5MIN_RAW_ID, ID_CVAR95_5MIN_RAW_ID,
    ID_CVAR95_RT_5MIN_RAW_ID, ID_RV_5MIN_RAW_ID, ID_VAR90_5MIN_RAW_ID, ID_VAR90_RT_5MIN_RAW_ID,
    ID_VAR95_5MIN_RAW_ID, ID_VAR95_RT_5MIN_RAW_ID, RV_5MIN_RAW_ID, VAR90_5MIN_RAW_ID,
    VAR90_RT_5MIN_RAW_ID, VAR95_5MIN_RAW_ID, VAR95_RT_5MIN_RAW_ID,
};
use crate::factor::common::{
    clean_intraday_value, quantile_linear, stock_minute_raw_spec, RequestedRawIds,
};
use crate::operators::{cs_zscore, ts_mean};

pub const RAW_VERSION: &str = "0.1.0";
pub const VERSION: &str = "0.1.0";

pub const WEEK_WINDOW: usize = 5;
pub const UNCERTAINTY_WINDOW: usize = 21;
pub const MIN_PERIODS: usize = 1;

const RAW_WINDOW_DAYS: usize = 1;
const FIVE_MINUTE_RETURN_COUNT: usize = 48;
const EPS: f64 = f64::EPSILON;
const DEPRECATED_FACTOR_IDS: &[&str] = &[
    "var90_week",
    "cvar90_week",
    "var90_rt_week",
    "cvar90_rt_week",
    "vovar90",
    "vocvar90",
    "vovar90_rt",
    "vocvar90_rt",
    "id_var90_week",
    "id_cvar90_week",
    "id_var90_rt_week",
    "id_cvar90_rt_week",
    "id_vovar90",
    "id_vocvar90",
    "id_vovar90_rt",
    "id_vocvar90_rt",
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DbzqPostProcess {
    WeekMean,
    Uncertainty,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DbzqRawFamily {
    Ordinary,
    Idiosyncratic,
}

#[derive(Clone, Copy, Debug)]
pub struct DbzqFactorDef {
    pub id: &'static str,
    pub alias: &'static str,
    pub name: &'static str,
    pub raw_id: &'static str,
    pub postprocess: DbzqPostProcess,
}

#[derive(Clone, Debug)]
struct InstrumentReturns {
    ts_code: String,
    returns: Vec<Option<f64>>,
}

#[derive(Clone, Copy, Debug, Default)]
struct DailyRiskStats {
    rv: Option<f64>,
    var90: Option<f64>,
    var95: Option<f64>,
    cvar90: Option<f64>,
    cvar95: Option<f64>,
    var90_rt: Option<f64>,
    var95_rt: Option<f64>,
    cvar90_rt: Option<f64>,
    cvar95_rt: Option<f64>,
}

#[derive(Clone, Copy, Debug, Default)]
struct RiskStatsRequest {
    rv: bool,
    var90: bool,
    var95: bool,
    cvar90: bool,
    cvar95: bool,
    var90_rt: bool,
    var95_rt: bool,
    cvar90_rt: bool,
    cvar95_rt: bool,
}

impl RiskStatsRequest {
    fn from_requested(requested: &RequestedRawIds<'_>, family: DbzqRawFamily) -> Self {
        match family {
            DbzqRawFamily::Ordinary => Self {
                rv: requested.contains(RV_5MIN_RAW_ID),
                var90: requested.contains(VAR90_5MIN_RAW_ID),
                var95: requested.contains(VAR95_5MIN_RAW_ID),
                cvar90: requested.contains(CVAR90_5MIN_RAW_ID),
                cvar95: requested.contains(CVAR95_5MIN_RAW_ID),
                var90_rt: requested.contains(VAR90_RT_5MIN_RAW_ID),
                var95_rt: requested.contains(VAR95_RT_5MIN_RAW_ID),
                cvar90_rt: requested.contains(CVAR90_RT_5MIN_RAW_ID),
                cvar95_rt: requested.contains(CVAR95_RT_5MIN_RAW_ID),
            },
            DbzqRawFamily::Idiosyncratic => Self {
                rv: requested.contains(ID_RV_5MIN_RAW_ID),
                var90: requested.contains(ID_VAR90_5MIN_RAW_ID),
                var95: requested.contains(ID_VAR95_5MIN_RAW_ID),
                cvar90: requested.contains(ID_CVAR90_5MIN_RAW_ID),
                cvar95: requested.contains(ID_CVAR95_5MIN_RAW_ID),
                var90_rt: requested.contains(ID_VAR90_RT_5MIN_RAW_ID),
                var95_rt: requested.contains(ID_VAR95_RT_5MIN_RAW_ID),
                cvar90_rt: requested.contains(ID_CVAR90_RT_5MIN_RAW_ID),
                cvar95_rt: requested.contains(ID_CVAR95_RT_5MIN_RAW_ID),
            },
        }
    }
}

pub fn all_raw_ids() -> [&'static str; 18] {
    [
        RV_5MIN_RAW_ID,
        VAR90_5MIN_RAW_ID,
        VAR95_5MIN_RAW_ID,
        CVAR90_5MIN_RAW_ID,
        CVAR95_5MIN_RAW_ID,
        VAR90_RT_5MIN_RAW_ID,
        VAR95_RT_5MIN_RAW_ID,
        CVAR90_RT_5MIN_RAW_ID,
        CVAR95_RT_5MIN_RAW_ID,
        ID_RV_5MIN_RAW_ID,
        ID_VAR90_5MIN_RAW_ID,
        ID_VAR95_5MIN_RAW_ID,
        ID_CVAR90_5MIN_RAW_ID,
        ID_CVAR95_5MIN_RAW_ID,
        ID_VAR90_RT_5MIN_RAW_ID,
        ID_VAR95_RT_5MIN_RAW_ID,
        ID_CVAR90_RT_5MIN_RAW_ID,
        ID_CVAR95_RT_5MIN_RAW_ID,
    ]
}

pub fn ordinary_raw_ids() -> [&'static str; 9] {
    [
        RV_5MIN_RAW_ID,
        VAR90_5MIN_RAW_ID,
        VAR95_5MIN_RAW_ID,
        CVAR90_5MIN_RAW_ID,
        CVAR95_5MIN_RAW_ID,
        VAR90_RT_5MIN_RAW_ID,
        VAR95_RT_5MIN_RAW_ID,
        CVAR90_RT_5MIN_RAW_ID,
        CVAR95_RT_5MIN_RAW_ID,
    ]
}

pub fn idiosyncratic_raw_ids() -> [&'static str; 9] {
    [
        ID_RV_5MIN_RAW_ID,
        ID_VAR90_5MIN_RAW_ID,
        ID_VAR95_5MIN_RAW_ID,
        ID_CVAR90_5MIN_RAW_ID,
        ID_CVAR95_5MIN_RAW_ID,
        ID_VAR90_RT_5MIN_RAW_ID,
        ID_VAR95_RT_5MIN_RAW_ID,
        ID_CVAR90_RT_5MIN_RAW_ID,
        ID_CVAR95_RT_5MIN_RAW_ID,
    ]
}

fn raw_ids_for_family(family: DbzqRawFamily) -> Vec<&'static str> {
    match family {
        DbzqRawFamily::Ordinary => ordinary_raw_ids().to_vec(),
        DbzqRawFamily::Idiosyncratic => idiosyncratic_raw_ids().to_vec(),
    }
}

pub fn raw_spec(raw_id: &str) -> IntradayDailyRawSpec {
    stock_minute_raw_spec(raw_id, RAW_VERSION, &["close"], RAW_WINDOW_DAYS)
}

pub fn raw_specs() -> Vec<IntradayDailyRawSpec> {
    all_raw_ids()
        .iter()
        .map(|raw_id| raw_spec(raw_id))
        .collect()
}

pub fn ordinary_raw_specs() -> Vec<IntradayDailyRawSpec> {
    ordinary_raw_ids()
        .iter()
        .map(|raw_id| raw_spec(raw_id))
        .collect()
}

pub fn idiosyncratic_raw_specs() -> Vec<IntradayDailyRawSpec> {
    idiosyncratic_raw_ids()
        .iter()
        .map(|raw_id| raw_spec(raw_id))
        .collect()
}

pub fn factor_spec(def: DbzqFactorDef) -> FactorSpec {
    let lookback = match def.postprocess {
        DbzqPostProcess::WeekMean => WEEK_WINDOW - 1,
        DbzqPostProcess::Uncertainty => UNCERTAINTY_WINDOW - 1,
    };
    FactorSpec {
        id: def.id.to_string(),
        aliases: vec![def.alias.to_string()],
        name: def.name.to_string(),
        asset_class: AssetClass::Stock,
        frequency: Frequency::Daily,
        version: VERSION.to_string(),
        tags: tags(def.id),
        description: format!(
            "{} based on 5-minute intraday log returns and neutralized by Barra SIZE and SW sector.",
            def.name
        ),
        dependencies: dependencies(),
        intraday_raw_dependencies: vec![IntradayDailyRawRequest::new(def.raw_id, lookback)],
        lookback: Lookback {
            trading_days: lookback,
        },
    }
}

pub fn compute_factor(def: DbzqFactorDef, data: &DataPool) -> Result<FactorSeries> {
    let panel = data.intraday_daily_raw_panel(def.raw_id)?;
    let raw = panel.column(def.raw_id)?;
    let post = match def.postprocess {
        DbzqPostProcess::WeekMean => raw.ts(|values| ts_mean(values, WEEK_WINDOW, MIN_PERIODS))?,
        DbzqPostProcess::Uncertainty => {
            let mean = raw.ts(|values| ts_mean(values, UNCERTAINTY_WINDOW, MIN_PERIODS))?;
            let std = raw.ts(|values| sample_std(values, UNCERTAINTY_WINDOW, MIN_PERIODS))?;
            std.zip_binary(&mean, safe_div)?
        }
    };
    let standardized = post.cs(cs_zscore)?;
    let factor = neutralize_size_sector(&standardized, &panel, data)?;
    Ok(factor.to_factor_series(factor_spec(def)))
}

#[macro_export]
macro_rules! define_dbzq_5min_factor {
    ($struct_name:ident, $id:expr, $alias:expr, $name:expr, $raw_id:expr, $postprocess:ident) => {
        const DEF: $crate::factor::common::dbzq_5min_risk::DbzqFactorDef =
            $crate::factor::common::dbzq_5min_risk::DbzqFactorDef {
                id: $id,
                alias: $alias,
                name: $name,
                raw_id: $raw_id,
                postprocess: $crate::factor::common::dbzq_5min_risk::DbzqPostProcess::$postprocess,
            };

        pub struct $struct_name;

        pub fn create() -> Box<dyn $crate::factor::Factor> {
            Box::new($struct_name)
        }

        impl $crate::factor::Factor for $struct_name {
            fn spec(&self) -> $crate::core::FactorSpec {
                $crate::factor::common::dbzq_5min_risk::factor_spec(DEF)
            }

            fn compute(
                &self,
                _context: &$crate::core::FactorContext,
                data: &$crate::data::DataPool,
            ) -> $crate::error::Result<$crate::core::FactorSeries> {
                $crate::factor::common::dbzq_5min_risk::compute_factor(DEF, data)
            }
        }
    };
}

pub fn minute_compute_many(
    raw_ids: &[String],
    context: &FactorContext,
    data: &DataPool,
) -> Result<Vec<IntradayDailyRawSeries>> {
    minute_compute_many_for(raw_ids, context, data, DbzqRawFamily::Ordinary)
}

pub fn minute_compute_many_for(
    raw_ids: &[String],
    context: &FactorContext,
    data: &DataPool,
    family: DbzqRawFamily,
) -> Result<Vec<IntradayDailyRawSeries>> {
    let family_raw_ids = raw_ids_for_family(family);
    let requested = RequestedRawIds::new(raw_ids, &family_raw_ids);
    if requested.is_empty() {
        return Ok(Vec::new());
    }
    let request = RiskStatsRequest::from_requested(&requested, family);

    let mut values = family_raw_ids
        .iter()
        .copied()
        .filter(|raw_id| requested.contains(raw_id))
        .map(|raw_id| (raw_id, Vec::<FactorValue>::new()))
        .collect::<BTreeMap<_, _>>();

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

        let mut instrument_returns = Vec::with_capacity(grouped.len());
        for (ts_code, indices) in grouped {
            instrument_returns.push(InstrumentReturns {
                ts_code,
                returns: five_minute_log_returns(&indices, trade_times, &close),
            });
        }
        let market_returns = matches!(family, DbzqRawFamily::Idiosyncratic)
            .then(|| market_mean_returns(&instrument_returns));

        for instrument in instrument_returns {
            let key = FactorRowKey::Daily {
                trade_date: *trade_date,
                ts_code: instrument.ts_code,
            };

            match family {
                DbzqRawFamily::Ordinary => {
                    let ordinary = daily_risk_stats(&instrument.returns, request);
                    push_ordinary_stats(&mut values, &requested, &key, ordinary);
                }
                DbzqRawFamily::Idiosyncratic => {
                    let Some(market_returns) = market_returns.as_ref() else {
                        continue;
                    };
                    let residuals = capm_residuals(&instrument.returns, market_returns);
                    let idiosyncratic = daily_risk_stats(&residuals, request);
                    push_idiosyncratic_stats(&mut values, &requested, &key, idiosyncratic);
                }
            }
        }
    }

    let mut output = Vec::new();
    for raw_id in family_raw_ids {
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

fn is_deprecated_factor_id(id: &str) -> bool {
    DEPRECATED_FACTOR_IDS.contains(&id)
}

fn tags(id: &str) -> Vec<String> {
    let mut values = [
        "price_volume",
        "return",
        "risk",
        "tail_risk",
        "intraday",
        "minute_agg",
        "neutralize",
        "barra",
        "size",
        "sector",
        "daily",
        "DBZQ",
    ]
    .iter()
    .map(|value| value.to_string())
    .collect::<Vec<_>>();
    if is_deprecated_factor_id(id) {
        values.push("deprecated".to_string());
    }
    values
}

fn dependencies() -> Vec<DataRequest> {
    vec![
        DataRequest::new(DatasetId::StockBarraDaily, &["SIZE"]),
        DataRequest::new(DatasetId::StockSwClassification, &["l1_code"]),
    ]
}

fn push_requested(
    values: &mut BTreeMap<&'static str, Vec<FactorValue>>,
    requested: &RequestedRawIds<'_>,
    raw_id: &'static str,
    key: &FactorRowKey,
    value: Option<f64>,
) {
    if !requested.contains(raw_id) {
        return;
    }
    values.entry(raw_id).or_default().push(FactorValue {
        key: key.clone(),
        value,
    });
}

fn push_ordinary_stats(
    values: &mut BTreeMap<&'static str, Vec<FactorValue>>,
    requested: &RequestedRawIds<'_>,
    key: &FactorRowKey,
    stats: DailyRiskStats,
) {
    push_requested(values, requested, RV_5MIN_RAW_ID, key, stats.rv);
    push_requested(values, requested, VAR90_5MIN_RAW_ID, key, stats.var90);
    push_requested(values, requested, VAR95_5MIN_RAW_ID, key, stats.var95);
    push_requested(values, requested, CVAR90_5MIN_RAW_ID, key, stats.cvar90);
    push_requested(values, requested, CVAR95_5MIN_RAW_ID, key, stats.cvar95);
    push_requested(values, requested, VAR90_RT_5MIN_RAW_ID, key, stats.var90_rt);
    push_requested(values, requested, VAR95_RT_5MIN_RAW_ID, key, stats.var95_rt);
    push_requested(
        values,
        requested,
        CVAR90_RT_5MIN_RAW_ID,
        key,
        stats.cvar90_rt,
    );
    push_requested(
        values,
        requested,
        CVAR95_RT_5MIN_RAW_ID,
        key,
        stats.cvar95_rt,
    );
}

fn push_idiosyncratic_stats(
    values: &mut BTreeMap<&'static str, Vec<FactorValue>>,
    requested: &RequestedRawIds<'_>,
    key: &FactorRowKey,
    stats: DailyRiskStats,
) {
    push_requested(values, requested, ID_RV_5MIN_RAW_ID, key, stats.rv);
    push_requested(values, requested, ID_VAR90_5MIN_RAW_ID, key, stats.var90);
    push_requested(values, requested, ID_VAR95_5MIN_RAW_ID, key, stats.var95);
    push_requested(values, requested, ID_CVAR90_5MIN_RAW_ID, key, stats.cvar90);
    push_requested(values, requested, ID_CVAR95_5MIN_RAW_ID, key, stats.cvar95);
    push_requested(
        values,
        requested,
        ID_VAR90_RT_5MIN_RAW_ID,
        key,
        stats.var90_rt,
    );
    push_requested(
        values,
        requested,
        ID_VAR95_RT_5MIN_RAW_ID,
        key,
        stats.var95_rt,
    );
    push_requested(
        values,
        requested,
        ID_CVAR90_RT_5MIN_RAW_ID,
        key,
        stats.cvar90_rt,
    );
    push_requested(
        values,
        requested,
        ID_CVAR95_RT_5MIN_RAW_ID,
        key,
        stats.cvar95_rt,
    );
}

fn five_minute_log_returns(
    indices: &[usize],
    trade_times: &[Option<String>],
    close: &[Option<f64>],
) -> Vec<Option<f64>> {
    let mut close_by_anchor = BTreeMap::<i32, f64>::new();
    for idx in indices {
        let Some(trade_time) = trade_times[*idx].as_deref() else {
            continue;
        };
        let Some(seconds) = time_to_seconds(trade_time) else {
            continue;
        };
        if !anchor_seconds().contains(&seconds) {
            continue;
        }
        let Some(close) = clean_intraday_value(close[*idx]).filter(|value| *value > 0.0) else {
            continue;
        };
        close_by_anchor.insert(seconds, close);
    }

    let anchors = anchor_seconds();
    let mut returns = Vec::with_capacity(FIVE_MINUTE_RETURN_COUNT);
    for pair in anchors.windows(2) {
        let (Some(previous), Some(current)) =
            (close_by_anchor.get(&pair[0]), close_by_anchor.get(&pair[1]))
        else {
            returns.push(None);
            continue;
        };
        returns.push(Some(current.ln() - previous.ln()));
    }
    returns
}

fn anchor_seconds() -> Vec<i32> {
    let mut anchors = Vec::with_capacity(FIVE_MINUTE_RETURN_COUNT + 1);
    anchors.push(seconds(9, 30));
    let mut minute = 35;
    while minute <= 150 {
        let (hour, minute_in_hour) = if minute < 60 {
            (9, minute)
        } else {
            (10 + (minute - 60) / 60, (minute - 60) % 60)
        };
        anchors.push(seconds(hour, minute_in_hour));
        minute += 5;
    }
    let mut afternoon_minute = 5;
    while afternoon_minute <= 120 {
        let (hour, minute_in_hour) = if afternoon_minute < 60 {
            (13, afternoon_minute)
        } else {
            (
                14 + (afternoon_minute - 60) / 60,
                (afternoon_minute - 60) % 60,
            )
        };
        anchors.push(seconds(hour, minute_in_hour));
        afternoon_minute += 5;
    }
    anchors
}

fn seconds(hour: i32, minute: i32) -> i32 {
    hour * 3600 + minute * 60
}

fn time_to_seconds(value: &str) -> Option<i32> {
    let value = value.trim();
    if value.is_empty() {
        return None;
    }
    let value = value
        .rsplit_once(' ')
        .map(|(_, right)| right)
        .or_else(|| value.rsplit_once('T').map(|(_, right)| right))
        .unwrap_or(value)
        .trim();
    if value.len() < 5 {
        return None;
    }
    let hour = value.get(0..2)?.parse::<i32>().ok()?;
    let minute = value.get(3..5)?.parse::<i32>().ok()?;
    let second = if value.len() >= 8 {
        value.get(6..8)?.parse::<i32>().ok()?
    } else {
        0
    };
    Some(hour * 3600 + minute * 60 + second)
}

fn market_mean_returns(instruments: &[InstrumentReturns]) -> Vec<Option<f64>> {
    let mut output = Vec::with_capacity(FIVE_MINUTE_RETURN_COUNT);
    for idx in 0..FIVE_MINUTE_RETURN_COUNT {
        let mut sum = 0.0;
        let mut count = 0usize;
        for instrument in instruments {
            if let Some(value) = instrument.returns.get(idx).and_then(|value| *value) {
                if value.is_finite() {
                    sum += value;
                    count += 1;
                }
            }
        }
        output.push((count > 0).then_some(sum / count as f64));
    }
    output
}

fn capm_residuals(stock: &[Option<f64>], market: &[Option<f64>]) -> Vec<Option<f64>> {
    let pairs = stock
        .iter()
        .zip(market.iter())
        .filter_map(|(stock, market)| match (*stock, *market) {
            (Some(stock), Some(market)) if stock.is_finite() && market.is_finite() => {
                Some((stock, market))
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    if pairs.len() < 3 {
        return vec![None; stock.len()];
    }
    let mean_stock = pairs.iter().map(|(stock, _)| *stock).sum::<f64>() / pairs.len() as f64;
    let mean_market = pairs.iter().map(|(_, market)| *market).sum::<f64>() / pairs.len() as f64;
    let variance_market = pairs
        .iter()
        .map(|(_, market)| {
            let diff = *market - mean_market;
            diff * diff
        })
        .sum::<f64>();
    if variance_market <= EPS {
        return vec![None; stock.len()];
    }
    let covariance = pairs
        .iter()
        .map(|(stock, market)| (*stock - mean_stock) * (*market - mean_market))
        .sum::<f64>();
    let beta = covariance / variance_market;
    let alpha = mean_stock - beta * mean_market;

    stock
        .iter()
        .zip(market.iter())
        .map(|(stock, market)| match (*stock, *market) {
            (Some(stock), Some(market)) if stock.is_finite() && market.is_finite() => {
                Some(stock - alpha - beta * market)
            }
            _ => None,
        })
        .collect()
}

fn daily_risk_stats(values: &[Option<f64>], request: RiskStatsRequest) -> DailyRiskStats {
    let valid = values
        .iter()
        .filter_map(|value| value.filter(|value| value.is_finite()))
        .collect::<Vec<_>>();
    if valid.is_empty() {
        return DailyRiskStats::default();
    }
    let rv = request
        .rv
        .then(|| valid.iter().map(|value| value * value).sum::<f64>());
    let var90 = request.var90.then(|| left_var(&valid, 0.10)).flatten();
    let var95 = request.var95.then(|| left_var(&valid, 0.05)).flatten();
    let cvar90 = request.cvar90.then(|| left_cvar(&valid, 0.10)).flatten();
    let cvar95 = request.cvar95.then(|| left_cvar(&valid, 0.05)).flatten();
    let var90_rt = request.var90_rt.then(|| right_var(&valid, 0.10)).flatten();
    let var95_rt = request.var95_rt.then(|| right_var(&valid, 0.05)).flatten();
    let cvar90_rt = request
        .cvar90_rt
        .then(|| right_cvar(&valid, 0.10))
        .flatten();
    let cvar95_rt = request
        .cvar95_rt
        .then(|| right_cvar(&valid, 0.05))
        .flatten();
    DailyRiskStats {
        rv,
        var90,
        var95,
        cvar90,
        cvar95,
        var90_rt,
        var95_rt,
        cvar90_rt,
        cvar95_rt,
    }
}

fn left_var(values: &[f64], alpha: f64) -> Option<f64> {
    let q = quantile(values, alpha)?;
    Some(-q)
}

fn right_var(values: &[f64], alpha: f64) -> Option<f64> {
    quantile(values, 1.0 - alpha)
}

fn left_cvar(values: &[f64], alpha: f64) -> Option<f64> {
    let q = quantile(values, alpha)?;
    let tail = values
        .iter()
        .copied()
        .filter(|value| *value <= q)
        .collect::<Vec<_>>();
    mean(&tail).map(|value| -value)
}

fn right_cvar(values: &[f64], alpha: f64) -> Option<f64> {
    let q = quantile(values, 1.0 - alpha)?;
    let tail = values
        .iter()
        .copied()
        .filter(|value| *value >= q)
        .collect::<Vec<_>>();
    mean(&tail)
}

fn quantile(values: &[f64], q: f64) -> Option<f64> {
    let mut values = values.to_vec();
    quantile_linear(&mut values, q)
}

fn mean(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        None
    } else {
        Some(values.iter().sum::<f64>() / values.len() as f64)
    }
}

fn sample_std(values: &[Option<f64>], window: usize, min_periods: usize) -> Vec<Option<f64>> {
    let mut output = vec![None; values.len()];
    let min_periods = min_periods.max(1).min(window);
    for idx in 0..values.len() {
        let start = (idx + 1).saturating_sub(window);
        let valid = values[start..=idx]
            .iter()
            .filter_map(|value| value.filter(|value| value.is_finite()))
            .collect::<Vec<_>>();
        if valid.len() < min_periods || valid.len() < 2 {
            continue;
        }
        let mean = valid.iter().sum::<f64>() / valid.len() as f64;
        let variance = valid
            .iter()
            .map(|value| {
                let diff = value - mean;
                diff * diff
            })
            .sum::<f64>()
            / (valid.len() - 1) as f64;
        output[idx] = Some(variance.sqrt());
    }
    output
}

fn safe_div(numerator: Option<f64>, denominator: Option<f64>) -> Option<f64> {
    match (numerator, denominator) {
        (Some(numerator), Some(denominator))
            if numerator.is_finite()
                && denominator.is_finite()
                && denominator.abs() > f64::EPSILON =>
        {
            Some(numerator / denominator)
        }
        _ => None,
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
    fn dbzq_anchor_seconds_generate_forty_eight_returns_with_0930_anchor() {
        let anchors = anchor_seconds();

        assert_eq!(anchors.len(), 49);
        assert_eq!(anchors[0], seconds(9, 30));
        assert_eq!(anchors[1], seconds(9, 35));
        assert_eq!(anchors[24], seconds(11, 30));
        assert_eq!(anchors[25], seconds(13, 5));
        assert_eq!(anchors[48], seconds(15, 0));
    }

    #[test]
    fn dbzq_five_minute_returns_use_log_close_differences() {
        let times = vec![
            Some("09:30:00".to_string()),
            Some("09:35:00".to_string()),
            Some("09:40:00".to_string()),
        ];
        let close = vec![Some(10.0), Some(11.0), Some(12.1)];
        let indices = vec![0, 1, 2];

        let returns = five_minute_log_returns(&indices, &times, &close);

        assert_close(returns[0], Some((11.0_f64 / 10.0).ln()));
        assert_close(returns[1], Some((12.1_f64 / 11.0).ln()));
        assert_eq!(returns.len(), 48);
    }

    #[test]
    fn dbzq_market_return_is_equal_weighted_cross_section_mean() {
        let instruments = vec![
            InstrumentReturns {
                ts_code: "a".to_string(),
                returns: vec![Some(0.01), None],
            },
            InstrumentReturns {
                ts_code: "b".to_string(),
                returns: vec![Some(0.03), Some(0.02)],
            },
        ];

        let market = market_mean_returns(&instruments);

        assert_close(market[0], Some(0.02));
        assert_close(market[1], Some(0.02));
    }

    #[test]
    fn dbzq_capm_residuals_use_intercept_and_beta() {
        let stock = vec![Some(2.0), Some(4.0), Some(6.0), Some(8.0)];
        let market = vec![Some(1.0), Some(2.0), Some(3.0), Some(4.0)];

        let residuals = capm_residuals(&stock, &market);

        assert!(residuals
            .into_iter()
            .all(|value| value.unwrap().abs() < 1e-10));
    }

    #[test]
    fn dbzq_capm_residuals_reject_zero_market_variance() {
        let stock = vec![Some(1.0), Some(2.0), Some(3.0)];
        let market = vec![Some(1.0), Some(1.0), Some(1.0)];

        assert_eq!(capm_residuals(&stock, &market), vec![None, None, None]);
    }

    #[test]
    fn dbzq_tail_metrics_match_left_and_right_definitions() {
        let values = vec![-4.0, -2.0, 1.0, 3.0, 5.0];

        assert_close(left_var(&values, 0.25), Some(2.0));
        assert_close(right_var(&values, 0.25), Some(3.0));
        assert_close(left_cvar(&values, 0.25), Some(3.0));
        assert_close(right_cvar(&values, 0.25), Some(4.0));
    }

    #[test]
    fn dbzq_daily_risk_stats_only_computes_requested_95_tail() {
        let values = vec![Some(-4.0), Some(-2.0), Some(1.0), Some(3.0), Some(5.0)];
        let request = RiskStatsRequest {
            var95: true,
            ..RiskStatsRequest::default()
        };

        let stats = daily_risk_stats(&values, request);

        assert!(stats.var95.is_some());
        assert_eq!(stats.rv, None);
        assert_eq!(stats.var90, None);
        assert_eq!(stats.cvar90, None);
        assert_eq!(stats.cvar95, None);
        assert_eq!(stats.var90_rt, None);
        assert_eq!(stats.var95_rt, None);
    }

    #[test]
    fn dbzq_daily_risk_stats_rv_only_skips_tail_metrics() {
        let values = vec![Some(-4.0), Some(-2.0), Some(1.0), Some(3.0), Some(5.0)];
        let request = RiskStatsRequest {
            rv: true,
            ..RiskStatsRequest::default()
        };

        let stats = daily_risk_stats(&values, request);

        assert_close(stats.rv, Some(55.0));
        assert_eq!(stats.var90, None);
        assert_eq!(stats.var95, None);
        assert_eq!(stats.cvar90, None);
        assert_eq!(stats.cvar95, None);
        assert_eq!(stats.var90_rt, None);
        assert_eq!(stats.var95_rt, None);
        assert_eq!(stats.cvar90_rt, None);
        assert_eq!(stats.cvar95_rt, None);
    }

    #[test]
    fn dbzq_request_mask_maps_only_requested_family_raw_ids() {
        let raw_ids = vec![VAR95_5MIN_RAW_ID.to_string()];
        let requested = RequestedRawIds::new(&raw_ids, &ordinary_raw_ids());
        let mask = RiskStatsRequest::from_requested(&requested, DbzqRawFamily::Ordinary);

        assert!(mask.var95);
        assert!(!mask.var90);
        assert!(!mask.cvar90);
        assert!(!mask.rv);
    }

    #[test]
    fn dbzq_sample_std_uses_n_minus_one_and_requires_two_values() {
        let values = vec![Some(1.0), Some(3.0), None, Some(5.0)];

        let std = sample_std(&values, 21, 1);

        assert_eq!(std[0], None);
        assert_close(std[1], Some(2.0_f64.sqrt()));
        assert_close(std[2], Some(2.0_f64.sqrt()));
        assert_close(std[3], Some(2.0));
    }

    #[test]
    fn dbzq_uncertainty_rejects_zero_mean() {
        assert_eq!(safe_div(Some(1.0), Some(0.0)), None);
        assert_close(safe_div(Some(2.0), Some(4.0)), Some(0.5));
    }

    #[test]
    fn dbzq_factor_spec_marks_only_90_parameter_variants_deprecated() {
        let deprecated = factor_spec(DbzqFactorDef {
            id: "var90_week",
            alias: "VaR90_week",
            name: "VaR90 Week",
            raw_id: VAR90_5MIN_RAW_ID,
            postprocess: DbzqPostProcess::WeekMean,
        });
        let retained = factor_spec(DbzqFactorDef {
            id: "var95_week",
            alias: "VaR95_week",
            name: "VaR95 Week",
            raw_id: VAR95_5MIN_RAW_ID,
            postprocess: DbzqPostProcess::WeekMean,
        });

        assert!(deprecated.tags.iter().any(|tag| tag == "deprecated"));
        assert!(!retained.tags.iter().any(|tag| tag == "deprecated"));
    }
}

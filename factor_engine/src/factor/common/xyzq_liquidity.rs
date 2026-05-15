use std::any::Any;
use std::collections::{BTreeMap, BTreeSet};

use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorRowKey, FactorSeries, FactorSpec,
    FactorValue, Frequency, IntradayDailyRawAuxiliaryRequest, IntradayDailyRawRequest,
    IntradayDailyRawSeries, IntradayDailyRawSpec, Lookback,
};
use crate::data::{DataPool, Table};
use crate::error::{err, Result};
use crate::factor::common::stock_daily_ops::{is_bj_stock, mask_bj, neutralize_ret20_size_sector};
use crate::factor::common::stock_daily_raw_ids::{
    AMIHUD_1MIN_RAW_ID, APBETA1_RAW_ID, APBETA2_5M_N_RAW_ID, APBETA2_5M_SUM_XY_RAW_ID,
    APBETA2_5M_SUM_X_RAW_ID, APBETA2_5M_SUM_Y_RAW_ID, APBETA2_5M_SUM_Z2_RAW_ID,
    APBETA2_5M_SUM_Z_RAW_ID, APBETA2_RAW_ID, APBETA3_5M_N_RAW_ID, APBETA3_5M_SUM_XY_RAW_ID,
    APBETA3_5M_SUM_X_RAW_ID, APBETA3_5M_SUM_Y_RAW_ID, APBETA3_5M_SUM_Z2_RAW_ID,
    APBETA3_5M_SUM_Z_RAW_ID, APBETA3_RAW_ID, APBETA4_RAW_ID, CLOSEVOLCORR_GAMMA3_RAW_ID,
    CLOSE_APBETA4_RAW_ID, GAMMA1_RAW_ID, GAMMA2_5M_DEN_RAW_ID, GAMMA2_5M_NUM_RAW_ID,
    GAMMA2_5M_N_RAW_ID, GAMMA2_RAW_ID, GAMMA3_RAW_ID, GAMMA4_RAW_ID, LIQRESID_RAW_ID,
    RSI_GAMMA3_RAW_ID, RV_APBETA3_RAW_ID, VARVAR_GAMMA3_RAW_ID, VOLRATIO_GAMMA4_RAW_ID,
};
use crate::factor::common::{
    intraday_time_in_range, stock_minute_raw_spec, ClassificationLevel, ClassificationMap,
    DailyPanel, PanelColumn,
};
use crate::factor::IntradayRawMaterializeMode;
use crate::operators::{ts_mean, ts_sum};

pub const VERSION: &str = "0.1.0";
pub const RAW_VERSION: &str = "0.1.0";
pub const ONE_MIN_PROVIDER_KEY: &str = "xyzq_liquidity_1min_provider";
pub const OPERATOR_PROVIDER_KEY: &str = "xyzq_liquidity_operator_provider";
pub const CROSSDAY_5M_PROVIDER_KEY: &str = "xyzq_liquidity_crossday_5m_provider";

const SAMPLE_START: &str = "09:31:00";
const SAMPLE_END: &str = "15:00:00";
const EPS: f64 = f64::EPSILON;
const INTRADAY_RAW_WINDOW_DAYS: usize = 1;
const CROSSDAY_5M_RAW_WINDOW_DAYS: usize = 2;
const FIVE_MINUTES_PER_DAY: usize = 48;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LiquidityFamily {
    OneMinute,
    Operator,
    Crossday5m,
}

#[derive(Clone, Copy, Debug)]
pub struct XyzqLiquidityFactorDef {
    pub id: &'static str,
    pub alias: &'static str,
    pub name: &'static str,
    pub raw_id: &'static str,
    pub family: LiquidityFamily,
    pub smooth_window: usize,
    pub lookback: usize,
}

pub fn one_minute_raw_ids() -> [&'static str; 7] {
    [
        AMIHUD_1MIN_RAW_ID,
        LIQRESID_RAW_ID,
        APBETA1_RAW_ID,
        APBETA4_RAW_ID,
        GAMMA1_RAW_ID,
        GAMMA3_RAW_ID,
        GAMMA4_RAW_ID,
    ]
}

pub fn operator_raw_ids() -> [&'static str; 6] {
    [
        CLOSE_APBETA4_RAW_ID,
        RV_APBETA3_RAW_ID,
        CLOSEVOLCORR_GAMMA3_RAW_ID,
        RSI_GAMMA3_RAW_ID,
        VARVAR_GAMMA3_RAW_ID,
        VOLRATIO_GAMMA4_RAW_ID,
    ]
}

pub fn crossday_5m_raw_ids() -> [&'static str; 15] {
    [
        APBETA2_5M_N_RAW_ID,
        APBETA2_5M_SUM_X_RAW_ID,
        APBETA2_5M_SUM_Y_RAW_ID,
        APBETA2_5M_SUM_XY_RAW_ID,
        APBETA2_5M_SUM_Z_RAW_ID,
        APBETA2_5M_SUM_Z2_RAW_ID,
        APBETA3_5M_N_RAW_ID,
        APBETA3_5M_SUM_X_RAW_ID,
        APBETA3_5M_SUM_Y_RAW_ID,
        APBETA3_5M_SUM_XY_RAW_ID,
        APBETA3_5M_SUM_Z_RAW_ID,
        APBETA3_5M_SUM_Z2_RAW_ID,
        GAMMA2_5M_N_RAW_ID,
        GAMMA2_5M_NUM_RAW_ID,
        GAMMA2_5M_DEN_RAW_ID,
    ]
}

pub fn crossday_5m_raw_ids_for_factor(raw_id: &str) -> Vec<&'static str> {
    match raw_id {
        APBETA2_RAW_ID => vec![
            APBETA2_5M_N_RAW_ID,
            APBETA2_5M_SUM_X_RAW_ID,
            APBETA2_5M_SUM_Y_RAW_ID,
            APBETA2_5M_SUM_XY_RAW_ID,
            APBETA2_5M_SUM_Z_RAW_ID,
            APBETA2_5M_SUM_Z2_RAW_ID,
        ],
        APBETA3_RAW_ID => vec![
            APBETA3_5M_N_RAW_ID,
            APBETA3_5M_SUM_X_RAW_ID,
            APBETA3_5M_SUM_Y_RAW_ID,
            APBETA3_5M_SUM_XY_RAW_ID,
            APBETA3_5M_SUM_Z_RAW_ID,
            APBETA3_5M_SUM_Z2_RAW_ID,
        ],
        GAMMA2_RAW_ID => vec![
            GAMMA2_5M_N_RAW_ID,
            GAMMA2_5M_NUM_RAW_ID,
            GAMMA2_5M_DEN_RAW_ID,
        ],
        _ => Vec::new(),
    }
}

pub fn raw_ids_for_family(family: LiquidityFamily) -> Vec<&'static str> {
    match family {
        LiquidityFamily::OneMinute => one_minute_raw_ids().to_vec(),
        LiquidityFamily::Operator => operator_raw_ids().to_vec(),
        LiquidityFamily::Crossday5m => crossday_5m_raw_ids().to_vec(),
    }
}

pub fn provider_key(family: LiquidityFamily) -> &'static str {
    match family {
        LiquidityFamily::OneMinute => ONE_MIN_PROVIDER_KEY,
        LiquidityFamily::Operator => OPERATOR_PROVIDER_KEY,
        LiquidityFamily::Crossday5m => CROSSDAY_5M_PROVIDER_KEY,
    }
}

pub fn raw_spec(raw_id: &str, family: LiquidityFamily) -> IntradayDailyRawSpec {
    let columns = match family {
        LiquidityFamily::OneMinute | LiquidityFamily::Crossday5m => {
            &["open", "close", "amount"][..]
        }
        LiquidityFamily::Operator => &["open", "close", "vol"][..],
    };
    let window_days = match family {
        LiquidityFamily::OneMinute | LiquidityFamily::Operator => INTRADAY_RAW_WINDOW_DAYS,
        LiquidityFamily::Crossday5m => CROSSDAY_5M_RAW_WINDOW_DAYS,
    };
    stock_minute_raw_spec(raw_id, RAW_VERSION, columns, window_days)
}

pub fn raw_specs_for_family(family: LiquidityFamily) -> Vec<IntradayDailyRawSpec> {
    raw_ids_for_family(family)
        .iter()
        .map(|raw_id| raw_spec(raw_id, family))
        .collect()
}

pub fn factor_spec(def: XyzqLiquidityFactorDef) -> FactorSpec {
    FactorSpec {
        id: def.id.to_string(),
        aliases: vec![def.alias.to_string()],
        name: def.name.to_string(),
        asset_class: AssetClass::Stock,
        frequency: Frequency::Daily,
        version: VERSION.to_string(),
        tags: tags(),
        description: format!(
            "{} XYZQ liquidity factor, neutralized by 20-day adjusted return, Barra SIZE, and SW L1 sector.",
            def.name
        ),
        dependencies: vec![
            DataRequest::new(DatasetId::StockDailyPv, &["close"]),
            DataRequest::new(DatasetId::StockAdjFactor, &["adj_factor"]),
            DataRequest::new(DatasetId::StockBarraDaily, &["SIZE"]),
            DataRequest::new(DatasetId::StockSwClassification, &["l1_code"]),
        ],
        intraday_raw_dependencies: intraday_raw_dependencies_for_factor(def),
        lookback: Lookback {
            trading_days: def.lookback.max(20),
        },
    }
}

fn intraday_raw_dependencies_for_factor(
    def: XyzqLiquidityFactorDef,
) -> Vec<IntradayDailyRawRequest> {
    match def.family {
        LiquidityFamily::Crossday5m => {
            let daily_lookback = match def.raw_id {
                APBETA3_RAW_ID => 6,
                APBETA2_RAW_ID | GAMMA2_RAW_ID => 2,
                _ => def.lookback.saturating_sub(1),
            };
            crossday_5m_raw_ids_for_factor(def.raw_id)
                .into_iter()
                .map(|raw_id| IntradayDailyRawRequest::new(raw_id, daily_lookback))
                .collect()
        }
        _ => vec![IntradayDailyRawRequest::new(
            def.raw_id,
            def.lookback.saturating_sub(1),
        )],
    }
}

pub fn compute_factor(def: XyzqLiquidityFactorDef, data: &DataPool) -> Result<FactorSeries> {
    if def.family == LiquidityFamily::Crossday5m {
        return compute_crossday_5m_factor(def, data);
    }
    let panel = data.intraday_daily_raw_panel(def.raw_id)?;
    let raw = panel.column(def.raw_id)?;
    let raw = mask_bj(&raw, &panel)?;
    let smoothed = if def.smooth_window > 1 {
        raw.ts(|values| ts_mean(values, def.smooth_window, 1))?
    } else {
        raw
    };
    let smoothed = mask_bj(&smoothed, &panel)?;
    let factor = neutralize_ret20_size_sector(&smoothed, &panel, data)?;
    Ok(factor.to_factor_series(factor_spec(def)))
}

fn compute_crossday_5m_factor(
    def: XyzqLiquidityFactorDef,
    data: &DataPool,
) -> Result<FactorSeries> {
    let (panel, raw) = match def.raw_id {
        APBETA2_RAW_ID => {
            let panel = data.intraday_daily_raw_panel(APBETA2_5M_N_RAW_ID)?;
            let raw = restore_moment_additive_factor(
                panel,
                APBETA2_5M_N_RAW_ID,
                APBETA2_5M_SUM_X_RAW_ID,
                APBETA2_5M_SUM_Y_RAW_ID,
                APBETA2_5M_SUM_XY_RAW_ID,
                APBETA2_5M_SUM_Z_RAW_ID,
                APBETA2_5M_SUM_Z2_RAW_ID,
                def.smooth_window,
            )?;
            (panel, raw)
        }
        APBETA3_RAW_ID => {
            let panel = data.intraday_daily_raw_panel(APBETA3_5M_N_RAW_ID)?;
            let raw = restore_moment_additive_factor(
                panel,
                APBETA3_5M_N_RAW_ID,
                APBETA3_5M_SUM_X_RAW_ID,
                APBETA3_5M_SUM_Y_RAW_ID,
                APBETA3_5M_SUM_XY_RAW_ID,
                APBETA3_5M_SUM_Z_RAW_ID,
                APBETA3_5M_SUM_Z2_RAW_ID,
                def.smooth_window,
            )?;
            (panel, raw)
        }
        GAMMA2_RAW_ID => {
            let panel = data.intraday_daily_raw_panel(GAMMA2_5M_N_RAW_ID)?;
            let raw = restore_gamma_additive_factor(
                panel,
                GAMMA2_5M_N_RAW_ID,
                GAMMA2_5M_NUM_RAW_ID,
                GAMMA2_5M_DEN_RAW_ID,
                def.smooth_window,
            )?;
            (panel, raw)
        }
        _ => {
            return Err(err(format!(
                "unsupported crossday liquidity raw: {}",
                def.raw_id
            )))
        }
    };
    let raw = mask_bj(&raw, panel)?;
    let factor = neutralize_ret20_size_sector(&raw, panel, data)?;
    Ok(factor.to_factor_series(factor_spec(def)))
}

fn restore_moment_additive_factor(
    panel: &DailyPanel,
    n_raw_id: &str,
    sum_x_raw_id: &str,
    sum_y_raw_id: &str,
    sum_xy_raw_id: &str,
    sum_z_raw_id: &str,
    sum_z2_raw_id: &str,
    window: usize,
) -> Result<PanelColumn> {
    let n = rolling_raw_sum(panel, n_raw_id, window)?;
    let sum_x = rolling_raw_sum(panel, sum_x_raw_id, window)?;
    let sum_y = rolling_raw_sum(panel, sum_y_raw_id, window)?;
    let sum_xy = rolling_raw_sum(panel, sum_xy_raw_id, window)?;
    let sum_z = rolling_raw_sum(panel, sum_z_raw_id, window)?;
    let sum_z2 = rolling_raw_sum(panel, sum_z2_raw_id, window)?;
    let values = n
        .values()
        .iter()
        .zip(sum_x.values())
        .zip(sum_y.values())
        .zip(sum_xy.values())
        .zip(sum_z.values())
        .zip(sum_z2.values())
        .map(|(((((n, sum_x), sum_y), sum_xy), sum_z), sum_z2)| {
            moment_ratio_from_sums(*n, *sum_x, *sum_y, *sum_xy, *sum_z, *sum_z2)
        })
        .collect();
    panel.column_from_values(values)
}

fn restore_gamma_additive_factor(
    panel: &DailyPanel,
    n_raw_id: &str,
    num_raw_id: &str,
    den_raw_id: &str,
    window: usize,
) -> Result<PanelColumn> {
    let n = rolling_raw_sum(panel, n_raw_id, window)?;
    let num = rolling_raw_sum(panel, num_raw_id, window)?;
    let den = rolling_raw_sum(panel, den_raw_id, window)?;
    let values = n
        .values()
        .iter()
        .zip(num.values())
        .zip(den.values())
        .map(|((n, num), den)| gamma_ratio_from_sums(*n, *num, *den))
        .collect();
    panel.column_from_values(values)
}

fn rolling_raw_sum(panel: &DailyPanel, raw_id: &str, window: usize) -> Result<PanelColumn> {
    panel.column(raw_id)?.ts(|values| ts_sum(values, window, 1))
}

fn moment_ratio_from_sums(
    n: Option<f64>,
    sum_x: Option<f64>,
    sum_y: Option<f64>,
    sum_xy: Option<f64>,
    sum_z: Option<f64>,
    sum_z2: Option<f64>,
) -> Option<f64> {
    let n = n.and_then(finite_value)?;
    if n < 2.0 {
        return None;
    }
    let sum_x = sum_x.and_then(finite_value)?;
    let sum_y = sum_y.and_then(finite_value)?;
    let sum_xy = sum_xy.and_then(finite_value)?;
    let sum_z = sum_z.and_then(finite_value)?;
    let sum_z2 = sum_z2.and_then(finite_value)?;
    let numerator = sum_xy - sum_x * sum_y / n;
    let denominator = sum_z2 - sum_z * sum_z / n;
    safe_div_value(numerator, denominator)
}

fn gamma_ratio_from_sums(n: Option<f64>, num: Option<f64>, den: Option<f64>) -> Option<f64> {
    let n = n.and_then(finite_value)?;
    if n < 1.0 {
        return None;
    }
    safe_div_value(num.and_then(finite_value)?, den.and_then(finite_value)?)
}

pub fn intraday_raw_auxiliary_requirements(
    raw_ids: &[String],
    family: LiquidityFamily,
) -> Vec<IntradayDailyRawAuxiliaryRequest> {
    let requested = requested_for_family(raw_ids, family);
    if requested.is_empty() {
        return Vec::new();
    }
    match family {
        LiquidityFamily::OneMinute | LiquidityFamily::Operator => vec![
            IntradayDailyRawAuxiliaryRequest::new(
                DataRequest::new(DatasetId::StockCiClassification, &["l1_code"]),
                0,
            ),
            IntradayDailyRawAuxiliaryRequest::new(
                DataRequest::new(DatasetId::StockDailyBasic, &["circ_mv"]),
                0,
            ),
        ],
        LiquidityFamily::Crossday5m => Vec::new(),
    }
}

pub fn requested_for_family(raw_ids: &[String], family: LiquidityFamily) -> BTreeSet<&'static str> {
    let allowed = raw_ids_for_family(family);
    raw_ids
        .iter()
        .filter_map(|raw_id| allowed.iter().copied().find(|allowed| *allowed == raw_id))
        .collect()
}

pub fn minute_compute_many_for_family(
    raw_ids: &[String],
    context: &FactorContext,
    data: &DataPool,
    family: LiquidityFamily,
) -> Result<Vec<IntradayDailyRawSeries>> {
    match family {
        LiquidityFamily::OneMinute => compute_one_minute_raws(raw_ids, context, data),
        LiquidityFamily::Operator => compute_operator_raws(raw_ids, context, data),
        LiquidityFamily::Crossday5m => Err(err(
            "crossday 5m liquidity raw must be materialized through stateful compute",
        )),
    }
}

pub fn crossday_materialize_mode() -> IntradayRawMaterializeMode {
    IntradayRawMaterializeMode::Stateful
}

pub fn initial_crossday_state() -> Box<dyn Any + Send> {
    Box::new(Crossday5mState::default())
}

pub fn minute_compute_crossday_stateful_many(
    raw_ids: &[String],
    context: &FactorContext,
    data: &DataPool,
    state: &mut dyn Any,
) -> Result<Vec<IntradayDailyRawSeries>> {
    let requested = requested_for_family(raw_ids, LiquidityFamily::Crossday5m);
    if requested.is_empty() {
        return Ok(Vec::new());
    }
    let state = state
        .downcast_mut::<Crossday5mState>()
        .ok_or_else(|| err("xyzq liquidity crossday state received incompatible state"))?;
    let trade_date = *context
        .target_dates
        .first()
        .ok_or_else(|| err("crossday liquidity raw requires one target date"))?;
    let Some(table) = data.minute(DatasetId::StockMinute1m, trade_date) else {
        state.prev_last_amihud_by_stock.clear();
        return Ok(series_from_crossday_values(
            trade_date,
            &requested,
            BTreeMap::new(),
        ));
    };
    let bars_by_stock = five_minute_bars_by_stock(table, true)?;
    let mut values = BTreeMap::<String, CrossdayAdditiveValues>::new();
    for (ts_code, bars) in &bars_by_stock {
        if !is_bj_stock(ts_code) && !bars.is_empty() {
            values.insert(ts_code.clone(), CrossdayAdditiveValues::default());
        }
    }
    let mut prev_amihud_by_stock = state.prev_last_amihud_by_stock.clone();
    let mut current_last_amihud_by_stock = BTreeMap::<String, f64>::new();
    let mut bar_values =
        Vec::<(&String, Option<f64>, Option<f64>)>::with_capacity(bars_by_stock.len());

    for bar_idx in 0..FIVE_MINUTES_PER_DAY {
        bar_values.clear();
        let mut market_ret_sum = 0.0;
        let mut market_ret_count = 0usize;
        let mut market_liq_sum = 0.0;
        let mut market_liq_count = 0usize;
        for (ts_code, bars) in &bars_by_stock {
            if is_bj_stock(ts_code) {
                continue;
            }
            let Some(bar) = bars.get(bar_idx) else {
                continue;
            };
            let prev_amihud = prev_amihud_by_stock.get(ts_code).copied();
            let delta_liq = log_diff(bar.amihud, prev_amihud);
            if let Some(amihud) = bar.amihud {
                prev_amihud_by_stock.insert(ts_code.clone(), amihud);
                current_last_amihud_by_stock.insert(ts_code.clone(), amihud);
            }
            if let Some(ret) = bar.ret {
                market_ret_sum += ret;
                market_ret_count += 1;
            }
            if let Some(delta_liq) = delta_liq {
                market_liq_sum += delta_liq;
                market_liq_count += 1;
            }
            bar_values.push((ts_code, bar.ret, delta_liq));
        }
        let market_ret = if market_ret_count == 0 {
            None
        } else {
            finite_value(market_ret_sum / market_ret_count as f64)
        };
        let market_liq = if market_liq_count == 0 {
            None
        } else {
            finite_value(market_liq_sum / market_liq_count as f64)
        };
        for (ts_code, ret, delta_liq) in &bar_values {
            if let Some(values) = values.get_mut(ts_code.as_str()) {
                values.push_bar(*ret, *delta_liq, market_ret, market_liq);
            }
        }
    }
    state.prev_last_amihud_by_stock = current_last_amihud_by_stock;

    for ts_code in bars_by_stock.keys() {
        if is_bj_stock(ts_code) {
            values.insert(ts_code.clone(), CrossdayAdditiveValues::default());
        }
    }

    Ok(series_from_crossday_values(trade_date, &requested, values))
}

#[macro_export]
macro_rules! define_xyzq_liquidity_factor {
    ($struct_name:ident, $id:expr, $alias:expr, $name:expr, $raw_id:expr, $family:expr, $smooth_window:expr, $lookback:expr) => {
        const DEF: $crate::factor::common::xyzq_liquidity::XyzqLiquidityFactorDef =
            $crate::factor::common::xyzq_liquidity::XyzqLiquidityFactorDef {
                id: $id,
                alias: $alias,
                name: $name,
                raw_id: $raw_id,
                family: $family,
                smooth_window: $smooth_window,
                lookback: $lookback,
            };

        pub struct $struct_name;

        pub fn create() -> Box<dyn $crate::factor::Factor> {
            Box::new($struct_name)
        }

        impl $crate::factor::Factor for $struct_name {
            fn spec(&self) -> $crate::core::FactorSpec {
                $crate::factor::common::xyzq_liquidity::factor_spec(DEF)
            }

            fn intraday_raw_specs(&self) -> Vec<$crate::core::IntradayDailyRawSpec> {
                match DEF.family {
                    $crate::factor::common::xyzq_liquidity::LiquidityFamily::Crossday5m => {
                        $crate::factor::common::xyzq_liquidity::crossday_5m_raw_ids_for_factor(DEF.raw_id)
                            .iter()
                            .map(|raw_id| {
                                $crate::factor::common::xyzq_liquidity::raw_spec(
                                    raw_id,
                                    DEF.family,
                                )
                            })
                            .collect()
                    }
                    _ => vec![$crate::factor::common::xyzq_liquidity::raw_spec(
                        DEF.raw_id, DEF.family,
                    )],
                }
            }

            fn intraday_raw_provider_key(&self, _raw_id: &str) -> String {
                $crate::factor::common::xyzq_liquidity::provider_key(DEF.family).to_string()
            }

            fn intraday_raw_materialize_mode(
                &self,
                _raw_ids: &[String],
            ) -> $crate::factor::IntradayRawMaterializeMode {
                match DEF.family {
                    $crate::factor::common::xyzq_liquidity::LiquidityFamily::Crossday5m => {
                        $crate::factor::common::xyzq_liquidity::crossday_materialize_mode()
                    }
                    _ => $crate::factor::IntradayRawMaterializeMode::Stateless,
                }
            }

            fn initial_intraday_raw_state(&self, _raw_ids: &[String]) -> Box<dyn std::any::Any + Send> {
                match DEF.family {
                    $crate::factor::common::xyzq_liquidity::LiquidityFamily::Crossday5m => {
                        $crate::factor::common::xyzq_liquidity::initial_crossday_state()
                    }
                    _ => Box::new(()),
                }
            }

            fn intraday_raw_auxiliary_requirements(
                &self,
                raw_ids: &[String],
            ) -> Vec<$crate::core::IntradayDailyRawAuxiliaryRequest> {
                $crate::factor::common::xyzq_liquidity::intraday_raw_auxiliary_requirements(
                    raw_ids, DEF.family,
                )
            }

            fn minute_compute_many(
                &self,
                raw_ids: &[String],
                context: &$crate::core::FactorContext,
                data: &$crate::data::DataPool,
            ) -> $crate::error::Result<Vec<$crate::core::IntradayDailyRawSeries>> {
                $crate::factor::common::xyzq_liquidity::minute_compute_many_for_family(
                    raw_ids, context, data, DEF.family,
                )
            }

            fn minute_compute_stateful_many(
                &self,
                raw_ids: &[String],
                context: &$crate::core::FactorContext,
                data: &$crate::data::DataPool,
                state: &mut dyn std::any::Any,
            ) -> $crate::error::Result<Vec<$crate::core::IntradayDailyRawSeries>> {
                match DEF.family {
                    $crate::factor::common::xyzq_liquidity::LiquidityFamily::Crossday5m => {
                        $crate::factor::common::xyzq_liquidity::minute_compute_crossday_stateful_many(
                            raw_ids, context, data, state,
                        )
                    }
                    _ => self.minute_compute_many(raw_ids, context, data),
                }
            }

            fn compute(
                &self,
                _context: &$crate::core::FactorContext,
                data: &$crate::data::DataPool,
            ) -> $crate::error::Result<$crate::core::FactorSeries> {
                $crate::factor::common::xyzq_liquidity::compute_factor(DEF, data)
            }
        }
    };
}

#[derive(Clone, Copy, Debug, Default)]
struct OneMinuteValues {
    amihud: Option<f64>,
    liqresid: Option<f64>,
    apbeta1: Option<f64>,
    apbeta4: Option<f64>,
    gamma1: Option<f64>,
    gamma3: Option<f64>,
    gamma4: Option<f64>,
}

#[derive(Clone, Copy, Debug, Default)]
struct OperatorValues {
    close_apbeta4: Option<f64>,
    rv_apbeta3: Option<f64>,
    closevolcorr_gamma3: Option<f64>,
    rsi_gamma3: Option<f64>,
    varvar_gamma3: Option<f64>,
    volratio_gamma4: Option<f64>,
}

#[derive(Clone, Copy, Debug, Default)]
struct CrossdayAdditiveValues {
    apbeta2: MomentAdditiveStats,
    apbeta3: MomentAdditiveStats,
    gamma2: GammaAdditiveStats,
}

#[derive(Clone, Copy, Debug, Default)]
struct MomentAdditiveStats {
    n: usize,
    sum_x: f64,
    sum_y: f64,
    sum_xy: f64,
    sum_z: f64,
    sum_z2: f64,
}

#[derive(Clone, Copy, Debug, Default)]
struct GammaAdditiveStats {
    n: usize,
    num: f64,
    den: f64,
}

#[derive(Clone, Copy, Debug, Default)]
struct MinutePoint {
    open: Option<f64>,
    close: Option<f64>,
    amount: Option<f64>,
    vol: Option<f64>,
}

#[derive(Clone, Copy, Debug)]
struct OneMinuteSeries {
    returns: [Option<f64>; 240],
    amounts: [Option<f64>; 240],
    amihud: [Option<f64>; 240],
    delta_amihud: [Option<f64>; 240],
    resid_delta_amihud: [Option<f64>; 240],
}

impl Default for OneMinuteSeries {
    fn default() -> Self {
        Self {
            returns: [None; 240],
            amounts: [None; 240],
            amihud: [None; 240],
            delta_amihud: [None; 240],
            resid_delta_amihud: [None; 240],
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct MinuteOperatorSeries {
    returns: [Option<f64>; 240],
    close: [Option<f64>; 240],
    vol: [Option<f64>; 240],
    close_delta: [Option<f64>; 240],
    rv_delta: [Option<f64>; 240],
    closevolcorr_delta: [Option<f64>; 240],
    rsi_delta: [Option<f64>; 240],
    varvar_delta: [Option<f64>; 240],
    volratio_delta: [Option<f64>; 240],
}

impl Default for MinuteOperatorSeries {
    fn default() -> Self {
        Self {
            returns: [None; 240],
            close: [None; 240],
            vol: [None; 240],
            close_delta: [None; 240],
            rv_delta: [None; 240],
            closevolcorr_delta: [None; 240],
            rsi_delta: [None; 240],
            varvar_delta: [None; 240],
            volratio_delta: [None; 240],
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct BenchmarkSeries {
    returns: [Option<f64>; 240],
    liquidity: [Option<f64>; 240],
    resid_liquidity: [Option<f64>; 240],
}

impl Default for BenchmarkSeries {
    fn default() -> Self {
        Self {
            returns: [None; 240],
            liquidity: [None; 240],
            resid_liquidity: [None; 240],
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct FiveMinuteBar {
    ret: Option<f64>,
    amihud: Option<f64>,
}

#[derive(Debug, Default)]
pub struct Crossday5mState {
    prev_last_amihud_by_stock: BTreeMap<String, f64>,
}

fn compute_one_minute_raws(
    raw_ids: &[String],
    context: &FactorContext,
    data: &DataPool,
) -> Result<Vec<IntradayDailyRawSeries>> {
    let requested = requested_for_family(raw_ids, LiquidityFamily::OneMinute);
    if requested.is_empty() {
        return Ok(Vec::new());
    }
    let mut output = one_minute_raw_map();
    for trade_date in &context.target_dates {
        let Some(table) = data.minute(DatasetId::StockMinute1m, *trade_date) else {
            continue;
        };
        let points = minute_points_by_stock(table, true)?;
        let groups = composite_groups(data, *trade_date, points.keys())?;
        let mut series_by_stock = BTreeMap::<String, OneMinuteSeries>::new();
        for (ts_code, minutes) in &points {
            if is_bj_stock(ts_code) {
                continue;
            }
            series_by_stock.insert(ts_code.clone(), one_minute_series(minutes));
        }
        let benchmarks = one_minute_benchmarks(&series_by_stock, &groups);
        let values = series_by_stock
            .iter()
            .map(|(ts_code, series)| {
                let benchmark = groups
                    .get(ts_code)
                    .and_then(|group| benchmarks.get(group))
                    .copied()
                    .unwrap_or_default();
                (ts_code.clone(), one_minute_values(series, &benchmark))
            })
            .collect::<BTreeMap<_, _>>();
        push_one_minute_values(*trade_date, &requested, &values, points.keys(), &mut output);
    }
    Ok(one_minute_series_from_map(output, &requested))
}

fn compute_operator_raws(
    raw_ids: &[String],
    context: &FactorContext,
    data: &DataPool,
) -> Result<Vec<IntradayDailyRawSeries>> {
    let requested = requested_for_family(raw_ids, LiquidityFamily::Operator);
    if requested.is_empty() {
        return Ok(Vec::new());
    }
    let mut output = operator_raw_map();
    for trade_date in &context.target_dates {
        let Some(table) = data.minute(DatasetId::StockMinute1m, *trade_date) else {
            continue;
        };
        let points = minute_points_by_stock(table, false)?;
        let groups = composite_groups(data, *trade_date, points.keys())?;
        let mut series_by_stock = BTreeMap::<String, MinuteOperatorSeries>::new();
        for (ts_code, minutes) in &points {
            if is_bj_stock(ts_code) {
                continue;
            }
            series_by_stock.insert(ts_code.clone(), operator_series(minutes));
        }
        let bench = operator_benchmarks(&series_by_stock, &groups);
        let values = series_by_stock
            .iter()
            .map(|(ts_code, series)| {
                let benchmark = groups
                    .get(ts_code)
                    .and_then(|group| bench.get(group))
                    .copied()
                    .unwrap_or_default();
                (ts_code.clone(), operator_values(series, &benchmark))
            })
            .collect::<BTreeMap<_, _>>();
        push_operator_values(*trade_date, &requested, &values, points.keys(), &mut output);
    }
    Ok(operator_series_from_map(output, &requested))
}

fn minute_points_by_stock(
    table: &Table,
    include_amount: bool,
) -> Result<BTreeMap<String, Vec<(String, MinutePoint)>>> {
    let ts_codes = table.required_utf8("ts_code")?;
    let trade_times = table.required_utf8("trade_time")?;
    let open = table.required_f64_cast("open")?;
    let close = table.required_f64_cast("close")?;
    let amount = if include_amount {
        Some(table.required_f64_cast("amount")?)
    } else {
        None
    };
    let vol = table.required_f64_cast("vol").ok();
    let mut grouped = BTreeMap::<String, Vec<(String, MinutePoint)>>::new();
    for idx in 0..table.len {
        let (Some(ts_code), Some(trade_time)) = (ts_codes[idx].clone(), trade_times[idx].clone())
        else {
            continue;
        };
        if !intraday_time_in_range(&trade_time, SAMPLE_START, SAMPLE_END) {
            continue;
        }
        grouped.entry(ts_code).or_default().push((
            trade_time,
            MinutePoint {
                open: clean_value(open[idx]),
                close: clean_value(close[idx]),
                amount: amount.as_ref().and_then(|values| clean_value(values[idx])),
                vol: vol.as_ref().and_then(|values| clean_value(values[idx])),
            },
        ));
    }
    for values in grouped.values_mut() {
        values.sort_by(|left, right| left.0.cmp(&right.0));
    }
    Ok(grouped)
}

fn one_minute_series(points: &[(String, MinutePoint)]) -> OneMinuteSeries {
    let mut series = OneMinuteSeries::default();
    for (idx, (_, point)) in points.iter().take(240).enumerate() {
        let ret = simple_return(point.open, point.close);
        let amount = point.amount.filter(|amount| *amount > EPS);
        let amihud = match (ret, amount) {
            (Some(ret), Some(amount)) => finite_value(ret.abs() / amount),
            _ => None,
        };
        series.returns[idx] = ret;
        series.amounts[idx] = amount;
        series.amihud[idx] = amihud;
    }
    for idx in 1..240 {
        series.delta_amihud[idx] = log_diff(series.amihud[idx], series.amihud[idx - 1]);
    }
    let residuals = ar1_residuals(&series.returns);
    let mut resid_amihud = [None; 240];
    for idx in 0..240 {
        resid_amihud[idx] = match (residuals[idx], series.amounts[idx]) {
            (Some(resid), Some(amount)) => finite_value(resid.abs() / amount),
            _ => None,
        };
    }
    for idx in 1..240 {
        series.resid_delta_amihud[idx] = log_diff(resid_amihud[idx], resid_amihud[idx - 1]);
    }
    series
}

fn one_minute_benchmarks(
    series_by_stock: &BTreeMap<String, OneMinuteSeries>,
    groups: &BTreeMap<String, String>,
) -> BTreeMap<String, BenchmarkSeries> {
    let mut sums = BTreeMap::<
        String,
        (
            [f64; 240],
            [usize; 240],
            [f64; 240],
            [usize; 240],
            [f64; 240],
            [usize; 240],
        ),
    >::new();
    for (ts_code, series) in series_by_stock {
        let Some(group) = groups.get(ts_code) else {
            continue;
        };
        let entry = sums.entry(group.clone()).or_insert((
            [0.0; 240], [0; 240], [0.0; 240], [0; 240], [0.0; 240], [0; 240],
        ));
        for idx in 0..240 {
            if let Some(value) = series.returns[idx] {
                entry.0[idx] += value;
                entry.1[idx] += 1;
            }
            if let Some(value) = series.delta_amihud[idx] {
                entry.2[idx] += value;
                entry.3[idx] += 1;
            }
            if let Some(value) = series.resid_delta_amihud[idx] {
                entry.4[idx] += value;
                entry.5[idx] += 1;
            }
        }
    }
    let mut output = BTreeMap::new();
    for (group, (ret_sum, ret_count, liq_sum, liq_count, resid_sum, resid_count)) in sums {
        let mut values = BenchmarkSeries::default();
        for idx in 0..240 {
            if ret_count[idx] > 0 {
                values.returns[idx] = finite_value(ret_sum[idx] / ret_count[idx] as f64);
            }
            if liq_count[idx] > 0 {
                values.liquidity[idx] = finite_value(liq_sum[idx] / liq_count[idx] as f64);
            }
            if resid_count[idx] > 0 {
                values.resid_liquidity[idx] =
                    finite_value(resid_sum[idx] / resid_count[idx] as f64);
            }
        }
        output.insert(group, values);
    }
    output
}

fn one_minute_values(series: &OneMinuteSeries, benchmark: &BenchmarkSeries) -> OneMinuteValues {
    let amihud = mean_option_slice(&series.amihud);
    let liqresid = liqresid_value(series, benchmark);
    let apbeta1 = covariance_ratio(
        &series.returns,
        &benchmark.returns,
        &benchmark.returns,
        &benchmark.liquidity,
    );
    let apbeta4 = covariance_ratio(
        &series.delta_amihud,
        &benchmark.liquidity,
        &benchmark.returns,
        &benchmark.liquidity,
    );
    let gamma1 = gamma_ratio(
        &series.returns,
        &benchmark.returns,
        &benchmark.liquidity,
        GammaKind::Rr,
    );
    let gamma3 = gamma_ratio(
        &series.returns,
        &benchmark.returns,
        &benchmark.liquidity,
        GammaKind::NegRL,
    );
    let gamma4 = gamma_ratio(
        &series.delta_amihud,
        &benchmark.returns,
        &benchmark.liquidity,
        GammaKind::LL,
    );
    OneMinuteValues {
        amihud,
        liqresid,
        apbeta1,
        apbeta4,
        gamma1,
        gamma3,
        gamma4,
    }
}

fn liqresid_value(series: &OneMinuteSeries, benchmark: &BenchmarkSeries) -> Option<f64> {
    let pairs = series
        .resid_delta_amihud
        .iter()
        .zip(benchmark.resid_liquidity.iter())
        .filter_map(|(left, right)| Some(((*left)?, (*right)?)))
        .collect::<Vec<_>>();
    if pairs.len() < 3 {
        return None;
    }
    let (alpha, beta) = ols_intercept_slope(&pairs)?;
    let residuals = pairs
        .iter()
        .map(|(left, right)| left - alpha - beta * right)
        .filter_map(finite_value)
        .collect::<Vec<_>>();
    if residuals.is_empty() {
        return None;
    }
    let q80 = quantile_sorted(residuals.clone(), 0.80)?;
    let tail = residuals
        .into_iter()
        .filter(|value| *value > q80)
        .collect::<Vec<_>>();
    if tail.is_empty() {
        return None;
    }
    finite_value(tail.iter().sum::<f64>() / tail.len() as f64)
}

#[derive(Clone, Copy, Debug)]
enum GammaKind {
    Rr,
    NegRL,
    LL,
}

fn covariance_ratio(
    left: &[Option<f64>; 240],
    numerator_right: &[Option<f64>; 240],
    denom_ret: &[Option<f64>; 240],
    denom_liq: &[Option<f64>; 240],
) -> Option<f64> {
    let rows = left
        .iter()
        .zip(numerator_right.iter())
        .zip(denom_ret.iter())
        .zip(denom_liq.iter())
        .filter_map(|(((left, numerator_right), denom_ret), denom_liq)| {
            Some(((*left)?, (*numerator_right)?, (*denom_ret)?, (*denom_liq)?))
        })
        .collect::<Vec<_>>();
    if rows.len() < 2 {
        return None;
    }
    let mean_left = rows.iter().map(|row| row.0).sum::<f64>() / rows.len() as f64;
    let mean_right = rows.iter().map(|row| row.1).sum::<f64>() / rows.len() as f64;
    let state_mean = rows.iter().map(|row| row.2 - row.3).sum::<f64>() / rows.len() as f64;
    let numerator = rows
        .iter()
        .map(|row| (row.0 - mean_left) * (row.1 - mean_right))
        .sum::<f64>();
    let denominator = rows
        .iter()
        .map(|row| {
            let centered = row.2 - row.3 - state_mean;
            centered * centered
        })
        .sum::<f64>();
    safe_div_value(numerator, denominator)
}

fn gamma_ratio(
    stock_value: &[Option<f64>; 240],
    bench_ret: &[Option<f64>; 240],
    bench_liq: &[Option<f64>; 240],
    kind: GammaKind,
) -> Option<f64> {
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    let mut count = 0usize;
    for idx in 0..240 {
        let (Some(stock), Some(ret), Some(liq)) =
            (stock_value[idx], bench_ret[idx], bench_liq[idx])
        else {
            continue;
        };
        if ret >= liq {
            continue;
        }
        numerator += match kind {
            GammaKind::Rr => stock * ret,
            GammaKind::NegRL => -stock * liq,
            GammaKind::LL => stock * liq,
        };
        let diff = ret - liq;
        denominator += diff * diff;
        count += 1;
    }
    if count == 0 {
        return None;
    }
    safe_div_value(numerator, denominator)
}

fn operator_series(points: &[(String, MinutePoint)]) -> MinuteOperatorSeries {
    let mut series = MinuteOperatorSeries::default();
    let mut day_vol_sum = 0.0;
    for (idx, (_, point)) in points.iter().take(240).enumerate() {
        series.returns[idx] = simple_return(point.open, point.close);
        series.close[idx] = point.close;
        series.vol[idx] = point.vol;
        if let Some(vol) = point.vol.filter(|vol| *vol >= 0.0) {
            day_vol_sum += vol;
        }
    }

    let close_feature = rolling_close_last(&series.close);
    let rv_feature = rolling_rv(&series.returns);
    let closevolcorr_feature = rolling_corr(&series.close, &series.vol, 5);
    let rsi_feature = rolling_rsi(&series.returns);
    let varvar_feature = rolling_varvar(&series.returns);
    let volratio_feature = rolling_volratio(&series.vol, day_vol_sum);

    series.close_delta = pct_change_series(&close_feature);
    series.rv_delta = pct_change_series(&rv_feature);
    series.closevolcorr_delta = pct_change_series(&closevolcorr_feature);
    series.rsi_delta = pct_change_series(&rsi_feature);
    series.varvar_delta = pct_change_series(&varvar_feature);
    series.volratio_delta = pct_change_series(&volratio_feature);
    series
}

fn operator_benchmarks(
    series_by_stock: &BTreeMap<String, MinuteOperatorSeries>,
    groups: &BTreeMap<String, String>,
) -> BTreeMap<String, OperatorBenchmarkSeries> {
    let mut sums = BTreeMap::<String, OperatorBenchmarkAccum>::new();
    for (ts_code, series) in series_by_stock {
        let Some(group) = groups.get(ts_code) else {
            continue;
        };
        let entry = sums.entry(group.clone()).or_default();
        for idx in 0..240 {
            entry.push_return(idx, series.returns[idx]);
            entry.push(OperatorFeature::Close, idx, series.close_delta[idx]);
            entry.push(OperatorFeature::Rv, idx, series.rv_delta[idx]);
            entry.push(
                OperatorFeature::CloseVolCorr,
                idx,
                series.closevolcorr_delta[idx],
            );
            entry.push(OperatorFeature::Rsi, idx, series.rsi_delta[idx]);
            entry.push(OperatorFeature::Varvar, idx, series.varvar_delta[idx]);
            entry.push(OperatorFeature::Volratio, idx, series.volratio_delta[idx]);
        }
    }
    sums.into_iter()
        .map(|(group, accum)| (group, accum.finalize()))
        .collect()
}

fn operator_values(
    series: &MinuteOperatorSeries,
    benchmark: &OperatorBenchmarkSeries,
) -> OperatorValues {
    OperatorValues {
        close_apbeta4: covariance_ratio(
            &series.close_delta,
            &benchmark.close_delta,
            &benchmark.returns,
            &benchmark.close_delta,
        ),
        rv_apbeta3: covariance_ratio(
            &series.returns,
            &benchmark.rv_delta,
            &benchmark.returns,
            &benchmark.rv_delta,
        ),
        closevolcorr_gamma3: gamma_ratio(
            &series.returns,
            &benchmark.returns,
            &benchmark.closevolcorr_delta,
            GammaKind::NegRL,
        ),
        rsi_gamma3: gamma_ratio(
            &series.returns,
            &benchmark.returns,
            &benchmark.rsi_delta,
            GammaKind::NegRL,
        ),
        varvar_gamma3: gamma_ratio(
            &series.returns,
            &benchmark.returns,
            &benchmark.varvar_delta,
            GammaKind::NegRL,
        ),
        volratio_gamma4: gamma_ratio(
            &series.volratio_delta,
            &benchmark.returns,
            &benchmark.volratio_delta,
            GammaKind::LL,
        ),
    }
}

#[derive(Clone, Copy, Debug)]
enum OperatorFeature {
    Close,
    Rv,
    CloseVolCorr,
    Rsi,
    Varvar,
    Volratio,
}

#[derive(Clone, Copy, Debug)]
struct OperatorBenchmarkSeries {
    returns: [Option<f64>; 240],
    close_delta: [Option<f64>; 240],
    rv_delta: [Option<f64>; 240],
    closevolcorr_delta: [Option<f64>; 240],
    rsi_delta: [Option<f64>; 240],
    varvar_delta: [Option<f64>; 240],
    volratio_delta: [Option<f64>; 240],
}

impl Default for OperatorBenchmarkSeries {
    fn default() -> Self {
        Self {
            returns: [None; 240],
            close_delta: [None; 240],
            rv_delta: [None; 240],
            closevolcorr_delta: [None; 240],
            rsi_delta: [None; 240],
            varvar_delta: [None; 240],
            volratio_delta: [None; 240],
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct SumCount {
    sum: [f64; 240],
    count: [usize; 240],
}

impl Default for SumCount {
    fn default() -> Self {
        Self {
            sum: [0.0; 240],
            count: [0; 240],
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct OperatorBenchmarkAccum {
    returns: SumCount,
    close_delta: SumCount,
    rv_delta: SumCount,
    closevolcorr_delta: SumCount,
    rsi_delta: SumCount,
    varvar_delta: SumCount,
    volratio_delta: SumCount,
}

impl OperatorBenchmarkAccum {
    fn push_return(&mut self, idx: usize, value: Option<f64>) {
        self.returns.push(idx, value);
    }

    fn push(&mut self, feature: OperatorFeature, idx: usize, value: Option<f64>) {
        match feature {
            OperatorFeature::Close => self.close_delta.push(idx, value),
            OperatorFeature::Rv => self.rv_delta.push(idx, value),
            OperatorFeature::CloseVolCorr => self.closevolcorr_delta.push(idx, value),
            OperatorFeature::Rsi => self.rsi_delta.push(idx, value),
            OperatorFeature::Varvar => self.varvar_delta.push(idx, value),
            OperatorFeature::Volratio => self.volratio_delta.push(idx, value),
        }
    }

    fn finalize(self) -> OperatorBenchmarkSeries {
        OperatorBenchmarkSeries {
            returns: self.returns.finalize(),
            close_delta: self.close_delta.finalize(),
            rv_delta: self.rv_delta.finalize(),
            closevolcorr_delta: self.closevolcorr_delta.finalize(),
            rsi_delta: self.rsi_delta.finalize(),
            varvar_delta: self.varvar_delta.finalize(),
            volratio_delta: self.volratio_delta.finalize(),
        }
    }
}

impl SumCount {
    fn push(&mut self, idx: usize, value: Option<f64>) {
        if let Some(value) = value.and_then(finite_value) {
            self.sum[idx] += value;
            self.count[idx] += 1;
        }
    }

    fn finalize(self) -> [Option<f64>; 240] {
        let mut output = [None; 240];
        for (idx, output) in output.iter_mut().enumerate() {
            if self.count[idx] > 0 {
                *output = finite_value(self.sum[idx] / self.count[idx] as f64);
            }
        }
        output
    }
}

impl CrossdayAdditiveValues {
    fn push_bar(
        &mut self,
        ret: Option<f64>,
        delta_liq: Option<f64>,
        market_ret: Option<f64>,
        market_liq: Option<f64>,
    ) {
        if let (Some(delta_liq), Some(market_ret), Some(market_liq)) =
            (delta_liq, market_ret, market_liq)
        {
            let z = market_ret - market_liq;
            self.apbeta2.push(delta_liq, market_ret, z);
            if market_ret < market_liq {
                self.gamma2.push(-delta_liq * market_ret, z * z);
            }
        }
        if let (Some(ret), Some(market_ret), Some(market_liq)) = (ret, market_ret, market_liq) {
            let z = market_ret - market_liq;
            self.apbeta3.push(ret, market_liq, z);
        }
    }
}

impl MomentAdditiveStats {
    fn push(&mut self, x: f64, y: f64, z: f64) {
        if x.is_finite() && y.is_finite() && z.is_finite() {
            self.n += 1;
            self.sum_x += x;
            self.sum_y += y;
            self.sum_xy += x * y;
            self.sum_z += z;
            self.sum_z2 += z * z;
        }
    }

    fn n_value(self) -> Option<f64> {
        (self.n > 0).then_some(self.n as f64)
    }

    fn value_if_nonempty(self, value: f64) -> Option<f64> {
        (self.n > 0).then_some(value).and_then(finite_value)
    }
}

impl GammaAdditiveStats {
    fn push(&mut self, num: f64, den: f64) {
        if num.is_finite() && den.is_finite() {
            self.n += 1;
            self.num += num;
            self.den += den;
        }
    }

    fn n_value(self) -> Option<f64> {
        (self.n > 0).then_some(self.n as f64)
    }

    fn value_if_nonempty(self, value: f64) -> Option<f64> {
        (self.n > 0).then_some(value).and_then(finite_value)
    }
}

fn five_minute_bars_by_stock(
    table: &Table,
    include_amount: bool,
) -> Result<BTreeMap<String, Vec<FiveMinuteBar>>> {
    let points = minute_points_by_stock(table, include_amount)?;
    let mut output = BTreeMap::new();
    for (ts_code, minutes) in points {
        let mut bars = Vec::new();
        for chunk in minutes.chunks(5).take(FIVE_MINUTES_PER_DAY) {
            if chunk.len() < 5 {
                continue;
            }
            let first = chunk[0].1;
            let last = chunk[chunk.len() - 1].1;
            let ret = simple_return(first.open, last.close);
            let amount_sum = chunk
                .iter()
                .filter_map(|(_, point)| point.amount)
                .sum::<f64>();
            let amihud = match (ret, finite_value(amount_sum).filter(|amount| *amount > EPS)) {
                (Some(ret), Some(amount)) => finite_value(ret.abs() / amount),
                _ => None,
            };
            bars.push(FiveMinuteBar { ret, amihud });
        }
        output.insert(ts_code, bars);
    }
    Ok(output)
}

fn composite_groups<'a, I>(
    data: &DataPool,
    trade_date: i32,
    ts_codes: I,
) -> Result<BTreeMap<String, String>>
where
    I: Iterator<Item = &'a String>,
{
    let sector_map = ClassificationMap::from_table(
        data.daily(DatasetId::StockCiClassification)?,
        ClassificationLevel::Sector,
    )?;
    let circ_mv = circ_mv_by_code(data.daily(DatasetId::StockDailyBasic)?, trade_date)?;
    let mut by_sector = BTreeMap::<String, Vec<(String, f64)>>::new();
    for ts_code in ts_codes {
        if is_bj_stock(ts_code) {
            continue;
        }
        let (Some(sector), Some(circ_mv)) = (
            sector_map
                .group_for(trade_date, ts_code)
                .map(str::to_string),
            circ_mv.get(ts_code).copied().flatten(),
        ) else {
            continue;
        };
        if circ_mv <= 0.0 || !circ_mv.is_finite() {
            continue;
        }
        by_sector
            .entry(sector)
            .or_default()
            .push((ts_code.clone(), circ_mv));
    }
    let mut output = BTreeMap::new();
    for (sector, mut stocks) in by_sector {
        stocks.sort_by(|left, right| {
            left.1
                .partial_cmp(&right.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| left.0.cmp(&right.0))
        });
        let len = stocks.len();
        if len == 0 {
            continue;
        }
        for (idx, (ts_code, _)) in stocks.into_iter().enumerate() {
            let bucket = (idx * 5) / len;
            output.insert(ts_code, format!("{sector}#{bucket}"));
        }
    }
    Ok(output)
}

fn circ_mv_by_code(table: &Table, trade_date: i32) -> Result<BTreeMap<String, Option<f64>>> {
    let trade_dates = table.required_i32("trade_date")?;
    let ts_codes = table.required_utf8("ts_code")?;
    let circ_mv = table.required_f64_cast("circ_mv")?;
    let mut output = BTreeMap::new();
    for idx in 0..table.len {
        if trade_dates[idx] != Some(trade_date) {
            continue;
        }
        let Some(ts_code) = ts_codes[idx].clone() else {
            continue;
        };
        output.insert(ts_code, clean_value(circ_mv[idx]));
    }
    Ok(output)
}

fn one_minute_raw_map() -> BTreeMap<&'static str, Vec<FactorValue>> {
    one_minute_raw_ids()
        .iter()
        .map(|raw_id| (*raw_id, Vec::new()))
        .collect()
}

fn operator_raw_map() -> BTreeMap<&'static str, Vec<FactorValue>> {
    operator_raw_ids()
        .iter()
        .map(|raw_id| (*raw_id, Vec::new()))
        .collect()
}

fn push_one_minute_values<'a, I>(
    trade_date: i32,
    requested: &BTreeSet<&'static str>,
    values: &BTreeMap<String, OneMinuteValues>,
    stock_keys: I,
    output: &mut BTreeMap<&'static str, Vec<FactorValue>>,
) where
    I: Iterator<Item = &'a String>,
{
    for ts_code in stock_keys {
        let key = FactorRowKey::Daily {
            trade_date,
            ts_code: ts_code.clone(),
        };
        let values = if is_bj_stock(ts_code) {
            OneMinuteValues::default()
        } else {
            values.get(ts_code).copied().unwrap_or_default()
        };
        push_raw_value(output, requested, AMIHUD_1MIN_RAW_ID, &key, values.amihud);
        push_raw_value(output, requested, LIQRESID_RAW_ID, &key, values.liqresid);
        push_raw_value(output, requested, APBETA1_RAW_ID, &key, values.apbeta1);
        push_raw_value(output, requested, APBETA4_RAW_ID, &key, values.apbeta4);
        push_raw_value(output, requested, GAMMA1_RAW_ID, &key, values.gamma1);
        push_raw_value(output, requested, GAMMA3_RAW_ID, &key, values.gamma3);
        push_raw_value(output, requested, GAMMA4_RAW_ID, &key, values.gamma4);
    }
}

fn push_operator_values<'a, I>(
    trade_date: i32,
    requested: &BTreeSet<&'static str>,
    values: &BTreeMap<String, OperatorValues>,
    stock_keys: I,
    output: &mut BTreeMap<&'static str, Vec<FactorValue>>,
) where
    I: Iterator<Item = &'a String>,
{
    for ts_code in stock_keys {
        let key = FactorRowKey::Daily {
            trade_date,
            ts_code: ts_code.clone(),
        };
        let values = if is_bj_stock(ts_code) {
            OperatorValues::default()
        } else {
            values.get(ts_code).copied().unwrap_or_default()
        };
        push_raw_value(
            output,
            requested,
            CLOSE_APBETA4_RAW_ID,
            &key,
            values.close_apbeta4,
        );
        push_raw_value(
            output,
            requested,
            RV_APBETA3_RAW_ID,
            &key,
            values.rv_apbeta3,
        );
        push_raw_value(
            output,
            requested,
            CLOSEVOLCORR_GAMMA3_RAW_ID,
            &key,
            values.closevolcorr_gamma3,
        );
        push_raw_value(
            output,
            requested,
            RSI_GAMMA3_RAW_ID,
            &key,
            values.rsi_gamma3,
        );
        push_raw_value(
            output,
            requested,
            VARVAR_GAMMA3_RAW_ID,
            &key,
            values.varvar_gamma3,
        );
        push_raw_value(
            output,
            requested,
            VOLRATIO_GAMMA4_RAW_ID,
            &key,
            values.volratio_gamma4,
        );
    }
}

fn series_from_crossday_values(
    trade_date: i32,
    requested: &BTreeSet<&'static str>,
    values: BTreeMap<String, CrossdayAdditiveValues>,
) -> Vec<IntradayDailyRawSeries> {
    let mut by_raw_id = crossday_5m_raw_ids()
        .iter()
        .map(|raw_id| (*raw_id, Vec::<FactorValue>::new()))
        .collect::<BTreeMap<_, _>>();
    for (ts_code, values) in values {
        let key = FactorRowKey::Daily {
            trade_date,
            ts_code,
        };
        let apbeta2 = values.apbeta2;
        push_raw_value(
            &mut by_raw_id,
            requested,
            APBETA2_5M_N_RAW_ID,
            &key,
            apbeta2.n_value(),
        );
        push_raw_value(
            &mut by_raw_id,
            requested,
            APBETA2_5M_SUM_X_RAW_ID,
            &key,
            apbeta2.value_if_nonempty(apbeta2.sum_x),
        );
        push_raw_value(
            &mut by_raw_id,
            requested,
            APBETA2_5M_SUM_Y_RAW_ID,
            &key,
            apbeta2.value_if_nonempty(apbeta2.sum_y),
        );
        push_raw_value(
            &mut by_raw_id,
            requested,
            APBETA2_5M_SUM_XY_RAW_ID,
            &key,
            apbeta2.value_if_nonempty(apbeta2.sum_xy),
        );
        push_raw_value(
            &mut by_raw_id,
            requested,
            APBETA2_5M_SUM_Z_RAW_ID,
            &key,
            apbeta2.value_if_nonempty(apbeta2.sum_z),
        );
        push_raw_value(
            &mut by_raw_id,
            requested,
            APBETA2_5M_SUM_Z2_RAW_ID,
            &key,
            apbeta2.value_if_nonempty(apbeta2.sum_z2),
        );
        let apbeta3 = values.apbeta3;
        push_raw_value(
            &mut by_raw_id,
            requested,
            APBETA3_5M_N_RAW_ID,
            &key,
            apbeta3.n_value(),
        );
        push_raw_value(
            &mut by_raw_id,
            requested,
            APBETA3_5M_SUM_X_RAW_ID,
            &key,
            apbeta3.value_if_nonempty(apbeta3.sum_x),
        );
        push_raw_value(
            &mut by_raw_id,
            requested,
            APBETA3_5M_SUM_Y_RAW_ID,
            &key,
            apbeta3.value_if_nonempty(apbeta3.sum_y),
        );
        push_raw_value(
            &mut by_raw_id,
            requested,
            APBETA3_5M_SUM_XY_RAW_ID,
            &key,
            apbeta3.value_if_nonempty(apbeta3.sum_xy),
        );
        push_raw_value(
            &mut by_raw_id,
            requested,
            APBETA3_5M_SUM_Z_RAW_ID,
            &key,
            apbeta3.value_if_nonempty(apbeta3.sum_z),
        );
        push_raw_value(
            &mut by_raw_id,
            requested,
            APBETA3_5M_SUM_Z2_RAW_ID,
            &key,
            apbeta3.value_if_nonempty(apbeta3.sum_z2),
        );
        let gamma2 = values.gamma2;
        push_raw_value(
            &mut by_raw_id,
            requested,
            GAMMA2_5M_N_RAW_ID,
            &key,
            gamma2.n_value(),
        );
        push_raw_value(
            &mut by_raw_id,
            requested,
            GAMMA2_5M_NUM_RAW_ID,
            &key,
            gamma2.value_if_nonempty(gamma2.num),
        );
        push_raw_value(
            &mut by_raw_id,
            requested,
            GAMMA2_5M_DEN_RAW_ID,
            &key,
            gamma2.value_if_nonempty(gamma2.den),
        );
    }
    crossday_5m_raw_ids()
        .iter()
        .filter(|raw_id| requested.contains(**raw_id))
        .map(|raw_id| IntradayDailyRawSeries {
            spec: raw_spec(raw_id, LiquidityFamily::Crossday5m),
            values: by_raw_id.remove(raw_id).unwrap_or_default(),
        })
        .collect()
}

fn one_minute_series_from_map(
    mut values: BTreeMap<&'static str, Vec<FactorValue>>,
    requested: &BTreeSet<&'static str>,
) -> Vec<IntradayDailyRawSeries> {
    one_minute_raw_ids()
        .iter()
        .filter(|raw_id| requested.contains(**raw_id))
        .map(|raw_id| IntradayDailyRawSeries {
            spec: raw_spec(raw_id, LiquidityFamily::OneMinute),
            values: values.remove(raw_id).unwrap_or_default(),
        })
        .collect()
}

fn operator_series_from_map(
    mut values: BTreeMap<&'static str, Vec<FactorValue>>,
    requested: &BTreeSet<&'static str>,
) -> Vec<IntradayDailyRawSeries> {
    operator_raw_ids()
        .iter()
        .filter(|raw_id| requested.contains(**raw_id))
        .map(|raw_id| IntradayDailyRawSeries {
            spec: raw_spec(raw_id, LiquidityFamily::Operator),
            values: values.remove(raw_id).unwrap_or_default(),
        })
        .collect()
}

fn push_raw_value(
    values: &mut BTreeMap<&'static str, Vec<FactorValue>>,
    requested: &BTreeSet<&'static str>,
    raw_id: &'static str,
    key: &FactorRowKey,
    value: Option<f64>,
) {
    if requested.contains(raw_id) {
        if let Some(column) = values.get_mut(raw_id) {
            column.push(FactorValue {
                key: key.clone(),
                value: value.and_then(finite_value),
            });
        }
    }
}

fn rolling_close_last(values: &[Option<f64>; 240]) -> [Option<f64>; 240] {
    let mut output = [None; 240];
    for idx in 4..240 {
        if values[idx - 4..=idx].iter().all(Option::is_some) {
            output[idx] = values[idx];
        }
    }
    output
}

fn rolling_rv(values: &[Option<f64>; 240]) -> [Option<f64>; 240] {
    let mut output = [None; 240];
    for idx in 4..240 {
        let finite = values[idx - 4..=idx]
            .iter()
            .filter_map(|value| *value)
            .collect::<Vec<_>>();
        if finite.len() == 5 {
            output[idx] = finite_value(finite.iter().map(|value| value * value).sum::<f64>());
        }
    }
    output
}

fn rolling_rsi(values: &[Option<f64>; 240]) -> [Option<f64>; 240] {
    let mut output = [None; 240];
    for idx in 4..240 {
        let finite = values[idx - 4..=idx]
            .iter()
            .filter_map(|value| *value)
            .collect::<Vec<_>>();
        if finite.len() != 5 {
            continue;
        }
        let gain = finite
            .iter()
            .filter(|value| **value > 0.0)
            .map(|value| *value)
            .sum::<f64>()
            / 5.0;
        let loss = finite
            .iter()
            .filter(|value| **value < 0.0)
            .map(|value| value.abs())
            .sum::<f64>()
            / 5.0;
        output[idx] = if gain <= EPS && loss <= EPS {
            Some(50.0)
        } else if loss <= EPS {
            Some(100.0)
        } else {
            finite_value(100.0 - 100.0 / (1.0 + gain / loss))
        };
    }
    output
}

fn rolling_varvar(values: &[Option<f64>; 240]) -> [Option<f64>; 240] {
    let mut std5 = [None; 240];
    for idx in 4..240 {
        std5[idx] = std_option_slice(&values[idx - 4..=idx]);
    }
    let mut output = [None; 240];
    for idx in 8..240 {
        output[idx] = std_option_slice(&std5[idx - 4..=idx]);
    }
    output
}

fn rolling_volratio(values: &[Option<f64>; 240], day_vol_sum: f64) -> [Option<f64>; 240] {
    let mut output = [None; 240];
    if day_vol_sum <= EPS || !day_vol_sum.is_finite() {
        return output;
    }
    for idx in 4..240 {
        let finite = values[idx - 4..=idx]
            .iter()
            .filter_map(|value| *value)
            .collect::<Vec<_>>();
        if finite.len() == 5 {
            output[idx] = finite_value(finite.iter().sum::<f64>() / day_vol_sum);
        }
    }
    output
}

fn rolling_corr(
    left: &[Option<f64>; 240],
    right: &[Option<f64>; 240],
    window: usize,
) -> [Option<f64>; 240] {
    let mut output = [None; 240];
    if window == 0 {
        return output;
    }
    for idx in window - 1..240 {
        output[idx] = corr_option_slices(
            &left[idx + 1 - window..=idx],
            &right[idx + 1 - window..=idx],
        );
    }
    output
}

fn pct_change_series(values: &[Option<f64>; 240]) -> [Option<f64>; 240] {
    let mut output = [None; 240];
    for idx in 1..240 {
        output[idx] = match (values[idx], values[idx - 1]) {
            (Some(current), Some(previous)) if previous.abs() > EPS => {
                finite_value((current - previous) / previous.abs())
            }
            _ => None,
        };
    }
    output
}

fn ar1_residuals(values: &[Option<f64>; 240]) -> [Option<f64>; 240] {
    let pairs = (1..240)
        .filter_map(|idx| Some((values[idx]?, values[idx - 1]?)))
        .collect::<Vec<_>>();
    let mut output = [None; 240];
    let Some((alpha, beta)) = ols_intercept_slope(&pairs) else {
        return output;
    };
    for idx in 1..240 {
        if let (Some(current), Some(lagged)) = (values[idx], values[idx - 1]) {
            output[idx] = finite_value(current - alpha - beta * lagged);
        }
    }
    output
}

fn ols_intercept_slope(pairs: &[(f64, f64)]) -> Option<(f64, f64)> {
    if pairs.len() < 2 {
        return None;
    }
    let n = pairs.len() as f64;
    let mean_y = pairs.iter().map(|(y, _)| *y).sum::<f64>() / n;
    let mean_x = pairs.iter().map(|(_, x)| *x).sum::<f64>() / n;
    let mut cov = 0.0;
    let mut var = 0.0;
    for (y, x) in pairs {
        cov += (x - mean_x) * (y - mean_y);
        var += (x - mean_x) * (x - mean_x);
    }
    if var <= EPS {
        return None;
    }
    let beta = cov / var;
    let alpha = mean_y - beta * mean_x;
    Some((alpha, beta)).filter(|(alpha, beta)| alpha.is_finite() && beta.is_finite())
}

fn simple_return(open: Option<f64>, close: Option<f64>) -> Option<f64> {
    match (open, close) {
        (Some(open), Some(close)) if open > EPS => finite_value(close / open - 1.0),
        _ => None,
    }
}

fn log_diff(current: Option<f64>, previous: Option<f64>) -> Option<f64> {
    match (current, previous) {
        (Some(current), Some(previous)) if current > 0.0 && previous > 0.0 => {
            finite_value(current.ln() - previous.ln())
        }
        _ => None,
    }
}

fn mean_option_slice(values: &[Option<f64>]) -> Option<f64> {
    let finite = values.iter().filter_map(|value| *value).collect::<Vec<_>>();
    if finite.is_empty() {
        return None;
    }
    finite_value(finite.iter().sum::<f64>() / finite.len() as f64)
}

fn std_option_slice(values: &[Option<f64>]) -> Option<f64> {
    let finite = values.iter().filter_map(|value| *value).collect::<Vec<_>>();
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
    finite_value(variance.sqrt())
}

fn corr_option_slices(left: &[Option<f64>], right: &[Option<f64>]) -> Option<f64> {
    let pairs = left
        .iter()
        .zip(right.iter())
        .filter_map(|(left, right)| Some(((*left)?, (*right)?)))
        .collect::<Vec<_>>();
    if pairs.len() < 2 {
        return None;
    }
    let mean_left = pairs.iter().map(|(left, _)| *left).sum::<f64>() / pairs.len() as f64;
    let mean_right = pairs.iter().map(|(_, right)| *right).sum::<f64>() / pairs.len() as f64;
    let mut cov: f64 = 0.0;
    let mut var_left: f64 = 0.0;
    let mut var_right: f64 = 0.0;
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

fn quantile_sorted(mut values: Vec<f64>, q: f64) -> Option<f64> {
    values.retain(|value| value.is_finite());
    if values.is_empty() {
        return None;
    }
    values.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
    if values.len() == 1 {
        return Some(values[0]);
    }
    let position = q.clamp(0.0, 1.0) * (values.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    if lower == upper {
        return Some(values[lower]);
    }
    let weight = position - lower as f64;
    finite_value(values[lower] * (1.0 - weight) + values[upper] * weight)
}

fn safe_div_value(numerator: f64, denominator: f64) -> Option<f64> {
    if denominator.abs() <= EPS || !numerator.is_finite() || !denominator.is_finite() {
        return None;
    }
    finite_value(numerator / denominator)
}

fn clean_value(value: Option<f64>) -> Option<f64> {
    value.and_then(finite_value)
}

fn finite_value(value: f64) -> Option<f64> {
    value.is_finite().then_some(value)
}

fn tags() -> Vec<String> {
    [
        "price_volume",
        "liquidity",
        "amihud",
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_specs_use_daily_window_for_stateless_liquidity_families() {
        let one_minute = raw_spec(AMIHUD_1MIN_RAW_ID, LiquidityFamily::OneMinute);
        let operator = raw_spec(CLOSE_APBETA4_RAW_ID, LiquidityFamily::Operator);
        let crossday = raw_spec(APBETA2_5M_N_RAW_ID, LiquidityFamily::Crossday5m);

        assert_eq!(one_minute.window_days, 1);
        assert_eq!(operator.window_days, 1);
        assert_eq!(crossday.window_days, 2);
    }

    #[test]
    fn amihud_uses_close_over_open_return_and_amount() {
        let points = (0..240)
            .map(|idx| {
                (
                    format!("09:{idx:02}:00"),
                    MinutePoint {
                        open: Some(100.0),
                        close: Some(101.0),
                        amount: Some(10.0),
                        vol: Some(1.0),
                    },
                )
            })
            .collect::<Vec<_>>();
        let series = one_minute_series(&points);
        assert!((series.amihud[0].unwrap() - 0.001).abs() < 1e-12);
    }

    #[test]
    fn apbeta_denominator_uses_centered_market_liquidity_spread() {
        let mut left = [None; 240];
        let mut right = [None; 240];
        let mut liq = [None; 240];
        left[0] = Some(1.0);
        left[1] = Some(3.0);
        right[0] = Some(2.0);
        right[1] = Some(4.0);
        liq[0] = Some(1.0);
        liq[1] = Some(1.0);
        assert!(covariance_ratio(&left, &right, &right, &liq).is_some());
    }

    #[test]
    fn crossday_moment_additive_stats_restore_direct_formula() {
        let rows = [(1.0, 4.0, 2.0), (3.0, 8.0, 5.0), (5.0, 7.0, 4.0)];
        let mut stats = MomentAdditiveStats::default();
        for (x, y, z) in rows {
            stats.push(x, y, z);
        }
        let restored = moment_ratio_from_sums(
            stats.n_value(),
            stats.value_if_nonempty(stats.sum_x),
            stats.value_if_nonempty(stats.sum_y),
            stats.value_if_nonempty(stats.sum_xy),
            stats.value_if_nonempty(stats.sum_z),
            stats.value_if_nonempty(stats.sum_z2),
        )
        .unwrap();
        let n = rows.len() as f64;
        let sum_x = rows.iter().map(|(x, _, _)| *x).sum::<f64>();
        let sum_y = rows.iter().map(|(_, y, _)| *y).sum::<f64>();
        let sum_xy = rows.iter().map(|(x, y, _)| x * y).sum::<f64>();
        let sum_z = rows.iter().map(|(_, _, z)| *z).sum::<f64>();
        let sum_z2 = rows.iter().map(|(_, _, z)| z * z).sum::<f64>();
        let expected = (sum_xy - sum_x * sum_y / n) / (sum_z2 - sum_z * sum_z / n);
        assert!((restored - expected).abs() < 1e-12);
    }

    #[test]
    fn crossday_gamma_additive_stats_restore_ratio() {
        let mut stats = GammaAdditiveStats::default();
        stats.push(2.0, 4.0);
        stats.push(3.0, 6.0);
        let restored = gamma_ratio_from_sums(
            stats.n_value(),
            stats.value_if_nonempty(stats.num),
            stats.value_if_nonempty(stats.den),
        )
        .unwrap();
        assert!((restored - 0.5).abs() < 1e-12);
    }

    #[test]
    fn pct_change_uses_absolute_previous_value() {
        let mut values = [None; 240];
        values[0] = Some(-2.0);
        values[1] = Some(-1.0);
        let changes = pct_change_series(&values);
        assert!((changes[1].unwrap() - 0.5).abs() < 1e-12);
    }
}

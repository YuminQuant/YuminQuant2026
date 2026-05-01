use std::collections::HashMap;

use crate::barra::BarraExposure;
use crate::core::{
    AssetClass, BarraSeries, BarraSpec, DataRequest, DatasetId, FactorContext, Frequency, Lookback,
};
use crate::data::{DataPool, Table};
use crate::error::{err, Result};
use crate::factor::common::{ClassificationLevel, ClassificationMap, DailyPanel, PanelColumn};
use crate::operators::{
    cs_winsorize_quantile, cs_zscore, ts_ew_regression_alpha_beta_residual_sigma, ts_ew_sum,
};

pub struct StockDailyBarraCne6Momentum;

const MODEL: &str = "CNE6";
const VERSION: &str = "0.1.0";
const MARKET_INDEX: &str = "000300.SH";
const STREV_WINDOW: usize = 21;
const STREV_HALF_LIFE: f64 = 5.0;
const TRADING_DAYS_PER_MONTH: usize = 21;
const TRADING_DAYS_PER_YEAR: usize = 252;
const SEASONALITY_YEARS: usize = 5;
const INDMOM_WINDOW: usize = 126;
const INDMOM_HALF_LIFE: f64 = 21.0;
const RSTR_WINDOW: usize = 252;
const RSTR_HALF_LIFE: f64 = 126.0;
const RSTR_LAG: usize = 11;
const RSTR_AVG_WINDOW: usize = 11;
const HALPHA_WINDOW: usize = 252;
const HALPHA_HALF_LIFE: f64 = 63.0;
const MAX_LOOKBACK: usize = TRADING_DAYS_PER_YEAR * SEASONALITY_YEARS + TRADING_DAYS_PER_MONTH;

pub fn create() -> Box<dyn BarraExposure> {
    Box::new(StockDailyBarraCne6Momentum)
}

impl BarraExposure for StockDailyBarraCne6Momentum {
    fn family_id(&self) -> &'static str {
        "MOMENTUM"
    }

    fn specs(&self) -> Vec<BarraSpec> {
        vec![
            momentum_spec(
                "Historical_Alpha",
                &["HALPHA"],
                "CNE6 historical alpha",
                "EW 252-day CAPM intercept against CSI 300 with half-life 63.",
            ),
            momentum_spec(
                "Relative_Strength",
                &["RSTR"],
                "CNE6 relative strength",
                "Lagged 11-day average of the 252-day EW log-return strength.",
            ),
            momentum_spec(
                "Short_Term_Reversal",
                &["STREV"],
                "CNE6 short-term reversal",
                "Past 21 trading days of EW log returns, excluding the current day.",
            ),
            momentum_spec(
                "Seasonality",
                &["SEASON"],
                "CNE6 seasonality",
                "Average of the past five trading-year approximate same-month returns.",
            ),
            momentum_spec(
                "Industry_Momentum",
                &["INDMOM"],
                "CNE6 industry momentum",
                "Difference between stock relative strength and CITIC L1 industry relative strength.",
            ),
            momentum_spec(
                "Momentum",
                &[] ,
                "CNE6 Momentum secondary exposure",
                "Equal-weight composite of Historical_Alpha and Relative_Strength.",
            ),
            momentum_spec(
                "MOMENTUM",
                &[],
                "CNE6 MOMENTUM style exposure",
                "Equal-weight composite of Short_Term_Reversal, Seasonality, Industry_Momentum, and Momentum.",
            ),
        ]
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<Vec<BarraSeries>> {
        let stock_panel = data.daily_panel(DatasetId::StockDailyPv)?;
        let index_panel = data.index_daily_panel(MARKET_INDEX)?;
        let ci_map = ClassificationMap::from_table(
            data.daily(DatasetId::StockCiClassification)?,
            ClassificationLevel::Sector,
        )?;

        let close = stock_panel.column("close")?;
        let pre_close = stock_panel.column("pre_close")?;
        let stock_returns = close.zip_binary(&pre_close, arithmetic_return)?;
        let stock_log_returns = close.zip_binary(&pre_close, log_return)?;

        let index_returns = index_panel
            .column("close")?
            .zip_binary(&index_panel.column("pre_close")?, arithmetic_return)?;
        let market_returns = expand_index_column(stock_panel, index_panel, &index_returns)?;

        let short_term_reversal = stock_log_returns
            .ts(short_term_reversal_raw)?
            .cs(standardize_cross_section)?;
        let seasonality = stock_log_returns
            .ts(seasonality_raw)?
            .cs(standardize_cross_section)?;
        let relative_strength = stock_log_returns
            .ts(relative_strength_raw)?
            .cs(standardize_cross_section)?;
        let historical_alpha = stock_returns
            .ts_binary(&market_returns, historical_alpha_raw)?
            .cs(standardize_cross_section)?;

        let industry_rs = stock_log_returns.ts(industry_relative_strength_raw)?;
        let circ_mv = align_daily_table_column(
            stock_panel,
            data.daily(DatasetId::StockDailyBasic)?,
            "circ_mv",
        )?;
        let sqrt_mv = industry_rs.zip_binary(&circ_mv, |rs, mv| match (clean(rs), clean(mv)) {
            (Some(_), Some(mv)) if mv > 0.0 => Some(mv.sqrt()),
            _ => None,
        })?;
        let industry_momentum =
            industry_momentum_column(stock_panel, &industry_rs, &sqrt_mv, &ci_map)?
                .cs(standardize_cross_section)?;

        let momentum_raw = historical_alpha.zip_binary(&relative_strength, average_two_values)?;
        let momentum = momentum_raw.cs(cs_zscore)?;

        let style_raw = short_term_reversal.zip_quaternary(
            &seasonality,
            &industry_momentum,
            &momentum,
            average_four_values,
        )?;
        let style = style_raw.cs(cs_zscore)?;

        let specs = self.specs();
        Ok(vec![
            historical_alpha.to_barra_series(specs[0].clone()),
            relative_strength.to_barra_series(specs[1].clone()),
            short_term_reversal.to_barra_series(specs[2].clone()),
            seasonality.to_barra_series(specs[3].clone()),
            industry_momentum.to_barra_series(specs[4].clone()),
            momentum.to_barra_series(specs[5].clone()),
            style.to_barra_series(specs[6].clone()),
        ])
    }
}

fn momentum_spec(id: &str, aliases: &[&str], name: &str, description: &str) -> BarraSpec {
    BarraSpec {
        id: id.to_string(),
        aliases: aliases.iter().map(|value| value.to_string()).collect(),
        name: name.to_string(),
        model: MODEL.to_string(),
        asset_class: AssetClass::Stock,
        frequency: Frequency::Daily,
        version: VERSION.to_string(),
        tags: ["barra", "cne6", "style", "momentum", "daily", "stock"]
            .iter()
            .map(|value| value.to_string())
            .collect(),
        description: description.to_string(),
        dependencies: vec![
            DataRequest::new(DatasetId::StockDailyPv, &["close", "pre_close"]),
            DataRequest::new(DatasetId::StockDailyBasic, &["circ_mv"]),
            DataRequest::index_daily(MARKET_INDEX, &["close", "pre_close"]),
            DataRequest::new(DatasetId::StockCiClassification, &["l1_code"]),
        ],
        lookback: Lookback {
            trading_days: MAX_LOOKBACK,
        },
    }
}

fn arithmetic_return(close: Option<f64>, pre_close: Option<f64>) -> Option<f64> {
    match (clean(close), clean(pre_close)) {
        (Some(close), Some(pre_close)) if pre_close.abs() > f64::EPSILON => {
            Some(close / pre_close - 1.0)
        }
        _ => None,
    }
}

fn log_return(close: Option<f64>, pre_close: Option<f64>) -> Option<f64> {
    match (clean(close), clean(pre_close)) {
        (Some(close), Some(pre_close)) if close > 0.0 && pre_close > 0.0 => {
            Some((close / pre_close).ln())
        }
        _ => None,
    }
}

fn short_term_reversal_raw(values: &[Option<f64>]) -> Vec<Option<f64>> {
    ts_ew_sum(
        &lag_values(values, 1),
        STREV_WINDOW,
        STREV_WINDOW,
        STREV_HALF_LIFE,
    )
}

fn seasonality_raw(values: &[Option<f64>]) -> Vec<Option<f64>> {
    let mut output = vec![None; values.len()];
    for idx in 0..values.len() {
        let mut total = 0.0;
        let mut valid = true;
        for year in 1..=SEASONALITY_YEARS {
            let Some(anchor) = idx.checked_sub(TRADING_DAYS_PER_YEAR * year) else {
                valid = false;
                break;
            };
            let Some(start) = anchor.checked_sub(TRADING_DAYS_PER_MONTH - 1) else {
                valid = false;
                break;
            };
            let mut log_sum = 0.0;
            for value in &values[start..=anchor] {
                let Some(value) = clean(*value) else {
                    valid = false;
                    break;
                };
                log_sum += value;
            }
            if !valid {
                break;
            }
            total += log_sum.exp_m1();
        }
        if valid {
            output[idx] = Some(total / SEASONALITY_YEARS as f64);
        }
    }
    output
}

fn industry_relative_strength_raw(values: &[Option<f64>]) -> Vec<Option<f64>> {
    ts_ew_sum(values, INDMOM_WINDOW, INDMOM_WINDOW, INDMOM_HALF_LIFE)
}

fn relative_strength_raw(values: &[Option<f64>]) -> Vec<Option<f64>> {
    let raw = ts_ew_sum(values, RSTR_WINDOW, RSTR_WINDOW, RSTR_HALF_LIFE);
    lagged_mean(&raw, RSTR_LAG, RSTR_AVG_WINDOW)
}

fn historical_alpha_raw(
    stock_returns: &[Option<f64>],
    market_returns: &[Option<f64>],
) -> Vec<Option<f64>> {
    let (alpha, _beta, _sigma) = ts_ew_regression_alpha_beta_residual_sigma(
        stock_returns,
        market_returns,
        HALPHA_WINDOW,
        HALPHA_WINDOW,
        HALPHA_HALF_LIFE,
    );
    alpha
}

fn industry_momentum_column(
    panel: &DailyPanel,
    rs: &PanelColumn,
    sqrt_mv: &PanelColumn,
    ci_map: &ClassificationMap,
) -> Result<PanelColumn> {
    let instrument_count = panel.instruments().len();
    let mut values = vec![None; panel.shape_len()];

    for (date_idx, trade_date) in panel.dates().iter().copied().enumerate() {
        let groups = ci_map.groups_for(trade_date, panel.instruments());
        let mut rs_section = Vec::with_capacity(instrument_count);
        let mut sqrt_mv_section = Vec::with_capacity(instrument_count);
        for instrument_idx in 0..instrument_count {
            let offset = date_idx * instrument_count + instrument_idx;
            rs_section.push(rs.values()[offset]);
            sqrt_mv_section.push(sqrt_mv.values()[offset]);
        }
        let computed = industry_momentum_cross_section(&rs_section, &sqrt_mv_section, &groups);
        for (instrument_idx, value) in computed.into_iter().enumerate() {
            values[date_idx * instrument_count + instrument_idx] = value;
        }
    }

    panel.column_from_values(values)
}

fn industry_momentum_cross_section(
    rs: &[Option<f64>],
    sqrt_mv: &[Option<f64>],
    groups: &[Option<String>],
) -> Vec<Option<f64>> {
    let mut group_denominator = HashMap::<&str, f64>::new();
    let mut group_weighted_rs = HashMap::<&str, f64>::new();

    for ((rs, sqrt_mv), group) in rs.iter().zip(sqrt_mv).zip(groups) {
        let (Some(rs), Some(sqrt_mv), Some(group)) =
            (clean(*rs), clean(*sqrt_mv), group.as_deref())
        else {
            continue;
        };
        if sqrt_mv <= 0.0 {
            continue;
        }
        *group_denominator.entry(group).or_default() += sqrt_mv;
        *group_weighted_rs.entry(group).or_default() += sqrt_mv * rs;
    }

    rs.iter()
        .zip(sqrt_mv)
        .zip(groups)
        .map(|((rs, sqrt_mv), group)| {
            let (Some(rs), Some(sqrt_mv), Some(group)) =
                (clean(*rs), clean(*sqrt_mv), group.as_deref())
            else {
                return None;
            };
            let denominator = *group_denominator.get(group)?;
            if denominator <= f64::EPSILON {
                return None;
            }
            let stock_weight = sqrt_mv / denominator;
            let industry_rs =
                group_weighted_rs.get(group).copied().unwrap_or_default() / denominator;
            Some(-(stock_weight * rs - industry_rs))
        })
        .collect()
}

fn expand_index_column(
    stock_panel: &DailyPanel,
    index_panel: &DailyPanel,
    index_column: &PanelColumn,
) -> Result<PanelColumn> {
    let index_instrument_count = index_panel.instruments().len();
    if index_instrument_count == 0 {
        return Err(err("index daily panel has no instruments"));
    }
    let mut by_date = HashMap::new();
    for (date_idx, trade_date) in index_panel.dates().iter().enumerate() {
        by_date.insert(
            *trade_date,
            index_column.values()[date_idx * index_instrument_count],
        );
    }

    let mut values = Vec::with_capacity(stock_panel.shape_len());
    for trade_date in stock_panel.dates() {
        let value = by_date.get(trade_date).copied().unwrap_or(None);
        for _ in stock_panel.instruments() {
            values.push(value);
        }
    }
    stock_panel.column_from_values(values)
}

fn align_daily_table_column(
    panel: &DailyPanel,
    table: &Table,
    column: &str,
) -> Result<PanelColumn> {
    let trade_dates = table.required_i32("trade_date")?;
    let ts_codes = table.required_utf8("ts_code")?;
    let column_values = table.required_f64_cast(column)?;
    let mut by_key = HashMap::new();
    for idx in 0..table.len {
        let (Some(trade_date), Some(ts_code)) = (trade_dates[idx], ts_codes[idx].as_deref()) else {
            continue;
        };
        by_key.insert((trade_date, ts_code), column_values[idx]);
    }

    let mut values = Vec::with_capacity(panel.shape_len());
    for trade_date in panel.dates() {
        for ts_code in panel.instruments() {
            values.push(
                by_key
                    .get(&(*trade_date, ts_code.as_str()))
                    .copied()
                    .unwrap_or(None),
            );
        }
    }
    panel.column_from_values(values)
}

fn lag_values(values: &[Option<f64>], lag: usize) -> Vec<Option<f64>> {
    let mut output = vec![None; values.len()];
    for idx in lag..values.len() {
        output[idx] = values[idx - lag];
    }
    output
}

fn lagged_mean(values: &[Option<f64>], lag: usize, window: usize) -> Vec<Option<f64>> {
    let mut output = vec![None; values.len()];
    if lag == 0 || window == 0 {
        return output;
    }
    for idx in 0..values.len() {
        let Some(end) = idx.checked_sub(lag) else {
            continue;
        };
        let Some(start) = end.checked_sub(window - 1) else {
            continue;
        };
        let mut sum = 0.0;
        let mut count = 0;
        for value in &values[start..=end] {
            let Some(value) = clean(*value) else {
                count = 0;
                break;
            };
            sum += value;
            count += 1;
        }
        if count == window {
            output[idx] = Some(sum / window as f64);
        }
    }
    output
}

fn average_two_values(left: Option<f64>, right: Option<f64>) -> Option<f64> {
    match (clean(left), clean(right)) {
        (Some(left), Some(right)) => Some((left + right) / 2.0),
        _ => None,
    }
}

fn average_four_values(
    first: Option<f64>,
    second: Option<f64>,
    third: Option<f64>,
    fourth: Option<f64>,
) -> Option<f64> {
    match (clean(first), clean(second), clean(third), clean(fourth)) {
        (Some(first), Some(second), Some(third), Some(fourth)) => {
            Some((first + second + third + fourth) / 4.0)
        }
        _ => None,
    }
}

fn standardize_cross_section(values: &[Option<f64>]) -> Vec<Option<f64>> {
    cs_zscore(&cs_winsorize_quantile(values, 0.01, 0.99))
}

fn clean(value: Option<f64>) -> Option<f64> {
    value.filter(|value| !value.is_nan())
}

#[cfg(test)]
mod tests {
    use crate::barra::BarraExposure;

    use super::{
        industry_momentum_cross_section, lagged_mean, seasonality_raw, short_term_reversal_raw,
        StockDailyBarraCne6Momentum, SEASONALITY_YEARS, STREV_HALF_LIFE, TRADING_DAYS_PER_MONTH,
        TRADING_DAYS_PER_YEAR,
    };

    fn assert_close(actual: f64, expected: f64) {
        assert!((actual - expected).abs() < 1e-10, "{actual} != {expected}");
    }

    #[test]
    fn cne6_momentum_family_registers_all_levels() {
        let exposure = StockDailyBarraCne6Momentum;
        let specs = exposure.specs();
        let ids = specs
            .iter()
            .map(|spec| spec.id.as_str())
            .collect::<Vec<_>>();

        assert_eq!(
            ids,
            vec![
                "Historical_Alpha",
                "Relative_Strength",
                "Short_Term_Reversal",
                "Seasonality",
                "Industry_Momentum",
                "Momentum",
                "MOMENTUM"
            ]
        );
    }

    #[test]
    fn short_term_reversal_excludes_current_return() {
        let mut values = vec![Some(1.0); 22];
        values[21] = Some(100.0);
        let output = short_term_reversal_raw(&values);
        let decay = 0.5_f64.powf(1.0 / STREV_HALF_LIFE);
        let expected = (0..21).map(|lag| decay.powi(lag)).sum::<f64>();

        assert_close(output[21].unwrap(), expected);
    }

    #[test]
    fn seasonality_uses_five_trading_year_month_windows() {
        let len = TRADING_DAYS_PER_YEAR * SEASONALITY_YEARS + TRADING_DAYS_PER_MONTH;
        let values = vec![Some(0.01_f64.ln_1p()); len];
        let output = seasonality_raw(&values);
        let expected = (TRADING_DAYS_PER_MONTH as f64 * 0.01_f64.ln_1p()).exp_m1();

        assert_eq!(output[len - 2], None);
        assert_close(output[len - 1].unwrap(), expected);
    }

    #[test]
    fn industry_momentum_uses_group_normalized_sqrt_market_value_weights() {
        let rs = vec![Some(1.0), Some(3.0), Some(10.0), None];
        let sqrt_mv = vec![Some(1.0), Some(3.0), Some(2.0), Some(1.0)];
        let groups = vec![
            Some("bank".to_string()),
            Some("bank".to_string()),
            Some("tech".to_string()),
            Some("tech".to_string()),
        ];
        let output = industry_momentum_cross_section(&rs, &sqrt_mv, &groups);

        assert_close(output[0].unwrap(), 2.25);
        assert_close(output[1].unwrap(), 0.25);
        assert_close(output[2].unwrap(), 0.0);
        assert_eq!(output[3], None);
    }

    #[test]
    fn lagged_mean_uses_lagged_window_not_current_values() {
        let values = (0..284).map(|value| Some(value as f64)).collect::<Vec<_>>();
        let output = lagged_mean(&values, 273, 11);

        assert_eq!(output[282], None);
        assert_close(output[283].unwrap(), 5.0);
    }
}

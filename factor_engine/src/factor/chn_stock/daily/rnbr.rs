use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::stock_daily_ops::is_bj_stock;
use crate::factor::common::vector::clean;
use crate::factor::common::{DailyPanel, PanelColumn};
use crate::factor::Factor;
use crate::operators::{cs_regression_residual, cs_zscore, ts_mean};

const VERSION: &str = "0.1.0";
const RET_WINDOW: usize = 10;
const TURNOVER_WINDOW: usize = 5;
const NEIGHBOR_COUNT: usize = 6;
const BALANCED_SIDE_COUNT: usize = 3;

pub struct StockDailyRnbr;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyRnbr)
}

impl Factor for StockDailyRnbr {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "rnbr".to_string(),
            aliases: vec![
                "RNBR".to_string(),
                "RNBR_ret".to_string(),
                "RNBR_tov".to_string(),
            ],
            name: "rnbr".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: tags(),
            description: "ZSZQ code-neighbor spillover factor: residualized six-neighbor 10-day average return spillover and six-neighbor 5-day turnover spillover, z-scored and equally combined without Barra or sector neutralization.".to_string(),
            dependencies: vec![
                DataRequest::new(DatasetId::StockDailyPv, &["close", "pre_close"]),
                DataRequest::new(DatasetId::StockDailyBasic, &["turnover_rate_f"]),
            ],
            intraday_raw_dependencies: Vec::new(),
            lookback: Lookback {
                trading_days: RET_WINDOW - 1,
            },
        }
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let panel = data.daily_panel(DatasetId::StockDailyPv)?;
        let close = panel.column("close")?;
        let pre_close = panel.column("pre_close")?;
        let daily_return = close.zip_binary(&pre_close, simple_return)?;
        let own_ret10 = daily_return.ts(|series| ts_mean(series, RET_WINDOW, RET_WINDOW))?;
        let nbr_ret10 = neighbor_average_by_date(&own_ret10, &daily_return, panel)?;
        let rnbr_ret = nbr_ret10.cs_binary(&own_ret10, cs_regression_residual)?;

        let turnover =
            panel.column_from_table(data.daily(DatasetId::StockDailyBasic)?, "turnover_rate_f")?;
        let own_tov5 = turnover.ts(|series| ts_mean(series, TURNOVER_WINDOW, TURNOVER_WINDOW))?;
        let nbr_tov5 = neighbor_average_by_date(&own_tov5, &turnover, panel)?;
        let rnbr_tov = nbr_tov5.cs_binary(&own_tov5, cs_regression_residual)?;

        let factor = average_pair(&rnbr_ret.cs(cs_zscore)?, &rnbr_tov.cs(cs_zscore)?)?;
        Ok(factor.to_factor_series(self.spec()))
    }
}

fn tags() -> Vec<String> {
    [
        "ZSZQ",
        "neighbor",
        "spillover",
        "return",
        "turnover",
        "residual",
        "zscore",
        "daily",
    ]
    .iter()
    .map(|value| value.to_string())
    .collect()
}

fn simple_return(close: Option<f64>, pre_close: Option<f64>) -> Option<f64> {
    match (clean(close), clean(pre_close)) {
        (Some(close), Some(pre_close)) if pre_close.abs() > f64::EPSILON => {
            let value = close / pre_close - 1.0;
            value.is_finite().then_some(value)
        }
        _ => None,
    }
}

fn neighbor_average_by_date(
    values: &PanelColumn,
    presence: &PanelColumn,
    panel: &DailyPanel,
) -> Result<PanelColumn> {
    let instrument_count = panel.instruments().len();
    let stock_codes = panel
        .instruments()
        .iter()
        .map(|ts_code| numeric_stock_code(ts_code))
        .collect::<Vec<_>>();
    let mut output = vec![None; panel.shape_len()];

    for date_idx in 0..panel.dates().len() {
        let offset = date_idx * instrument_count;
        let mut stocks = Vec::with_capacity(instrument_count);
        for instrument_idx in 0..instrument_count {
            let panel_idx = offset + instrument_idx;
            if presence.values()[panel_idx].is_none() {
                continue;
            }
            if let Some(code) = stock_codes[instrument_idx] {
                stocks.push((code, instrument_idx));
            }
        }
        stocks.sort_by_key(|(code, instrument_idx)| (*code, *instrument_idx));

        for position in 0..stocks.len() {
            let instrument_idx = stocks[position].1;
            let mut sum = 0.0;
            let mut count = 0usize;
            for neighbor_position in neighbor_indices(position, stocks.len()) {
                let neighbor_instrument_idx = stocks[neighbor_position].1;
                if let Some(value) = clean(values.values()[offset + neighbor_instrument_idx]) {
                    sum += value;
                    count += 1;
                }
            }
            if count > 0 {
                let value = sum / count as f64;
                if value.is_finite() {
                    output[offset + instrument_idx] = Some(value);
                }
            }
        }
    }

    panel.column_from_values(output)
}

fn numeric_stock_code(ts_code: &str) -> Option<u32> {
    if is_bj_stock(ts_code) {
        return None;
    }
    let prefix = ts_code.get(0..6)?;
    if !prefix.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    prefix.parse::<u32>().ok()
}

fn neighbor_indices(position: usize, len: usize) -> Vec<usize> {
    if len <= 1 || position >= len {
        return Vec::new();
    }
    let window = (NEIGHBOR_COUNT + 1).min(len);
    let (start, end) = if len <= NEIGHBOR_COUNT + 1 {
        (0, len)
    } else if position < BALANCED_SIDE_COUNT {
        (0, window)
    } else if position + BALANCED_SIDE_COUNT >= len {
        (len - window, len)
    } else {
        (
            position - BALANCED_SIDE_COUNT,
            position + BALANCED_SIDE_COUNT + 1,
        )
    };
    (start..end).filter(|idx| *idx != position).collect()
}

fn average_pair(left: &PanelColumn, right: &PanelColumn) -> Result<PanelColumn> {
    left.zip_binary(right, |left, right| match (clean(left), clean(right)) {
        (Some(left), Some(right)) => {
            let value = (left + right) / 2.0;
            value.is_finite().then_some(value)
        }
        _ => None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rnbr_numeric_stock_code_excludes_bj_and_invalid_codes() {
        assert_eq!(numeric_stock_code("000001.SZ"), Some(1));
        assert_eq!(numeric_stock_code("600000.SH"), Some(600000));
        assert_eq!(numeric_stock_code("920001.BJ"), None);
        assert_eq!(numeric_stock_code("ABC001.SZ"), None);
    }

    #[test]
    fn rnbr_neighbor_indices_use_right_side_at_left_boundary() {
        assert_eq!(neighbor_indices(0, 10), vec![1, 2, 3, 4, 5, 6]);
        assert_eq!(neighbor_indices(1, 10), vec![0, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn rnbr_neighbor_indices_are_balanced_in_middle() {
        assert_eq!(neighbor_indices(4, 10), vec![1, 2, 3, 5, 6, 7]);
    }

    #[test]
    fn rnbr_neighbor_indices_use_left_side_at_right_boundary() {
        assert_eq!(neighbor_indices(9, 10), vec![3, 4, 5, 6, 7, 8]);
        assert_eq!(neighbor_indices(8, 10), vec![3, 4, 5, 6, 7, 9]);
    }

    #[test]
    fn rnbr_neighbor_indices_use_available_stocks_when_universe_is_small() {
        assert_eq!(neighbor_indices(1, 3), vec![0, 2]);
    }

    #[test]
    fn rnbr_simple_return_rejects_invalid_denominator() {
        let value = simple_return(Some(11.0), Some(10.0)).expect("return");
        assert!((value - 0.1).abs() < 1e-12);
        assert_eq!(simple_return(Some(11.0), Some(0.0)), None);
    }

    #[test]
    fn rnbr_spec_has_zszq_tag_and_no_neutralization_dependency() {
        let spec = StockDailyRnbr.spec();
        assert_eq!(spec.id, "rnbr");
        assert!(spec.tags.iter().any(|tag| tag == "ZSZQ"));
        assert!(!spec.tags.iter().any(|tag| tag == "neutralize"));
        assert_eq!(spec.lookback.trading_days, RET_WINDOW - 1);
    }
}

use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    IntradayDailyRawRequest, Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::chn_stock::daily::rpv::OPEN_AUCTION_TURNOVER_RAW_ID;
use crate::factor::common::{vector::clean, PanelColumn};
use crate::factor::Factor;
use crate::operators::{cs_zscore, ts_delay};

const VERSION: &str = "0.1.0";
const WINDOW: usize = 20;
const GROUP_COUNT: usize = 5;
const GROUP_SIZE: usize = WINDOW / GROUP_COUNT;

pub struct StockDailyNewMomentum;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyNewMomentum)
}

impl Factor for StockDailyNewMomentum {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "new_momentum".to_string(),
            aliases: vec!["NEW_Momentum".to_string(), "NEW_MOMENTUM".to_string()],
            name: "NEW_Momentum".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: [
                "price_volume",
                "return",
                "momentum",
                "intraday_return",
                "overnight_return",
                "turnover",
                "daily",
                "DWZQ",
            ]
            .iter()
            .map(|value| value.to_string())
            .collect(),
            description: "New momentum factor combining turnover-sorted intraday and prior-turnover-sorted overnight return components.".to_string(),
            dependencies: vec![
                DataRequest::new(DatasetId::StockDailyPv, &["open", "close", "pre_close"]),
                DataRequest::new(DatasetId::StockDailyBasic, &["turnover_rate_f"]),
            ],
            intraday_raw_dependencies: vec![IntradayDailyRawRequest::new(
                OPEN_AUCTION_TURNOVER_RAW_ID,
                WINDOW,
            )],
            lookback: Lookback {
                trading_days: WINDOW,
            },
        }
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let panel = data.intraday_daily_raw_panel(OPEN_AUCTION_TURNOVER_RAW_ID)?;
        let pv_table = data.daily(DatasetId::StockDailyPv)?;
        let basic_table = data.daily(DatasetId::StockDailyBasic)?;
        let open = panel.column_from_table(pv_table, "open")?;
        let close = panel.column_from_table(pv_table, "close")?;
        let pre_close = panel.column_from_table(pv_table, "pre_close")?;
        let full_turnover = panel
            .column_from_table(basic_table, "turnover_rate_f")?
            .map_values(percent_to_decimal);
        let open_auction_turnover = panel.column(OPEN_AUCTION_TURNOVER_RAW_ID)?;
        let intraday_turnover =
            full_turnover.zip_binary(&open_auction_turnover, intraday_turnover)?;
        let prior_turnover = full_turnover.ts(|values| ts_delay(values, 1))?;

        let intraday_return = close.zip_binary(&open, ret)?;
        let overnight_return = open.zip_binary(&pre_close, ret)?;

        let intraday_part1 = rolling_group_mean(&intraday_return, &intraday_turnover, 0)?;
        let intraday_part5 = rolling_group_mean(&intraday_return, &intraday_turnover, 4)?;
        let new_intraday = weighted_pair(
            &intraday_part1.cs(cs_zscore)?,
            &intraday_part5.cs(cs_zscore)?,
            -1.0,
            1.0,
        )?;

        let overnight_part1 = rolling_group_mean(&overnight_return, &prior_turnover, 0)?;
        let overnight_part5 = rolling_group_mean(&overnight_return, &prior_turnover, 4)?;
        let new_overnight = weighted_pair(
            &overnight_part1.cs(cs_zscore)?,
            &overnight_part5.cs(cs_zscore)?,
            1.0,
            -1.0,
        )?;

        let factor = add_pair(&new_intraday.cs(cs_zscore)?, &new_overnight.cs(cs_zscore)?)?;
        Ok(factor.to_factor_series(self.spec()))
    }
}

fn rolling_group_mean(
    returns: &PanelColumn,
    sort_values: &PanelColumn,
    group_idx: usize,
) -> Result<PanelColumn> {
    returns.ts_binary(sort_values, |returns, sort_values| {
        grouped_part_series(returns, sort_values, group_idx)
    })
}

fn grouped_part_series(
    returns: &[Option<f64>],
    sort_values: &[Option<f64>],
    group_idx: usize,
) -> Vec<Option<f64>> {
    let mut output = vec![None; returns.len()];
    if group_idx >= GROUP_COUNT {
        return output;
    }

    for idx in 0..returns.len() {
        if idx + 1 < WINDOW {
            continue;
        }
        let start = idx + 1 - WINDOW;
        let mut pairs = Vec::<(f64, usize, f64)>::with_capacity(WINDOW);
        for window_idx in start..=idx {
            let (Some(return_value), Some(sort_value)) =
                (clean(returns[window_idx]), clean(sort_values[window_idx]))
            else {
                continue;
            };
            pairs.push((sort_value, window_idx, return_value));
        }
        if pairs.len() != WINDOW {
            continue;
        }
        pairs.sort_by(|left, right| {
            left.0
                .total_cmp(&right.0)
                .then_with(|| left.1.cmp(&right.1))
        });
        let group_start = group_idx * GROUP_SIZE;
        let group_end = group_start + GROUP_SIZE;
        let sum = pairs[group_start..group_end]
            .iter()
            .map(|(_, _, return_value)| *return_value)
            .sum::<f64>();
        output[idx] = Some(sum / GROUP_SIZE as f64);
    }
    output
}

fn percent_to_decimal(value: Option<f64>) -> Option<f64> {
    clean(value).map(|value| value / 100.0)
}

fn intraday_turnover(full: Option<f64>, open_auction: Option<f64>) -> Option<f64> {
    match (clean(full), clean(open_auction)) {
        (Some(full), Some(open_auction)) => {
            let value = full - open_auction;
            (value >= 0.0).then_some(value)
        }
        _ => None,
    }
}

fn ret(numerator: Option<f64>, denominator: Option<f64>) -> Option<f64> {
    match (clean(numerator), clean(denominator)) {
        (Some(numerator), Some(denominator)) if denominator.abs() > f64::EPSILON => {
            Some(numerator / denominator - 1.0)
        }
        _ => None,
    }
}

fn weighted_pair(
    left: &PanelColumn,
    right: &PanelColumn,
    left_weight: f64,
    right_weight: f64,
) -> Result<PanelColumn> {
    left.zip_binary(right, |left, right| match (clean(left), clean(right)) {
        (Some(left), Some(right)) => Some(left * left_weight + right * right_weight),
        _ => None,
    })
}

fn add_pair(left: &PanelColumn, right: &PanelColumn) -> Result<PanelColumn> {
    weighted_pair(left, right, 1.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: Option<f64>, expected: f64) {
        let actual = actual.expect("expected value");
        assert!(
            (actual - expected).abs() < 1e-12,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn grouped_part_series_sorts_twenty_days_into_equal_quintiles() {
        let returns = (0..20).map(|idx| Some(idx as f64)).collect::<Vec<_>>();
        let sort_values = (0..20)
            .rev()
            .map(|idx| Some(idx as f64))
            .collect::<Vec<_>>();

        let low = grouped_part_series(&returns, &sort_values, 0);
        let high = grouped_part_series(&returns, &sort_values, 4);

        assert_close(low[19], 17.5);
        assert_close(high[19], 1.5);
    }

    #[test]
    fn grouped_part_series_requires_twenty_valid_pairs() {
        let returns = vec![Some(1.0); 20];
        let mut sort_values = vec![Some(1.0); 20];
        sort_values[3] = None;

        let output = grouped_part_series(&returns, &sort_values, 0);

        assert_eq!(output[19], None);
    }

    #[test]
    fn intraday_turnover_rejects_negative_values() {
        assert_eq!(intraday_turnover(Some(0.02), Some(0.005)), Some(0.015));
        assert_eq!(intraday_turnover(Some(0.001), Some(0.005)), None);
    }

    #[test]
    fn return_rejects_zero_denominator() {
        assert_close(ret(Some(11.0), Some(10.0)), 0.1);
        assert_eq!(ret(Some(11.0), Some(0.0)), None);
    }
}

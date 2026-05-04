use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::{vector::clean, PanelColumn};
use crate::factor::Factor;

const VERSION: &str = "0.1.0";
const WINDOW: usize = 20;
const GROUP_COUNT: usize = 5;
const GROUP_SIZE: usize = WINDOW / GROUP_COUNT;

pub struct StockDailyNewSmallMom;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyNewSmallMom)
}

impl Factor for StockDailyNewSmallMom {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "new_small_mom".to_string(),
            aliases: vec!["NEW_SMALL_MOM".to_string()],
            name: "new_small_mom".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: [
                "price_volume",
                "return",
                "momentum",
                "moneyflow",
                "small_order",
                "daily",
                "DWZQ",
            ]
            .iter()
            .map(|value| value.to_string())
            .collect(),
            description: "Small-trader momentum factor sorting 20-day daily returns by small-order trading ratio and subtracting the high-small-ratio group from the low-small-ratio group.".to_string(),
            dependencies: vec![
                DataRequest::new(DatasetId::StockDailyPv, &["close", "pre_close", "amount"]),
                DataRequest::new(
                    DatasetId::StockMoneyflow,
                    &["buy_sm_amount", "sell_sm_amount"],
                ),
            ],
            intraday_raw_dependencies: Vec::new(),
            lookback: Lookback {
                trading_days: WINDOW - 1,
            },
        }
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let panel = data.daily_panel(DatasetId::StockDailyPv)?;
        let close = panel.column("close")?;
        let pre_close = panel.column("pre_close")?;
        let amount = panel.column("amount")?;
        let moneyflow_table = data.daily(DatasetId::StockMoneyflow)?;
        let buy_sm_amount = panel.column_from_table(moneyflow_table, "buy_sm_amount")?;
        let sell_sm_amount = panel.column_from_table(moneyflow_table, "sell_sm_amount")?;

        let daily_return = close.zip_binary(&pre_close, ret)?;
        let small_ratio = buy_sm_amount.zip_ternary(&sell_sm_amount, &amount, small_trade_ratio)?;
        let part1 = rolling_group_mean(&daily_return, &small_ratio, 0)?;
        let part5 = rolling_group_mean(&daily_return, &small_ratio, 4)?;
        let factor = part1.zip_binary(&part5, subtract)?;
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

fn ret(numerator: Option<f64>, denominator: Option<f64>) -> Option<f64> {
    match (clean(numerator), clean(denominator)) {
        (Some(numerator), Some(denominator)) if denominator.abs() > f64::EPSILON => {
            Some(numerator / denominator - 1.0)
        }
        _ => None,
    }
}

fn small_trade_ratio(
    buy_sm_amount: Option<f64>,
    sell_sm_amount: Option<f64>,
    amount: Option<f64>,
) -> Option<f64> {
    match (
        clean(buy_sm_amount),
        clean(sell_sm_amount),
        clean(amount).map(|value| value / 10.0),
    ) {
        (Some(buy), Some(sell), Some(total_amount)) if total_amount.abs() > f64::EPSILON => {
            Some(((buy + sell) / 2.0) / total_amount)
        }
        _ => None,
    }
}

fn subtract(left: Option<f64>, right: Option<f64>) -> Option<f64> {
    match (clean(left), clean(right)) {
        (Some(left), Some(right)) => Some(left - right),
        _ => None,
    }
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
    fn small_ratio_uses_existing_moneyflow_amount_unit_conversion() {
        assert_close(small_trade_ratio(Some(20.0), Some(40.0), Some(300.0)), 1.0);
        assert_eq!(small_trade_ratio(Some(20.0), Some(40.0), Some(0.0)), None);
    }

    #[test]
    fn grouped_part_series_sorts_twenty_days_by_small_ratio() {
        let returns = (0..20).map(|idx| Some(idx as f64)).collect::<Vec<_>>();
        let small_ratios = (0..20).map(|idx| Some(idx as f64)).collect::<Vec<_>>();

        let low = grouped_part_series(&returns, &small_ratios, 0);
        let high = grouped_part_series(&returns, &small_ratios, 4);

        assert_close(low[19], 1.5);
        assert_close(high[19], 17.5);
    }

    #[test]
    fn grouped_part_series_requires_twenty_valid_pairs() {
        let returns = vec![Some(1.0); 20];
        let mut small_ratios = vec![Some(1.0); 20];
        small_ratios[7] = None;

        let output = grouped_part_series(&returns, &small_ratios, 0);

        assert_eq!(output[19], None);
    }

    #[test]
    fn subtract_requires_both_parts() {
        assert_close(subtract(Some(0.03), Some(0.01)), 0.02);
        assert_eq!(subtract(Some(0.03), None), None);
    }
}

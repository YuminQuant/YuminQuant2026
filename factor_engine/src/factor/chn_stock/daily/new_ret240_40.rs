use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::vector::clean;
use crate::factor::Factor;

const VERSION: &str = "0.1.0";
const TOTAL_WINDOW: usize = 240;
const SKIP_RECENT: usize = 40;
const SIGNAL_WINDOW: usize = TOTAL_WINDOW - SKIP_RECENT;
const HALF_WINDOW: usize = SIGNAL_WINDOW / 2;

pub struct StockDailyNewRet24040;

#[derive(Clone, Copy, Debug)]
struct Observation {
    position: usize,
    daily_return: f64,
    small_ratio: f64,
    turnover: f64,
}

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyNewRet24040)
}

impl Factor for StockDailyNewRet24040 {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "new_ret240_40".to_string(),
            aliases: vec!["New_Ret240_40".to_string()],
            name: "New_Ret240_40".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: [
                "price_volume",
                "return",
                "momentum",
                "moneyflow",
                "small_order",
                "turnover",
                "daily",
                "GSZQ",
            ]
            .iter()
            .map(|value| value.to_string())
            .collect(),
            description: "New Ret240_40 momentum factor averaging returns from the older 200 days in a 240-day window that are both high small-order ratio and low turnover.".to_string(),
            dependencies: vec![
                DataRequest::new(DatasetId::StockDailyPv, &["close", "pre_close", "amount"]),
                DataRequest::new(
                    DatasetId::StockMoneyflow,
                    &["buy_sm_amount", "sell_sm_amount"],
                ),
                DataRequest::new(DatasetId::StockDailyBasic, &["turnover_rate_f"]),
            ],
            intraday_raw_dependencies: Vec::new(),
            lookback: Lookback {
                trading_days: TOTAL_WINDOW - 1,
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
        let turnover = panel
            .column_from_table(data.daily(DatasetId::StockDailyBasic)?, "turnover_rate_f")?
            .map_values(percent_to_decimal);

        let daily_return = close.zip_binary(&pre_close, ret)?;
        let small_ratio = buy_sm_amount.zip_ternary(&sell_sm_amount, &amount, small_trade_ratio)?;
        let factor = daily_return.ts_ternary(&small_ratio, &turnover, new_ret240_40_series)?;
        Ok(factor.to_factor_series(self.spec()))
    }
}

fn new_ret240_40_series(
    returns: &[Option<f64>],
    small_ratios: &[Option<f64>],
    turnovers: &[Option<f64>],
) -> Vec<Option<f64>> {
    let mut output = vec![None; returns.len()];
    for idx in 0..returns.len() {
        if idx + 1 < TOTAL_WINDOW {
            continue;
        }
        let start = idx + 1 - TOTAL_WINDOW;
        let end = idx - SKIP_RECENT;
        let mut observations = Vec::with_capacity(SIGNAL_WINDOW);
        for (position, window_idx) in (start..=end).enumerate() {
            let (Some(daily_return), Some(small_ratio), Some(turnover)) = (
                finite(returns[window_idx]),
                finite(small_ratios[window_idx]),
                finite(turnovers[window_idx]),
            ) else {
                continue;
            };
            observations.push(Observation {
                position,
                daily_return,
                small_ratio,
                turnover,
            });
        }
        if observations.len() != SIGNAL_WINDOW {
            continue;
        }

        let mut high_small = vec![false; SIGNAL_WINDOW];
        let mut by_small = observations.clone();
        by_small.sort_by(|left, right| {
            left.small_ratio
                .total_cmp(&right.small_ratio)
                .then_with(|| left.position.cmp(&right.position))
        });
        for row in by_small.iter().skip(HALF_WINDOW) {
            high_small[row.position] = true;
        }

        let mut low_turnover = vec![false; SIGNAL_WINDOW];
        let mut by_turnover = observations.clone();
        by_turnover.sort_by(|left, right| {
            left.turnover
                .total_cmp(&right.turnover)
                .then_with(|| left.position.cmp(&right.position))
        });
        for row in by_turnover.iter().take(HALF_WINDOW) {
            low_turnover[row.position] = true;
        }

        let mut sum = 0.0;
        let mut count = 0;
        for row in &observations {
            if high_small[row.position] && low_turnover[row.position] {
                sum += row.daily_return;
                count += 1;
            }
        }
        if count > 0 {
            output[idx] = Some(sum / count as f64);
        }
    }
    output
}

fn ret(numerator: Option<f64>, denominator: Option<f64>) -> Option<f64> {
    match (finite(numerator), finite(denominator)) {
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
        finite(buy_sm_amount),
        finite(sell_sm_amount),
        finite(amount).map(|value| value / 10.0),
    ) {
        (Some(buy), Some(sell), Some(total_amount)) if total_amount.abs() > f64::EPSILON => {
            Some(((buy + sell) / 2.0) / total_amount)
        }
        _ => None,
    }
}

fn percent_to_decimal(value: Option<f64>) -> Option<f64> {
    finite(value).map(|value| value / 100.0)
}

fn finite(value: Option<f64>) -> Option<f64> {
    clean(value).filter(|value| value.is_finite())
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
    fn small_ratio_uses_new_small_mom_unit_conversion() {
        assert_close(small_trade_ratio(Some(20.0), Some(40.0), Some(300.0)), 1.0);
        assert_eq!(small_trade_ratio(Some(20.0), Some(40.0), Some(0.0)), None);
    }

    #[test]
    fn new_ret240_40_uses_older_two_hundred_days_and_skips_recent_forty() {
        let returns = (0..TOTAL_WINDOW)
            .map(|idx| Some(idx as f64))
            .collect::<Vec<_>>();
        let small = (0..TOTAL_WINDOW)
            .map(|idx| {
                if idx < SIGNAL_WINDOW {
                    Some(idx as f64)
                } else {
                    Some(10_000.0)
                }
            })
            .collect::<Vec<_>>();
        let turnover = (0..TOTAL_WINDOW)
            .map(|idx| {
                if idx < 100 {
                    Some(1_000.0 + idx as f64)
                } else if idx < SIGNAL_WINDOW {
                    Some((idx - 100) as f64)
                } else {
                    Some(-10_000.0)
                }
            })
            .collect::<Vec<_>>();

        let output = new_ret240_40_series(&returns, &small, &turnover);

        assert_close(output[TOTAL_WINDOW - 1], 149.5);
    }

    #[test]
    fn new_ret240_40_requires_complete_valid_signal_window() {
        let returns = vec![Some(1.0); TOTAL_WINDOW];
        let mut small = vec![Some(1.0); TOTAL_WINDOW];
        let turnover = vec![Some(1.0); TOTAL_WINDOW];
        small[42] = None;

        let output = new_ret240_40_series(&returns, &small, &turnover);

        assert_eq!(output[TOTAL_WINDOW - 1], None);
    }

    #[test]
    fn new_ret240_40_returns_none_when_high_small_and_low_turnover_do_not_overlap() {
        let returns = vec![Some(1.0); TOTAL_WINDOW];
        let small = (0..TOTAL_WINDOW)
            .map(|idx| Some(idx as f64))
            .collect::<Vec<_>>();
        let turnover = (0..TOTAL_WINDOW)
            .map(|idx| Some(idx as f64))
            .collect::<Vec<_>>();

        let output = new_ret240_40_series(&returns, &small, &turnover);

        assert_eq!(output[TOTAL_WINDOW - 1], None);
    }

    #[test]
    fn percent_to_decimal_converts_daily_turnover() {
        assert_close(percent_to_decimal(Some(2.5)), 0.025);
        assert_eq!(percent_to_decimal(Some(f64::INFINITY)), None);
    }
}

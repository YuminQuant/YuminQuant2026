use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::stock_daily_ops::neutralize_size_sector;
use crate::factor::common::{vector::clean, DailyPanel, PanelColumn};
use crate::factor::Factor;

const VERSION: &str = "0.1.0";
const WINDOW: usize = 20;
const CUT_RATIO: f64 = 0.10;

pub struct StockDailyActPositive10;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyActPositive10)
}

impl Factor for StockDailyActPositive10 {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "act_positive_10".to_string(),
            aliases: vec!["ACT_positive_10".to_string()],
            name: "act_positive_10".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: tags(),
            description: "KYZQ positive ACT factor: mean active big+medium net buy ratio on the top 10% adjusted-return days in a rolling 20-day window, neutralized by Barra SIZE and SW sector.".to_string(),
            dependencies: vec![
                DataRequest::new(DatasetId::StockDailyPv, &["close"]),
                DataRequest::new(DatasetId::StockAdjFactor, &["adj_factor"]),
                DataRequest::new(
                    DatasetId::StockMoneyflow,
                    &[
                        "buy_lg_amount",
                        "sell_lg_amount",
                        "buy_md_amount",
                        "sell_md_amount",
                    ],
                ),
                DataRequest::new(DatasetId::StockBarraDaily, &["SIZE"]),
                DataRequest::new(DatasetId::StockSwClassification, &["l1_code"]),
            ],
            intraday_raw_dependencies: Vec::new(),
            lookback: Lookback {
                trading_days: WINDOW,
            },
        }
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let panel = data.daily_panel(DatasetId::StockDailyPv)?;
        let moneyflow = data.daily(DatasetId::StockMoneyflow)?;
        let buy_lg = panel.column_from_table(moneyflow, "buy_lg_amount")?;
        let sell_lg = panel.column_from_table(moneyflow, "sell_lg_amount")?;
        let buy_md = panel.column_from_table(moneyflow, "buy_md_amount")?;
        let sell_md = panel.column_from_table(moneyflow, "sell_md_amount")?;

        let act = buy_lg.zip_quaternary(&sell_lg, &buy_md, &sell_md, active_buy_sell_ratio)?;
        let adjusted_return = adjusted_daily_return(&panel, data)?;
        let raw = rolling_top_return_act_mean(&act, &adjusted_return)?;
        let factor = neutralize_size_sector(&raw, &panel, data)?;
        Ok(factor.to_factor_series(self.spec()))
    }
}

fn tags() -> Vec<String> {
    [
        "KYZQ",
        "moneyflow",
        "active_buy",
        "large_order",
        "medium_order",
        "return",
        "neutralize",
        "barra",
        "size",
        "sector",
        "daily",
    ]
    .iter()
    .map(|value| value.to_string())
    .collect()
}

fn adjusted_daily_return(panel: &DailyPanel, data: &DataPool) -> Result<PanelColumn> {
    let close = panel.column("close")?;
    let adj_factor =
        panel.column_from_table(data.daily(DatasetId::StockAdjFactor)?, "adj_factor")?;
    let adj_close = close.zip_binary(&adj_factor, multiply_pair)?;
    adj_close.ts(one_day_return_series)
}

fn multiply_pair(left: Option<f64>, right: Option<f64>) -> Option<f64> {
    match (clean(left), clean(right)) {
        (Some(left), Some(right)) => Some(left * right),
        _ => None,
    }
}

fn one_day_return_series(values: &[Option<f64>]) -> Vec<Option<f64>> {
    let mut output = vec![None; values.len()];
    for idx in 1..values.len() {
        let (Some(current), Some(previous)) = (clean(values[idx]), clean(values[idx - 1])) else {
            continue;
        };
        if previous.abs() > f64::EPSILON {
            let value = current / previous - 1.0;
            if value.is_finite() {
                output[idx] = Some(value);
            }
        }
    }
    output
}

fn active_buy_sell_ratio(
    buy_lg: Option<f64>,
    sell_lg: Option<f64>,
    buy_md: Option<f64>,
    sell_md: Option<f64>,
) -> Option<f64> {
    let (Some(buy_lg), Some(sell_lg), Some(buy_md), Some(sell_md)) =
        (clean(buy_lg), clean(sell_lg), clean(buy_md), clean(sell_md))
    else {
        return None;
    };
    let buy = buy_lg + buy_md;
    let sell = sell_lg + sell_md;
    let denominator = buy + sell;
    if denominator.abs() <= f64::EPSILON {
        return None;
    }
    let value = (buy - sell) / denominator;
    value.is_finite().then_some(value)
}

fn rolling_top_return_act_mean(
    act: &PanelColumn,
    adjusted_return: &PanelColumn,
) -> Result<PanelColumn> {
    act.ts_binary(adjusted_return, top_return_act_mean_series)
}

fn top_return_act_mean_series(
    act: &[Option<f64>],
    adjusted_return: &[Option<f64>],
) -> Vec<Option<f64>> {
    let mut output = vec![None; act.len()];
    for idx in 0..act.len() {
        let start = (idx + 1).saturating_sub(WINDOW);
        let mut pairs = Vec::<(f64, f64)>::with_capacity(WINDOW);
        for window_idx in start..=idx {
            let (Some(act_value), Some(return_value)) =
                (clean(act[window_idx]), clean(adjusted_return[window_idx]))
            else {
                continue;
            };
            pairs.push((return_value, act_value));
        }
        if pairs.is_empty() {
            continue;
        }
        pairs.sort_by(|left, right| right.0.total_cmp(&left.0));
        let take_count = cut_count(pairs.len());
        output[idx] =
            Some(pairs[..take_count].iter().map(|(_, act)| *act).sum::<f64>() / take_count as f64);
    }
    output
}

fn cut_count(valid_count: usize) -> usize {
    ((valid_count as f64) * CUT_RATIO).ceil().max(1.0) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: Option<f64>, expected: f64) {
        let actual = actual.expect("value");
        assert!(
            (actual - expected).abs() < 1e-12,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn kyzq_act_ratio_uses_large_and_medium_active_amounts() {
        assert_close(
            active_buy_sell_ratio(Some(5.0), Some(1.0), Some(2.0), Some(2.0)),
            0.4,
        );
        assert_eq!(
            active_buy_sell_ratio(Some(0.0), Some(0.0), Some(0.0), Some(0.0)),
            None
        );
    }

    #[test]
    fn kyzq_act_positive_10_selects_top_return_decile_with_ceiling() {
        let act = (1..=11).map(|value| Some(value as f64)).collect::<Vec<_>>();
        let returns = (1..=11).map(|value| Some(value as f64)).collect::<Vec<_>>();

        let output = top_return_act_mean_series(&act, &returns);

        assert_close(output[10], (11.0 + 10.0) / 2.0);
    }

    #[test]
    fn kyzq_act_positive_10_spec_has_kyzq_tag() {
        let spec = StockDailyActPositive10.spec();
        assert_eq!(spec.id, "act_positive_10");
        assert!(spec.tags.iter().any(|tag| tag == "KYZQ"));
    }
}

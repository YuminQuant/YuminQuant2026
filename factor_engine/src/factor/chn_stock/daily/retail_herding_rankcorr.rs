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

pub struct StockDailyRetailHerdingRankcorr;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyRetailHerdingRankcorr)
}

impl Factor for StockDailyRetailHerdingRankcorr {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "retail_herding_rankcorr".to_string(),
            aliases: vec!["Retail Herding RankCorr".to_string()],
            name: "retail_herding_rankcorr".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: tags(),
            description: "KYZQ retail herding factor: rolling rank correlation between adjusted daily return R_t and next-day small-order net inflow S_{t+1}, using only information available through the target date, neutralized by Barra SIZE and SW sector.".to_string(),
            dependencies: vec![
                DataRequest::new(DatasetId::StockDailyPv, &["close"]),
                DataRequest::new(DatasetId::StockAdjFactor, &["adj_factor"]),
                DataRequest::new(
                    DatasetId::StockMoneyflow,
                    &["buy_sm_amount", "sell_sm_amount"],
                ),
                DataRequest::new(DatasetId::StockBarraDaily, &["SIZE"]),
                DataRequest::new(DatasetId::StockSwClassification, &["l1_code"]),
            ],
            intraday_raw_dependencies: Vec::new(),
            lookback: Lookback {
                trading_days: WINDOW + 1,
            },
        }
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let panel = data.daily_panel(DatasetId::StockDailyPv)?;
        let moneyflow = data.daily(DatasetId::StockMoneyflow)?;
        let buy_sm = panel.column_from_table(moneyflow, "buy_sm_amount")?;
        let sell_sm = panel.column_from_table(moneyflow, "sell_sm_amount")?;

        let small_net = buy_sm.zip_binary(&sell_sm, net_flow)?;
        let adjusted_return = adjusted_daily_return(&panel, data)?;
        let raw = adjusted_return.ts_binary(&small_net, retail_rankcorr_series)?;
        let factor = neutralize_size_sector(&raw, &panel, data)?;
        Ok(factor.to_factor_series(self.spec()))
    }
}

fn tags() -> Vec<String> {
    [
        "KYZQ",
        "moneyflow",
        "small_order",
        "rankcorr",
        "herding",
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

fn net_flow(buy: Option<f64>, sell: Option<f64>) -> Option<f64> {
    match (clean(buy), clean(sell)) {
        (Some(buy), Some(sell)) => Some(buy - sell),
        _ => None,
    }
}

fn retail_rankcorr_series(
    adjusted_return: &[Option<f64>],
    small_net: &[Option<f64>],
) -> Vec<Option<f64>> {
    let mut output = vec![None; adjusted_return.len()];
    for idx in 0..adjusted_return.len() {
        let start = idx.saturating_sub(WINDOW);
        let mut return_values = Vec::with_capacity(WINDOW);
        let mut next_small_net = Vec::with_capacity(WINDOW);
        for return_idx in start..idx {
            let next_idx = return_idx + 1;
            if next_idx > idx {
                continue;
            }
            let (Some(ret), Some(net)) = (
                clean(adjusted_return[return_idx]),
                clean(small_net[next_idx]),
            ) else {
                continue;
            };
            return_values.push(ret);
            next_small_net.push(net);
        }
        output[idx] = rank_corr(&return_values, &next_small_net);
    }
    output
}

fn rank_corr(left: &[f64], right: &[f64]) -> Option<f64> {
    if left.len() != right.len() || left.len() < 2 {
        return None;
    }
    let left_ranks = average_ranks(left);
    let right_ranks = average_ranks(right);
    pearson_corr(&left_ranks, &right_ranks)
}

fn average_ranks(values: &[f64]) -> Vec<f64> {
    let mut indexed = values
        .iter()
        .enumerate()
        .map(|(idx, value)| (idx, *value))
        .collect::<Vec<_>>();
    indexed.sort_by(|left, right| {
        left.1
            .total_cmp(&right.1)
            .then_with(|| left.0.cmp(&right.0))
    });

    let mut ranks = vec![0.0; values.len()];
    let mut start = 0usize;
    while start < indexed.len() {
        let mut end = start + 1;
        while end < indexed.len() && indexed[end].1 == indexed[start].1 {
            end += 1;
        }
        let avg_rank = (start + 1 + end) as f64 / 2.0;
        for (idx, _) in &indexed[start..end] {
            ranks[*idx] = avg_rank;
        }
        start = end;
    }
    ranks
}

fn pearson_corr(left: &[f64], right: &[f64]) -> Option<f64> {
    if left.len() != right.len() || left.len() < 2 {
        return None;
    }
    let left_mean = left.iter().sum::<f64>() / left.len() as f64;
    let right_mean = right.iter().sum::<f64>() / right.len() as f64;
    let mut covariance = 0.0;
    let mut left_var = 0.0;
    let mut right_var = 0.0;
    for (left, right) in left.iter().zip(right) {
        let left_delta = *left - left_mean;
        let right_delta = *right - right_mean;
        covariance += left_delta * right_delta;
        left_var += left_delta * left_delta;
        right_var += right_delta * right_delta;
    }
    let denominator = (left_var * right_var).sqrt();
    if denominator <= f64::EPSILON {
        return None;
    }
    let value = covariance / denominator;
    value.is_finite().then_some(value)
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
    fn kyzq_retail_herding_aligns_return_t_with_small_net_t_plus_one() {
        let returns = vec![Some(1.0), Some(2.0), Some(100.0)];
        let small_net = vec![Some(999.0), Some(10.0), Some(20.0)];

        let output = retail_rankcorr_series(&returns, &small_net);

        assert_close(output[2], 1.0);
    }

    #[test]
    fn kyzq_retail_herding_requires_two_valid_pairs() {
        let returns = vec![Some(1.0), None];
        let small_net = vec![Some(1.0), Some(2.0)];

        let output = retail_rankcorr_series(&returns, &small_net);

        assert_eq!(output[1], None);
    }

    #[test]
    fn kyzq_retail_rankcorr_uses_average_tie_ranks() {
        assert_eq!(average_ranks(&[2.0, 1.0, 2.0]), vec![2.5, 1.0, 2.5]);
    }

    #[test]
    fn kyzq_retail_herding_spec_has_kyzq_tag() {
        let spec = StockDailyRetailHerdingRankcorr.spec();
        assert_eq!(spec.id, "retail_herding_rankcorr");
        assert!(spec.tags.iter().any(|tag| tag == "KYZQ"));
    }
}

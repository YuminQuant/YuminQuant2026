use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::stock_daily_ops::neutralize_size_sector;
use crate::factor::common::{vector::clean, PanelColumn};
use crate::factor::Factor;

const VERSION: &str = "0.1.0";
const WINDOW: usize = 20;

pub struct StockDailyCnir;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyCnir)
}

impl Factor for StockDailyCnir {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "cnir".to_string(),
            aliases: vec!["CNIR".to_string()],
            name: "cnir".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: tags(),
            description: "KYZQ CNIR factor from broad principal moneyflow imbalance after MOD return residualization, rolling 20-day corrected net inflow over corrected turnover, neutralized by Barra SIZE and SW sector.".to_string(),
            dependencies: vec![
                DataRequest::new(DatasetId::StockDailyPv, &["close", "pre_close"]),
                DataRequest::new(
                    DatasetId::StockMoneyflow,
                    &[
                        "buy_elg_amount",
                        "sell_elg_amount",
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
                trading_days: WINDOW - 1,
            },
        }
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let panel = data.daily_panel(DatasetId::StockDailyPv)?;
        let moneyflow = data.daily(DatasetId::StockMoneyflow)?;
        let buy_elg = panel.column_from_table(moneyflow, "buy_elg_amount")?;
        let buy_lg = panel.column_from_table(moneyflow, "buy_lg_amount")?;
        let buy_md = panel.column_from_table(moneyflow, "buy_md_amount")?;
        let sell_elg = panel.column_from_table(moneyflow, "sell_elg_amount")?;
        let sell_lg = panel.column_from_table(moneyflow, "sell_lg_amount")?;
        let sell_md = panel.column_from_table(moneyflow, "sell_md_amount")?;

        let buy = buy_elg.zip_ternary(&buy_lg, &buy_md, sum_three_positive)?;
        let sell = sell_elg.zip_ternary(&sell_lg, &sell_md, sum_three_positive)?;
        let total = buy.zip_binary(&sell, sum_positive_pair)?;
        let imbalance = buy.zip_binary(&sell, imbalance_log_ratio)?;
        let ret = panel
            .column("close")?
            .zip_binary(&panel.column("pre_close")?, daily_return)?;
        let epsilon = imbalance.cs_neutralize_regression(&[&ret], None)?;
        let corrected_net = epsilon.zip_binary(&total, corrected_net_from_epsilon_total)?;
        let raw = rolling_cnir(&corrected_net, &total)?;
        let factor = neutralize_size_sector(&raw, &panel, data)?;
        Ok(factor.to_factor_series(self.spec()))
    }
}

fn tags() -> Vec<String> {
    [
        "KYZQ",
        "moneyflow",
        "principal",
        "imbalance",
        "return_residual",
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

fn sum_three_positive(first: Option<f64>, second: Option<f64>, third: Option<f64>) -> Option<f64> {
    let (Some(first), Some(second), Some(third)) = (clean(first), clean(second), clean(third))
    else {
        return None;
    };
    let value = first + second + third;
    value.is_finite().then_some(value)
}

fn sum_positive_pair(left: Option<f64>, right: Option<f64>) -> Option<f64> {
    match (clean(left), clean(right)) {
        (Some(left), Some(right)) => {
            let value = left + right;
            (value > f64::EPSILON && value.is_finite()).then_some(value)
        }
        _ => None,
    }
}

fn imbalance_log_ratio(buy: Option<f64>, sell: Option<f64>) -> Option<f64> {
    match (clean(buy), clean(sell)) {
        (Some(buy), Some(sell)) if buy > 0.0 && sell > 0.0 => {
            let value = (buy / sell).ln();
            value.is_finite().then_some(value)
        }
        _ => None,
    }
}

fn daily_return(close: Option<f64>, pre_close: Option<f64>) -> Option<f64> {
    match (clean(close), clean(pre_close)) {
        (Some(close), Some(pre_close)) if pre_close.abs() > f64::EPSILON => {
            let value = close / pre_close - 1.0;
            value.is_finite().then_some(value)
        }
        _ => None,
    }
}

fn corrected_net_from_epsilon_total(epsilon: Option<f64>, total: Option<f64>) -> Option<f64> {
    let (Some(epsilon), Some(total)) = (clean(epsilon), clean(total)) else {
        return None;
    };
    let (buy_hat, sell_hat) = corrected_amounts(epsilon, total)?;
    Some(buy_hat - sell_hat)
}

fn corrected_amounts(epsilon: f64, total: f64) -> Option<(f64, f64)> {
    if !epsilon.is_finite() || !total.is_finite() || total <= f64::EPSILON {
        return None;
    }
    let buy_share = if epsilon >= 0.0 {
        1.0 / (1.0 + (-epsilon).exp())
    } else {
        let exp_value = epsilon.exp();
        exp_value / (1.0 + exp_value)
    };
    let buy = buy_share * total;
    let sell = total - buy;
    (buy.is_finite() && sell.is_finite()).then_some((buy, sell))
}

fn rolling_cnir(corrected_net: &PanelColumn, total: &PanelColumn) -> Result<PanelColumn> {
    corrected_net.ts_binary(total, rolling_cnir_series)
}

fn rolling_cnir_series(corrected_net: &[Option<f64>], total: &[Option<f64>]) -> Vec<Option<f64>> {
    let mut output = vec![None; corrected_net.len()];
    for idx in 0..corrected_net.len() {
        let start = (idx + 1).saturating_sub(WINDOW);
        let mut numerator = 0.0;
        let mut denominator = 0.0;
        for window_idx in start..=idx {
            let (Some(net), Some(total)) =
                (clean(corrected_net[window_idx]), clean(total[window_idx]))
            else {
                continue;
            };
            numerator += net;
            denominator += total;
        }
        if denominator > f64::EPSILON {
            let value = numerator / denominator;
            if value.is_finite() {
                output[idx] = Some(value);
            }
        }
    }
    output
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
    fn kyzq_cnir_merges_broad_principal_amounts() {
        assert_close(sum_three_positive(Some(1.0), Some(2.0), Some(3.0)), 6.0);
        assert_eq!(sum_three_positive(Some(1.0), None, Some(3.0)), None);
    }

    #[test]
    fn kyzq_cnir_log_imbalance_requires_positive_buy_and_sell() {
        assert_close(imbalance_log_ratio(Some(4.0), Some(2.0)), 2.0_f64.ln());
        assert_eq!(imbalance_log_ratio(Some(4.0), Some(0.0)), None);
    }

    #[test]
    fn kyzq_cnir_corrected_amounts_preserve_total() {
        let (buy, sell) = corrected_amounts(2.0, 100.0).expect("amounts");
        assert!((buy + sell - 100.0).abs() < 1e-12);
        assert!(buy > sell);
    }

    #[test]
    fn kyzq_cnir_rolling_ratio_uses_corrected_net_over_total() {
        let net = vec![Some(2.0), Some(-1.0), Some(3.0)];
        let total = vec![Some(10.0), Some(10.0), Some(20.0)];
        let output = rolling_cnir_series(&net, &total);

        assert_close(output[2], 4.0 / 40.0);
    }

    #[test]
    fn kyzq_cnir_factor_spec_has_kyzq_tag() {
        let spec = StockDailyCnir.spec();
        assert_eq!(spec.id, "cnir");
        assert!(spec.tags.iter().any(|tag| tag == "KYZQ"));
    }
}

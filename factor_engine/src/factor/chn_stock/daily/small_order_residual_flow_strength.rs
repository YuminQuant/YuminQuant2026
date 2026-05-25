use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::stock_daily_ops::{adjusted_20d_return, neutralize_size_sector};
use crate::factor::common::{vector::clean, PanelColumn};
use crate::factor::Factor;
use crate::operators::cs_regression_residual;

const VERSION: &str = "0.1.0";
const WINDOW: usize = 20;

pub struct StockDailySmallOrderResidualFlowStrength;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailySmallOrderResidualFlowStrength)
}

impl Factor for StockDailySmallOrderResidualFlowStrength {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "small_order_residual_flow_strength".to_string(),
            aliases: vec!["Small Order Residual Flow Strength".to_string()],
            name: "small_order_residual_flow_strength".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: tags(),
            description: "KYZQ small-order residual flow strength: rolling 20-day sum(net small buy-sell) over sum(abs(net)), residualized against adjusted Ret20 with intercept, then neutralized by Barra SIZE and SW sector. The report's negative-direction raw value is preserved.".to_string(),
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
                trading_days: WINDOW,
            },
        }
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let panel = data.daily_panel(DatasetId::StockDailyPv)?;
        let moneyflow = data.daily(DatasetId::StockMoneyflow)?;
        let buy = panel.column_from_table(moneyflow, "buy_sm_amount")?;
        let sell = panel.column_from_table(moneyflow, "sell_sm_amount")?;

        let net = buy.zip_binary(&sell, net_flow)?;
        let strength = rolling_flow_strength(&net)?;
        let ret20 = adjusted_20d_return(data, &panel)?;
        let residual = strength.cs_binary(&ret20, cs_regression_residual)?;
        let factor = neutralize_size_sector(&residual, &panel, data)?;
        Ok(factor.to_factor_series(self.spec()))
    }
}

fn tags() -> Vec<String> {
    [
        "KYZQ",
        "moneyflow",
        "small_order",
        "residual",
        "ret20",
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

fn net_flow(buy: Option<f64>, sell: Option<f64>) -> Option<f64> {
    match (clean(buy), clean(sell)) {
        (Some(buy), Some(sell)) => Some(buy - sell),
        _ => None,
    }
}

fn rolling_flow_strength(net: &PanelColumn) -> Result<PanelColumn> {
    net.ts(flow_strength_series)
}

fn flow_strength_series(net: &[Option<f64>]) -> Vec<Option<f64>> {
    let mut output = vec![None; net.len()];
    for idx in 0..net.len() {
        let start = (idx + 1).saturating_sub(WINDOW);
        let mut numerator = 0.0;
        let mut denominator = 0.0;
        for value in net[start..=idx].iter().filter_map(|value| clean(*value)) {
            numerator += value;
            denominator += value.abs();
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
    fn kyzq_small_flow_strength_uses_sum_net_over_sum_abs_net() {
        let net = vec![Some(-2.0), Some(1.0), Some(-3.0)];
        let output = flow_strength_series(&net);

        assert_close(output[2], -4.0 / 6.0);
    }

    #[test]
    fn kyzq_small_flow_strength_rejects_zero_abs_flow() {
        let net = vec![Some(0.0), Some(0.0)];
        let output = flow_strength_series(&net);

        assert_eq!(output[1], None);
    }

    #[test]
    fn kyzq_small_flow_strength_spec_has_kyzq_tag() {
        let spec = StockDailySmallOrderResidualFlowStrength.spec();
        assert_eq!(spec.id, "small_order_residual_flow_strength");
        assert!(spec.tags.iter().any(|tag| tag == "KYZQ"));
    }
}

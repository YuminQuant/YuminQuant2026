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
const MIN_VALID_DAYS: usize = 10;
const CUT_RATIO: f64 = 0.25;

pub struct StockDailyTFct25;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyTFct25)
}

impl Factor for StockDailyTFct25 {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "t_fct_25".to_string(),
            aliases: vec!["T_fct_25".to_string()],
            name: "t_fct_25".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: tags(),
            description: "KYZQ ideal turnover factor: 20-day high-close turnover mean minus low-close turnover mean using a 25% close-price split, neutralized by Barra SIZE and SW sector.".to_string(),
            dependencies: vec![
                DataRequest::new(DatasetId::StockDailyPv, &["close"]),
                DataRequest::new(DatasetId::StockDailyBasic, &["turnover_rate_f"]),
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
        let close = panel.column("close")?;
        let turnover =
            panel.column_from_table(data.daily(DatasetId::StockDailyBasic)?, "turnover_rate_f")?;

        let raw = rolling_spread_by_close(&turnover, &close)?;
        let factor = neutralize_size_sector(&raw, &panel, data)?;
        Ok(factor.to_factor_series(self.spec()))
    }
}

fn tags() -> Vec<String> {
    [
        "KYZQ",
        "price_volume",
        "turnover",
        "liquidity",
        "hidden_structure",
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

fn rolling_spread_by_close(metric: &PanelColumn, close: &PanelColumn) -> Result<PanelColumn> {
    metric.ts_binary(close, spread_by_close_series)
}

fn spread_by_close_series(metric: &[Option<f64>], close: &[Option<f64>]) -> Vec<Option<f64>> {
    let mut output = vec![None; metric.len()];
    for idx in 0..metric.len() {
        let start = (idx + 1).saturating_sub(WINDOW);
        let mut pairs = Vec::<(f64, f64)>::with_capacity(WINDOW);
        for window_idx in start..=idx {
            let (Some(metric_value), Some(close_value)) =
                (clean(metric[window_idx]), clean(close[window_idx]))
            else {
                continue;
            };
            pairs.push((close_value, metric_value));
        }
        if pairs.len() < MIN_VALID_DAYS {
            continue;
        }
        pairs.sort_by(|left, right| left.0.total_cmp(&right.0));
        let take_count = cut_count(pairs.len());
        let low_mean = mean_metric(&pairs[..take_count]);
        let high_mean = mean_metric(&pairs[pairs.len() - take_count..]);
        output[idx] = Some(high_mean - low_mean);
    }
    output
}

fn cut_count(valid_count: usize) -> usize {
    ((valid_count as f64) * CUT_RATIO).ceil().max(1.0) as usize
}

fn mean_metric(pairs: &[(f64, f64)]) -> f64 {
    pairs.iter().map(|(_, metric)| *metric).sum::<f64>() / pairs.len() as f64
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
    fn kyzq_t_fct_25_uses_ceiled_quarter_split_by_close() {
        let close = (1..=10).map(|value| Some(value as f64)).collect::<Vec<_>>();
        let metric = (11..=20)
            .map(|value| Some(value as f64))
            .collect::<Vec<_>>();

        let output = spread_by_close_series(&metric, &close);

        assert_close(
            output[9],
            (18.0 + 19.0 + 20.0) / 3.0 - (11.0 + 12.0 + 13.0) / 3.0,
        );
    }

    #[test]
    fn kyzq_t_fct_25_requires_ten_valid_days() {
        let close = vec![Some(1.0); 9];
        let metric = vec![Some(2.0); 9];

        let output = spread_by_close_series(&metric, &close);

        assert_eq!(output[8], None);
    }

    #[test]
    fn kyzq_t_fct_25_spec_has_kyzq_tag() {
        let spec = StockDailyTFct25.spec();
        assert_eq!(spec.id, "t_fct_25");
        assert!(spec.tags.iter().any(|tag| tag == "KYZQ"));
    }
}

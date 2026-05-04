use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::{vector::clean, PanelColumn};
use crate::factor::common::{ClassificationLevel, ClassificationMap};
use crate::factor::Factor;

const VERSION: &str = "0.1.0";
const MA_WINDOWS: [usize; 6] = [1, 5, 10, 20, 60, 120];
const MAX_WINDOW: usize = 120;

pub struct StockDailyAcf;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyAcf)
}

impl Factor for StockDailyAcf {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "acf".to_string(),
            aliases: vec!["ACF".to_string()],
            name: "ACF".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: [
                "price_volume",
                "amount",
                "convergence",
                "moving_average",
                "neutralize",
                "barra",
                "size",
                "sector",
                "daily",
                "KYZQ",
            ]
            .iter()
            .map(|value| value.to_string())
            .collect(),
            description: "Amount Convergence Factor based on the cross-period standard deviation of daily amount moving averages, neutralized by Barra SIZE and SW sector.".to_string(),
            dependencies: vec![
                DataRequest::new(DatasetId::StockDailyPv, &["amount"]),
                DataRequest::new(DatasetId::StockBarraDaily, &["SIZE"]),
                DataRequest::new(DatasetId::StockSwClassification, &["l1_code"]),
            ],
            intraday_raw_dependencies: Vec::new(),
            lookback: Lookback {
                trading_days: MAX_WINDOW - 1,
            },
        }
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let sector_map = ClassificationMap::from_table(
            data.daily(DatasetId::StockSwClassification)?,
            ClassificationLevel::Sector,
        )?;
        let panel = data.daily_panel(DatasetId::StockDailyPv)?;
        let amount = panel.column("amount")?;
        let size = panel.column_from_table(data.daily(DatasetId::StockBarraDaily)?, "SIZE")?;

        let raw_factor = convergence_score(&amount)?;
        let factor = raw_factor.cs_neutralize_regression_by_group(
            &[&size],
            None,
            |trade_date, ts_codes| sector_map.groups_for(trade_date, ts_codes),
        )?;
        Ok(factor.to_factor_series(self.spec()))
    }
}

fn convergence_score(input: &PanelColumn) -> Result<PanelColumn> {
    input.ts(convergence_score_series)
}

fn convergence_score_series(values: &[Option<f64>]) -> Vec<Option<f64>> {
    let means = MA_WINDOWS
        .iter()
        .map(|window| rolling_mean_full(values, *window))
        .collect::<Vec<_>>();
    (0..values.len())
        .map(|idx| {
            let mut ma_values = Vec::with_capacity(MA_WINDOWS.len());
            for mean in &means {
                ma_values.push(clean(mean[idx])?);
            }
            let std = std_dev(&ma_values);
            Some(-(1.0 + std).ln())
        })
        .collect()
}

fn rolling_mean_full(values: &[Option<f64>], window: usize) -> Vec<Option<f64>> {
    let mut output = vec![None; values.len()];
    if window == 0 {
        return output;
    }
    let mut sum = 0.0;
    let mut count = 0usize;
    for idx in 0..values.len() {
        if let Some(value) = clean(values[idx]) {
            sum += value;
            count += 1;
        }
        if idx >= window {
            if let Some(value) = clean(values[idx - window]) {
                sum -= value;
                count -= 1;
            }
        }
        if count == window {
            output[idx] = Some(sum / window as f64);
        }
    }
    output
}

fn std_dev(values: &[f64]) -> f64 {
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|value| {
            let diff = value - mean;
            diff * diff
        })
        .sum::<f64>()
        / values.len() as f64;
    variance.sqrt()
}

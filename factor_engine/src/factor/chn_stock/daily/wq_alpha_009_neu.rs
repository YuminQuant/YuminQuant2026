use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::{ClassificationLevel, ClassificationMap};
use crate::factor::Factor;
use crate::operators::{ts_diff, ts_max, ts_min};

pub struct StockDailyWQAlpha009Neu;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyWQAlpha009Neu)
}

impl Factor for StockDailyWQAlpha009Neu {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "WQAlpha009_neu".to_string(),
            aliases: Vec::new(),
            name: "WQAlpha009 neutralized by sector and SIZE".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: "0.1.0".to_string(),
            tags: [
                "worldquant101alpha_neutralized",
                "price_volume",
                "neutralize",
                "barra",
                "size",
                "sector",
                "daily",
            ]
            .iter()
            .map(|value| value.to_string())
            .collect(),
            description: "WQAlpha009 residualized against SW sector dummies and Barra SIZE."
                .to_string(),
            dependencies: vec![
                DataRequest::new(DatasetId::StockDailyPv, &["close"]),
                DataRequest::new(DatasetId::StockBarraDaily, &["SIZE"]),
                DataRequest::new(DatasetId::StockSwClassification, &["l1_code"]),
            ],
            intraday_raw_dependencies: Vec::new(),
            lookback: Lookback { trading_days: 5 },
        }
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let sector_map = ClassificationMap::from_table(
            data.daily(DatasetId::StockSwClassification)?,
            ClassificationLevel::Sector,
        )?;
        let panel = data.daily_panel(DatasetId::StockDailyPv)?;
        let size = panel.column_from_table(data.daily(DatasetId::StockBarraDaily)?, "SIZE")?;

        let delta_close = panel.column("close")?.ts(|values| ts_diff(values, 1))?;
        let min_delta = delta_close.ts(|values| ts_min(values, 5, 5))?;
        let max_delta = delta_close.ts(|values| ts_max(values, 5, 5))?;
        let raw = delta_close.ts_ternary(&min_delta, &max_delta, |delta, min, max| {
            delta
                .iter()
                .zip(min)
                .zip(max)
                .map(
                    |((delta, min), max)| match (clean(*delta), clean(*min), clean(*max)) {
                        (Some(delta), Some(min), Some(max)) => Some(if 0.0 < min || max < 0.0 {
                            delta
                        } else {
                            -delta
                        }),
                        _ => None,
                    },
                )
                .collect()
        })?;
        let neutralized =
            raw.cs_neutralize_regression_by_group(&[&size], None, |trade_date, ts_codes| {
                sector_map.groups_for(trade_date, ts_codes)
            })?;
        Ok(neutralized.to_factor_series(self.spec()))
    }
}

fn clean(value: Option<f64>) -> Option<f64> {
    value.filter(|value| !value.is_nan())
}

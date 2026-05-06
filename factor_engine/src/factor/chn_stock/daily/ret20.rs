use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::chip;
use crate::factor::Factor;

const VERSION: &str = "0.1.0";

pub struct StockDailyRet20;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyRet20)
}

impl Factor for StockDailyRet20 {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "ret20".to_string(),
            aliases: vec!["Ret20".to_string()],
            name: "Ret20".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: ["price", "return", "reversal", "general", "daily"]
                .iter()
                .map(|value| value.to_string())
                .collect(),
            description: "20-day adjusted close return using available history with min_periods=1."
                .to_string(),
            dependencies: vec![
                DataRequest::new(DatasetId::StockDailyPv, &["close"]),
                DataRequest::new(DatasetId::StockAdjFactor, &["adj_factor"]),
            ],
            intraday_raw_dependencies: Vec::new(),
            lookback: Lookback {
                trading_days: chip::RET_WINDOW,
            },
        }
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let panel = data.daily_panel(DatasetId::StockDailyPv)?;
        let close = panel.column("close")?;
        let adj_factor =
            panel.column_from_table(data.daily(DatasetId::StockAdjFactor)?, "adj_factor")?;
        let adj_close = chip::adjusted_close(&close, &adj_factor)?;
        let factor = chip::ret20_from_adjusted_close(&adj_close)?;
        Ok(factor.to_factor_series(self.spec()))
    }
}

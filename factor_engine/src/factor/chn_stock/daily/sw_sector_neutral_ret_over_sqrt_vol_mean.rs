use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    IntradayDailyRawRequest, Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::chn_stock::daily::ret_over_sqrt_vol_mean::RAW_ID as RET_OVER_SQRT_VOL_RAW_ID;
use crate::factor::common::{ClassificationLevel, ClassificationMap};
use crate::factor::Factor;
use crate::operators::cs_neutralize;

pub struct StockDailySwSectorNeutralRetOverSqrtVolMean;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailySwSectorNeutralRetOverSqrtVolMean)
}

impl Factor for StockDailySwSectorNeutralRetOverSqrtVolMean {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "sw_sector_neutral_ret_over_sqrt_vol_mean".to_string(),
            aliases: Vec::new(),
            name: "Stock SW sector-neutral intraday mean return over sqrt volume".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: "0.1.0".to_string(),
            tags: [
                "price_volume",
                "return",
                "volume",
                "intraday",
                "minute_agg",
                "cross_section",
                "neutralize",
                "sector",
                "sw",
                "daily",
            ]
            .iter()
            .map(|value| value.to_string())
            .collect(),
            description: "SW sector-neutralized mean intraday return divided by sqrt(volume)."
                .to_string(),
            dependencies: vec![DataRequest::new(
                DatasetId::StockSwClassification,
                &["l1_code"],
            )],
            intraday_raw_dependencies: vec![IntradayDailyRawRequest::new(
                RET_OVER_SQRT_VOL_RAW_ID,
                0,
            )],
            lookback: Lookback { trading_days: 0 },
        }
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let sector_map = ClassificationMap::from_table(
            data.daily(DatasetId::StockSwClassification)?,
            ClassificationLevel::Sector,
        )?;
        let panel = data.intraday_daily_raw_panel(RET_OVER_SQRT_VOL_RAW_ID)?;
        let raw = panel.column(RET_OVER_SQRT_VOL_RAW_ID)?;
        let factor = raw.cs_by_group(
            |trade_date, ts_codes| sector_map.groups_for(trade_date, ts_codes),
            cs_neutralize,
        )?;
        Ok(factor.to_factor_series(self.spec()))
    }
}

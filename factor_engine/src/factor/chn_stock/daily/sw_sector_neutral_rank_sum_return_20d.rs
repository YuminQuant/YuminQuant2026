use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::{ClassificationLevel, ClassificationMap, DailyPanel};
use crate::factor::Factor;
use crate::operators::{cs_neutralize, cs_pctrank, ts_pctchg, ts_sum};

pub struct StockDailySwSectorNeutralRankSumReturn20d;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailySwSectorNeutralRankSumReturn20d)
}

impl Factor for StockDailySwSectorNeutralRankSumReturn20d {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "sw_sector_neutral_rank_sum_return_20d".to_string(),
            aliases: vec!["stock.daily.pv.sw_sector_neutral_rank_sum_return_20d".to_string()],
            name: "Stock SW sector-neutral ranked 20-day summed return".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: "0.1.0".to_string(),
            tags: [
                "price_volume",
                "return",
                "momentum",
                "cross_section",
                "neutralize",
                "sector",
                "sw",
                "daily",
            ]
            .iter()
            .map(|value| value.to_string())
            .collect(),
            description:
                "SW sector-neutralized cross-sectional rank of trailing 20-day summed close return."
                    .to_string(),
            dependencies: vec![
                DataRequest::new(DatasetId::StockDailyPv, &["close"]),
                DataRequest::new(DatasetId::StockSwClassification, &["l1_code"]),
            ],
            intraday_raw_dependencies: Vec::new(),
            lookback: Lookback { trading_days: 20 },
        }
    }

    fn compute(&self, context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let sector_map = ClassificationMap::from_table(
            data.daily(DatasetId::StockSwClassification)?,
            ClassificationLevel::Sector,
        )?;

        let panel = DailyPanel::from_table(data.daily(DatasetId::StockDailyPv)?, context)?;
        let factor = panel
            .column("close")?
            .ts(|values| ts_pctchg(values, 1))?
            .ts(|values| ts_sum(values, 20, 20))?
            .cs(|values| cs_pctrank(values, true))?
            .cs_by_group(
                |trade_date, ts_codes| sector_map.groups_for(trade_date, ts_codes),
                cs_neutralize,
            )?;

        Ok(factor.to_factor_series(self.spec()))
    }
}

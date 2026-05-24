use crate::core::{FactorContext, FactorSeries, FactorSpec, IntradayDailyRawSpec};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::mszq_price_volume_tension::{self, MszqPriceVolumeTensionFactorDef};
use crate::factor::Factor;

const DEF: MszqPriceVolumeTensionFactorDef = MszqPriceVolumeTensionFactorDef {
    id: "price_volume_tension",
    alias: "price_volume_tension",
    name: "Price Volume Tension",
};

pub struct StockDailyPriceVolumeTension;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyPriceVolumeTension)
}

impl Factor for StockDailyPriceVolumeTension {
    fn spec(&self) -> FactorSpec {
        mszq_price_volume_tension::factor_spec(DEF)
    }

    fn intraday_raw_specs(&self) -> Vec<IntradayDailyRawSpec> {
        mszq_price_volume_tension::raw_specs()
    }

    fn intraday_raw_provider_key(&self, _raw_id: &str) -> String {
        mszq_price_volume_tension::PROVIDER_KEY.to_string()
    }

    fn minute_compute_many(
        &self,
        raw_ids: &[String],
        context: &FactorContext,
        data: &DataPool,
    ) -> Result<Vec<crate::core::IntradayDailyRawSeries>> {
        mszq_price_volume_tension::minute_compute_many(raw_ids, context, data)
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        mszq_price_volume_tension::compute_factor(DEF, data)
    }
}

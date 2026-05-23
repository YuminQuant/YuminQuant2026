use crate::core::{FactorContext, FactorSeries, FactorSpec, IntradayDailyRawSpec};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::dbzq_intraday_volume_distribution::{
    self, DbzqIntradayVolumeDistributionFactorDef, DbzqIntradayVolumeDistributionKind,
};
use crate::factor::Factor;

const DEF: DbzqIntradayVolumeDistributionFactorDef = DbzqIntradayVolumeDistributionFactorDef {
    id: "volume_price_distribution",
    alias: "volume_price_distribution",
    name: "Volume Price Distribution",
    kind: DbzqIntradayVolumeDistributionKind::VolumePrice,
};

pub struct StockDailyVolumePriceDistribution;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyVolumePriceDistribution)
}

impl Factor for StockDailyVolumePriceDistribution {
    fn spec(&self) -> FactorSpec {
        dbzq_intraday_volume_distribution::factor_spec(DEF)
    }

    fn intraday_raw_specs(&self) -> Vec<IntradayDailyRawSpec> {
        dbzq_intraday_volume_distribution::raw_specs_for_kind(DEF.kind)
    }

    fn intraday_raw_provider_key(&self, _raw_id: &str) -> String {
        dbzq_intraday_volume_distribution::PROVIDER_KEY.to_string()
    }

    fn minute_compute_many(
        &self,
        raw_ids: &[String],
        context: &FactorContext,
        data: &DataPool,
    ) -> Result<Vec<crate::core::IntradayDailyRawSeries>> {
        dbzq_intraday_volume_distribution::minute_compute_many(raw_ids, context, data)
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        dbzq_intraday_volume_distribution::compute_factor(DEF, data)
    }
}

pub mod chn_stock;
pub mod common;
pub mod future;
pub mod registry;

use crate::core::{
    DataRequest, FactorContext, FactorSeries, FactorSpec, IntradayDailyRawSeries,
    IntradayDailyRawSpec,
};
use crate::data::DataPool;
use crate::error::Result;

pub trait Factor: Send + Sync {
    fn spec(&self) -> FactorSpec;

    fn requirements(&self) -> Vec<DataRequest> {
        self.spec().dependencies
    }

    fn intraday_raw_specs(&self) -> Vec<IntradayDailyRawSpec> {
        Vec::new()
    }

    fn minute_compute(
        &self,
        _raw_id: &str,
        _context: &FactorContext,
        _data: &DataPool,
    ) -> Result<Option<IntradayDailyRawSeries>> {
        Ok(None)
    }

    fn compute(&self, context: &FactorContext, data: &DataPool) -> Result<FactorSeries>;
}

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

    fn minute_compute_many(
        &self,
        raw_ids: &[String],
        context: &FactorContext,
        data: &DataPool,
    ) -> Result<Vec<IntradayDailyRawSeries>> {
        let mut output = Vec::new();
        for raw_id in raw_ids {
            if let Some(series) = self.minute_compute(raw_id, context, data)? {
                output.push(series);
            }
        }
        Ok(output)
    }

    fn compute(&self, context: &FactorContext, data: &DataPool) -> Result<FactorSeries>;
}

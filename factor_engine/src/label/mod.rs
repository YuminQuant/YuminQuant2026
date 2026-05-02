pub mod chn_stock;
pub mod engine;
pub mod registry;

use crate::core::{
    FactorContext, IntradayDailyRawRequest, IntradayDailyRawSeries, IntradayDailyRawSpec,
    LabelSeries, LabelSpec,
};
use crate::data::DataPool;
use crate::error::Result;

pub trait Label: Send + Sync {
    fn spec(&self) -> LabelSpec;

    fn intraday_raw_specs(&self) -> Vec<IntradayDailyRawSpec> {
        Vec::new()
    }

    fn intraday_raw_dependencies(&self) -> Vec<IntradayDailyRawRequest> {
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

    fn compute(&self, context: &FactorContext, data: &DataPool) -> Result<LabelSeries>;
}

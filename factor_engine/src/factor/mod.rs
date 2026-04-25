pub mod chn_stock;
pub mod common;
pub mod future;
pub mod registry;

use crate::core::{DataRequest, FactorContext, FactorSeries, FactorSpec};
use crate::data::DataPool;
use crate::error::Result;

pub trait Factor: Send + Sync {
    fn spec(&self) -> FactorSpec;

    fn requirements(&self) -> Vec<DataRequest> {
        self.spec().dependencies
    }

    fn compute(&self, context: &FactorContext, data: &DataPool) -> Result<FactorSeries>;
}

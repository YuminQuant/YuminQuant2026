pub mod chn_stock;
pub mod engine;
pub mod registry;

use crate::core::{FactorContext, LabelSeries, LabelSpec};
use crate::data::DataPool;
use crate::error::Result;

pub trait Label: Send + Sync {
    fn spec(&self) -> LabelSpec;

    fn compute(&self, context: &FactorContext, data: &DataPool) -> Result<LabelSeries>;
}

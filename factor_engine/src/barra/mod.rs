pub mod chn_stock;
pub mod engine;
pub mod registry;

use crate::core::{BarraSeries, BarraSpec, FactorContext};
use crate::data::DataPool;
use crate::error::Result;

pub trait BarraExposure: Send + Sync {
    fn specs(&self) -> Vec<BarraSpec>;

    fn compute(&self, context: &FactorContext, data: &DataPool) -> Result<Vec<BarraSeries>>;
}

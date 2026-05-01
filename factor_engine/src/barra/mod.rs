pub mod chn_stock;
pub mod common;
pub mod engine;
pub mod registry;

use crate::core::{BarraSeries, BarraSpec, FactorContext};
use crate::data::DataPool;
use crate::error::Result;

pub trait BarraExposure: Send + Sync {
    fn family_id(&self) -> &'static str;

    fn specs(&self) -> Vec<BarraSpec>;

    fn compute(&self, context: &FactorContext, data: &DataPool) -> Result<Vec<BarraSeries>>;
}

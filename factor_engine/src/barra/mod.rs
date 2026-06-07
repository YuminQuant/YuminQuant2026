pub mod chn_stock;
pub mod common;
pub mod engine;
pub mod registry;

use std::any::Any;
use std::collections::BTreeSet;

use crate::core::{BarraSeries, BarraSpec, FactorContext};
use crate::data::DataPool;
use crate::error::Result;

pub trait BarraExposure: Send + Sync {
    fn family_id(&self) -> &'static str;

    fn specs(&self) -> Vec<BarraSpec>;

    fn compute(&self, context: &FactorContext, data: &DataPool) -> Result<Vec<BarraSeries>>;

    fn compute_provider_key(&self) -> String {
        self.family_id().to_string()
    }

    fn initial_compute_state(&self, _selected_ids: &BTreeSet<String>) -> Box<dyn Any + Send> {
        Box::new(())
    }

    fn compute_stateful(
        &self,
        context: &FactorContext,
        data: &DataPool,
        _state: &mut (dyn Any + Send),
    ) -> Result<Vec<BarraSeries>> {
        self.compute(context, data)
    }
}

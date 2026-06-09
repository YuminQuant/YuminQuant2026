pub mod chn_stock;
pub mod common;
pub mod future;
pub mod registry;

use std::any::Any;

use crate::core::{
    DataRequest, FactorContext, FactorSeries, FactorSpec, IntradayDailyRawAuxiliaryRequest,
    IntradayDailyRawSeries, IntradayDailyRawSpec,
};
use crate::data::DataPool;
use crate::error::Result;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IntradayRawMaterializeMode {
    Stateless,
    Stateful,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FactorUpdatePolicy {
    Daily,
    FinancialEventSnapshot,
    FinancialEventStateDailyFast,
}

pub trait Factor: Send + Sync {
    fn spec(&self) -> FactorSpec;

    fn provided_specs(&self) -> Vec<FactorSpec> {
        vec![self.spec()]
    }

    fn compute_provider_key(&self) -> String {
        self.spec().registry_key()
    }

    fn update_policy(&self) -> FactorUpdatePolicy {
        FactorUpdatePolicy::Daily
    }

    fn requirements(&self) -> Vec<DataRequest> {
        self.spec().dependencies
    }

    fn requirements_for_context(&self, context: &FactorContext) -> Vec<DataRequest> {
        self.requirements()
            .into_iter()
            .map(|request| {
                let dates = request.resolved_dates(context);
                request.with_explicit_dates(dates)
            })
            .collect()
    }

    fn intraday_raw_specs(&self) -> Vec<IntradayDailyRawSpec> {
        Vec::new()
    }

    fn intraday_raw_provider_key(&self, _raw_id: &str) -> String {
        self.spec().registry_key()
    }

    fn intraday_raw_materialize_mode(&self, _raw_ids: &[String]) -> IntradayRawMaterializeMode {
        IntradayRawMaterializeMode::Stateless
    }

    fn initial_intraday_raw_state(&self, _raw_ids: &[String]) -> Box<dyn Any + Send> {
        Box::new(())
    }

    fn intraday_raw_auxiliary_requirements(
        &self,
        _raw_ids: &[String],
    ) -> Vec<IntradayDailyRawAuxiliaryRequest> {
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

    fn minute_compute_stateful_many(
        &self,
        raw_ids: &[String],
        context: &FactorContext,
        data: &DataPool,
        _state: &mut dyn Any,
    ) -> Result<Vec<IntradayDailyRawSeries>> {
        self.minute_compute_many(raw_ids, context, data)
    }

    fn compute(&self, context: &FactorContext, data: &DataPool) -> Result<FactorSeries>;

    fn compute_many(
        &self,
        requested_ids: &[String],
        context: &FactorContext,
        data: &DataPool,
    ) -> Result<Vec<FactorSeries>> {
        let spec = self.spec();
        if requested_ids.iter().any(|id| id == &spec.id) {
            Ok(vec![self.compute(context, data)?])
        } else {
            Ok(Vec::new())
        }
    }

    fn initial_compute_state(&self, _requested_ids: &[String]) -> Box<dyn Any + Send> {
        Box::new(())
    }

    fn compute_many_stateful(
        &self,
        requested_ids: &[String],
        context: &FactorContext,
        data: &DataPool,
        _state: &mut (dyn Any + Send),
    ) -> Result<Vec<FactorSeries>> {
        self.compute_many(requested_ids, context, data)
    }
}

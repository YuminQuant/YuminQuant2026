pub mod chn_stock;
pub mod common;
pub mod engine;
pub mod registry;

use std::any::Any;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};

use crate::core::{BarraSeries, BarraSpec, DatasetId, FactorContext};
use crate::data::{ColumnData, DataPool, Table};
use crate::error::Result;
use crate::factor::common::{PitFinancialData, ReportTypePreference};

#[derive(Default)]
pub struct BarraSharedCache {
    pit_financial: Mutex<BTreeMap<(DatasetId, ReportTypePreference), Arc<PitFinancialData>>>,
}

impl BarraSharedCache {
    pub fn pit_financial_data(
        &self,
        data: &DataPool,
        dataset: DatasetId,
        preference: ReportTypePreference,
    ) -> Result<Arc<PitFinancialData>> {
        let key = (dataset, preference.clone());
        let mut cache = self
            .pit_financial
            .lock()
            .expect("Barra shared financial PIT cache lock");
        if let Some(existing) = cache.get(&key) {
            return Ok(Arc::clone(existing));
        }
        let table = data.daily(dataset)?;
        let value_columns = financial_value_columns(table);
        let value_column_refs = value_columns.iter().map(String::as_str).collect::<Vec<_>>();
        let parsed = Arc::new(PitFinancialData::from_table(
            table,
            &value_column_refs,
            preference,
        )?);
        cache.insert(key, Arc::clone(&parsed));
        Ok(parsed)
    }
}

fn financial_value_columns(table: &Table) -> Vec<String> {
    const META_COLUMNS: &[&str] = &[
        "ts_code",
        "ann_date",
        "f_ann_date",
        "end_date",
        "update_flag",
        "report_type",
        "quarter",
    ];
    table
        .columns
        .iter()
        .filter_map(|(name, column)| {
            (!META_COLUMNS.contains(&name.as_str())
                && matches!(
                    column,
                    ColumnData::I32(_)
                        | ColumnData::I64(_)
                        | ColumnData::F32(_)
                        | ColumnData::F64(_)
                ))
            .then(|| name.clone())
        })
        .collect()
}

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
        _shared_cache: &BarraSharedCache,
    ) -> Result<Vec<BarraSeries>> {
        self.compute(context, data)
    }
}

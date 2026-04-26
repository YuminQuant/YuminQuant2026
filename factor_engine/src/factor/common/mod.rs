pub mod cross_section;
pub mod daily;
pub mod minute;
pub mod panel;
pub mod vector;

use std::collections::BTreeMap;

use crate::data::{ColumnData, Table};
use crate::error::Result;

pub use cross_section::{
    compute_daily_cross_section, ClassificationLevel, ClassificationMap, DailyCrossSection,
};
pub use daily::{compute_daily_by_instrument, DailySeries};
pub use minute::{compute_minute_by_instrument, MinuteSeries};
pub use panel::{DailyPanel, PanelColumn};

fn collect_numeric_columns(
    table: &Table,
    key_columns: &[&str],
) -> Result<BTreeMap<String, Vec<Option<f64>>>> {
    let mut columns = BTreeMap::new();
    for (name, column) in &table.columns {
        if key_columns.iter().any(|key| *key == name.as_str()) {
            continue;
        }
        if matches!(
            column,
            ColumnData::I32(_) | ColumnData::I64(_) | ColumnData::F32(_) | ColumnData::F64(_)
        ) {
            columns.insert(name.clone(), table.required_f64_cast(name)?);
        }
    }
    Ok(columns)
}

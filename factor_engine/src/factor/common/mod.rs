pub mod cross_section;
pub mod daily;
pub mod financial;
pub mod intraday_daily;
pub mod intraday_raw;
pub mod minute;
pub mod panel;
pub mod price_volume;
pub mod stock_daily_ops;
pub mod stock_daily_raw_ids;
pub mod umr;
pub mod vector;

use std::collections::BTreeMap;

use crate::data::{ColumnData, Table};
use crate::error::Result;

pub use cross_section::{
    compute_daily_cross_section, ClassificationLevel, ClassificationMap, DailyCrossSection,
};
pub use daily::{compute_daily_by_instrument, DailySeries};
pub use financial::{
    DeadlinePolicy, PitFinancialData, PitFinancialRecord, QuarterMatrix, QuarterValue,
    ReportTypePreference,
};
pub use intraday_daily::{
    intraday_time_in_range, IntradayDailyPanel, IntradaySeries, IntradayWindow,
};
pub use intraday_raw::{
    clean as clean_intraday_value, intraday_daily_raw_series_to_table, mean as intraday_mean,
    pct_change_at, quantile_linear, stock_minute_raw_spec,
};
pub use minute::{compute_minute_by_instrument, MinuteSeries};
pub use panel::{DailyPanel, PanelColumn};
pub use price_volume::{daily_vwap_from_amount_vol, minute_vwap_from_amount_vol};

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

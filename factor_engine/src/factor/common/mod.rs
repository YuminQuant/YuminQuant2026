pub mod chip;
pub mod cross_section;
pub mod daily;
pub mod dbzq_5min_risk;
pub mod dbzq_coskewness;
pub mod dbzq_financial_efficiency;
pub mod dbzq_intraday_volume_distribution;
pub mod dbzq_profit_skew;
pub mod financial;
pub mod financial_similarity;
pub mod gaussian_financial;
pub mod gfzq_5min_salience;
pub mod gfzq_apl_beta;
pub mod gfzq_behavioral;
pub mod gfzq_hf_resid_std;
pub mod gfzq_jump_vol_5min;
pub mod intraday_daily;
pub mod intraday_raw;
pub mod kyzq_apm;
pub mod kyzq_peak_valley;
pub mod minute;
pub mod mszq_gravity_field;
pub mod mszq_main_force_volatility;
pub mod mszq_momentum_pulse;
pub mod mszq_price_volume_tension;
pub mod panel;
pub mod price_volume;
pub mod stock_daily_ops;
pub mod stock_daily_raw_ids;
pub mod umr;
pub mod vector;
pub mod xyzq_crossday_spillover;
pub mod xyzq_domain_structure;
pub mod xyzq_extreme_gmm;
pub mod xyzq_flow_structure;
pub mod xyzq_industry_spillover;
pub mod xyzq_intraday_contrast;
pub mod xyzq_intraday_distribution;
pub mod xyzq_liquidity;
pub mod xyzq_serial_structure;
pub mod xyzq_volume_shape;
pub mod xyzq_vshape_structure;

use std::collections::BTreeMap;

use crate::data::{ColumnData, Table};
use crate::error::Result;

pub use cross_section::{
    compute_daily_cross_section, ClassificationLevel, ClassificationMap, DailyCrossSection,
};
pub use daily::{compute_daily_by_instrument, DailySeries};
pub use financial::{
    cached_financial_stock_snapshots, cached_financial_stock_snapshots_for_date,
    compute_financial_event_snapshot_streaming, factor_series_to_panel_column,
    financial_event_trade_dates, DividendIndex, DividendReader, EventDrivenCrossSectionCache,
    FinancialEventMarker, FinancialEventMarkerBuilder, FinancialEventSchedule, FinancialPitIndex,
    FinancialPitReader, FinancialRecordMarker, FinancialStatementDataset,
    FinancialStockSnapshotCache, FinancialSyntheticMarker, InstrumentAlignedSnapshotCache,
    PitFinancialRecordView, ReportTypePreference,
};
pub use intraday_daily::{
    intraday_time_in_range, IntradayDailyPanel, IntradaySeries, IntradayWindow,
};
pub use intraday_raw::{
    clean as clean_intraday_value, intraday_daily_raw_series_to_table, mean as intraday_mean,
    pct_change_at, quantile_linear, stock_derived_bar_raw_spec, stock_minute_raw_spec,
    RequestedRawIds,
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

use std::collections::BTreeMap;
use std::path::PathBuf;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use yq_factor_engine::backtest::request::NeutralizeSpec;
use yq_factor_engine::core::{AssetClass, Frequency};
use yq_factor_engine::neutralize::{neutralize_daily_values, NeutralizeDailyValuesRequest};

#[pyfunction]
#[pyo3(signature = (
    trade_date,
    ts_code,
    score,
    neutralize,
    asset = "stock",
    frequency = "daily",
    start_date = None,
    end_date = None,
    project_config_path = None
))]
fn neutralize_daily(
    trade_date: Vec<i32>,
    ts_code: Vec<String>,
    score: Vec<Option<f64>>,
    neutralize: &str,
    asset: &str,
    frequency: &str,
    start_date: Option<i32>,
    end_date: Option<i32>,
    project_config_path: Option<String>,
) -> PyResult<Vec<Option<f64>>> {
    let asset_class = AssetClass::parse(&asset.to_ascii_lowercase())
        .ok_or_else(|| PyValueError::new_err(format!("invalid asset: {asset}")))?;
    let frequency = Frequency::parse(&frequency.to_ascii_lowercase())
        .ok_or_else(|| PyValueError::new_err(format!("invalid frequency: {frequency}")))?;
    let neutralize = NeutralizeSpec::parse(neutralize)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let start_date = start_date.unwrap_or_else(|| trade_date.iter().copied().min().unwrap_or(0));
    let end_date = end_date.unwrap_or_else(|| trade_date.iter().copied().max().unwrap_or(0));
    let request = NeutralizeDailyValuesRequest {
        trade_dates: trade_date,
        ts_codes: ts_code,
        scores: score,
        neutralize,
        asset_class,
        frequency,
        start_date,
        end_date,
        project_config_path: project_config_path.map(PathBuf::from),
        sector: None,
        barra: BTreeMap::new(),
    };
    neutralize_daily_values(&request).map_err(|error| PyValueError::new_err(error.to_string()))
}

#[pymodule]
fn yq_factor_engine_py(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(neutralize_daily, module)?)?;
    Ok(())
}

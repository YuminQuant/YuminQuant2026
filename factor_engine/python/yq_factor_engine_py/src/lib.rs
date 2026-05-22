use std::collections::BTreeMap;
use std::convert::TryInto;
use std::path::PathBuf;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use yq_factor_engine::backtest::request::NeutralizeSpec;
use yq_factor_engine::core::{AssetClass, Frequency};
use yq_factor_engine::logsig_signature::{logsig_signature_batch_from_volume, signature_width};
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

#[pyfunction]
fn logsig_signature_batch(
    py: Python<'_>,
    volume: &Bound<'_, PyAny>,
    order: usize,
) -> PyResult<PyObject> {
    let numpy = py.import_bound("numpy")?;
    let float64 = numpy.getattr("float64")?;
    let array = numpy.call_method1("ascontiguousarray", (volume, float64))?;
    let ndim = array.getattr("ndim")?.extract::<usize>()?;
    if ndim != 2 {
        return Err(PyValueError::new_err(format!(
            "logsig_signature_batch expects a 2D volume array, got ndim={ndim}"
        )));
    }
    let shape = array.getattr("shape")?.extract::<Vec<usize>>()?;
    if shape.len() != 2 {
        return Err(PyValueError::new_err("could not read volume array shape"));
    }
    let rows = shape[0];
    let cols = shape[1];
    let raw = array.call_method0("tobytes")?.extract::<Vec<u8>>()?;
    let expected_bytes = rows
        .checked_mul(cols)
        .and_then(|count| count.checked_mul(std::mem::size_of::<f64>()))
        .ok_or_else(|| PyValueError::new_err("volume array is too large"))?;
    if raw.len() != expected_bytes {
        return Err(PyValueError::new_err(format!(
            "volume byte length {} does not match shape {rows}x{cols}",
            raw.len()
        )));
    }
    let mut values = Vec::with_capacity(rows * cols);
    for chunk in raw.chunks_exact(std::mem::size_of::<f64>()) {
        let bytes: [u8; 8] = chunk.try_into().expect("chunks_exact yields 8 bytes");
        values.push(f64::from_ne_bytes(bytes));
    }

    let width = signature_width(order).map_err(|error| PyValueError::new_err(error.to_string()))?;
    let signatures = py
        .allow_threads(|| logsig_signature_batch_from_volume(&values, rows, cols, order))
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let signature_bytes = unsafe {
        std::slice::from_raw_parts(
            signatures.as_ptr() as *const u8,
            signatures.len() * std::mem::size_of::<f32>(),
        )
    };
    let buffer = PyBytes::new_bound(py, signature_bytes);
    let float32 = numpy.getattr("float32")?;
    let output = numpy.call_method1("frombuffer", (&buffer, float32))?;
    let reshaped = output.call_method1("reshape", (rows, width))?;
    Ok(reshaped.into_py(py))
}

#[pymodule]
fn yq_factor_engine_py(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(neutralize_daily, module)?)?;
    module.add_function(wrap_pyfunction!(logsig_signature_batch, module)?)?;
    Ok(())
}

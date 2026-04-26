use std::collections::BTreeMap;
use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, BooleanArray, Float32Array, Float64Array, Int32Array, Int64Array, StringArray,
};
use arrow_schema::{DataType, Field};

use crate::error::{err, Result};

#[derive(Clone, Debug)]
pub enum ColumnData {
    Utf8(Vec<Option<String>>),
    I32(Vec<Option<i32>>),
    I64(Vec<Option<i64>>),
    F32(Vec<Option<f32>>),
    F64(Vec<Option<f64>>),
    Bool(Vec<Option<bool>>),
}

impl ColumnData {
    pub fn len(&self) -> usize {
        match self {
            Self::Utf8(values) => values.len(),
            Self::I32(values) => values.len(),
            Self::I64(values) => values.len(),
            Self::F32(values) => values.len(),
            Self::F64(values) => values.len(),
            Self::Bool(values) => values.len(),
        }
    }

    pub fn take(&self, indices: &[usize]) -> Self {
        match self {
            Self::Utf8(values) => {
                Self::Utf8(indices.iter().map(|idx| values[*idx].clone()).collect())
            }
            Self::I32(values) => Self::I32(indices.iter().map(|idx| values[*idx]).collect()),
            Self::I64(values) => Self::I64(indices.iter().map(|idx| values[*idx]).collect()),
            Self::F32(values) => Self::F32(indices.iter().map(|idx| values[*idx]).collect()),
            Self::F64(values) => Self::F64(indices.iter().map(|idx| values[*idx]).collect()),
            Self::Bool(values) => Self::Bool(indices.iter().map(|idx| values[*idx]).collect()),
        }
    }

    pub fn append(&mut self, other: &Self) -> Result<()> {
        match self {
            Self::Utf8(left) => match other {
                Self::Utf8(right) => left.extend(right.iter().cloned()),
                _ => return Err(err("cannot append non-utf8 column into utf8 column")),
            },
            Self::Bool(left) => match other {
                Self::Bool(right) => left.extend(right.iter().copied()),
                _ => return Err(err("cannot append non-bool column into bool column")),
            },
            Self::I32(left) => match other {
                Self::I32(right) => left.extend(right.iter().copied()),
                Self::I64(right) => {
                    let mut promoted = left
                        .iter()
                        .map(|value| value.map(i64::from))
                        .collect::<Vec<_>>();
                    promoted.extend(right.iter().copied());
                    *self = Self::I64(promoted);
                }
                Self::F32(right) => {
                    let mut promoted = left
                        .iter()
                        .map(|value| value.map(|v| v as f32))
                        .collect::<Vec<_>>();
                    promoted.extend(right.iter().copied());
                    *self = Self::F32(promoted);
                }
                Self::F64(right) => {
                    let mut promoted = left
                        .iter()
                        .map(|value| value.map(f64::from))
                        .collect::<Vec<_>>();
                    promoted.extend(right.iter().copied());
                    *self = Self::F64(promoted);
                }
                _ => return Err(err("cannot append non-numeric column into int32 column")),
            },
            Self::I64(left) => match other {
                Self::I32(right) => left.extend(right.iter().map(|value| value.map(i64::from))),
                Self::I64(right) => left.extend(right.iter().copied()),
                Self::F32(right) => {
                    let mut promoted = left
                        .iter()
                        .map(|value| value.map(|v| v as f64))
                        .collect::<Vec<_>>();
                    promoted.extend(right.iter().map(|value| value.map(f64::from)));
                    *self = Self::F64(promoted);
                }
                Self::F64(right) => {
                    let mut promoted = left
                        .iter()
                        .map(|value| value.map(|v| v as f64))
                        .collect::<Vec<_>>();
                    promoted.extend(right.iter().copied());
                    *self = Self::F64(promoted);
                }
                _ => return Err(err("cannot append non-numeric column into int64 column")),
            },
            Self::F32(left) => match other {
                Self::I32(right) => left.extend(right.iter().map(|value| value.map(|v| v as f32))),
                Self::I64(right) => {
                    let mut promoted = left
                        .iter()
                        .map(|value| value.map(f64::from))
                        .collect::<Vec<_>>();
                    promoted.extend(right.iter().map(|value| value.map(|v| v as f64)));
                    *self = Self::F64(promoted);
                }
                Self::F32(right) => left.extend(right.iter().copied()),
                Self::F64(right) => {
                    let mut promoted = left
                        .iter()
                        .map(|value| value.map(f64::from))
                        .collect::<Vec<_>>();
                    promoted.extend(right.iter().copied());
                    *self = Self::F64(promoted);
                }
                _ => return Err(err("cannot append non-numeric column into float32 column")),
            },
            Self::F64(left) => match other {
                Self::I32(right) => left.extend(right.iter().map(|value| value.map(f64::from))),
                Self::I64(right) => left.extend(right.iter().map(|value| value.map(|v| v as f64))),
                Self::F32(right) => left.extend(right.iter().map(|value| value.map(f64::from))),
                Self::F64(right) => left.extend(right.iter().copied()),
                _ => return Err(err("cannot append non-numeric column into float64 column")),
            },
        }
        Ok(())
    }

    pub fn field(&self, name: &str) -> Field {
        let data_type = match self {
            Self::Utf8(_) => DataType::Utf8,
            Self::I32(_) => DataType::Int32,
            Self::I64(_) => DataType::Int64,
            Self::F32(_) => DataType::Float32,
            Self::F64(_) => DataType::Float64,
            Self::Bool(_) => DataType::Boolean,
        };
        Field::new(name, data_type, true)
    }

    pub fn to_arrow(&self) -> ArrayRef {
        match self {
            Self::Utf8(values) => Arc::new(StringArray::from(values.clone())),
            Self::I32(values) => Arc::new(Int32Array::from(values.clone())),
            Self::I64(values) => Arc::new(Int64Array::from(values.clone())),
            Self::F32(values) => Arc::new(Float32Array::from(values.clone())),
            Self::F64(values) => Arc::new(Float64Array::from(values.clone())),
            Self::Bool(values) => Arc::new(BooleanArray::from(values.clone())),
        }
    }

    pub fn from_arrow(array: &ArrayRef) -> Result<Self> {
        if let Some(values) = array.as_any().downcast_ref::<StringArray>() {
            return Ok(Self::Utf8(
                (0..values.len())
                    .map(|idx| (!values.is_null(idx)).then(|| values.value(idx).to_string()))
                    .collect(),
            ));
        }
        if let Some(values) = array.as_any().downcast_ref::<Int32Array>() {
            return Ok(Self::I32(
                (0..values.len())
                    .map(|idx| (!values.is_null(idx)).then(|| values.value(idx)))
                    .collect(),
            ));
        }
        if let Some(values) = array.as_any().downcast_ref::<Int64Array>() {
            return Ok(Self::I64(
                (0..values.len())
                    .map(|idx| (!values.is_null(idx)).then(|| values.value(idx)))
                    .collect(),
            ));
        }
        if let Some(values) = array.as_any().downcast_ref::<Float32Array>() {
            return Ok(Self::F32(
                (0..values.len())
                    .map(|idx| (!values.is_null(idx)).then(|| values.value(idx)))
                    .collect(),
            ));
        }
        if let Some(values) = array.as_any().downcast_ref::<Float64Array>() {
            return Ok(Self::F64(
                (0..values.len())
                    .map(|idx| (!values.is_null(idx)).then(|| values.value(idx)))
                    .collect(),
            ));
        }
        if let Some(values) = array.as_any().downcast_ref::<BooleanArray>() {
            return Ok(Self::Bool(
                (0..values.len())
                    .map(|idx| (!values.is_null(idx)).then(|| values.value(idx)))
                    .collect(),
            ));
        }
        Err(err(format!(
            "unsupported arrow data type in parquet reader: {:?}",
            array.data_type()
        )))
    }
}

#[derive(Clone, Debug, Default)]
pub struct Table {
    pub columns: BTreeMap<String, ColumnData>,
    pub len: usize,
}

impl Table {
    pub fn new(columns: BTreeMap<String, ColumnData>) -> Result<Self> {
        let len = columns.values().next().map(ColumnData::len).unwrap_or(0);
        for (name, column) in &columns {
            if column.len() != len {
                return Err(err(format!(
                    "column {} has length {}, expected {}",
                    name,
                    column.len(),
                    len
                )));
            }
        }
        Ok(Self { columns, len })
    }

    pub fn empty() -> Self {
        Self::default()
    }

    pub fn column_names(&self) -> Vec<String> {
        self.columns.keys().cloned().collect()
    }

    pub fn insert(&mut self, name: impl Into<String>, column: ColumnData) -> Result<()> {
        if self.columns.is_empty() {
            self.len = column.len();
        } else if column.len() != self.len {
            return Err(err(format!(
                "inserted column has length {}, expected {}",
                column.len(),
                self.len
            )));
        }
        self.columns.insert(name.into(), column);
        Ok(())
    }

    pub fn append(&mut self, other: &Table) -> Result<()> {
        if self.columns.is_empty() {
            *self = other.clone();
            return Ok(());
        }
        for (name, column) in &other.columns {
            let target = self
                .columns
                .get_mut(name)
                .ok_or_else(|| err(format!("missing column {} while appending table", name)))?;
            target.append(column)?;
        }
        self.len += other.len;
        Ok(())
    }

    pub fn take(&self, indices: &[usize]) -> Result<Self> {
        let mut columns = BTreeMap::new();
        for (name, column) in &self.columns {
            columns.insert(name.clone(), column.take(indices));
        }
        Self::new(columns)
    }

    pub fn filter_i32_range(&self, column_name: &str, start: i32, end: i32) -> Result<Self> {
        let dates = self.required_i32(column_name)?;
        let indices: Vec<usize> = dates
            .iter()
            .enumerate()
            .filter_map(|(idx, value)| {
                value.and_then(|date| (date >= start && date <= end).then_some(idx))
            })
            .collect();
        self.take(&indices)
    }

    pub fn required_i32(&self, name: &str) -> Result<&Vec<Option<i32>>> {
        match self.columns.get(name) {
            Some(ColumnData::I32(values)) => Ok(values),
            Some(ColumnData::I64(_)) => {
                return Err(err(format!(
                    "column {} is i64; call required_i64_cast for casted access",
                    name
                )))
            }
            Some(_) => Err(err(format!("column {} is not int32", name))),
            None => Err(err(format!("missing required column {}", name))),
        }
    }

    pub fn required_i64_cast(&self, name: &str) -> Result<Vec<Option<i64>>> {
        match self.columns.get(name) {
            Some(ColumnData::I64(values)) => Ok(values.clone()),
            Some(ColumnData::I32(values)) => {
                Ok(values.iter().map(|value| value.map(i64::from)).collect())
            }
            Some(_) => Err(err(format!("column {} cannot be cast to int64", name))),
            None => Err(err(format!("missing required column {}", name))),
        }
    }

    pub fn required_i32_date_cast(&self, name: &str) -> Result<Vec<Option<i32>>> {
        match self.columns.get(name) {
            Some(ColumnData::I32(values)) => Ok(values.clone()),
            Some(ColumnData::I64(values)) => values
                .iter()
                .map(|value| {
                    value
                        .map(|value| {
                            i32::try_from(value).map_err(|_| {
                                err(format!(
                                    "date column {} value {} is outside int32 range",
                                    name, value
                                ))
                            })
                        })
                        .transpose()
                })
                .collect(),
            Some(ColumnData::Utf8(values)) => values
                .iter()
                .map(|value| {
                    value
                        .as_deref()
                        .and_then(normalize_date_text)
                        .map(|value| {
                            value.parse::<i32>().map_err(|_| {
                                err(format!(
                                    "date column {} value {} cannot be parsed as YYYYMMDD",
                                    name, value
                                ))
                            })
                        })
                        .transpose()
                })
                .collect(),
            Some(_) => Err(err(format!(
                "column {} cannot be cast to YYYYMMDD date",
                name
            ))),
            None => Err(err(format!("missing required column {}", name))),
        }
    }

    pub fn required_utf8(&self, name: &str) -> Result<&Vec<Option<String>>> {
        match self.columns.get(name) {
            Some(ColumnData::Utf8(values)) => Ok(values),
            Some(_) => Err(err(format!("column {} is not utf8", name))),
            None => Err(err(format!("missing required column {}", name))),
        }
    }

    pub fn required_f64_cast(&self, name: &str) -> Result<Vec<Option<f64>>> {
        match self.columns.get(name) {
            Some(ColumnData::F64(values)) => Ok(values.clone()),
            Some(ColumnData::F32(values)) => {
                Ok(values.iter().map(|value| value.map(f64::from)).collect())
            }
            Some(ColumnData::I64(values)) => {
                Ok(values.iter().map(|value| value.map(|v| v as f64)).collect())
            }
            Some(ColumnData::I32(values)) => {
                Ok(values.iter().map(|value| value.map(f64::from)).collect())
            }
            Some(_) => Err(err(format!("column {} cannot be cast to float64", name))),
            None => Err(err(format!("missing required column {}", name))),
        }
    }
}

fn normalize_date_text(value: &str) -> Option<&str> {
    let value = value.trim();
    if value.is_empty() {
        return None;
    }
    match value.to_ascii_lowercase().as_str() {
        "nan" | "none" | "null" | "nat" => None,
        _ => Some(value),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::{ColumnData, Table};

    #[test]
    fn date_cast_treats_common_text_missing_values_as_none() {
        let table = Table::new(BTreeMap::from([(
            "date".to_string(),
            ColumnData::Utf8(vec![
                Some("20260424".to_string()),
                Some("".to_string()),
                Some("nan".to_string()),
                Some("NaN".to_string()),
                Some("None".to_string()),
                Some("null".to_string()),
                Some("NaT".to_string()),
                None,
            ]),
        )]))
        .expect("valid table");

        assert_eq!(
            table.required_i32_date_cast("date").expect("date cast"),
            vec![Some(20260424), None, None, None, None, None, None, None]
        );
    }
}

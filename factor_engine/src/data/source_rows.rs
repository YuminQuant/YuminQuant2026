use std::borrow::Cow;
use std::ops::Index;
use std::sync::Arc;

use crate::data::{ColumnData, Table};
use crate::error::{err, Result};

/// A bounded row selection over immutable yearly sources, without a merged table.
#[derive(Clone, Debug)]
pub struct SourceRows {
    sources: Vec<Arc<Table>>,
    rows: Vec<(usize, usize)>,
}

impl SourceRows {
    pub fn new(
        sources: Vec<Arc<Table>>,
        date_column: &str,
        lower: i32,
        upper: i32,
    ) -> Result<Self> {
        let mut rows = Vec::new();
        for (source, table) in sources.iter().enumerate() {
            for (row, date) in table
                .required_i32_date_cast(date_column)?
                .into_iter()
                .enumerate()
            {
                if date.is_some_and(|date| date >= lower && date <= upper) {
                    rows.push((source, row));
                }
            }
        }
        Ok(Self { sources, rows })
    }

    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn reader(&self) -> SourceRowsReader<'_> {
        SourceRowsReader {
            sources: self.sources.iter().map(Arc::as_ref).collect(),
            rows: Cow::Borrowed(&self.rows),
            len: self.len(),
        }
    }
}

pub struct SourceRowsReader<'a> {
    sources: Vec<&'a Table>,
    rows: Cow<'a, [(usize, usize)]>,
    pub len: usize,
}

impl<'a> From<&'a Table> for SourceRowsReader<'a> {
    fn from(table: &'a Table) -> Self {
        Self {
            sources: vec![table],
            rows: Cow::Owned((0..table.len).map(|row| (0, row)).collect()),
            len: table.len,
        }
    }
}

pub struct SourceStrings<'a>(Vec<&'a Option<String>>);

impl<'a> Index<usize> for SourceStrings<'a> {
    type Output = &'a Option<String>;
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<'a> SourceRowsReader<'a> {
    pub fn required_utf8(&self, name: &str) -> Result<SourceStrings<'a>> {
        let columns = self
            .sources
            .iter()
            .map(|table| table.required_utf8(name))
            .collect::<Result<Vec<_>>>()?;
        Ok(SourceStrings(
            self.rows
                .iter()
                .map(|&(source, row)| &columns[source][row])
                .collect(),
        ))
    }

    pub fn required_f64_cast(&self, name: &str) -> Result<Vec<Option<f64>>> {
        let columns = self
            .sources
            .iter()
            .map(|table| match table.columns.get(name) {
                Some(
                    column @ (ColumnData::F64(_)
                    | ColumnData::F32(_)
                    | ColumnData::I64(_)
                    | ColumnData::I32(_)),
                ) => Ok(column),
                Some(_) => Err(err(format!("column {name} cannot be cast to float64"))),
                None => Err(err(format!("missing required column {name}"))),
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(self
            .rows
            .iter()
            .map(|&(source, row)| match columns[source] {
                ColumnData::F64(values) => values[row],
                ColumnData::F32(values) => values[row].map(f64::from),
                ColumnData::I64(values) => values[row].map(|v| v as f64),
                ColumnData::I32(values) => values[row].map(f64::from),
                _ => unreachable!("validated numeric column"),
            })
            .collect())
    }

    pub fn required_i32_date_cast(&self, name: &str) -> Result<Vec<Option<i32>>> {
        let columns = self
            .sources
            .iter()
            .map(|table| match table.columns.get(name) {
                Some(ColumnData::I32(values)) => Ok(Cow::Borrowed(values.as_slice())),
                _ => table.required_i32_date_cast(name).map(Cow::Owned),
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(self
            .rows
            .iter()
            .map(|&(source, row)| columns[source][row])
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn source(dates: &[i32], code: &str) -> Arc<Table> {
        Arc::new(
            Table::new(BTreeMap::from([
                (
                    "report_date".into(),
                    ColumnData::I32(dates.iter().copied().map(Some).collect()),
                ),
                (
                    "ts_code".into(),
                    ColumnData::Utf8(vec![Some(code.into()); dates.len()]),
                ),
                (
                    "np".into(),
                    ColumnData::F32((0..dates.len()).map(|i| Some(i as f32 + 0.5)).collect()),
                ),
            ]))
            .unwrap(),
        )
    }

    #[test]
    fn analyst_multi_sources_match_merged_filter_without_copying_strings() {
        let a = source(&[20231201, 20231231], "A");
        let b = source(&[20240101, 20240501], "B");
        let rows = SourceRows::new(
            vec![a.clone(), b.clone()],
            "report_date",
            20231231,
            20240101,
        )
        .unwrap();
        let reader = rows.reader();
        let mut merged = (*a).clone();
        merged.append(&b).unwrap();
        let expected = merged
            .filter_i32_range("report_date", 20231231, 20240101)
            .unwrap();
        assert_eq!(reader.len, expected.len);
        assert_eq!(
            reader.required_f64_cast("np").unwrap(),
            expected.required_f64_cast("np").unwrap()
        );
        assert_eq!(
            reader.required_i32_date_cast("report_date").unwrap(),
            expected.required_i32_date_cast("report_date").unwrap()
        );
        let strings = reader.required_utf8("ts_code").unwrap();
        assert_eq!(strings[0].as_deref(), Some("A"));
        assert_eq!(strings[1].as_deref(), Some("B"));
        assert!(std::ptr::eq(
            strings[0],
            &a.required_utf8("ts_code").unwrap()[1]
        ));
        assert!(matches!(reader.rows, Cow::Borrowed(_)));
        let weak = Arc::downgrade(&a);
        drop(reader);
        drop(rows);
        drop(a);
        assert!(weak.upgrade().is_none());
    }

    #[test]
    fn analyst_multi_sources_reject_missing_columns_even_on_empty_selection() {
        let a = source(&[20240101], "A");
        let mut missing = (*a).clone();
        missing.columns.remove("np");
        let rows = SourceRows::new(
            vec![a, Arc::new(missing)],
            "report_date",
            20250101,
            20251231,
        )
        .unwrap();
        assert_eq!(rows.len(), 0);
        assert!(rows.reader().required_f64_cast("np").is_err());
        assert!(rows.reader().required_utf8("np").is_err());
    }
}

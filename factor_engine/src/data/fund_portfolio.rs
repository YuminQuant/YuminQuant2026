use std::path::PathBuf;
use std::sync::Arc;

use crate::data::{ColumnData, Table};
use crate::error::{err, Result};

pub const FUND_PORTFOLIO_COLUMNS: &[&str] = &[
    "ts_code",
    "ann_date",
    "end_date",
    "symbol",
    "mkv",
    "amount",
    "stk_mkv_ratio",
    "stk_float_ratio",
];

#[derive(Clone, Debug)]
struct DisclosureGroup {
    start: usize,
    end: usize,
    ann_date: i32,
    end_date: i32,
}

/// Preference-free raw disclosure groups. No deduplication, conflict policy or top-N filter.
#[derive(Debug)]
pub struct FundPortfolioIndex {
    source: Arc<Table>,
    rows: Vec<usize>,
    groups: Vec<DisclosureGroup>,
}

impl FundPortfolioIndex {
    pub fn from_table(source: Arc<Table>) -> Result<Self> {
        let funds = source.required_utf8("ts_code")?;
        source.required_utf8("symbol")?;
        let announcements = source.required_i32_date_cast("ann_date")?;
        let periods = source.required_i32_date_cast("end_date")?;
        for name in &FUND_PORTFOLIO_COLUMNS[4..] {
            match source.columns.get(*name) {
                Some(
                    ColumnData::F64(_)
                    | ColumnData::F32(_)
                    | ColumnData::I32(_)
                    | ColumnData::I64(_),
                ) => {}
                _ => {
                    return Err(err(format!(
                        "missing or nonnumeric fund portfolio column {name}"
                    )))
                }
            }
        }
        let mut rows = (0..source.len).collect::<Vec<_>>();
        for &row in &rows {
            if funds[row].as_ref().is_none_or(String::is_empty)
                || announcements[row].is_none()
                || periods[row].is_none()
            {
                return Err(err("fund portfolio row has a missing disclosure key"));
            }
            if announcements[row] < periods[row] {
                return Err(err("fund portfolio ann_date precedes end_date"));
            }
        }
        rows.sort_by(|&a, &b| {
            (announcements[a], &funds[a], periods[a]).cmp(&(
                announcements[b],
                &funds[b],
                periods[b],
            ))
        });
        let mut groups = Vec::new();
        let mut start = 0;
        while start < rows.len() {
            let row = rows[start];
            let mut end = start + 1;
            while end < rows.len() {
                let next = rows[end];
                if (announcements[next], &funds[next], periods[next])
                    != (announcements[row], &funds[row], periods[row])
                {
                    break;
                }
                end += 1;
            }
            groups.push(DisclosureGroup {
                start,
                end,
                ann_date: announcements[row].unwrap(),
                end_date: periods[row].unwrap(),
            });
            start = end;
        }
        rows.shrink_to_fit();
        groups.shrink_to_fit();
        Ok(Self {
            source,
            rows,
            groups,
        })
    }

    pub fn len(&self) -> usize {
        self.rows.len()
    }

    fn disclosures_between(
        &self,
        after: Option<i32>,
        until: i32,
    ) -> impl Iterator<Item = FundDisclosureView<'_>> {
        let start = after.map_or(0, |after| {
            self.groups.partition_point(|g| g.ann_date <= after)
        });
        let end = self.groups.partition_point(|g| g.ann_date <= until);
        self.groups[start.min(end)..end]
            .iter()
            .map(|group| FundDisclosureView { index: self, group })
    }
}

#[derive(Clone, Copy)]
pub struct FundDisclosureView<'a> {
    index: &'a FundPortfolioIndex,
    group: &'a DisclosureGroup,
}

impl<'a> FundDisclosureView<'a> {
    pub fn fund_code(&self) -> &'a str {
        self.index.source.required_utf8("ts_code").unwrap()[self.rows()[0]]
            .as_deref()
            .unwrap()
    }
    pub fn ann_date(&self) -> i32 {
        self.group.ann_date
    }
    pub fn end_date(&self) -> i32 {
        self.group.end_date
    }
    pub fn rows(&self) -> &'a [usize] {
        &self.index.rows[self.group.start..self.group.end]
    }
    pub fn symbol(&self, row: usize) -> Option<&'a str> {
        self.index.source.required_utf8("symbol").unwrap()[row].as_deref()
    }
    pub fn value(&self, row: usize, column: &str) -> Option<f64> {
        match self.index.source.columns.get(column)? {
            ColumnData::F64(v) => v[row],
            ColumnData::F32(v) => v[row].map(f64::from),
            ColumnData::I32(v) => v[row].map(f64::from),
            ColumnData::I64(v) => v[row].map(|v| v as f64),
            _ => None,
        }
    }
}

#[derive(Debug)]
pub struct FundPortfolioBatchIndex {
    partitions: Vec<Arc<FundPortfolioIndex>>,
    history_files: Vec<PathBuf>,
    upper: i32,
}

impl FundPortfolioBatchIndex {
    pub fn new(
        partitions: Vec<Arc<FundPortfolioIndex>>,
        history_files: Vec<PathBuf>,
        upper: i32,
    ) -> Self {
        Self {
            partitions,
            history_files,
            upper,
        }
    }
    pub fn len(&self) -> usize {
        self.partitions.iter().map(|p| p.len()).sum()
    }
    pub fn reader(&self) -> FundPortfolioReader<'_> {
        FundPortfolioReader { index: self }
    }
}

#[derive(Clone, Copy)]
pub struct FundPortfolioReader<'a> {
    index: &'a FundPortfolioBatchIndex,
}

impl FundPortfolioReader<'_> {
    pub fn disclosures_between(
        &self,
        after: Option<i32>,
        until: i32,
    ) -> Vec<FundDisclosureView<'_>> {
        if after.is_some_and(|after| after >= until) {
            return Vec::new();
        }
        let mut groups = self
            .index
            .partitions
            .iter()
            .flat_map(|p| p.disclosures_between(after, until.min(self.index.upper)))
            .collect::<Vec<_>>();
        groups.sort_by(|a, b| {
            (a.ann_date(), a.fund_code(), a.end_date()).cmp(&(
                b.ann_date(),
                b.fund_code(),
                b.end_date(),
            ))
        });
        groups
    }

    /// Historical sources are consumed one year at a time and never retained by the batch cache.
    pub fn bootstrap_until(
        &self,
        until: i32,
        mut consume: impl FnMut(FundDisclosureView<'_>) -> Result<()>,
    ) -> Result<()> {
        let columns = FUND_PORTFOLIO_COLUMNS
            .iter()
            .map(|c| c.to_string())
            .collect::<Vec<_>>();
        for file in &self.index.history_files {
            let table = Arc::new(crate::data::parquet_io::read_parquet(file, Some(&columns))?);
            let index = FundPortfolioIndex::from_table(table)?;
            for group in index.disclosures_between(None, until.min(self.index.upper)) {
                consume(group)?;
            }
        }
        for group in self.disclosures_between(None, until) {
            consume(group)?;
        }
        Ok(())
    }
}

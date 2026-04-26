use std::collections::BTreeMap;

use crate::data::Table;
use crate::error::Result;

#[derive(Clone, Debug)]
pub struct PitFinancialRecord {
    pub end_date: i32,
    pub disclosure_date: i32,
    pub update_flag: i64,
    columns: BTreeMap<String, Option<f64>>,
}

impl PitFinancialRecord {
    pub fn column(&self, name: &str) -> Option<f64> {
        self.columns.get(name).copied().flatten()
    }
}

#[derive(Clone, Debug, Default)]
pub struct PitFinancialData {
    by_ts_code: BTreeMap<String, BTreeMap<i32, Vec<PitFinancialRecord>>>,
}

impl PitFinancialData {
    pub fn from_table(table: &Table, value_columns: &[&str]) -> Result<Self> {
        let ts_codes = table.required_utf8("ts_code")?;
        let ann_dates = table.required_i32_date_cast("ann_date")?;
        let f_ann_dates = table.required_i32_date_cast("f_ann_date")?;
        let end_dates = table.required_i32_date_cast("end_date")?;
        let update_flags = table.required_i64_cast("update_flag")?;
        let mut value_data = BTreeMap::new();
        for column in value_columns {
            value_data.insert((*column).to_string(), table.required_f64_cast(column)?);
        }

        let mut by_ts_code: BTreeMap<String, BTreeMap<i32, Vec<PitFinancialRecord>>> =
            BTreeMap::new();
        for idx in 0..table.len {
            let (Some(ts_code), Some(end_date), Some(disclosure_date)) = (
                ts_codes[idx].clone(),
                end_dates[idx],
                f_ann_dates[idx].or(ann_dates[idx]),
            ) else {
                continue;
            };
            let columns = value_data
                .iter()
                .map(|(name, values)| (name.clone(), values[idx]))
                .collect::<BTreeMap<_, _>>();
            by_ts_code
                .entry(ts_code)
                .or_default()
                .entry(end_date)
                .or_default()
                .push(PitFinancialRecord {
                    end_date,
                    disclosure_date,
                    update_flag: update_flags[idx].unwrap_or(0),
                    columns,
                });
        }

        for by_end_date in by_ts_code.values_mut() {
            for versions in by_end_date.values_mut() {
                versions.sort_by(|left, right| {
                    right
                        .disclosure_date
                        .cmp(&left.disclosure_date)
                        .then_with(|| right.update_flag.cmp(&left.update_flag))
                });
            }
        }

        Ok(Self { by_ts_code })
    }

    pub fn latest_quarters(
        &self,
        ts_code: &str,
        trade_date: i32,
        count: usize,
    ) -> Vec<&PitFinancialRecord> {
        let Some(by_end_date) = self.by_ts_code.get(ts_code) else {
            return Vec::new();
        };
        let mut output = Vec::new();
        for versions in by_end_date.values().rev() {
            let Some(record) = versions
                .iter()
                .find(|record| record.disclosure_date <= trade_date)
            else {
                continue;
            };
            output.push(record);
            if output.len() == count {
                break;
            }
        }
        output
    }

    pub fn record_for_end_date(
        &self,
        ts_code: &str,
        trade_date: i32,
        end_date: i32,
    ) -> Option<&PitFinancialRecord> {
        self.by_ts_code
            .get(ts_code)?
            .get(&end_date)?
            .iter()
            .find(|record| record.disclosure_date <= trade_date)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use crate::data::{ColumnData, Table};

    use super::PitFinancialData;

    #[test]
    fn pit_financial_data_uses_only_disclosed_versions() {
        let table = Table::new(BTreeMap::from([
            (
                "ts_code".to_string(),
                ColumnData::Utf8(vec![
                    Some("000001.SZ".to_string()),
                    Some("000001.SZ".to_string()),
                    Some("000001.SZ".to_string()),
                ]),
            ),
            (
                "ann_date".to_string(),
                ColumnData::I32(vec![Some(20250331), Some(20250430), Some(20250430)]),
            ),
            (
                "f_ann_date".to_string(),
                ColumnData::I32(vec![Some(20250331), Some(20250430), Some(20250430)]),
            ),
            (
                "end_date".to_string(),
                ColumnData::I32(vec![Some(20241231), Some(20241231), Some(20240930)]),
            ),
            (
                "update_flag".to_string(),
                ColumnData::I32(vec![Some(0), Some(1), Some(0)]),
            ),
            (
                "value".to_string(),
                ColumnData::F64(vec![Some(10.0), Some(12.0), Some(8.0)]),
            ),
        ]))
        .expect("valid table");
        let data = PitFinancialData::from_table(&table, &["value"]).expect("pit data");

        let before_revision = data.latest_quarters("000001.SZ", 20250401, 1);
        assert_eq!(before_revision[0].column("value"), Some(10.0));

        let after_revision = data.latest_quarters("000001.SZ", 20250501, 1);
        assert_eq!(after_revision[0].column("value"), Some(12.0));
    }

    #[test]
    fn pit_financial_data_prefers_update_flag_on_same_disclosure_date() {
        let table = Table::new(BTreeMap::from([
            (
                "ts_code".to_string(),
                ColumnData::Utf8(vec![
                    Some("000001.SZ".to_string()),
                    Some("000001.SZ".to_string()),
                ]),
            ),
            (
                "ann_date".to_string(),
                ColumnData::I32(vec![Some(20250331), Some(20250331)]),
            ),
            (
                "f_ann_date".to_string(),
                ColumnData::I32(vec![Some(20250331), Some(20250331)]),
            ),
            (
                "end_date".to_string(),
                ColumnData::I32(vec![Some(20241231), Some(20241231)]),
            ),
            (
                "update_flag".to_string(),
                ColumnData::I32(vec![Some(0), Some(1)]),
            ),
            (
                "value".to_string(),
                ColumnData::F64(vec![Some(10.0), Some(11.0)]),
            ),
        ]))
        .expect("valid table");
        let data = PitFinancialData::from_table(&table, &["value"]).expect("pit data");

        let records = data.latest_quarters("000001.SZ", 20250401, 1);
        assert_eq!(records[0].column("value"), Some(11.0));
    }
}

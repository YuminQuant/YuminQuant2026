use std::collections::{BTreeMap, HashSet};

use crate::core::{FactorContext, FactorRowKey, FactorSeries, FactorSpec, FactorValue};
use crate::data::Table;
use crate::error::{err, Result};

use super::collect_numeric_columns;

pub struct DailyCrossSection {
    pub trade_date: i32,
    pub ts_codes: Vec<String>,
    columns: BTreeMap<String, Vec<Option<f64>>>,
}

impl DailyCrossSection {
    pub fn column(&self, name: &str) -> Result<&[Option<f64>]> {
        self.columns.get(name).map(Vec::as_slice).ok_or_else(|| {
            err(format!(
                "missing daily cross-section column {} for {}",
                name, self.trade_date
            ))
        })
    }

    pub fn ts_codes(&self) -> &[String] {
        &self.ts_codes
    }
}

pub fn compute_daily_cross_section<F>(
    spec: FactorSpec,
    context: &FactorContext,
    table: &Table,
    mut expr: F,
) -> Result<FactorSeries>
where
    F: FnMut(&DailyCrossSection) -> Result<Vec<Option<f64>>>,
{
    let target_dates = context.target_dates.iter().copied().collect::<HashSet<_>>();
    let ts_codes = table.required_utf8("ts_code")?;
    let trade_dates = table.required_i32("trade_date")?;
    let value_columns = collect_numeric_columns(table, &["trade_date", "ts_code"])?;
    let mut grouped: BTreeMap<i32, Vec<usize>> = BTreeMap::new();

    for idx in 0..table.len {
        let (Some(_ts_code), Some(trade_date)) = (ts_codes[idx].clone(), trade_dates[idx]) else {
            continue;
        };
        grouped.entry(trade_date).or_default().push(idx);
    }

    let mut values = Vec::new();
    for (trade_date, mut indices) in grouped {
        if !target_dates.contains(&trade_date) {
            continue;
        }
        indices.sort_by(|left, right| ts_codes[*left].cmp(&ts_codes[*right]));
        let section_ts_codes = indices
            .iter()
            .filter_map(|idx| ts_codes[*idx].clone())
            .collect::<Vec<_>>();
        let columns = value_columns
            .iter()
            .map(|(name, column)| {
                (
                    name.clone(),
                    indices.iter().map(|idx| column[*idx]).collect::<Vec<_>>(),
                )
            })
            .collect::<BTreeMap<_, _>>();
        let section = DailyCrossSection {
            trade_date,
            ts_codes: section_ts_codes,
            columns,
        };
        let computed = expr(&section)?;
        if computed.len() != section.ts_codes.len() {
            return Err(err(format!(
                "factor {} returned {} values for {}, expected {}",
                spec.id,
                computed.len(),
                section.trade_date,
                section.ts_codes.len()
            )));
        }

        for (idx, ts_code) in section.ts_codes.iter().enumerate() {
            values.push(FactorValue {
                key: FactorRowKey::Daily {
                    trade_date: section.trade_date,
                    ts_code: ts_code.clone(),
                },
                value: computed[idx],
            });
        }
    }

    Ok(FactorSeries { spec, values })
}

#[derive(Clone, Copy, Debug)]
pub enum ClassificationLevel {
    Sector,
    Industry,
    Subindustry,
}

impl ClassificationLevel {
    fn code_column(self) -> &'static str {
        match self {
            Self::Sector => "l1_code",
            Self::Industry => "l2_code",
            Self::Subindustry => "l3_code",
        }
    }
}

#[derive(Clone, Debug)]
struct ClassificationInterval {
    in_date: i32,
    out_date: i32,
    code: String,
}

#[derive(Clone, Debug, Default)]
pub struct ClassificationMap {
    by_ts_code: BTreeMap<String, Vec<ClassificationInterval>>,
}

impl ClassificationMap {
    pub fn from_table(table: &Table, level: ClassificationLevel) -> Result<Self> {
        let ts_codes = table.required_utf8("ts_code")?;
        let in_dates = table.required_i32_date_cast("in_date")?;
        let out_dates = table.required_i32_date_cast("out_date")?;
        let classification_codes = table.required_utf8(level.code_column())?;
        let mut by_ts_code: BTreeMap<String, Vec<ClassificationInterval>> = BTreeMap::new();

        for idx in 0..table.len {
            let (Some(ts_code), Some(in_date), Some(code)) = (
                ts_codes[idx].clone(),
                in_dates[idx],
                classification_codes[idx].clone(),
            ) else {
                continue;
            };
            if code.is_empty() {
                continue;
            }
            let out_date = out_dates[idx].unwrap_or(99_991_231);
            by_ts_code
                .entry(ts_code)
                .or_default()
                .push(ClassificationInterval {
                    in_date,
                    out_date,
                    code,
                });
        }

        for intervals in by_ts_code.values_mut() {
            intervals.sort_by_key(|interval| interval.in_date);
        }

        Ok(Self { by_ts_code })
    }

    pub fn group_for(&self, trade_date: i32, ts_code: &str) -> Option<&str> {
        self.by_ts_code
            .get(ts_code)?
            .iter()
            .rev()
            .find(|interval| interval.in_date <= trade_date && trade_date <= interval.out_date)
            .map(|interval| interval.code.as_str())
    }

    pub fn groups_for(&self, trade_date: i32, ts_codes: &[String]) -> Vec<Option<String>> {
        ts_codes
            .iter()
            .map(|ts_code| {
                self.group_for(trade_date, ts_code)
                    .map(|group| group.to_string())
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use crate::core::{AssetClass, FactorContext, FactorRowKey, FactorSpec, Frequency, Lookback};
    use crate::data::{ColumnData, Table};

    use super::{compute_daily_cross_section, ClassificationLevel, ClassificationMap};

    #[test]
    fn classification_map_finds_active_sector_interval_from_string_dates() {
        let table = Table::new(BTreeMap::from([
            (
                "ts_code".to_string(),
                ColumnData::Utf8(vec![
                    Some("000001.SZ".to_string()),
                    Some("000001.SZ".to_string()),
                    Some("000002.SZ".to_string()),
                ]),
            ),
            (
                "in_date".to_string(),
                ColumnData::Utf8(vec![
                    Some("20200101".to_string()),
                    Some("20260110".to_string()),
                    Some("20200101".to_string()),
                ]),
            ),
            (
                "out_date".to_string(),
                ColumnData::Utf8(vec![
                    Some("20260109".to_string()),
                    Some("99991231".to_string()),
                    Some("nan".to_string()),
                ]),
            ),
            (
                "l1_code".to_string(),
                ColumnData::Utf8(vec![
                    Some("old".to_string()),
                    Some("new".to_string()),
                    Some("active".to_string()),
                ]),
            ),
        ]))
        .expect("valid table");

        let map = ClassificationMap::from_table(&table, ClassificationLevel::Sector).expect("map");

        assert_eq!(map.group_for(20260105, "000001.SZ"), Some("old"));
        assert_eq!(map.group_for(20260130, "000001.SZ"), Some("new"));
        assert_eq!(map.group_for(20260424, "000002.SZ"), Some("active"));
    }

    #[test]
    fn daily_cross_section_helper_groups_by_target_date_and_sorts_codes() {
        let table = Table::new(BTreeMap::from([
            (
                "trade_date".to_string(),
                ColumnData::I32(vec![Some(20260105), Some(20260105), Some(20260106)]),
            ),
            (
                "ts_code".to_string(),
                ColumnData::Utf8(vec![
                    Some("000002.SZ".to_string()),
                    Some("000001.SZ".to_string()),
                    Some("000001.SZ".to_string()),
                ]),
            ),
            (
                "close".to_string(),
                ColumnData::F64(vec![Some(2.0), Some(1.0), Some(3.0)]),
            ),
        ]))
        .expect("valid table");
        let context = FactorContext {
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            start_date: 20260105,
            end_date: 20260105,
            load_start_date: 20260105,
            target_dates: vec![20260105],
        };
        let spec = FactorSpec {
            id: "test.cross_section".to_string(),
            aliases: Vec::new(),
            name: "test".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: "0.1.0".to_string(),
            tags: Vec::new(),
            description: String::new(),
            dependencies: Vec::new(),
            lookback: Lookback { trading_days: 0 },
        };

        let output = compute_daily_cross_section(spec, &context, &table, |section| {
            assert_eq!(
                section.ts_codes(),
                &["000001.SZ".to_string(), "000002.SZ".to_string()]
            );
            Ok(section.column("close")?.to_vec())
        })
        .expect("computed");

        assert_eq!(output.values.len(), 2);
        assert_eq!(
            output.values[0].key,
            FactorRowKey::Daily {
                trade_date: 20260105,
                ts_code: "000001.SZ".to_string()
            }
        );
        assert_eq!(output.values[0].value, Some(1.0));
    }
}

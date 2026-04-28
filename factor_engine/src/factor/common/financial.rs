use std::collections::BTreeMap;

use crate::data::Table;
use crate::error::{err, Result};

use super::{DailyPanel, PanelColumn};

#[derive(Clone, Debug)]
pub struct PitFinancialRecord {
    pub end_date: i32,
    pub disclosure_date: i32,
    pub report_type: i64,
    pub update_flag: i64,
    columns: BTreeMap<String, Option<f64>>,
}

impl PitFinancialRecord {
    pub fn column(&self, name: &str) -> Option<f64> {
        self.columns
            .get(name)
            .copied()
            .flatten()
            .filter(|value| !value.is_nan())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReportTypePreference {
    order: Vec<i64>,
}

impl ReportTypePreference {
    pub fn new(order: &[i64]) -> Self {
        Self {
            order: order.to_vec(),
        }
    }

    pub fn income_single_quarter() -> Self {
        Self::new(&[3, 2])
    }

    pub fn balance_sheet_consolidated() -> Self {
        Self::new(&[1, 4])
    }

    fn contains(&self, report_type: i64) -> bool {
        self.order.contains(&report_type)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DeadlinePolicy {
    RequiredAfterDeadline,
}

#[derive(Clone, Debug)]
pub struct PitFinancialData {
    preference: ReportTypePreference,
    by_ts_code: BTreeMap<String, BTreeMap<i32, BTreeMap<i64, Vec<PitFinancialRecord>>>>,
}

impl PitFinancialData {
    pub fn from_table(
        table: &Table,
        value_columns: &[&str],
        preference: ReportTypePreference,
    ) -> Result<Self> {
        let ts_codes = table.required_utf8("ts_code")?;
        let ann_dates = table.required_i32_date_cast("ann_date")?;
        let f_ann_dates = table.required_i32_date_cast("f_ann_date")?;
        let end_dates = table.required_i32_date_cast("end_date")?;
        let update_flags = table.required_i64_cast("update_flag")?;
        let report_types = if table.columns.contains_key("report_type") {
            table.required_i64_cast("report_type")?
        } else {
            vec![Some(1); table.len]
        };
        let mut value_data = BTreeMap::new();
        for column in value_columns {
            value_data.insert((*column).to_string(), table.required_f64_cast(column)?);
        }

        let mut by_ts_code: BTreeMap<
            String,
            BTreeMap<i32, BTreeMap<i64, Vec<PitFinancialRecord>>>,
        > = BTreeMap::new();
        for idx in 0..table.len {
            let (Some(ts_code), Some(end_date), Some(disclosure_date), Some(report_type)) = (
                ts_codes[idx].clone(),
                end_dates[idx],
                f_ann_dates[idx].or(ann_dates[idx]),
                report_types[idx],
            ) else {
                continue;
            };
            if !preference.contains(report_type) {
                continue;
            }
            let columns = value_data
                .iter()
                .map(|(name, values)| (name.clone(), values[idx]))
                .collect::<BTreeMap<_, _>>();
            by_ts_code
                .entry(ts_code)
                .or_default()
                .entry(end_date)
                .or_default()
                .entry(report_type)
                .or_default()
                .push(PitFinancialRecord {
                    end_date,
                    disclosure_date,
                    report_type,
                    update_flag: update_flags[idx].unwrap_or(0),
                    columns,
                });
        }

        for by_end_date in by_ts_code.values_mut() {
            for by_report_type in by_end_date.values_mut() {
                for versions in by_report_type.values_mut() {
                    versions.sort_by(|left, right| {
                        right
                            .disclosure_date
                            .cmp(&left.disclosure_date)
                            .then_with(|| right.update_flag.cmp(&left.update_flag))
                    });
                }
            }
        }

        Ok(Self {
            preference,
            by_ts_code,
        })
    }

    pub fn record_for_end_date(
        &self,
        ts_code: &str,
        trade_date: i32,
        end_date: i32,
    ) -> Option<&PitFinancialRecord> {
        let by_report_type = self.by_ts_code.get(ts_code)?.get(&end_date)?;
        for report_type in &self.preference.order {
            let Some(versions) = by_report_type.get(report_type) else {
                continue;
            };
            if let Some(record) = versions
                .iter()
                .find(|record| record.disclosure_date <= trade_date)
            {
                return Some(record);
            }
        }
        None
    }

    pub fn quarters<'a>(
        &'a self,
        panel: &'a DailyPanel,
        column: &str,
        count: usize,
        policy: DeadlinePolicy,
    ) -> Result<QuarterMatrix<'a>> {
        let mut values = Vec::with_capacity(panel.shape_len());
        for trade_date in panel.dates() {
            for ts_code in panel.instruments() {
                let row = self
                    .quarter_chain(ts_code, *trade_date, count, policy)
                    .map(|records| {
                        records
                            .into_iter()
                            .map(|record| {
                                Some(QuarterValue {
                                    end_date: record.end_date,
                                    value: record.column(column),
                                })
                            })
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_else(|| vec![None; count]);
                values.push(row);
            }
        }
        Ok(QuarterMatrix {
            panel,
            quarter_count: count,
            values,
        })
    }

    pub fn quarters_like<'a>(
        &'a self,
        panel: &'a DailyPanel,
        column: &str,
        template: &QuarterMatrix<'a>,
    ) -> Result<QuarterMatrix<'a>> {
        template.require_same_panel(panel)?;
        let mut values = Vec::with_capacity(panel.shape_len());
        for (date_idx, trade_date) in panel.dates().iter().enumerate() {
            for (instrument_idx, ts_code) in panel.instruments().iter().enumerate() {
                let offset = panel_offset(panel, date_idx, instrument_idx);
                let row = template.values[offset]
                    .iter()
                    .map(|quarter| {
                        let end_date = quarter.as_ref()?.end_date;
                        Some(QuarterValue {
                            end_date,
                            value: self
                                .record_for_end_date(ts_code, *trade_date, end_date)
                                .and_then(|record| record.column(column)),
                        })
                    })
                    .collect::<Vec<_>>();
                values.push(row);
            }
        }
        Ok(QuarterMatrix {
            panel,
            quarter_count: template.quarter_count,
            values,
        })
    }

    fn quarter_chain(
        &self,
        ts_code: &str,
        trade_date: i32,
        count: usize,
        _policy: DeadlinePolicy,
    ) -> Option<Vec<&PitFinancialRecord>> {
        if count == 0 {
            return Some(Vec::new());
        }
        let by_end_date = self.by_ts_code.get(ts_code)?;
        let required_anchor = required_anchor_end_date(trade_date);
        let latest_possible = latest_possible_end_date(trade_date);

        for (&anchor, _) in by_end_date.iter().rev() {
            if anchor > latest_possible || anchor < required_anchor {
                continue;
            }
            let mut current = anchor;
            let mut records = Vec::with_capacity(count);
            let mut complete = true;
            for _ in 0..count {
                let Some(record) = self.record_for_end_date(ts_code, trade_date, current) else {
                    complete = false;
                    break;
                };
                records.push(record);
                current = previous_quarter_end_date(current)?;
            }
            if complete {
                return Some(records);
            }
        }
        None
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QuarterValue {
    pub end_date: i32,
    pub value: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct QuarterMatrix<'a> {
    panel: &'a DailyPanel,
    quarter_count: usize,
    values: Vec<Vec<Option<QuarterValue>>>,
}

impl<'a> QuarterMatrix<'a> {
    pub fn binary<F>(&self, other: &Self, mut f: F) -> Result<Self>
    where
        F: FnMut(f64, f64) -> Option<f64>,
    {
        self.require_same_shape(other)?;
        let values = self
            .values
            .iter()
            .zip(&other.values)
            .map(|(left_row, right_row)| {
                left_row
                    .iter()
                    .zip(right_row)
                    .map(|(left, right)| match (left, right) {
                        (Some(left), Some(right)) if left.end_date == right.end_date => {
                            Some(QuarterValue {
                                end_date: left.end_date,
                                value: match (left.value, right.value) {
                                    (Some(left), Some(right))
                                        if !left.is_nan() && !right.is_nan() =>
                                    {
                                        f(left, right)
                                    }
                                    _ => None,
                                },
                            })
                        }
                        _ => None,
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        Ok(Self {
            panel: self.panel,
            quarter_count: self.quarter_count,
            values,
        })
    }

    pub fn mean(&self) -> Result<PanelColumn> {
        let values = self
            .values
            .iter()
            .map(|row| {
                if row.len() != self.quarter_count {
                    return None;
                }
                let mut sum = 0.0;
                for quarter in row {
                    let value = quarter.as_ref()?.value?;
                    if value.is_nan() {
                        return None;
                    }
                    sum += value;
                }
                (self.quarter_count > 0).then_some(sum / self.quarter_count as f64)
            })
            .collect::<Vec<_>>();
        self.panel.column_from_values(values)
    }

    fn require_same_panel(&self, panel: &DailyPanel) -> Result<()> {
        if self.panel.dates() == panel.dates() && self.panel.instruments() == panel.instruments() {
            return Ok(());
        }
        Err(err(
            "quarter matrix and panel use different indexes and cannot be combined",
        ))
    }

    fn require_same_shape(&self, other: &Self) -> Result<()> {
        self.require_same_panel(other.panel)?;
        if self.quarter_count == other.quarter_count && self.values.len() == other.values.len() {
            return Ok(());
        }
        Err(err(
            "quarter matrices use different shapes and cannot be combined",
        ))
    }
}

fn panel_offset(panel: &DailyPanel, date_idx: usize, instrument_idx: usize) -> usize {
    date_idx * panel.instruments().len() + instrument_idx
}

fn latest_possible_end_date(trade_date: i32) -> i32 {
    let year = trade_date / 10_000;
    let mmdd = trade_date % 10_000;
    match mmdd {
        1231..=9999 => year * 10_000 + 1231,
        930..=1230 => year * 10_000 + 930,
        630..=929 => year * 10_000 + 630,
        331..=629 => year * 10_000 + 331,
        _ => (year - 1) * 10_000 + 1231,
    }
}

fn required_anchor_end_date(trade_date: i32) -> i32 {
    let year = trade_date / 10_000;
    let mmdd = trade_date % 10_000;
    match mmdd {
        1031..=9999 => year * 10_000 + 930,
        831..=1030 => year * 10_000 + 630,
        430..=830 => year * 10_000 + 331,
        _ => (year - 1) * 10_000 + 930,
    }
}

fn previous_quarter_end_date(end_date: i32) -> Option<i32> {
    let year = end_date / 10_000;
    match end_date % 10_000 {
        331 => Some((year - 1) * 10_000 + 1231),
        630 => Some(year * 10_000 + 331),
        930 => Some(year * 10_000 + 630),
        1231 => Some(year * 10_000 + 930),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use crate::core::{AssetClass, FactorContext, Frequency};
    use crate::data::{ColumnData, Table};

    use super::{latest_possible_end_date, required_anchor_end_date};
    use super::{DeadlinePolicy, PitFinancialData, ReportTypePreference};

    fn financial_table(rows: &[(i32, i32, i64, i64, f64)]) -> Table {
        Table::new(BTreeMap::from([
            (
                "ts_code".to_string(),
                ColumnData::Utf8(rows.iter().map(|_| Some("000001.SZ".to_string())).collect()),
            ),
            (
                "ann_date".to_string(),
                ColumnData::I32(rows.iter().map(|(_, date, _, _, _)| Some(*date)).collect()),
            ),
            (
                "f_ann_date".to_string(),
                ColumnData::I32(rows.iter().map(|(_, date, _, _, _)| Some(*date)).collect()),
            ),
            (
                "end_date".to_string(),
                ColumnData::I32(rows.iter().map(|(end, _, _, _, _)| Some(*end)).collect()),
            ),
            (
                "report_type".to_string(),
                ColumnData::I64(
                    rows.iter()
                        .map(|(_, _, report_type, _, _)| Some(*report_type))
                        .collect(),
                ),
            ),
            (
                "update_flag".to_string(),
                ColumnData::I64(
                    rows.iter()
                        .map(|(_, _, _, update_flag, _)| Some(*update_flag))
                        .collect(),
                ),
            ),
            (
                "value".to_string(),
                ColumnData::F64(
                    rows.iter()
                        .map(|(_, _, _, _, value)| Some(*value))
                        .collect(),
                ),
            ),
        ]))
        .expect("valid table")
    }

    fn panel(dates: &[i32]) -> crate::factor::common::DailyPanel {
        let table = Table::new(BTreeMap::from([
            (
                "trade_date".to_string(),
                ColumnData::I32(dates.iter().map(|date| Some(*date)).collect()),
            ),
            (
                "ts_code".to_string(),
                ColumnData::Utf8(
                    dates
                        .iter()
                        .map(|_| Some("000001.SZ".to_string()))
                        .collect(),
                ),
            ),
            (
                "dummy".to_string(),
                ColumnData::F64(dates.iter().map(|_| Some(1.0)).collect()),
            ),
        ]))
        .expect("valid table");
        let context = FactorContext {
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            start_date: *dates.first().unwrap(),
            end_date: *dates.last().unwrap(),
            load_start_date: *dates.first().unwrap(),
            load_dates: dates.to_vec(),
            target_dates: dates.to_vec(),
        };
        crate::factor::common::DailyPanel::from_table(&table, &context).expect("panel")
    }

    fn statutory_disclosure_date(end_date: i32) -> i32 {
        let year = end_date / 10_000;
        match end_date % 10_000 {
            331 => year * 10_000 + 430,
            630 => year * 10_000 + 831,
            930 => year * 10_000 + 1031,
            1231 => (year + 1) * 10_000 + 430,
            _ => end_date,
        }
    }

    #[test]
    fn report_type_preference_uses_order_before_fallback() {
        let table = financial_table(&[
            (20250331, 20250420, 1, 0, 1.0),
            (20250331, 20250420, 2, 0, 2.0),
            (20250331, 20250420, 3, 0, 3.0),
            (20250331, 20250420, 4, 0, 4.0),
        ]);
        let income = PitFinancialData::from_table(
            &table,
            &["value"],
            ReportTypePreference::income_single_quarter(),
        )
        .expect("income");
        let balance = PitFinancialData::from_table(
            &table,
            &["value"],
            ReportTypePreference::balance_sheet_consolidated(),
        )
        .expect("balance");

        assert_eq!(
            income
                .record_for_end_date("000001.SZ", 20250501, 20250331)
                .and_then(|record| record.column("value")),
            Some(3.0)
        );
        assert_eq!(
            balance
                .record_for_end_date("000001.SZ", 20250501, 20250331)
                .and_then(|record| record.column("value")),
            Some(1.0)
        );
    }

    #[test]
    fn pit_financial_data_uses_only_disclosed_versions() {
        let table = financial_table(&[
            (20241231, 20250331, 3, 0, 10.0),
            (20241231, 20250430, 3, 1, 12.0),
        ]);
        let data = PitFinancialData::from_table(
            &table,
            &["value"],
            ReportTypePreference::income_single_quarter(),
        )
        .expect("pit data");

        assert_eq!(
            data.record_for_end_date("000001.SZ", 20250401, 20241231)
                .and_then(|record| record.column("value")),
            Some(10.0)
        );
        assert_eq!(
            data.record_for_end_date("000001.SZ", 20250501, 20241231)
                .and_then(|record| record.column("value")),
            Some(12.0)
        );
    }

    #[test]
    fn deadline_policy_falls_back_before_deadline_but_requires_latest_after_deadline() {
        let mut rows = Vec::new();
        for end_date in [
            20231231, 20240331, 20240630, 20240930, 20241231, 20250331, 20250630, 20250930,
            20251231,
        ] {
            rows.push((end_date, statutory_disclosure_date(end_date), 3, 0, 1.0));
        }
        rows.push((20260331, 20260507, 3, 0, 2.0));
        let data = PitFinancialData::from_table(
            &financial_table(&rows),
            &["value"],
            ReportTypePreference::income_single_quarter(),
        )
        .expect("pit data");
        let panel = panel(&[20260429, 20260506, 20260508]);
        let quarters = data
            .quarters(&panel, "value", 8, DeadlinePolicy::RequiredAfterDeadline)
            .expect("quarters");
        let mean = quarters.mean().expect("mean");

        assert_eq!(mean.values()[0], Some(1.0));
        assert_eq!(mean.values()[1], None);
        assert_eq!(mean.values()[2], Some(1.125));
    }

    #[test]
    fn quarter_deadlines_map_to_expected_required_periods() {
        assert_eq!(required_anchor_end_date(20260429), 20250930);
        assert_eq!(required_anchor_end_date(20260430), 20260331);
        assert_eq!(required_anchor_end_date(20260831), 20260630);
        assert_eq!(required_anchor_end_date(20261031), 20260930);
        assert_eq!(latest_possible_end_date(20260429), 20260331);
    }
}

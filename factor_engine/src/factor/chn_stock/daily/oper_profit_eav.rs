use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::financial::previous_quarter_end_date;
use crate::factor::common::stock_daily_ops::{is_bj_stock, neutralize_size_sector};
use crate::factor::common::{
    cached_financial_stock_snapshots_for_date, FinancialEventMarker, FinancialEventMarkerBuilder,
    FinancialEventSchedule, FinancialPitReader, FinancialStatementDataset,
    InstrumentAlignedSnapshotCache, ReportTypePreference,
};
use crate::factor::{Factor, FactorUpdatePolicy};

const QUARTERS: usize = 12;
const MIN_PERIODS: usize = 4;
const COLUMN: &str = "operate_profit";

pub struct StockDailyOperProfitEav;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyOperProfitEav)
}

impl Factor for StockDailyOperProfitEav {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "oper_profit_eav".into(),
            aliases: vec!["OPER_PROFIT_EAV".into()],
            name: "Operating Profit Acceleration EAV".into(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: "0.1.0".into(),
            tags: ["HAZQ", "fundamental", "financial", "pit", "profit_acceleration", "size_neutralize", "sector_neutralize", "daily"].into_iter().map(str::to_string).collect(),
            description: "PIT single-quarter operating profit acceleration: (OP_t-OP_t-4)/sample_std(OP_t..OP_t-7) minus (OP_t-4-OP_t-8)/sample_std(OP_t-4..OP_t-11). Requires four finite observations per eight-quarter window and valid OP_t, OP_t-4, OP_t-8; retains negative profit, excludes BJ, and neutralizes daily by SW level-1 sector and Barra SIZE.".into(),
            dependencies: vec![
                DataRequest::financial_quarters(DatasetId::StockIncome, &[COLUMN], QUARTERS),
                DataRequest::new(DatasetId::StockBarraDaily, &["SIZE"]),
                DataRequest::new(DatasetId::StockSwClassification, &["l1_code"]),
            ],
            intraday_raw_dependencies: Vec::new(),
            lookback: Lookback { trading_days: 0 },
        }
    }

    fn update_policy(&self) -> FactorUpdatePolicy {
        FactorUpdatePolicy::FinancialEventStateDailyFast
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let panel = data.stock_universe_panel()?;
        let income = data.financial_reader(
            DatasetId::StockIncome,
            ReportTypePreference::income_single_quarter(),
        )?;
        // Batch-local cache prevents overlapping batches from reusing future snapshots.
        let mut cache = InstrumentAlignedSnapshotCache::<f64>::default();
        let schedule = FinancialEventSchedule::from_pit_readers(&[income.clone()]);
        let mut snapshots = vec![None; panel.instruments().len()];
        let mut values = vec![None; panel.shape_len()];
        let mut last_date = None;
        for (date_idx, date) in panel.dates().iter().copied().enumerate() {
            let start = date_idx * panel.instruments().len();
            let presence_changed = date_idx > 0
                && (0..panel.instruments().len()).any(|i| {
                    panel.is_present_offset(start + i)
                        != panel.is_present_offset(start + i - panel.instruments().len())
                });
            if last_date.is_none()
                || presence_changed
                || schedule.has_event_after_until(last_date, date)
            {
                snapshots = cached_financial_stock_snapshots_for_date(
                    &panel,
                    date,
                    &mut cache,
                    |_, code, offset| is_bj_stock(code) || !panel.is_present_offset(offset),
                    |date, code, _| marker(&income, code, date),
                    |date, code, _| snapshot(&income, code, date),
                );
            }
            for (i, value) in snapshots.iter().enumerate() {
                if panel.is_present_offset(start + i) {
                    values[start + i] = *value;
                }
            }
            last_date = Some(date);
        }
        let raw = panel.column_from_values(values)?;
        Ok(neutralize_size_sector(&raw, &panel, data)?.to_factor_series(self.spec()))
    }
}

fn quarter_ends(anchor: i32) -> Option<[i32; QUARTERS]> {
    let mut ends = [anchor; QUARTERS];
    for i in 1..QUARTERS {
        ends[i] = previous_quarter_end_date(ends[i - 1])?;
    }
    Some(ends)
}

fn marker(income: &FinancialPitReader<'_>, code: &str, date: i32) -> Option<FinancialEventMarker> {
    let ends = quarter_ends(income.latest_quarter_end_date(code, date)?)?;
    let mut builder = FinancialEventMarkerBuilder::new();
    for end in ends {
        builder.include_reader_record_for_end_date(
            FinancialStatementDataset::Income,
            income,
            code,
            date,
            end,
        );
    }
    builder.build()
}

fn snapshot(income: &FinancialPitReader<'_>, code: &str, date: i32) -> Option<f64> {
    let ends = quarter_ends(income.latest_quarter_end_date(code, date)?)?;
    let mut profits = [None; QUARTERS];
    for (i, end) in ends.into_iter().enumerate() {
        profits[i] = income
            .record_for_end_date(code, date, end)
            .and_then(|record| record.column(COLUMN));
    }
    acceleration(&profits)
}

fn acceleration(profits: &[Option<f64>; QUARTERS]) -> Option<f64> {
    let op = profits.map(|value| value.filter(|v| v.is_finite()));
    let (current, prior, earlier) = (op[0]?, op[4]?, op[8]?);
    let std = |slice: &[Option<f64>]| -> Option<f64> {
        let count = slice.iter().flatten().count();
        if count < MIN_PERIODS {
            return None;
        }
        let mean = slice.iter().flatten().sum::<f64>() / count as f64;
        let sd = (slice
            .iter()
            .flatten()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>()
            / (count - 1) as f64)
            .sqrt();
        (sd.is_finite() && sd > 1e-12).then_some(sd)
    };
    let result = (current - prior) / std(&op[..8])? - (prior - earlier) / std(&op[4..])?;
    result.is_finite().then_some(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eav_two_overlapping_windows_use_sample_std() {
        let op = std::array::from_fn(|i| Some(((12 - i) as f64).powi(2)));
        // Squares 5..12 have sample variance 1758; squares 1..8 have 510.
        let expected = 80.0 / 1758.0_f64.sqrt() - 48.0 / 510.0_f64.sqrt();
        assert!((acceleration(&op).unwrap() - expected).abs() < 1e-12);
        let negative = op.map(|v| v.map(|x| -x));
        assert!((acceleration(&negative).unwrap() + expected).abs() < 1e-12);
    }

    #[test]
    fn eav_rejects_missing_nonfinite_and_zero_variance() {
        assert_eq!(acceleration(&[Some(1.0); 12]), None);
        for i in [0, 4, 8] {
            let mut op = std::array::from_fn(|j| Some((j * j) as f64));
            op[i] = None;
            assert_eq!(acceleration(&op), None);
            op[i] = Some(f64::NAN);
            assert_eq!(acceleration(&op), None);
        }
    }

    #[test]
    fn eav_requires_four_values_in_each_window_without_shifting_quarters() {
        let mut op = [None; QUARTERS];
        for (i, value) in [(0, 10.0), (1, 8.0), (4, 6.0), (5, 4.0), (8, 2.0), (9, 0.0)] {
            op[i] = Some(value);
        }
        // Each window has four observations with sample variance 20/3.
        assert!(acceleration(&op).unwrap().abs() < 1e-12);
        op[0] = Some(12.0);
        let expected = 6.0 / (35.0_f64 / 3.0).sqrt() - 4.0 / (20.0_f64 / 3.0).sqrt();
        assert!((acceleration(&op).unwrap() - expected).abs() < 1e-12);
        for i in [1, 9] {
            let mut missing = op;
            missing[i] = None;
            assert_eq!(acceleration(&missing), None);
            missing[i] = Some(f64::INFINITY);
            assert_eq!(acceleration(&missing), None);
        }
        op[2] = Some(f64::NAN);
        assert!((acceleration(&op).unwrap() - expected).abs() < 1e-12);
    }

    #[test]
    fn eav_uses_twelve_calendar_quarters_and_no_pv_anchor() {
        let ends = quarter_ends(20250630).unwrap();
        assert_eq!((ends[4], ends[8], ends[11]), (20240630, 20230630, 20220930));
        let spec = StockDailyOperProfitEav.spec();
        for tag in ["HAZQ", "fundamental"] {
            assert!(spec.tags.contains(&tag.to_string()));
        }
        assert!(!spec
            .dependencies
            .iter()
            .any(|r| r.dataset == DatasetId::StockDailyPv));
    }
}

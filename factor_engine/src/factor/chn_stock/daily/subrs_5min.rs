use std::collections::{BTreeMap, BTreeSet};

use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorRowKey, FactorSeries, FactorSpec,
    FactorValue, Frequency, IntradayDailyRawRequest, IntradayDailyRawSeries, IntradayDailyRawSpec,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::stock_daily_ops::neutralize_size_sector;
use crate::factor::common::stock_daily_raw_ids::{
    SUBRHS_5MIN_RAW_ID, SUBRHT_5MIN_RAW_ID, SUBRK_5MIN_RAW_ID, SUBRS_5MIN_RAW_ID,
};
use crate::factor::common::{clean_intraday_value, intraday_time_in_range, stock_minute_raw_spec};
use crate::factor::Factor;
use crate::operators::{cs_zscore, ts_mean};

const RAW_VERSION: &str = "0.1.0";
const VERSION: &str = "0.1.0";
const SMOOTH_WINDOW: usize = 5;
const MIN_PERIODS: usize = 1;
const GRID_COUNT: usize = 5;
const EPS: f64 = f64::EPSILON;

pub struct StockDailySubrs5min;

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct DailySubsampleMoments {
    pub(crate) rs: Option<f64>,
    pub(crate) rk: Option<f64>,
    pub(crate) rhs: Option<f64>,
    pub(crate) rht: Option<f64>,
}

#[derive(Clone, Copy, Debug, Default)]
struct SubgridSums {
    n: usize,
    rv: f64,
    sum3: f64,
    sum4: f64,
    sum5: f64,
    sum6: f64,
}

#[derive(Clone, Copy, Debug, Default)]
struct MomentAccumulator {
    sum: f64,
    count: usize,
}

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailySubrs5min)
}

pub(crate) fn raw_spec(raw_id: &str) -> IntradayDailyRawSpec {
    stock_minute_raw_spec(raw_id, RAW_VERSION, &["close"], 1)
}

pub(crate) fn all_raw_ids() -> [&'static str; 4] {
    [
        SUBRS_5MIN_RAW_ID,
        SUBRK_5MIN_RAW_ID,
        SUBRHS_5MIN_RAW_ID,
        SUBRHT_5MIN_RAW_ID,
    ]
}

pub(crate) fn subr_tags() -> Vec<String> {
    [
        "price_volume",
        "price",
        "return",
        "realized_moment",
        "intraday",
        "minute_agg",
        "neutralize",
        "barra",
        "size",
        "sector",
        "daily",
        "DBZQ",
    ]
    .iter()
    .map(|value| value.to_string())
    .collect()
}

pub(crate) fn subr_dependencies() -> Vec<DataRequest> {
    vec![
        DataRequest::new(DatasetId::StockBarraDaily, &["SIZE"]),
        DataRequest::new(DatasetId::StockSwClassification, &["l1_code"]),
    ]
}

impl Factor for StockDailySubrs5min {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "subrs_5min".to_string(),
            aliases: vec!["subRS_5min".to_string(), "SUBRS_5MIN".to_string()],
            name: "subRS 5min".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: subr_tags(),
            description: "Downsampled 5-minute realized skewness, smoothed over 5 days, z-scored, and neutralized by SIZE and SW sector.".to_string(),
            dependencies: subr_dependencies(),
            intraday_raw_dependencies: vec![IntradayDailyRawRequest::new(
                SUBRS_5MIN_RAW_ID,
                SMOOTH_WINDOW - 1,
            )],
            lookback: Lookback {
                trading_days: SMOOTH_WINDOW - 1,
            },
        }
    }

    fn intraday_raw_specs(&self) -> Vec<IntradayDailyRawSpec> {
        all_raw_ids()
            .iter()
            .map(|raw_id| raw_spec(raw_id))
            .collect()
    }

    fn minute_compute(
        &self,
        raw_id: &str,
        context: &FactorContext,
        data: &DataPool,
    ) -> Result<Option<IntradayDailyRawSeries>> {
        let raw_ids = vec![raw_id.to_string()];
        Ok(self
            .minute_compute_many(&raw_ids, context, data)?
            .into_iter()
            .next())
    }

    fn minute_compute_many(
        &self,
        raw_ids: &[String],
        context: &FactorContext,
        data: &DataPool,
    ) -> Result<Vec<IntradayDailyRawSeries>> {
        let requested = raw_ids
            .iter()
            .map(String::as_str)
            .filter(|raw_id| all_raw_ids().contains(raw_id))
            .collect::<BTreeSet<_>>();
        if requested.is_empty() {
            return Ok(Vec::new());
        }

        let mut rs_values = Vec::new();
        let mut rk_values = Vec::new();
        let mut rhs_values = Vec::new();
        let mut rht_values = Vec::new();

        for trade_date in &context.target_dates {
            let Some(table) = data.minute(DatasetId::StockMinute1m, *trade_date) else {
                continue;
            };
            let ts_codes = table.required_utf8("ts_code")?;
            let trade_times = table.required_utf8("trade_time")?;
            let close = table.required_f64_cast("close")?;

            let mut grouped = BTreeMap::<String, Vec<usize>>::new();
            for idx in 0..table.len {
                let Some(ts_code) = ts_codes[idx].clone() else {
                    continue;
                };
                if trade_times[idx].is_none() {
                    continue;
                }
                grouped.entry(ts_code).or_default().push(idx);
            }

            for (ts_code, mut indices) in grouped {
                indices.sort_by(|left, right| trade_times[*left].cmp(&trade_times[*right]));
                let moments = daily_subsample_moments(&indices, trade_times, &close);
                let key = FactorRowKey::Daily {
                    trade_date: *trade_date,
                    ts_code,
                };
                if requested.contains(SUBRS_5MIN_RAW_ID) {
                    rs_values.push(FactorValue {
                        key: key.clone(),
                        value: moments.rs,
                    });
                }
                if requested.contains(SUBRK_5MIN_RAW_ID) {
                    rk_values.push(FactorValue {
                        key: key.clone(),
                        value: moments.rk,
                    });
                }
                if requested.contains(SUBRHS_5MIN_RAW_ID) {
                    rhs_values.push(FactorValue {
                        key: key.clone(),
                        value: moments.rhs,
                    });
                }
                if requested.contains(SUBRHT_5MIN_RAW_ID) {
                    rht_values.push(FactorValue {
                        key,
                        value: moments.rht,
                    });
                }
            }
        }

        let mut output = Vec::new();
        if requested.contains(SUBRS_5MIN_RAW_ID) {
            output.push(IntradayDailyRawSeries {
                spec: raw_spec(SUBRS_5MIN_RAW_ID),
                values: rs_values,
            });
        }
        if requested.contains(SUBRK_5MIN_RAW_ID) {
            output.push(IntradayDailyRawSeries {
                spec: raw_spec(SUBRK_5MIN_RAW_ID),
                values: rk_values,
            });
        }
        if requested.contains(SUBRHS_5MIN_RAW_ID) {
            output.push(IntradayDailyRawSeries {
                spec: raw_spec(SUBRHS_5MIN_RAW_ID),
                values: rhs_values,
            });
        }
        if requested.contains(SUBRHT_5MIN_RAW_ID) {
            output.push(IntradayDailyRawSeries {
                spec: raw_spec(SUBRHT_5MIN_RAW_ID),
                values: rht_values,
            });
        }
        Ok(output)
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        compute_subr_factor(self.spec(), SUBRS_5MIN_RAW_ID, data)
    }
}

pub(crate) fn compute_subr_factor(
    spec: FactorSpec,
    raw_id: &str,
    data: &DataPool,
) -> Result<FactorSeries> {
    let panel = data.intraday_daily_raw_panel(raw_id)?;
    let raw = panel.column(raw_id)?;
    let smoothed = raw.ts(|values| ts_mean(values, SMOOTH_WINDOW, MIN_PERIODS))?;
    let standardized = smoothed.cs(cs_zscore)?;
    let factor = neutralize_size_sector(&standardized, panel, data)?;
    Ok(factor.to_factor_series(spec))
}

fn daily_subsample_moments(
    indices: &[usize],
    trade_times: &[Option<String>],
    close: &[Option<f64>],
) -> DailySubsampleMoments {
    let mut log_prices = Vec::new();
    for idx in indices {
        let Some(trade_time) = trade_times[*idx].as_deref() else {
            continue;
        };
        if !intraday_time_in_range(trade_time, "09:31:00", "15:00:00") {
            continue;
        }
        log_prices.push(positive_log(clean_intraday_value(close[*idx])));
    }
    moments_from_log_prices(&log_prices)
}

fn moments_from_log_prices(log_prices: &[Option<f64>]) -> DailySubsampleMoments {
    let mut rs = MomentAccumulator::default();
    let mut rk = MomentAccumulator::default();
    let mut rhs = MomentAccumulator::default();
    let mut rht = MomentAccumulator::default();

    for offset in 0..GRID_COUNT {
        let sums = subgrid_sums_from_log_prices(log_prices, offset);
        let moments = sums.moments();
        rs.add(moments.rs);
        rk.add(moments.rk);
        rhs.add(moments.rhs);
        rht.add(moments.rht);
    }

    DailySubsampleMoments {
        rs: rs.mean(),
        rk: rk.mean(),
        rhs: rhs.mean(),
        rht: rht.mean(),
    }
}

fn subgrid_sums_from_log_prices(log_prices: &[Option<f64>], offset: usize) -> SubgridSums {
    let mut sums = SubgridSums::default();
    let mut pos = offset + GRID_COUNT;
    while pos < log_prices.len() {
        if let (Some(previous), Some(current)) = (log_prices[pos - GRID_COUNT], log_prices[pos]) {
            sums.add_return(current - previous);
        }
        pos += GRID_COUNT;
    }
    sums
}

impl SubgridSums {
    fn add_return(&mut self, value: f64) {
        if !value.is_finite() {
            return;
        }
        let r2 = value * value;
        let r3 = r2 * value;
        self.n += 1;
        self.rv += r2;
        self.sum3 += r3;
        self.sum4 += r2 * r2;
        self.sum5 += r3 * r2;
        self.sum6 += r3 * r3;
    }

    fn moments(self) -> DailySubsampleMoments {
        if self.n == 0 || self.rv <= EPS || !self.rv.is_finite() {
            return DailySubsampleMoments::default();
        }
        let n = self.n as f64;
        DailySubsampleMoments {
            rs: finite_value(n.sqrt() * self.sum3 / self.rv.powf(1.5)),
            rk: finite_value(n * self.sum4 / self.rv.powi(2)),
            rhs: finite_value(n.powf(1.5) * self.sum5 / self.rv.powf(2.5)),
            rht: finite_value(n.powi(2) * self.sum6 / self.rv.powi(3)),
        }
    }
}

impl MomentAccumulator {
    fn add(&mut self, value: Option<f64>) {
        let Some(value) = value.filter(|value| value.is_finite()) else {
            return;
        };
        self.sum += value;
        self.count += 1;
    }

    fn mean(self) -> Option<f64> {
        if self.count > 0 {
            Some(self.sum / self.count as f64)
        } else {
            None
        }
    }
}

fn positive_log(value: Option<f64>) -> Option<f64> {
    value
        .filter(|value| value.is_finite() && *value > 0.0)
        .map(f64::ln)
}

fn finite_value(value: f64) -> Option<f64> {
    if value.is_finite() {
        Some(value)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: Option<f64>, expected: Option<f64>) {
        match (actual, expected) {
            (Some(actual), Some(expected)) => assert!(
                (actual - expected).abs() < 1e-10,
                "expected {expected}, got {actual}"
            ),
            (None, None) => {}
            _ => panic!("expected {:?}, got {:?}", expected, actual),
        }
    }

    #[test]
    fn subgrid_sums_use_offset_plus_five_steps() {
        let logs = (1..=11).map(|value| Some(value as f64)).collect::<Vec<_>>();
        let first_grid = subgrid_sums_from_log_prices(&logs, 0);
        let second_grid = subgrid_sums_from_log_prices(&logs, 1);

        assert_eq!(first_grid.n, 2);
        assert_close(Some(first_grid.rv), Some(50.0));
        assert_eq!(second_grid.n, 1);
        assert_close(Some(second_grid.rv), Some(25.0));
    }

    #[test]
    fn missing_grid_point_breaks_only_that_pair() {
        let logs = vec![
            Some(0.0),
            Some(0.0),
            Some(0.0),
            Some(0.0),
            Some(0.0),
            None,
            Some(0.0),
            Some(0.0),
            Some(0.0),
            Some(0.0),
            Some(10.0),
        ];

        let first_grid = subgrid_sums_from_log_prices(&logs, 0);

        assert_eq!(first_grid.n, 0);
    }

    #[test]
    fn realized_moment_formulas_match_hand_calculation() {
        let mut sums = SubgridSums::default();
        sums.add_return(1.0);
        sums.add_return(2.0);

        let moments = sums.moments();
        let rv = 5.0_f64;
        assert_close(moments.rs, Some(2.0_f64.sqrt() * 9.0 / rv.powf(1.5)));
        assert_close(moments.rk, Some(2.0 * 17.0 / rv.powi(2)));
        assert_close(moments.rhs, Some(2.0_f64.powf(1.5) * 33.0 / rv.powf(2.5)));
        assert_close(moments.rht, Some(4.0 * 65.0 / rv.powi(3)));
    }

    #[test]
    fn zero_realized_variance_outputs_none() {
        let mut sums = SubgridSums::default();
        sums.add_return(0.0);
        sums.add_return(0.0);

        let moments = sums.moments();

        assert_eq!(moments.rs, None);
        assert_eq!(moments.rk, None);
        assert_eq!(moments.rhs, None);
        assert_eq!(moments.rht, None);
    }

    #[test]
    fn daily_moments_average_valid_subgrids() {
        let logs = vec![
            Some(1.0),
            Some(1.0),
            Some(1.0),
            Some(1.0),
            Some(1.0),
            Some(2.0),
            Some(1.0),
            Some(1.0),
            Some(1.0),
            Some(1.0),
            Some(4.0),
        ];

        let moments = moments_from_log_prices(&logs);

        assert!(moments.rs.is_some());
        assert!(moments.rk.is_some());
        assert!(moments.rhs.is_some());
        assert!(moments.rht.is_some());
    }
}

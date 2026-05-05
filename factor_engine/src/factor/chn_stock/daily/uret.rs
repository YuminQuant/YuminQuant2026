use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    IntradayDailyRawRequest, Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::stock_daily_raw_ids::VOLUME_CV_RAW_ID;
use crate::factor::common::{vector::clean, PanelColumn};
use crate::factor::Factor;

const VERSION: &str = "0.1.0";
const WINDOW: usize = 20;
const GROUP_COUNT: usize = 5;
const GROUP_SIZE: usize = WINDOW / GROUP_COUNT;

pub struct StockDailyUret;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyUret)
}

impl Factor for StockDailyUret {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "uret".to_string(),
            aliases: vec!["URet".to_string(), "URET".to_string()],
            name: "URet".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: [
                "price_volume",
                "return",
                "volume",
                "distribution",
                "intraday",
                "minute_agg",
                "daily",
                "DWZQ",
            ]
            .iter()
            .map(|value| value.to_string())
            .collect(),
            description: "Information-distribution return factor that sorts 20-day daily returns by intraday volume coefficient of variation and uses the highest-Z return quintile.".to_string(),
            dependencies: vec![DataRequest::new(
                DatasetId::StockDailyPv,
                &["close", "pre_close"],
            )],
            intraday_raw_dependencies: vec![IntradayDailyRawRequest::new(
                VOLUME_CV_RAW_ID,
                WINDOW - 1,
            )],
            lookback: Lookback {
                trading_days: WINDOW - 1,
            },
        }
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let panel = data.intraday_daily_raw_panel(VOLUME_CV_RAW_ID)?;
        let pv_table = data.daily(DatasetId::StockDailyPv)?;
        let close = panel.column_from_table(pv_table, "close")?;
        let pre_close = panel.column_from_table(pv_table, "pre_close")?;
        let z = panel.column(VOLUME_CV_RAW_ID)?;

        let daily_return = close.zip_binary(&pre_close, ret)?;
        let factor = rolling_group_mean(&daily_return, &z, GROUP_COUNT - 1)?;
        Ok(factor.to_factor_series(self.spec()))
    }
}

fn rolling_group_mean(
    returns: &PanelColumn,
    sort_values: &PanelColumn,
    group_idx: usize,
) -> Result<PanelColumn> {
    returns.ts_binary(sort_values, |returns, sort_values| {
        grouped_part_series(returns, sort_values, group_idx)
    })
}

fn grouped_part_series(
    returns: &[Option<f64>],
    sort_values: &[Option<f64>],
    group_idx: usize,
) -> Vec<Option<f64>> {
    let mut output = vec![None; returns.len()];
    if group_idx >= GROUP_COUNT {
        return output;
    }

    for idx in 0..returns.len() {
        if idx + 1 < WINDOW {
            continue;
        }
        let start = idx + 1 - WINDOW;
        let mut pairs = Vec::<(f64, usize, f64)>::with_capacity(WINDOW);
        for window_idx in start..=idx {
            let (Some(return_value), Some(sort_value)) =
                (clean(returns[window_idx]), clean(sort_values[window_idx]))
            else {
                continue;
            };
            pairs.push((sort_value, window_idx, return_value));
        }
        if pairs.len() != WINDOW {
            continue;
        }
        pairs.sort_by(|left, right| {
            left.0
                .total_cmp(&right.0)
                .then_with(|| left.1.cmp(&right.1))
        });
        let group_start = group_idx * GROUP_SIZE;
        let group_end = group_start + GROUP_SIZE;
        let sum = pairs[group_start..group_end]
            .iter()
            .map(|(_, _, return_value)| *return_value)
            .sum::<f64>();
        output[idx] = Some(sum / GROUP_SIZE as f64);
    }
    output
}

fn ret(numerator: Option<f64>, denominator: Option<f64>) -> Option<f64> {
    match (clean(numerator), clean(denominator)) {
        (Some(numerator), Some(denominator)) if denominator.abs() > f64::EPSILON => {
            Some(numerator / denominator - 1.0)
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: Option<f64>, expected: f64) {
        let actual = actual.expect("expected value");
        assert!(
            (actual - expected).abs() < 1e-12,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn grouped_part_series_takes_highest_z_quintile() {
        let returns = (0..20).map(|idx| Some(idx as f64)).collect::<Vec<_>>();
        let z = (0..20).map(|idx| Some(idx as f64)).collect::<Vec<_>>();

        let output = grouped_part_series(&returns, &z, GROUP_COUNT - 1);

        assert_close(output[19], 17.5);
    }

    #[test]
    fn grouped_part_series_requires_twenty_valid_pairs() {
        let returns = vec![Some(1.0); 20];
        let mut z = vec![Some(1.0); 20];
        z[3] = None;

        let output = grouped_part_series(&returns, &z, GROUP_COUNT - 1);

        assert_eq!(output[19], None);
    }

    #[test]
    fn grouped_part_series_uses_date_order_to_break_z_ties() {
        let returns = (0..20).map(|idx| Some(idx as f64)).collect::<Vec<_>>();
        let z = vec![Some(1.0); 20];

        let output = grouped_part_series(&returns, &z, GROUP_COUNT - 1);

        assert_close(output[19], 17.5);
    }

    #[test]
    fn return_rejects_zero_denominator() {
        assert_close(ret(Some(11.0), Some(10.0)), 0.1);
        assert_eq!(ret(Some(11.0), Some(0.0)), None);
    }
}

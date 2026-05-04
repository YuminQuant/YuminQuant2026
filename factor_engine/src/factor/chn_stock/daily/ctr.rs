use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    IntradayDailyRawRequest, Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::chn_stock::daily::rpv::OPEN_AUCTION_TURNOVER_RAW_ID;
use crate::factor::common::vector::clean;
use crate::factor::Factor;

const VERSION: &str = "0.1.0";
const WINDOW: usize = 20;
const TAKE_COUNT: usize = 4;

pub struct StockDailyCtr;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyCtr)
}

impl Factor for StockDailyCtr {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "ctr".to_string(),
            aliases: vec!["CTR".to_string()],
            name: "CTR".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: [
                "price_volume",
                "turnover",
                "overnight",
                "smart",
                "neutralize",
                "barra",
                "size",
                "daily",
                "DWZQ",
            ]
            .iter()
            .map(|value| value.to_string())
            .collect(),
            description: "Cutlets of Turnover Rate factor using the prior intraday turnover of the four lowest overnight-smart days, neutralized by SIZE.".to_string(),
            dependencies: vec![
                DataRequest::new(DatasetId::StockDailyPv, &["open", "pre_close"]),
                DataRequest::new(DatasetId::StockDailyBasic, &["turnover_rate_f"]),
                DataRequest::new(DatasetId::StockBarraDaily, &["SIZE"]),
            ],
            intraday_raw_dependencies: vec![IntradayDailyRawRequest::new(
                OPEN_AUCTION_TURNOVER_RAW_ID,
                WINDOW,
            )],
            lookback: Lookback {
                trading_days: WINDOW,
            },
        }
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let panel = data.intraday_daily_raw_panel(OPEN_AUCTION_TURNOVER_RAW_ID)?;
        let pv_table = data.daily(DatasetId::StockDailyPv)?;
        let basic_table = data.daily(DatasetId::StockDailyBasic)?;
        let open = panel.column_from_table(pv_table, "open")?;
        let pre_close = panel.column_from_table(pv_table, "pre_close")?;
        let full_turnover = panel
            .column_from_table(basic_table, "turnover_rate_f")?
            .map_values(percent_to_decimal);
        let overnight_turnover = panel.column(OPEN_AUCTION_TURNOVER_RAW_ID)?;
        let size = panel.column_from_table(data.daily(DatasetId::StockBarraDaily)?, "SIZE")?;

        let overnight_return = open.zip_binary(&pre_close, overnight_return)?;
        let intraday_turnover = full_turnover.zip_binary(&overnight_turnover, intraday_turnover)?;
        let raw = overnight_return.ts_ternary(
            &overnight_turnover,
            &intraday_turnover,
            ctr_cutlets_series,
        )?;
        let factor = raw.cs_neutralize_regression(&[&size], None)?;
        Ok(factor.to_factor_series(self.spec()))
    }
}

fn ctr_cutlets_series(
    overnight_returns: &[Option<f64>],
    overnight_turnovers: &[Option<f64>],
    intraday_turnovers: &[Option<f64>],
) -> Vec<Option<f64>> {
    let mut output = vec![None; overnight_returns.len()];
    for idx in 0..overnight_returns.len() {
        if idx < WINDOW {
            continue;
        }
        let start = idx + 1 - WINDOW;
        let Some((min_return, max_return)) = window_min_max(overnight_returns, start, idx) else {
            continue;
        };
        let range = max_return - min_return;
        if range.abs() <= f64::EPSILON {
            continue;
        }

        let mut candidates = Vec::<(f64, usize, f64)>::new();
        for smart_idx in start..=idx {
            if smart_idx == 0 {
                continue;
            }
            let (Some(return_value), Some(turnover_value), Some(prior_intraday)) = (
                clean(overnight_returns[smart_idx]),
                clean(overnight_turnovers[smart_idx]),
                clean(intraday_turnovers[smart_idx - 1]),
            ) else {
                continue;
            };
            if turnover_value <= f64::EPSILON {
                continue;
            }
            let smart = ((return_value - min_return) / range) / turnover_value;
            if smart.is_finite() {
                candidates.push((smart, smart_idx, prior_intraday));
            }
        }

        if candidates.len() < TAKE_COUNT {
            continue;
        }
        candidates.sort_by(|left, right| {
            left.0
                .total_cmp(&right.0)
                .then_with(|| left.1.cmp(&right.1))
        });
        let sum = candidates
            .iter()
            .take(TAKE_COUNT)
            .map(|(_, _, value)| *value)
            .sum::<f64>();
        output[idx] = Some(sum / TAKE_COUNT as f64);
    }
    output
}

fn window_min_max(values: &[Option<f64>], start: usize, end: usize) -> Option<(f64, f64)> {
    let mut min_value = f64::INFINITY;
    let mut max_value = f64::NEG_INFINITY;
    let mut count = 0usize;
    for value in values.iter().take(end + 1).skip(start) {
        let value = clean(*value)?;
        min_value = min_value.min(value);
        max_value = max_value.max(value);
        count += 1;
    }
    (count == WINDOW).then_some((min_value, max_value))
}

fn percent_to_decimal(value: Option<f64>) -> Option<f64> {
    clean(value).map(|value| value / 100.0)
}

fn overnight_return(open: Option<f64>, pre_close: Option<f64>) -> Option<f64> {
    match (clean(open), clean(pre_close)) {
        (Some(open), Some(pre_close)) if pre_close.abs() > f64::EPSILON => {
            Some(open / pre_close - 1.0)
        }
        _ => None,
    }
}

fn intraday_turnover(full: Option<f64>, overnight: Option<f64>) -> Option<f64> {
    match (clean(full), clean(overnight)) {
        (Some(full), Some(overnight)) => {
            let intraday = full - overnight;
            (intraday >= 0.0).then_some(intraday)
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: Option<f64>, expected: f64) {
        let actual = actual.expect("value");
        assert!(
            (actual - expected).abs() < 1e-10,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn intraday_turnover_rejects_negative_values() {
        assert_close(intraday_turnover(Some(2.0), Some(0.5)), 1.5);
        assert_eq!(intraday_turnover(Some(0.1), Some(0.2)), None);
        assert_eq!(intraday_turnover(Some(0.1), None), None);
    }

    #[test]
    fn ctr_uses_prior_intraday_turnover_of_lowest_smart_days() {
        let returns = (0..=20).map(|value| Some(value as f64)).collect::<Vec<_>>();
        let overnight_turnovers = vec![Some(1.0); 21];
        let intraday_turnovers = (100..=120)
            .map(|value| Some(value as f64))
            .collect::<Vec<_>>();

        let factor = ctr_cutlets_series(&returns, &overnight_turnovers, &intraday_turnovers);

        assert_close(factor[20], (100.0 + 101.0 + 102.0 + 103.0) / 4.0);
    }

    #[test]
    fn ctr_requires_twenty_valid_returns_and_four_candidates() {
        let mut returns = vec![Some(1.0); 21];
        returns[10] = None;
        let overnight_turnovers = vec![Some(1.0); 21];
        let intraday_turnovers = vec![Some(1.0); 21];
        assert_eq!(
            ctr_cutlets_series(&returns, &overnight_turnovers, &intraday_turnovers)[20],
            None
        );

        let returns = (0..=20).map(|value| Some(value as f64)).collect::<Vec<_>>();
        let mut overnight_turnovers = vec![Some(1.0); 21];
        for idx in 1..18 {
            overnight_turnovers[idx] = None;
        }
        assert_eq!(
            ctr_cutlets_series(&returns, &overnight_turnovers, &intraday_turnovers)[20],
            None
        );
    }
}

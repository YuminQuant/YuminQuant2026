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
const TAKE_COUNT: usize = 3;

pub struct StockDailyJumpCtr;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyJumpCtr)
}

impl Factor for StockDailyJumpCtr {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "jump_ctr".to_string(),
            aliases: vec!["JumpCTR".to_string(), "JUMPCTR".to_string()],
            name: "JumpCTR".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: [
                "price_volume",
                "turnover",
                "overnight",
                "smart",
                "jump",
                "neutralize",
                "barra",
                "size",
                "daily",
                "DWZQ",
            ]
            .iter()
            .map(|value| value.to_string())
            .collect(),
            description: "Jump the Gun CTR factor using prior intraday turnovers ranked by next-day overnight smart money plus current intraday turnover, neutralized by SIZE.".to_string(),
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
            jump_ctr_series,
        )?;
        let factor = raw.cs_neutralize_regression(&[&size], None)?;
        Ok(factor.to_factor_series(self.spec()))
    }
}

fn jump_ctr_series(
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
        let Some(current_intraday) = clean(intraday_turnovers[idx]) else {
            continue;
        };
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
            let prior_idx = smart_idx - 1;
            let (Some(return_value), Some(turnover_value), Some(prior_intraday)) = (
                clean(overnight_returns[smart_idx]),
                clean(overnight_turnovers[smart_idx]),
                clean(intraday_turnovers[prior_idx]),
            ) else {
                continue;
            };
            if turnover_value <= f64::EPSILON {
                continue;
            }
            let smart = ((return_value - min_return) / range) / turnover_value;
            if smart.is_finite() {
                candidates.push((smart, prior_idx, prior_intraday));
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
        let selected_sum = candidates
            .iter()
            .take(TAKE_COUNT)
            .map(|(_, _, value)| *value)
            .sum::<f64>();
        output[idx] = Some((selected_sum + current_intraday) / (TAKE_COUNT as f64 + 1.0));
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
    fn jump_ctr_aligns_candidate_with_next_day_smart() {
        let returns = (0..=20).map(|value| Some(value as f64)).collect::<Vec<_>>();
        let overnight_turnovers = vec![Some(1.0); 21];
        let intraday_turnovers = (100..=120)
            .map(|value| Some(value as f64))
            .collect::<Vec<_>>();

        let factor = jump_ctr_series(&returns, &overnight_turnovers, &intraday_turnovers);

        assert_close(factor[20], (100.0 + 101.0 + 102.0 + 120.0) / 4.0);
    }

    #[test]
    fn jump_ctr_requires_current_intraday_turnover() {
        let returns = (0..=20).map(|value| Some(value as f64)).collect::<Vec<_>>();
        let overnight_turnovers = vec![Some(1.0); 21];
        let mut intraday_turnovers = vec![Some(1.0); 21];
        intraday_turnovers[20] = None;

        assert_eq!(
            jump_ctr_series(&returns, &overnight_turnovers, &intraday_turnovers)[20],
            None
        );
    }

    #[test]
    fn jump_ctr_rejects_zero_range_and_zero_overnight_turnover() {
        let returns = vec![Some(1.0); 21];
        let overnight_turnovers = vec![Some(1.0); 21];
        let intraday_turnovers = vec![Some(1.0); 21];
        assert_eq!(
            jump_ctr_series(&returns, &overnight_turnovers, &intraday_turnovers)[20],
            None
        );

        let returns = (0..=20).map(|value| Some(value as f64)).collect::<Vec<_>>();
        let overnight_turnovers = vec![Some(0.0); 21];
        assert_eq!(
            jump_ctr_series(&returns, &overnight_turnovers, &intraday_turnovers)[20],
            None
        );
    }
}

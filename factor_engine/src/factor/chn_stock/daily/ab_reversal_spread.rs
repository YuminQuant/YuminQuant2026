use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::stock_daily_ops::neutralize_size_sector;
use crate::factor::common::vector::clean;
use crate::factor::Factor;
use crate::operators::ts_mean;

const VERSION: &str = "0.1.0";
const EVENT_WINDOW: usize = 20;
const AB_WINDOW: usize = 240;
const MIN_PERIODS: usize = 1;

pub struct StockDailyAbReversalSpread;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyAbReversalSpread)
}

impl Factor for StockDailyAbReversalSpread {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "ab_reversal_spread".to_string(),
            aliases: vec!["YuLi".to_string(), "AB_NR_minus_AB_PR".to_string()],
            name: "ab_reversal_spread".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: tags(),
            description: "ZSZQ daily intraday reversal anomaly spread: AB_NR minus AB_PR, where NR/PR are 20-day frequencies of overnight-up intraday-down and overnight-down intraday-up days, scaled by their 240-day rolling means, then neutralized by Barra SIZE and SW sector.".to_string(),
            dependencies: vec![
                DataRequest::new(DatasetId::StockDailyPv, &["open", "close", "pre_close"]),
                DataRequest::new(DatasetId::StockBarraDaily, &["SIZE"]),
                DataRequest::new(DatasetId::StockSwClassification, &["l1_code"]),
            ],
            intraday_raw_dependencies: Vec::new(),
            lookback: Lookback { trading_days: 258 },
        }
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let panel = data.daily_panel(DatasetId::StockDailyPv)?;
        let open = panel.column("open")?;
        let close = panel.column("close")?;
        let pre_close = panel.column("pre_close")?;

        let nr_event = open
            .zip_binary(&pre_close, overnight_up_event_base)?
            .zip_binary(
                &close.zip_binary(&open, intraday_down_event_base)?,
                event_and,
            )?;
        let pr_event = open
            .zip_binary(&pre_close, overnight_down_event_base)?
            .zip_binary(&close.zip_binary(&open, intraday_up_event_base)?, event_and)?;

        let nr_frequency = nr_event.ts(|series| ts_mean(series, EVENT_WINDOW, MIN_PERIODS))?;
        let pr_frequency = pr_event.ts(|series| ts_mean(series, EVENT_WINDOW, MIN_PERIODS))?;
        let ab_nr = abnormal_component(&nr_frequency)?;
        let ab_pr = abnormal_component(&pr_frequency)?;
        let raw = ab_nr.zip_binary(&ab_pr, spread)?;
        let factor = neutralize_size_sector(&raw, &panel, data)?;
        Ok(factor.to_factor_series(self.spec()))
    }
}

fn tags() -> Vec<String> {
    [
        "ZSZQ",
        "deprecated",
        "reversal",
        "overnight",
        "intraday",
        "abnormal",
        "frequency",
        "neutralize",
        "barra",
        "size",
        "sector",
        "daily",
    ]
    .iter()
    .map(|value| value.to_string())
    .collect()
}

fn overnight_up_event_base(open: Option<f64>, pre_close: Option<f64>) -> Option<f64> {
    return_condition(open, pre_close, |ret| ret > 0.0)
}

fn overnight_down_event_base(open: Option<f64>, pre_close: Option<f64>) -> Option<f64> {
    return_condition(open, pre_close, |ret| ret < 0.0)
}

fn intraday_up_event_base(close: Option<f64>, open: Option<f64>) -> Option<f64> {
    return_condition(close, open, |ret| ret > 0.0)
}

fn intraday_down_event_base(close: Option<f64>, open: Option<f64>) -> Option<f64> {
    return_condition(close, open, |ret| ret < 0.0)
}

fn return_condition<F>(
    numerator: Option<f64>,
    denominator: Option<f64>,
    predicate: F,
) -> Option<f64>
where
    F: FnOnce(f64) -> bool,
{
    match (clean(numerator), clean(denominator)) {
        (Some(numerator), Some(denominator)) if denominator > f64::EPSILON => {
            let ret = numerator / denominator - 1.0;
            ret.is_finite()
                .then_some(if predicate(ret) { 1.0 } else { 0.0 })
        }
        _ => None,
    }
}

fn event_and(first: Option<f64>, second: Option<f64>) -> Option<f64> {
    match (clean(first), clean(second)) {
        (Some(first), Some(second)) => Some(if first > 0.0 && second > 0.0 {
            1.0
        } else {
            0.0
        }),
        _ => None,
    }
}

fn abnormal_component(
    frequency: &crate::factor::common::PanelColumn,
) -> Result<crate::factor::common::PanelColumn> {
    let mean = frequency.ts(|series| ts_mean(series, AB_WINDOW, MIN_PERIODS))?;
    frequency.zip_binary(&mean, safe_div)
}

fn safe_div(numerator: Option<f64>, denominator: Option<f64>) -> Option<f64> {
    match (clean(numerator), clean(denominator)) {
        (Some(numerator), Some(denominator)) if denominator.abs() > f64::EPSILON => {
            let value = numerator / denominator;
            value.is_finite().then_some(value)
        }
        _ => None,
    }
}

fn spread(left: Option<f64>, right: Option<f64>) -> Option<f64> {
    match (clean(left), clean(right)) {
        (Some(left), Some(right)) => {
            let value = left - right;
            value.is_finite().then_some(value)
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
            (actual - expected).abs() < 1e-12,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn zszq_ab_reversal_spread_events_match_overnight_and_intraday_signs() {
        let overnight_up = overnight_up_event_base(Some(11.0), Some(10.0));
        let overnight_down = overnight_down_event_base(Some(9.0), Some(10.0));
        let intraday_down = intraday_down_event_base(Some(9.5), Some(10.0));
        let intraday_up = intraday_up_event_base(Some(10.5), Some(10.0));

        assert_eq!(event_and(overnight_up, intraday_down), Some(1.0));
        assert_eq!(event_and(overnight_down, intraday_up), Some(1.0));
        assert_eq!(
            event_and(
                overnight_up_event_base(Some(11.0), Some(10.0)),
                intraday_down_event_base(Some(10.5), Some(10.0)),
            ),
            Some(0.0)
        );
    }

    #[test]
    fn zszq_ab_reversal_spread_zero_return_is_valid_non_event() {
        assert_eq!(overnight_up_event_base(Some(10.0), Some(10.0)), Some(0.0));
        assert_eq!(intraday_down_event_base(Some(10.0), Some(10.0)), Some(0.0));
    }

    #[test]
    fn zszq_ab_reversal_spread_invalid_prices_are_missing() {
        assert_eq!(overnight_up_event_base(Some(10.0), Some(0.0)), None);
        assert_eq!(intraday_down_event_base(None, Some(10.0)), None);
    }

    #[test]
    fn zszq_ab_reversal_spread_rolling_frequency_and_ab_component() {
        let mut events = vec![Some(0.0); 260];
        events[18] = Some(1.0);
        events[19] = Some(1.0);
        let frequency = ts_mean(&events, EVENT_WINDOW, MIN_PERIODS);
        assert_close(frequency[19], 0.1);

        let ab_mean = ts_mean(&frequency, AB_WINDOW, MIN_PERIODS);
        let ab_value = safe_div(frequency[19], ab_mean[19]);
        assert!(ab_value.expect("ab") > 0.0);
    }

    #[test]
    fn zszq_ab_reversal_spread_zero_ab_denominator_is_missing() {
        assert_eq!(safe_div(Some(1.0), Some(0.0)), None);
    }

    #[test]
    fn zszq_ab_reversal_spread_spec_has_zszq_tag() {
        let spec = StockDailyAbReversalSpread.spec();
        assert_eq!(spec.id, "ab_reversal_spread");
        assert!(spec.tags.iter().any(|tag| tag == "ZSZQ"));
        assert_eq!(spec.lookback.trading_days, 258);
    }
}

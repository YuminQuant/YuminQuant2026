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

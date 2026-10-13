use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::stock_daily_ops::is_bj_stock;
use crate::factor::common::vector::clean;
use crate::factor::common::{DailyPanel, PanelColumn};
use crate::factor::Factor;
use crate::operators::cs_regression_residual;

const WINDOW: usize = 160;
const REVERSE_WINDOW: usize = 20;
const LOW_QUANTILE: f64 = 0.70;

pub struct StockDailyLongMom2;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyLongMom2)
}

impl Factor for StockDailyLongMom2 {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "longmom2".into(),
            aliases: vec!["LongMom2".into(), "LONGMOM2".into()],
            name: "Long Momentum 2.0".into(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: "0.1.0".into(),
            tags: ["KYZQ", "price_volume", "momentum", "amplitude", "excess_return",
                "reversal_neutralize", "daily"].into_iter().map(str::to_string).collect(),
            description: "KYZQ LongMom2: first take 160 trading dates inclusive of today, then discard missing/nontraded and closing-limit-up/down observations without extending the window. Amplitude=(high-low)/pre_close. Sum daily pct_chg/100 minus the equal-weight return of present, traded, valid non-BJ stocks over amplitudes <= the linearly interpolated 70th percentile (boundary ties retained). Closing-limit stocks participate in the market mean. Finally take cross-sectional OLS residual against the compounded last 20 daily returns with intercept. Both rolling windows use min_periods=1, including partial early windows; skip missing observations, with all-missing windows remaining null. Excludes BJ; no industry/SIZE neutralization or final standardization.".into(),
            dependencies: vec![
                DataRequest::new(DatasetId::StockDailyPv,
                    &["close", "high", "low", "pre_close", "pct_chg", "vol"]).with_panel_only(),
                DataRequest::new(DatasetId::StockDailyLimit, &["up_limit", "down_limit"]).with_panel_only(),
            ],
            intraday_raw_dependencies: vec![],
            lookback: Lookback { trading_days: WINDOW - 1 },
        }
    }

    fn compute(&self, _: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let panel = data.daily_panel(DatasetId::StockDailyPv)?;
        let close = panel.column("close")?;
        let high = panel.column("high")?;
        let low = panel.column("low")?;
        let pre_close = panel.column("pre_close")?;
        let pct_chg = panel.column("pct_chg")?;
        let volume = panel.column("vol")?;
        let limits = data.daily_panel(DatasetId::StockDailyLimit)?;
        let up_limit = panel.column_from_panel(limits, "up_limit")?;
        let down_limit = panel.column_from_panel(limits, "down_limit")?;
        let market = market_returns(panel, &close, &pre_close, &pct_chg, &volume);
        let n = panel.instruments().len();
        let mut raw = vec![None; panel.shape_len()];
        let mut reverse = vec![None; panel.shape_len()];
        for (stock, code) in panel.instruments().iter().enumerate() {
            if is_bj_stock(code) {
                continue;
            }
            // Keep per-stock history locally rather than materializing amplitude/excess panels.
            let mut returns = vec![None; panel.dates().len()];
            let mut amplitudes = vec![None; panel.dates().len()];
            let mut excess = vec![None; panel.dates().len()];
            for day in 0..panel.dates().len() {
                let offset = day * n + stock;
                if !panel.is_present_offset(offset) {
                    continue;
                }
                returns[day] = daily_return(
                    close.values()[offset],
                    pre_close.values()[offset],
                    pct_chg.values()[offset],
                );
                amplitudes[day] = eligible_amplitude(
                    close.values()[offset],
                    high.values()[offset],
                    low.values()[offset],
                    pre_close.values()[offset],
                    volume.values()[offset],
                    up_limit.values()[offset],
                    down_limit.values()[offset],
                );
                excess[day] = returns[day]
                    .zip(market[day])
                    .and_then(|(r, m)| clean(Some(r - m)));
            }
            let sums = low_amplitude_series(&amplitudes, &excess);
            let ret20 = compounded_return_series(&returns);
            for (day, date) in panel.dates().iter().enumerate() {
                if panel.is_target_date(*date) {
                    raw[day * n + stock] = sums[day];
                    reverse[day * n + stock] = ret20[day];
                }
            }
        }
        let raw = panel.column_from_values(raw)?;
        let reverse = panel.column_from_values(reverse)?;
        Ok(raw
            .cs_binary(&reverse, cs_regression_residual)?
            .to_factor_series(self.spec()))
    }
}

fn positive(value: Option<f64>) -> Option<f64> {
    clean(value).filter(|v| *v > 0.0)
}

fn daily_return(close: Option<f64>, pre_close: Option<f64>, pct_chg: Option<f64>) -> Option<f64> {
    positive(close)?;
    positive(pre_close)?;
    clean(pct_chg).filter(|v| *v >= -100.0).map(|v| v / 100.0)
}

fn eligible_amplitude(
    close: Option<f64>,
    high: Option<f64>,
    low: Option<f64>,
    pre_close: Option<f64>,
    volume: Option<f64>,
    up_limit: Option<f64>,
    down_limit: Option<f64>,
) -> Option<f64> {
    let close = positive(close)?;
    let high = positive(high)?;
    let low = positive(low)?;
    let pre_close = positive(pre_close)?;
    positive(volume)?;
    let up = positive(up_limit)?;
    let down = positive(down_limit)?;
    // Daily prices are quoted in cents; rounding handles float32 parquet representation.
    let cents = |value: f64| (value * 100.0).round();
    if high < low || up <= down || cents(close) >= cents(up) || cents(close) <= cents(down) {
        return None;
    }
    clean(Some((high - low) / pre_close))
}

fn market_returns(
    panel: &DailyPanel,
    close: &PanelColumn,
    pre_close: &PanelColumn,
    pct_chg: &PanelColumn,
    volume: &PanelColumn,
) -> Vec<Option<f64>> {
    let n = panel.instruments().len();
    (0..panel.dates().len())
        .map(|day| {
            let mut sum = 0.0;
            let mut count = 0usize;
            for (stock, code) in panel.instruments().iter().enumerate() {
                let offset = day * n + stock;
                if is_bj_stock(code)
                    || !panel.is_present_offset(offset)
                    || positive(volume.values()[offset]).is_none()
                {
                    continue;
                }
                if let Some(r) = daily_return(
                    close.values()[offset],
                    pre_close.values()[offset],
                    pct_chg.values()[offset],
                ) {
                    sum += r;
                    count += 1;
                }
            }
            if count == 0 {
                None
            } else {
                clean(Some(sum / count as f64))
            }
        })
        .collect()
}

fn low_amplitude_series(amplitudes: &[Option<f64>], excess: &[Option<f64>]) -> Vec<Option<f64>> {
    let mut output = vec![None; amplitudes.len()];
    let mut samples = Vec::with_capacity(WINDOW);
    for end in 0..amplitudes.len() {
        samples.clear();
        for q in (end + 1).saturating_sub(WINDOW)..=end {
            if let (Some(a), Some(r)) = (clean(amplitudes[q]), clean(excess[q])) {
                samples.push((a, r));
            }
        }
        output[end] = low_amplitude_sum(&mut samples);
    }
    output
}


fn low_amplitude_sum(samples: &mut [(f64, f64)]) -> Option<f64> {
    if samples.is_empty() {
        return None;
    }
    let position = (samples.len() - 1) as f64 * LOW_QUANTILE;
    let lower = position.floor() as usize;
    let fraction = position - lower as f64;
    let cutoff = {
        let (_, mid, upper) = samples.select_nth_unstable_by(lower, |a, b| a.0.total_cmp(&b.0));
        let upper_value = if fraction > 0.0 {
            upper.iter().map(|v| v.0).reduce(f64::min)?
        } else {
            mid.0
        };
        mid.0 + fraction * (upper_value - mid.0)
    };
    clean(Some(
        samples
            .iter()
            .filter(|(a, _)| *a <= cutoff)
            .map(|(_, r)| r)
            .sum(),
    ))
}

fn compounded_return_series(returns: &[Option<f64>]) -> Vec<Option<f64>> {
    let mut output = vec![None; returns.len()];
    for end in 0..returns.len() {
        let mut gross = 1.0;
        let mut count = 0usize;
        for value in &returns[(end + 1).saturating_sub(REVERSE_WINDOW)..=end] {
            let Some(value) = clean(*value) else {
                continue;
            };
            gross *= 1.0 + value;
            count += 1;
        }
        if count >= 1 {
            output[end] = clean(Some(gross - 1.0));
        }
    }
    output
}

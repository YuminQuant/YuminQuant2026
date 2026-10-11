use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::stock_daily_ops::is_bj_stock;
use crate::factor::common::vector::clean;
use crate::factor::common::DailyPanel;
use crate::factor::Factor;
use crate::operators::cs_zscore;

const VERSION: &str = "0.1.0";
const WINDOW: usize = 20;
const MIN_PERIODS: usize = 10;

pub struct StockDailyCc;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyCc)
}

impl Factor for StockDailyCc {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "cc".to_string(),
            aliases: vec!["CC".to_string()],
            name: "cc".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: tags(),
            description: "HXZQ cross-sectional network centrality factor. It combines SCC, the 20-day correlation of each stock log return with the equal-weight market return excluding itself, and TCC, the inverse 20-day mean squared market z-deviation, after daily cross-sectional z-score normalization. BJ stocks are excluded and the raw CC composite is not Barra or sector neutralized.".to_string(),
            dependencies: vec![DataRequest::new(
                DatasetId::StockDailyPv,
                &["close", "pre_close"],
            )],
            intraday_raw_dependencies: Vec::new(),
            lookback: Lookback {
                trading_days: WINDOW - 1,
            },
        }
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let panel = data.daily_panel(DatasetId::StockDailyPv)?;
        let close = panel.column("close")?;
        let pre_close = panel.column("pre_close")?;
        let returns = close.zip_binary(&pre_close, log_return)?;
        let eligible = eligible_instruments(&panel);
        let market_stats = market_stats_by_date(&panel, returns.values(), &eligible);
        let scc = panel.column_from_values(scc_values(
            &panel,
            returns.values(),
            &eligible,
            &market_stats,
        ))?;
        let tcc = panel.column_from_values(tcc_values(
            &panel,
            returns.values(),
            &eligible,
            &market_stats,
        ))?;
        let factor = average_pair(&scc.cs(cs_zscore)?, &tcc.cs(cs_zscore)?, &panel)?;
        Ok(factor.to_factor_series(self.spec()))
    }
}

fn tags() -> Vec<String> {
    [
        "price_volume",
        "HXZQ",
        "cs_network",
        "correlation",
        "centrality",
        "return",
        "zscore",
        "daily",
    ]
    .iter()
    .map(|value| value.to_string())
    .collect()
}

fn log_return(close: Option<f64>, pre_close: Option<f64>) -> Option<f64> {
    let (Some(close), Some(pre_close)) = (clean(close), clean(pre_close)) else {
        return None;
    };
    if close <= f64::EPSILON || pre_close <= f64::EPSILON {
        return None;
    }
    let value = (close / pre_close).ln();
    value.is_finite().then_some(value)
}

fn eligible_instruments(panel: &DailyPanel) -> Vec<bool> {
    panel
        .instruments()
        .iter()
        .map(|ts_code| !is_bj_stock(ts_code))
        .collect()
}

#[derive(Clone, Copy, Debug, Default)]
struct MarketStats {
    sum: f64,
    count: usize,
    mean: Option<f64>,
    std: Option<f64>,
}

fn market_stats_by_date(
    panel: &DailyPanel,
    returns: &[Option<f64>],
    eligible: &[bool],
) -> Vec<MarketStats> {
    let instrument_count = panel.instruments().len();
    let mut output = Vec::with_capacity(panel.dates().len());

    for date_idx in 0..panel.dates().len() {
        let offset = date_idx * instrument_count;
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        let mut count = 0usize;
        for instrument_idx in 0..instrument_count {
            if !eligible[instrument_idx] {
                continue;
            }
            let Some(value) = clean(returns[offset + instrument_idx]) else {
                continue;
            };
            sum += value;
            sum_sq += value * value;
            count += 1;
        }

        if count == 0 {
            output.push(MarketStats::default());
            continue;
        }

        let mean = sum / count as f64;
        let variance = (sum_sq / count as f64) - mean * mean;
        let std = if count >= 2 && variance > f64::EPSILON {
            Some(variance.max(0.0).sqrt())
        } else {
            None
        };
        output.push(MarketStats {
            sum,
            count,
            mean: Some(mean),
            std,
        });
    }

    output
}

fn scc_values(
    panel: &DailyPanel,
    returns: &[Option<f64>],
    eligible: &[bool],
    market_stats: &[MarketStats],
) -> Vec<Option<f64>> {
    let instrument_count = panel.instruments().len();
    let date_count = panel.dates().len();
    let mut output = vec![None; panel.shape_len()];

    for instrument_idx in 0..instrument_count {
        if !eligible[instrument_idx] {
            continue;
        }

        let mut state = RollingCorrelation::default();
        for date_idx in 0..date_count {
            let offset = date_idx * instrument_count;
            if let Some(pair) =
                scc_pair_at(returns, market_stats, offset + instrument_idx, date_idx)
            {
                state.add(pair.0, pair.1);
            }
            if date_idx >= WINDOW {
                let remove_offset = (date_idx - WINDOW) * instrument_count;
                if let Some(pair) = scc_pair_at(
                    returns,
                    market_stats,
                    remove_offset + instrument_idx,
                    date_idx - WINDOW,
                ) {
                    state.remove(pair.0, pair.1);
                }
            }

            if let Some(rho_bar) = state.correlation(MIN_PERIODS) {
                let denominator = 2.0 * (1.0 - rho_bar);
                if denominator > f64::EPSILON {
                    let value = 1.0 / denominator;
                    if value.is_finite() {
                        output[offset + instrument_idx] = Some(value);
                    }
                }
            }
        }
    }

    output
}

fn scc_pair_at(
    returns: &[Option<f64>],
    market_stats: &[MarketStats],
    panel_idx: usize,
    date_idx: usize,
) -> Option<(f64, f64)> {
    let stock_return = clean(returns[panel_idx])?;
    let stats = market_stats[date_idx];
    if stats.count < 2 {
        return None;
    }
    let market_ex_self = (stats.sum - stock_return) / (stats.count - 1) as f64;
    market_ex_self
        .is_finite()
        .then_some((stock_return, market_ex_self))
}

fn tcc_values(
    panel: &DailyPanel,
    returns: &[Option<f64>],
    eligible: &[bool],
    market_stats: &[MarketStats],
) -> Vec<Option<f64>> {
    let instrument_count = panel.instruments().len();
    let date_count = panel.dates().len();
    let mut z_square = vec![None; panel.shape_len()];

    for date_idx in 0..date_count {
        let offset = date_idx * instrument_count;
        let Some(mean) = market_stats[date_idx].mean else {
            continue;
        };
        let Some(std) = market_stats[date_idx].std else {
            continue;
        };
        if std <= f64::EPSILON {
            continue;
        }
        for instrument_idx in 0..instrument_count {
            if !eligible[instrument_idx] {
                continue;
            }
            let Some(value) = clean(returns[offset + instrument_idx]) else {
                continue;
            };
            let z = (value - mean) / std;
            if z.is_finite() {
                z_square[offset + instrument_idx] = Some(z * z);
            }
        }
    }

    rolling_inverse_mean(&z_square, panel, eligible, WINDOW, MIN_PERIODS)
}

fn rolling_inverse_mean(
    values: &[Option<f64>],
    panel: &DailyPanel,
    eligible: &[bool],
    window: usize,
    min_periods: usize,
) -> Vec<Option<f64>> {
    let instrument_count = panel.instruments().len();
    let date_count = panel.dates().len();
    let mut output = vec![None; panel.shape_len()];

    for instrument_idx in 0..instrument_count {
        if !eligible[instrument_idx] {
            continue;
        }

        let mut sum = 0.0;
        let mut count = 0usize;
        for date_idx in 0..date_count {
            let offset = date_idx * instrument_count + instrument_idx;
            if let Some(value) = clean(values[offset]) {
                sum += value;
                count += 1;
            }
            if date_idx >= window {
                let remove_offset = (date_idx - window) * instrument_count + instrument_idx;
                if let Some(value) = clean(values[remove_offset]) {
                    sum -= value;
                    count -= 1;
                }
            }
            if count >= min_periods {
                let mean = sum / count as f64;
                if mean > f64::EPSILON {
                    let value = 1.0 / mean;
                    if value.is_finite() {
                        output[offset] = Some(value);
                    }
                }
            }
        }
    }

    output
}

#[derive(Default)]
struct RollingCorrelation {
    count: usize,
    sum_x: f64,
    sum_y: f64,
    sum_x2: f64,
    sum_y2: f64,
    sum_xy: f64,
}

impl RollingCorrelation {
    fn add(&mut self, x: f64, y: f64) {
        self.count += 1;
        self.sum_x += x;
        self.sum_y += y;
        self.sum_x2 += x * x;
        self.sum_y2 += y * y;
        self.sum_xy += x * y;
    }

    fn remove(&mut self, x: f64, y: f64) {
        self.count = self.count.saturating_sub(1);
        self.sum_x -= x;
        self.sum_y -= y;
        self.sum_x2 -= x * x;
        self.sum_y2 -= y * y;
        self.sum_xy -= x * y;
        if self.sum_x.abs() < 1e-12 {
            self.sum_x = 0.0;
        }
        if self.sum_y.abs() < 1e-12 {
            self.sum_y = 0.0;
        }
        if self.sum_x2.abs() < 1e-12 {
            self.sum_x2 = 0.0;
        }
        if self.sum_y2.abs() < 1e-12 {
            self.sum_y2 = 0.0;
        }
        if self.sum_xy.abs() < 1e-12 {
            self.sum_xy = 0.0;
        }
    }

    fn correlation(&self, min_periods: usize) -> Option<f64> {
        if self.count < min_periods || self.count < 2 {
            return None;
        }
        let count = self.count as f64;
        let cov = self.sum_xy - self.sum_x * self.sum_y / count;
        let var_x = self.sum_x2 - self.sum_x * self.sum_x / count;
        let var_y = self.sum_y2 - self.sum_y * self.sum_y / count;
        if var_x <= f64::EPSILON || var_y <= f64::EPSILON {
            return None;
        }
        let corr = cov / (var_x * var_y).sqrt();
        corr.is_finite().then_some(corr.clamp(-1.0, 1.0))
    }
}

fn average_pair(
    left: &crate::factor::common::PanelColumn,
    right: &crate::factor::common::PanelColumn,
    panel: &DailyPanel,
) -> Result<crate::factor::common::PanelColumn> {
    let values = left
        .values()
        .iter()
        .zip(right.values())
        .map(|(left, right)| match (clean(*left), clean(*right)) {
            (Some(left), Some(right)) => {
                let value = 0.5 * (left + right);
                value.is_finite().then_some(value)
            }
            _ => None,
        })
        .collect();
    panel.column_from_values(values)
}

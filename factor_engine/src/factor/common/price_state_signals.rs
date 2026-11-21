use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::{err, Result};
use crate::factor::common::stock_daily_ops::{mask_bj, neutralize_size_sector_with_inputs};
use crate::factor::common::{ClassificationLevel, ClassificationMap, PanelColumn};
use crate::factor::Factor;
use crate::operators::{
    cs_pctrank, cs_regression_residual, cs_zscore, ts_delay, ts_max, ts_min, ts_std_dev, ts_sum,
};

const IDS: [&str; 4] = [
    "path_reversal",
    "position_reversal",
    "composite_momentum",
    "rebound_ivol60",
];

pub struct PriceState(pub usize);

impl Factor for PriceState {
    fn spec(&self) -> FactorSpec {
        spec(self.0)
    }

    fn compute_provider_key(&self) -> String {
        "stock|daily|price_state_signals".into()
    }

    fn compute(&self, ctx: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        self.compute_many(&[IDS[self.0].into()], ctx, data)?
            .pop()
            .ok_or_else(|| err("price state provider returned no output"))
    }

    fn compute_many(
        &self,
        requested: &[String],
        _: &FactorContext,
        data: &DataPool,
    ) -> Result<Vec<FactorSeries>> {
        let [path, position, momentum, volatility] =
            IDS.map(|id| requested.iter().any(|value| value == id));
        if !(path || position || momentum || volatility) {
            return Ok(Vec::new());
        }
        let panel = data.daily_panel(DatasetId::StockDailyPv)?;
        let adj =
            panel.column_from_panel(data.daily_panel(DatasetId::StockAdjFactor)?, "adj_factor")?;
        let price = |name| {
            let adjusted = panel
                .column(name)?
                .zip_binary(&adj, |p, a| positive(positive(p)? * positive(a)?))?;
            mask_bj(&adjusted, panel)
        };
        let close = price("close")?;
        let prev = close.ts_parallel(|s| ts_delay(s, 1))?;
        let size =
            panel.column_from_panel(data.daily_panel(DatasetId::StockBarraDaily)?, "SIZE")?;
        let sector = ClassificationMap::from_table(
            data.daily(DatasetId::StockSwClassification)?,
            ClassificationLevel::Sector,
        )?;
        let mut output = Vec::new();

        if path || position || momentum {
            let log_return = close.zip_binary(&prev, |a, b| finite(a?.ln() - b?.ln()))?;
            let ret20 = sum(&log_return, 20)?;
            let z20 = ret20.cs_parallel(standardize)?;
            if path {
                let delta = close.zip_binary(&prev, |a, b| finite(a? - b?))?;
                let length = sum(&delta.map_values(|v| v.map(f64::abs)), 20)?;
                let efficiency = sum(&delta, 20)?.zip_binary(&length, ratio)?;
                let z_efficiency = efficiency.cs_parallel(standardize)?;
                let resid = z20.cs_binary_parallel(&z_efficiency, cs_regression_residual)?;
                output.push((0, resid.map_values(|x| x.map(|v| -v))));
            }
            if position || momentum {
                let z60 = sum(&log_return, 60)?.cs_parallel(standardize)?;
                let low = price("low")?;
                let high = price("high")?;
                let range_low = low.ts_parallel(|s| ts_min(s, 60, 60))?;
                let range_high = high.ts_parallel(|s| ts_max(s, 60, 60))?;
                let range = range_high.zip_binary(&range_low, difference)?;
                let location = close
                    .zip_binary(&range_low, difference)?
                    .zip_binary(&range, ratio)?;
                let residual = state_residual(&location, &z20, &z60)?;
                if position {
                    let z_position = residual.cs_parallel(cs_zscore)?;
                    output.push((
                        1,
                        z_position.zip_binary(&z20, |p, r| finite((p? - r?) * 0.5))?,
                    ));
                }
                if momentum {
                    let group = ret20.cs_parallel(|s| cs_pctrank(s, true))?;
                    let strong_day = log_return
                        .cs_parallel(|s| cs_pctrank(s, true))?
                        .map_values(|v| v.map(|rank| if rank > 0.5 { 1.0 } else { 0.0 }));
                    let gap = price("open")?.zip_binary(&prev, difference)?;
                    let mut combined = component_score(&residual, &group)?;
                    for window in [60, 120] {
                        let interval = if window == 60 {
                            range.clone()
                        } else {
                            let lo = low.ts_parallel(|s| ts_min(s, window, window))?;
                            let hi = high.ts_parallel(|s| ts_max(s, window, window))?;
                            let interval = hi.zip_binary(&lo, difference)?;
                            let loc = close
                                .zip_binary(&lo, difference)?
                                .zip_binary(&interval, ratio)?;
                            let component = state_residual(&loc, &z20, &z60)?;
                            combined = add_component(combined, &component, &group)?;
                            interval
                        };
                        let close_rank = close.ts_parallel(|s| latest_percentile(s, window))?;
                        let strong =
                            sum(&strong_day, window)?.map_values(|v| v.map(|n| n / window as f64));
                        let gap_sum = sum(&gap, window)?;
                        let overnight = gap_sum.zip_binary(&interval, ratio)?;
                        for raw in [close_rank, strong, overnight] {
                            let component = state_residual(&raw, &z20, &z60)?;
                            combined = add_component(combined, &component, &group)?;
                        }
                    }
                    output.push((2, combined.map_values(|v| v.map(|s| s / 8.0))));
                }
            }
        }
        if volatility {
            let pb =
                panel.column_from_panel(data.daily_panel(DatasetId::StockDailyBasic)?, "pb")?;
            let bp = pb.map_values(|v| {
                let p = v?;
                if p == 0.0 {
                    None
                } else {
                    finite(p.recip())
                }
            });
            let daily_return = close.zip_binary(&prev, |a, b| finite(a? / b? - 1.0))?;
            let residual = daily_return.cs_neutralize_regression_by_group(
                &[&size, &bp],
                None,
                |date, codes| sector.groups_for(date, codes),
            )?;
            let ivol = residual.ts_parallel(|s| ts_std_dev(s, 60, 60))?;
            let old_close = close.ts_parallel(|s| ts_delay(s, 20))?;
            let old_low = price("low")?.ts_parallel(|s| ts_delay(&ts_min(s, 230, 230), 20))?;
            let rebound = old_close.zip_binary(&old_low, |a, b| finite(ratio(a, b)? - 1.0))?;
            let adjusted = ivol
                .cs_parallel(standardize)?
                .cs_binary_parallel(&rebound.cs_parallel(standardize)?, cs_regression_residual)?;
            output.push((3, adjusted.map_values(|v| v.map(|s| -s))));
        }
        output
            .into_iter()
            .map(|(i, values)| {
                Ok(
                    neutralize_size_sector_with_inputs(&values, panel, &size, &sector)?
                        .to_factor_series(spec(i)),
                )
            })
            .collect()
    }
}

fn spec(i: usize) -> FactorSpec {
    let columns: &[&str] = match i {
        0 => &["close"],
        1 => &["close", "high", "low"],
        2 => &["open", "close", "high", "low"],
        _ => &["close", "low"],
    };
    let mut dependencies = vec![
        DataRequest::new(DatasetId::StockDailyPv, columns).with_panel_only(),
        DataRequest::new(DatasetId::StockAdjFactor, &["adj_factor"]).with_panel_only(),
        DataRequest::new(DatasetId::StockBarraDaily, &["SIZE"]).with_panel_only(),
        DataRequest::new(DatasetId::StockSwClassification, &["l1_code"]),
    ];
    if i == 3 {
        dependencies.push(DataRequest::new(DatasetId::StockDailyBasic, &["pb"]).with_panel_only());
    }
    let formula = match i {
        0 => "Negative residual of 20-day log return on signed path efficiency sum(delta close)/sum(abs(delta close)).",
        1 => "Equal-weight standardized 60-day price-location residual (on 20/60-day log returns) and negative 20-day log return; no reversal group mapping.",
        2 => "Eight equal-weight standardized signals: 60/120-day price location, temporal close rank, fraction of market-top-half daily returns and summed overnight gaps/price range. Regress each on 20/60-day log returns, then group by top-20% 20-day return and map within-group residual percentile to [0,0.8] vs [0.2,1]. Require all eight.",
        _ => "Negative residual of 60-day population std of daily return residuals (SW L1, Barra SIZE, 1/pb) on close[t-20]/min(low[t-249..t-20])-1. B/P is 1/pb, retaining finite negative ratios.",
    };
    FactorSpec {
        id: IDS[i].into(),
        aliases: vec![],
        name: IDS[i].into(),
        asset_class: AssetClass::Stock,
        frequency: Frequency::Daily,
        version: "0.1.0".into(),
        tags: ["CTZQ", "price_volume", "price_state", "daily", "neutralize", "barra", "size", "sector"]
            .into_iter().map(str::to_string).collect(),
        description: format!("{formula} Adjusted OHLC, complete windows, average ties, non-BJ. State regressions use median +/- 5 raw MAD clipping (skip if MAD is zero) then cross-sectional z-score. Final SW L1 + Barra SIZE neutralization."),
        dependencies,
        intraday_raw_dependencies: vec![],
        lookback: Lookback { trading_days: [20, 60, 120, 249][i] },
    }
}

fn finite(v: f64) -> Option<f64> {
    v.is_finite().then_some(v)
}

fn positive(v: impl Into<Option<f64>>) -> Option<f64> {
    v.into().filter(|x| x.is_finite() && *x > 0.0)
}

fn ratio(numerator: Option<f64>, denominator: Option<f64>) -> Option<f64> {
    finite(numerator? / positive(denominator)?)
}

fn difference(a: Option<f64>, b: Option<f64>) -> Option<f64> {
    finite(a? - b?)
}

fn sum(values: &PanelColumn, window: usize) -> Result<PanelColumn> {
    values.ts_parallel(|s| ts_sum(s, window, window))
}

fn latest_percentile(values: &[Option<f64>], window: usize) -> Vec<Option<f64>> {
    let mut out = vec![None; values.len()];
    for (end, slice) in values.windows(window).enumerate() {
        let Some(last) = slice[window - 1] else {
            continue;
        };
        let mut rank = 0.0;
        let mut valid = true;
        for value in &slice[..window - 1] {
            match value {
                Some(v) if *v < last => rank += 1.0,
                Some(v) if *v == last => rank += 0.5,
                Some(_) => (),
                None => {
                    valid = false;
                    break;
                }
            }
        }
        if valid {
            out[end + window - 1] = Some(rank / (window - 1) as f64);
        }
    }
    out
}

fn median(values: &mut [f64]) -> f64 {
    let mid = values.len() / 2;
    let even = values.len() % 2 == 0;
    let (left, pivot, _) = values.select_nth_unstable_by(mid, f64::total_cmp);
    if even {
        0.5 * (*pivot + left.iter().copied().max_by(f64::total_cmp).unwrap())
    } else {
        *pivot
    }
}

fn standardize(values: &[Option<f64>]) -> Vec<Option<f64>> {
    let mut valid: Vec<f64> = values
        .iter()
        .filter_map(|v| v.filter(|x| x.is_finite()))
        .collect();
    if valid.is_empty() {
        return vec![None; values.len()];
    }
    let center = median(&mut valid);
    valid.iter_mut().for_each(|v| *v = (*v - center).abs());
    let width = 5.0 * median(&mut valid);
    if width == 0.0 || !width.is_finite() {
        return cs_zscore(values);
    }
    let clipped: Vec<_> = values
        .iter()
        .map(|v| {
            v.filter(|x| x.is_finite())
                .map(|x| x.clamp(center - width, center + width))
        })
        .collect();
    cs_zscore(&clipped)
}

fn state_residual(raw: &PanelColumn, z20: &PanelColumn, z60: &PanelColumn) -> Result<PanelColumn> {
    raw.cs_parallel(standardize)?
        .cs_neutralize_regression(&[z20, z60], None)
}

fn component_score(residual: &PanelColumn, group: &PanelColumn) -> Result<PanelColumn> {
    residual.cs_binary_parallel(group, |values, groups| {
        let ranked = [false, true].map(|top| {
            let masked: Vec<_> = values
                .iter()
                .zip(groups)
                .map(|(v, g)| g.filter(|rank| (*rank > 0.8) == top).and(*v))
                .collect();
            cs_pctrank(&masked, true)
        });
        let mapped: Vec<_> = ranked[0]
            .iter()
            .zip(&ranked[1])
            .map(|(rest, top)| rest.map(|p| 0.8 * p + 0.2).or_else(|| top.map(|p| 0.8 * p)))
            .collect();
        cs_zscore(&mapped)
    })
}

fn add_component(
    total: PanelColumn,
    residual: &PanelColumn,
    group: &PanelColumn,
) -> Result<PanelColumn> {
    total.zip_binary(&component_score(residual, group)?, |a, b| finite(a? + b?))
}

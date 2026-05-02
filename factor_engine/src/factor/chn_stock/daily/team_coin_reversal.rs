use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::{ClassificationLevel, ClassificationMap, PanelColumn};
use crate::factor::Factor;
use crate::operators::{cs_mean, cs_zscore, ts_delay, ts_diff, ts_mean, ts_pctchg, ts_std_dev};

const WINDOW: usize = 20;

pub struct StockDailyTeamCoinReversal;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyTeamCoinReversal)
}

impl Factor for StockDailyTeamCoinReversal {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "team_coin_reversal".to_string(),
            aliases: Vec::new(),
            name: "Team Coin Reversal".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: "0.2.0".to_string(),
            tags: [
                "price_volume",
                "return",
                "reversal",
                "turnover",
                "volatility",
                "intraday_return",
                "overnight_return",
                "composite",
                "neutralize",
                "barra",
                "size",
                "sector",
                "daily",
                "FZZQ",
            ]
            .iter()
            .map(|value| value.to_string())
            .collect(),
            description: "Composite reversal factor built from interday, intraday, and overnight volatility/turnover flip signals, neutralized by Barra SIZE and SW sector.".to_string(),
            dependencies: vec![
                DataRequest::new(DatasetId::StockDailyPv, &["close", "open"]),
                DataRequest::new(DatasetId::StockAdjFactor, &["adj_factor"]),
                DataRequest::new(DatasetId::StockDailyBasic, &["turnover_rate_f"]),
                DataRequest::new(DatasetId::StockBarraDaily, &["SIZE"]),
                DataRequest::new(DatasetId::StockSwClassification, &["l1_code"]),
            ],
            intraday_raw_dependencies: Vec::new(),
            lookback: Lookback { trading_days: 21 },
        }
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let sector_map = ClassificationMap::from_table(
            data.daily(DatasetId::StockSwClassification)?,
            ClassificationLevel::Sector,
        )?;
        let panel = data.daily_panel(DatasetId::StockDailyPv)?;
        let close = panel.column("close")?;
        let open = panel.column("open")?;
        let adj_factor =
            panel.column_from_table(data.daily(DatasetId::StockAdjFactor)?, "adj_factor")?;
        let adj_close = close.zip_binary(&adj_factor, mul)?;
        let adj_open = open.zip_binary(&adj_factor, mul)?;
        let turnover = panel
            .column_from_table(data.daily(DatasetId::StockDailyBasic)?, "turnover_rate_f")?
            .map_values(|value| clean(value).map(|value| value / 100.0));
        let size = panel.column_from_table(data.daily(DatasetId::StockBarraDaily)?, "SIZE")?;

        let close_returns = adj_close.ts(|values| ts_pctchg(values, 1))?;
        let intraday_returns = close.zip_binary(&open, ret)?;
        let overnight_returns = overnight_returns(&adj_open, &adj_close)?;
        let turnover_delta = turnover.ts(|values| ts_diff(values, 1))?;

        let interday_vol_flip = volatility_flip(&close_returns)?;
        let interday_turnover_flip = turnover_flip_mean20(&close_returns, &turnover_delta)?;
        let intraday_vol_flip = volatility_flip(&intraday_returns)?;
        let intraday_turnover_flip = turnover_flip_mean20(&intraday_returns, &turnover_delta)?;
        let overnight_vol_flip = overnight_volatility_flip(&overnight_returns)?;
        let overnight_turnover_flip = overnight_turnover_flip(&overnight_returns, &turnover)?;

        let interday_component =
            average_pair(&interday_vol_flip, &interday_turnover_flip)?.cs(cs_zscore)?;
        let intraday_component =
            average_pair(&intraday_vol_flip, &intraday_turnover_flip)?.cs(cs_zscore)?;
        let overnight_component =
            average_pair(&overnight_vol_flip, &overnight_turnover_flip)?.cs(cs_zscore)?;
        let raw_factor = average_three(
            &interday_component,
            &intraday_component,
            &overnight_component,
        )?
        .cs(cs_zscore)?
        .map_values(|value| clean(value).map(|value| -value));
        let neutralized = raw_factor.cs_neutralize_regression_by_group(
            &[&size],
            None,
            |trade_date, ts_codes| sector_map.groups_for(trade_date, ts_codes),
        )?;

        Ok(neutralized.to_factor_series(self.spec()))
    }
}

fn volatility_flip(returns: &PanelColumn) -> Result<PanelColumn> {
    let mean20 = returns.ts(|values| ts_mean(values, WINDOW, WINDOW))?;
    let std20 = returns.ts(|values| ts_std_dev(values, WINDOW, WINDOW))?;
    let std20_mean = std20.cs(cs_mean)?;
    mean20.zip_binary(
        &std20.zip_binary(&std20_mean, less_than)?,
        |mean20, flip| match (clean(mean20), clean(flip)) {
            (Some(mean20), Some(flip)) => Some(if flip > 0.0 { -mean20 } else { mean20 }),
            _ => None,
        },
    )
}

fn turnover_flip_mean20(
    returns: &PanelColumn,
    turnover_delta: &PanelColumn,
) -> Result<PanelColumn> {
    let turnover_delta_mean = turnover_delta.cs(cs_mean)?;
    let flipped = returns.zip_binary(
        &turnover_delta.zip_binary(&turnover_delta_mean, less_than)?,
        |ret, flip| match (clean(ret), clean(flip)) {
            (Some(ret), Some(flip)) => Some(if flip > 0.0 { -ret } else { ret }),
            _ => None,
        },
    )?;
    flipped.ts(|values| ts_mean(values, WINDOW, WINDOW))
}

fn overnight_returns(adj_open: &PanelColumn, adj_close: &PanelColumn) -> Result<PanelColumn> {
    let prev_adj_close = adj_close.ts(|values| ts_delay(values, 1))?;
    adj_open.zip_binary(&prev_adj_close, ret)
}

fn overnight_distance(overnight_returns: &PanelColumn) -> Result<PanelColumn> {
    let overnight_mean = overnight_returns.cs(cs_mean)?;
    overnight_returns.zip_binary(&overnight_mean, abs_diff)
}

fn overnight_volatility_flip(overnight_returns: &PanelColumn) -> Result<PanelColumn> {
    let distance = overnight_distance(overnight_returns)?;
    let distance_mean20 = distance.ts(|values| ts_mean(values, WINDOW, WINDOW))?;
    let distance_std20 = distance.ts(|values| ts_std_dev(values, WINDOW, WINDOW))?;
    let distance_std20_mean = distance_std20.cs(cs_mean)?;
    distance_mean20.zip_binary(
        &distance_std20.zip_binary(&distance_std20_mean, less_than)?,
        |mean20, flip| match (clean(mean20), clean(flip)) {
            (Some(mean20), Some(flip)) => Some(if flip > 0.0 { -mean20 } else { mean20 }),
            _ => None,
        },
    )
}

fn overnight_turnover_flip(
    overnight_returns: &PanelColumn,
    turnover: &PanelColumn,
) -> Result<PanelColumn> {
    let distance = overnight_distance(overnight_returns)?;
    let turnover_delta_lag1 = turnover
        .ts(|values| ts_diff(values, 1))?
        .ts(|values| ts_delay(values, 1))?;
    let turnover_delta_lag1_mean = turnover_delta_lag1.cs(cs_mean)?;
    let turnover_distance = turnover_delta_lag1.zip_binary(&turnover_delta_lag1_mean, abs_diff)?;
    let turnover_distance_mean = turnover_distance.cs(cs_mean)?;
    let flipped = distance.zip_binary(
        &turnover_distance.zip_binary(&turnover_distance_mean, less_than)?,
        |distance, flip| match (clean(distance), clean(flip)) {
            (Some(distance), Some(flip)) => Some(if flip > 0.0 { -distance } else { distance }),
            _ => None,
        },
    )?;
    flipped.ts(|values| ts_mean(values, WINDOW, WINDOW))
}

fn average_pair(left: &PanelColumn, right: &PanelColumn) -> Result<PanelColumn> {
    left.zip_binary(right, |left, right| match (clean(left), clean(right)) {
        (Some(left), Some(right)) => Some((left + right) / 2.0),
        _ => None,
    })
}

fn average_three(
    first: &PanelColumn,
    second: &PanelColumn,
    third: &PanelColumn,
) -> Result<PanelColumn> {
    first.zip_ternary(second, third, |first, second, third| {
        match (clean(first), clean(second), clean(third)) {
            (Some(first), Some(second), Some(third)) => Some((first + second + third) / 3.0),
            _ => None,
        }
    })
}

fn mul(left: Option<f64>, right: Option<f64>) -> Option<f64> {
    match (clean(left), clean(right)) {
        (Some(left), Some(right)) => Some(left * right),
        _ => None,
    }
}

fn ret(numerator: Option<f64>, denominator: Option<f64>) -> Option<f64> {
    match (clean(numerator), clean(denominator)) {
        (Some(numerator), Some(denominator)) if denominator.abs() > f64::EPSILON => {
            Some(numerator / denominator - 1.0)
        }
        _ => None,
    }
}

fn abs_diff(left: Option<f64>, right: Option<f64>) -> Option<f64> {
    match (clean(left), clean(right)) {
        (Some(left), Some(right)) => Some((left - right).abs()),
        _ => None,
    }
}

fn less_than(left: Option<f64>, right: Option<f64>) -> Option<f64> {
    match (clean(left), clean(right)) {
        (Some(left), Some(right)) => Some((left < right) as i32 as f64),
        _ => None,
    }
}

fn clean(value: Option<f64>) -> Option<f64> {
    value.filter(|value| !value.is_nan())
}

#[cfg(test)]
mod tests {
    use crate::factor::common::DailyPanel;

    use super::{average_pair, average_three};

    #[test]
    fn composite_averages_require_all_inputs() {
        let panel = DailyPanel::from_index(
            vec![20260102],
            vec!["000001.SZ".to_string(), "000002.SZ".to_string()],
            &[20260102],
            vec![true, true],
        )
        .expect("panel");
        let first = panel
            .column_from_values(vec![Some(1.0), Some(2.0)])
            .expect("first");
        let second = panel
            .column_from_values(vec![Some(3.0), None])
            .expect("second");
        let third = panel
            .column_from_values(vec![Some(5.0), Some(6.0)])
            .expect("third");

        let pair = average_pair(&first, &second).expect("pair");
        assert_eq!(pair.values(), &[Some(2.0), None]);

        let triple = average_three(&first, &second, &third).expect("triple");
        assert_eq!(triple.values(), &[Some(3.0), None]);
    }
}

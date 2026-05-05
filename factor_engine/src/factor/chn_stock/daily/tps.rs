use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::stock_daily_ops::{
    multiply_pair, nonnegative_shift, plus_deturn, plus_factor, turn_deplus,
    PLUS_TURNOVER_MIN_PERIODS, PLUS_TURNOVER_WINDOW,
};
use crate::factor::Factor;
use crate::operators::{cs_zscore, ts_mean};

const VERSION: &str = "0.2.0";

pub struct StockDailyTps;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyTps)
}

impl Factor for StockDailyTps {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "tps".to_string(),
            aliases: vec!["TPS".to_string()],
            name: "TPS".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: [
                "price_volume",
                "turnover",
                "price",
                "regression",
                "composite",
                "daily",
                "DWZQ",
            ]
            .iter()
            .map(|value| value.to_string())
            .collect(),
            description: "Turn20 conformed by PLUS, combining pure turnover and pure PLUS after cross-sectional residualization and non-negative shifting.".to_string(),
            dependencies: vec![
                DataRequest::new(
                    DatasetId::StockDailyPv,
                    &["close", "high", "low", "pre_close"],
                ),
                DataRequest::new(DatasetId::StockDailyBasic, &["turnover_rate_f"]),
            ],
            intraday_raw_dependencies: Vec::new(),
            lookback: Lookback {
                trading_days: PLUS_TURNOVER_WINDOW - 1,
            },
        }
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let pv = data.daily_panel(DatasetId::StockDailyPv)?;
        let close = pv.column("close")?;
        let high = pv.column("high")?;
        let low = pv.column("low")?;
        let pre_close = pv.column("pre_close")?;
        let turnover =
            pv.column_from_table(data.daily(DatasetId::StockDailyBasic)?, "turnover_rate_f")?;

        let plus = plus_factor(&close, &high, &low, &pre_close)?;
        let turn_deplus20 = turn_deplus(&turnover, &plus)?
            .ts(|values| ts_mean(values, PLUS_TURNOVER_WINDOW, PLUS_TURNOVER_MIN_PERIODS))?
            .cs(cs_zscore)?
            .cs(nonnegative_shift)?;
        let plus_deturn20 = plus_deturn(&plus, &turnover)?
            .ts(|values| ts_mean(values, PLUS_TURNOVER_WINDOW, PLUS_TURNOVER_MIN_PERIODS))?
            .cs(cs_zscore)?
            .cs(nonnegative_shift)?;
        let factor = multiply_pair(&turn_deplus20, &plus_deturn20)?;
        Ok(factor.to_factor_series(self.spec()))
    }
}

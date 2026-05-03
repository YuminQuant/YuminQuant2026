use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::chn_stock::daily::tps::{
    multiply_pair, nonnegative_shift, plus_deturn, plus_factor, turn_deplus, WINDOW,
};
use crate::factor::Factor;
use crate::operators::{cs_zscore, ts_mean, ts_std_dev};

const VERSION: &str = "0.1.0";

pub struct StockDailySps;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailySps)
}

impl Factor for StockDailySps {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "sps".to_string(),
            aliases: vec!["SPS".to_string()],
            name: "SPS".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: [
                "price_volume",
                "turnover",
                "price",
                "stability",
                "regression",
                "composite",
                "daily",
                "DWZQ",
            ]
            .iter()
            .map(|value| value.to_string())
            .collect(),
            description: "STR conformed by PLUS, combining pure turnover stability and pure PLUS after cross-sectional residualization and non-negative shifting.".to_string(),
            dependencies: vec![
                DataRequest::new(
                    DatasetId::StockDailyPv,
                    &["close", "high", "low", "pre_close"],
                ),
                DataRequest::new(DatasetId::StockDailyBasic, &["turnover_rate_f"]),
            ],
            intraday_raw_dependencies: Vec::new(),
            lookback: Lookback {
                trading_days: WINDOW - 1,
            },
        }
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let pv = data.daily_panel(DatasetId::StockDailyPv)?;
        let basic = data.daily_panel(DatasetId::StockDailyBasic)?;
        let close = pv.column("close")?;
        let high = pv.column("high")?;
        let low = pv.column("low")?;
        let pre_close = pv.column("pre_close")?;
        let turnover = basic.column("turnover_rate_f")?;

        let plus = plus_factor(&close, &high, &low, &pre_close)?;
        let str_deplus = turn_deplus(&turnover, &plus)?
            .ts(|values| ts_std_dev(values, WINDOW, WINDOW))?
            .cs(cs_zscore)?
            .cs(nonnegative_shift)?;
        let plus_deturn20 = plus_deturn(&plus, &turnover)?
            .ts(|values| ts_mean(values, WINDOW, WINDOW))?
            .cs(cs_zscore)?
            .cs(nonnegative_shift)?;
        let factor = multiply_pair(&str_deplus, &plus_deturn20)?;
        Ok(factor.to_factor_series(self.spec()))
    }
}

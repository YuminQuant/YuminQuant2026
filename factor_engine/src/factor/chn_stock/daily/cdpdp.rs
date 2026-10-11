use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    IntradayDailyRawRequest, Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::stock_daily_ops::rolling_mean_desize;
use crate::factor::common::stock_daily_raw_ids::{
    DP_NEG_NEXT_DP_NEG_CORR_RAW_ID, DP_POS_NEXT_DP_POS_CORR_RAW_ID,
};
use crate::factor::common::vector::clean;
use crate::factor::common::PanelColumn;
use crate::factor::Factor;
use crate::operators::cs_zscore;

const VERSION: &str = "0.1.0";
const WINDOW: usize = 20;

pub struct StockDailyCdpdp;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyCdpdp)
}

impl Factor for StockDailyCdpdp {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "cdpdp".to_string(),
            aliases: vec!["CDPDP".to_string()],
            name: "CDPDP".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: [
                "price_volume",
                "price",
                "correlation",
                "intraday",
                "minute_agg",
                "neutralize",
                "barra",
                "size",
                "daily",
                "DWZQ",
            ]
            .iter()
            .map(|value| value.to_string())
            .collect(),
            description: "Correlation of adjacent intraday Delta Price values, split by positive-positive and negative-negative delta regimes and neutralized by SIZE.".to_string(),
            dependencies: vec![DataRequest::new(DatasetId::StockBarraDaily, &["SIZE"])],
            intraday_raw_dependencies: vec![
                IntradayDailyRawRequest::new(DP_POS_NEXT_DP_POS_CORR_RAW_ID, WINDOW - 1),
                IntradayDailyRawRequest::new(DP_NEG_NEXT_DP_NEG_CORR_RAW_ID, WINDOW - 1),
            ],
            lookback: Lookback {
                trading_days: WINDOW - 1,
            },
        }
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let panel = data.intraday_daily_raw_panel(DP_POS_NEXT_DP_POS_CORR_RAW_ID)?;
        let size = panel.column_from_table(data.daily(DatasetId::StockBarraDaily)?, "SIZE")?;

        let positive = rolling_mean_desize(panel.column(DP_POS_NEXT_DP_POS_CORR_RAW_ID)?, &size)?;
        let negative = rolling_mean_desize(panel.column(DP_NEG_NEXT_DP_NEG_CORR_RAW_ID)?, &size)?;
        let factor = add_pair(&positive.cs(cs_zscore)?, &negative.cs(cs_zscore)?)?;
        Ok(factor.to_factor_series(self.spec()))
    }
}

fn add_pair(left: &PanelColumn, right: &PanelColumn) -> Result<PanelColumn> {
    left.zip_binary(right, |left, right| match (clean(left), clean(right)) {
        (Some(left), Some(right)) => Some(left + right),
        _ => None,
    })
}

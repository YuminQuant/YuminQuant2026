use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    IntradayDailyRawRequest, Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::stock_daily_ops::neutralize_size_sector;
use crate::factor::common::stock_daily_raw_ids::SUBRHT_5MIN_RAW_ID;
use crate::factor::Factor;
use crate::operators::{cs_zscore, ts_mean};

const VERSION: &str = "0.1.0";
const SMOOTH_WINDOW: usize = 5;
const MIN_PERIODS: usize = 1;

pub struct StockDailySubrht5min;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailySubrht5min)
}

impl Factor for StockDailySubrht5min {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "subrht_5min".to_string(),
            aliases: vec!["subRHT_5min".to_string(), "SUBRHT_5MIN".to_string()],
            name: "subRHT 5min".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: subr_tags(),
            description: "Downsampled 5-minute realized hypertailness, smoothed over 5 days, z-scored, and neutralized by SIZE and SW sector.".to_string(),
            dependencies: subr_dependencies(),
            intraday_raw_dependencies: vec![IntradayDailyRawRequest::new(
                SUBRHT_5MIN_RAW_ID,
                SMOOTH_WINDOW - 1,
            )],
            lookback: Lookback {
                trading_days: SMOOTH_WINDOW - 1,
            },
        }
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        compute_subr_factor(self.spec(), SUBRHT_5MIN_RAW_ID, data)
    }
}

fn subr_tags() -> Vec<String> {
    [
        "price_volume",
        "price",
        "return",
        "realized_moment",
        "intraday",
        "minute_agg",
        "neutralize",
        "barra",
        "size",
        "sector",
        "daily",
        "DBZQ",
    ]
    .iter()
    .map(|value| value.to_string())
    .collect()
}

fn subr_dependencies() -> Vec<DataRequest> {
    vec![
        DataRequest::new(DatasetId::StockBarraDaily, &["SIZE"]),
        DataRequest::new(DatasetId::StockSwClassification, &["l1_code"]),
    ]
}

fn compute_subr_factor(spec: FactorSpec, raw_id: &str, data: &DataPool) -> Result<FactorSeries> {
    let panel = data.intraday_daily_raw_panel(raw_id)?;
    let raw = panel.column(raw_id)?;
    let smoothed = raw.ts(|values| ts_mean(values, SMOOTH_WINDOW, MIN_PERIODS))?;
    let standardized = smoothed.cs(cs_zscore)?;
    let factor = neutralize_size_sector(&standardized, panel, data)?;
    Ok(factor.to_factor_series(spec))
}

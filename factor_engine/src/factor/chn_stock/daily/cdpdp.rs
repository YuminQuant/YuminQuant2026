use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    IntradayDailyRawRequest, Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::chn_stock::daily::cdpp::{
    rolling_mean_desize, DP_NEG_NEXT_DP_NEG_CORR_RAW_ID, DP_POS_NEXT_DP_POS_CORR_RAW_ID,
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    use crate::core::{AssetClass, Frequency};
    use crate::data::{ColumnData, Table};
    use crate::factor::common::DailyPanel;

    fn assert_close(actual: Option<f64>, expected: Option<f64>) {
        match (actual, expected) {
            (Some(actual), Some(expected)) => assert!(
                (actual - expected).abs() < 1e-10,
                "expected {expected}, got {actual}"
            ),
            (None, None) => {}
            _ => panic!("expected {:?}, got {:?}", expected, actual),
        }
    }

    #[test]
    fn spec_consumes_cdpp_raw_provider_dependencies() {
        let raw_ids = StockDailyCdpdp
            .spec()
            .intraday_raw_dependencies
            .into_iter()
            .map(|request| request.raw_id)
            .collect::<Vec<_>>();

        assert_eq!(
            raw_ids,
            vec![
                DP_POS_NEXT_DP_POS_CORR_RAW_ID.to_string(),
                DP_NEG_NEXT_DP_NEG_CORR_RAW_ID.to_string(),
            ]
        );
    }

    #[test]
    fn add_pair_requires_both_sides() {
        let table = Table::new(BTreeMap::from([
            (
                "trade_date".to_string(),
                ColumnData::I32(vec![Some(20260101)]),
            ),
            (
                "ts_code".to_string(),
                ColumnData::Utf8(vec![Some("000001.SZ".to_string())]),
            ),
            ("left".to_string(), ColumnData::F64(vec![Some(1.25)])),
            ("right".to_string(), ColumnData::F64(vec![Some(-0.25)])),
        ]))
        .expect("table");
        let context = FactorContext {
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            start_date: 20260101,
            end_date: 20260101,
            load_start_date: 20260101,
            load_dates: vec![20260101],
            target_dates: vec![20260101],
        };
        let panel = DailyPanel::from_table(&table, &context).expect("panel");
        let left = panel.column("left").expect("left");
        let right = panel.column("right").expect("right");

        let sum = add_pair(&left, &right).expect("sum");
        assert_close(sum.values()[0], Some(1.0));
    }
}

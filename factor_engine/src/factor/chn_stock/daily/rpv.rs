use std::collections::{BTreeMap, HashMap};

use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorRowKey, FactorSeries, FactorSpec,
    FactorValue, Frequency, IntradayDailyRawAuxiliaryRequest, IntradayDailyRawRequest,
    IntradayDailyRawSeries, IntradayDailyRawSpec, Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::vector::clean;
use crate::factor::common::{
    clean_intraday_value, intraday_time_in_range, stock_minute_raw_spec, DailyPanel, PanelColumn,
};
use crate::factor::Factor;
use crate::operators::{cs_zscore, ts_corr, ts_delay};

pub const OPEN_AUCTION_TURNOVER_RAW_ID: &str = "daily_open_auction_turnover";

const RAW_VERSION: &str = "0.1.0";
const VERSION: &str = "0.1.0";
const WINDOW: usize = 20;
const FLOAT_SHARE_UNIT: f64 = 10_000.0;

pub struct StockDailyRpv;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyRpv)
}

fn raw_spec() -> IntradayDailyRawSpec {
    stock_minute_raw_spec(OPEN_AUCTION_TURNOVER_RAW_ID, RAW_VERSION, &["vol"], 1)
}

impl Factor for StockDailyRpv {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "rpv".to_string(),
            aliases: vec!["RPV".to_string()],
            name: "RPV".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: VERSION.to_string(),
            tags: [
                "price_volume",
                "turnover",
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
            description: "Renewed Correlation of Price and Volume factor combining intraday and overnight price-turnover correlations with final SIZE neutralization.".to_string(),
            dependencies: vec![
                DataRequest::new(DatasetId::StockDailyPv, &["open", "close", "pre_close"]),
                DataRequest::new(DatasetId::StockDailyBasic, &["turnover_rate_f"]),
                DataRequest::new(DatasetId::StockBarraDaily, &["SIZE"]),
            ],
            intraday_raw_dependencies: vec![IntradayDailyRawRequest::new(
                OPEN_AUCTION_TURNOVER_RAW_ID,
                WINDOW,
            )],
            lookback: Lookback {
                trading_days: WINDOW,
            },
        }
    }

    fn intraday_raw_specs(&self) -> Vec<IntradayDailyRawSpec> {
        vec![raw_spec()]
    }

    fn intraday_raw_auxiliary_requirements(
        &self,
        raw_ids: &[String],
    ) -> Vec<IntradayDailyRawAuxiliaryRequest> {
        if raw_ids
            .iter()
            .any(|raw_id| raw_id == OPEN_AUCTION_TURNOVER_RAW_ID)
        {
            vec![IntradayDailyRawAuxiliaryRequest::new(
                DataRequest::new(DatasetId::StockDailyBasic, &["float_share"]),
                0,
            )]
        } else {
            Vec::new()
        }
    }

    fn minute_compute(
        &self,
        raw_id: &str,
        context: &FactorContext,
        data: &DataPool,
    ) -> Result<Option<IntradayDailyRawSeries>> {
        if raw_id != OPEN_AUCTION_TURNOVER_RAW_ID {
            return Ok(None);
        }

        let basic_panel = data.daily_panel(DatasetId::StockDailyBasic)?;
        let float_share = panel_column_map(basic_panel, &basic_panel.column("float_share")?);

        let mut values = Vec::new();
        for trade_date in &context.target_dates {
            let Some(table) = data.minute(DatasetId::StockMinute1m, *trade_date) else {
                continue;
            };
            let ts_codes = table.required_utf8("ts_code")?;
            let trade_times = table.required_utf8("trade_time")?;
            let volume = table.required_f64_cast("vol")?;

            let mut grouped = BTreeMap::<String, Vec<usize>>::new();
            for idx in 0..table.len {
                let Some(ts_code) = ts_codes[idx].clone() else {
                    continue;
                };
                if trade_times[idx].is_none() {
                    continue;
                }
                grouped.entry(ts_code).or_default().push(idx);
            }

            for (ts_code, mut indices) in grouped {
                indices.sort_by(|left, right| trade_times[*left].cmp(&trade_times[*right]));
                let share = float_share
                    .get(&(*trade_date, ts_code.clone()))
                    .copied()
                    .flatten();
                values.push(FactorValue {
                    key: FactorRowKey::Daily {
                        trade_date: *trade_date,
                        ts_code,
                    },
                    value: open_auction_turnover(&indices, trade_times, &volume, share),
                });
            }
        }

        Ok(Some(IntradayDailyRawSeries {
            spec: raw_spec(),
            values,
        }))
    }

    fn compute(&self, _context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let panel = data.intraday_daily_raw_panel(OPEN_AUCTION_TURNOVER_RAW_ID)?;
        let pv_table = data.daily(DatasetId::StockDailyPv)?;
        let basic_table = data.daily(DatasetId::StockDailyBasic)?;
        let open = panel.column_from_table(pv_table, "open")?;
        let close = panel.column_from_table(pv_table, "close")?;
        let pre_close = panel.column_from_table(pv_table, "pre_close")?;
        let turnover = panel
            .column_from_table(basic_table, "turnover_rate_f")?
            .map_values(percent_to_decimal);
        let auction_turnover = panel.column(OPEN_AUCTION_TURNOVER_RAW_ID)?;
        let size = panel.column_from_table(data.daily(DatasetId::StockBarraDaily)?, "SIZE")?;

        let co = close.zip_binary(&open, subtract)?;
        let iv = turnover.zip_binary(&auction_turnover, subtract)?;
        let ccoiv = co.ts_binary(&iv, |co, iv| ts_corr(co, iv, WINDOW, WINDOW))?;

        let oyc = open.zip_binary(&pre_close, subtract)?;
        let yv = turnover.ts(|values| ts_delay(values, 1))?;
        let cov = oyc.ts_binary(&yv, |oyc, yv| ts_corr(oyc, yv, WINDOW, WINDOW))?;

        let raw = subtract_pair(&ccoiv.cs(cs_zscore)?, &cov.cs(cs_zscore)?)?;
        let factor = raw.cs_neutralize_regression(&[&size], None)?;
        Ok(factor.to_factor_series(self.spec()))
    }
}

fn panel_column_map(
    panel: &DailyPanel,
    column: &PanelColumn,
) -> HashMap<(i32, String), Option<f64>> {
    let mut output = HashMap::new();
    let code_count = panel.instruments().len();
    for (date_idx, trade_date) in panel.dates().iter().enumerate() {
        for (code_idx, ts_code) in panel.instruments().iter().enumerate() {
            output.insert(
                (*trade_date, ts_code.clone()),
                column.values()[date_idx * code_count + code_idx],
            );
        }
    }
    output
}

fn open_auction_turnover(
    indices: &[usize],
    trade_times: &[Option<String>],
    volume: &[Option<f64>],
    float_share: Option<f64>,
) -> Option<f64> {
    let float_share = clean(float_share)?;
    if float_share <= 0.0 {
        return None;
    }
    let denominator = float_share * FLOAT_SHARE_UNIT;
    if denominator <= f64::EPSILON {
        return None;
    }
    for idx in indices {
        let Some(trade_time) = trade_times[*idx].as_deref() else {
            continue;
        };
        if !intraday_time_in_range(trade_time, "09:30:00", "09:30:00") {
            continue;
        }
        let volume = clean_intraday_value(volume[*idx])?;
        return Some(volume / denominator);
    }
    None
}

fn percent_to_decimal(value: Option<f64>) -> Option<f64> {
    clean(value).map(|value| value / 100.0)
}

fn subtract(left: Option<f64>, right: Option<f64>) -> Option<f64> {
    match (clean(left), clean(right)) {
        (Some(left), Some(right)) => Some(left - right),
        _ => None,
    }
}

fn subtract_pair(left: &PanelColumn, right: &PanelColumn) -> Result<PanelColumn> {
    left.zip_binary(right, subtract)
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn open_auction_turnover_uses_0930_volume_over_float_share_shares() {
        let indices = vec![0, 1, 2];
        let times = vec![
            Some("09:29:00".to_string()),
            Some("09:30:00".to_string()),
            Some("09:31:00".to_string()),
        ];
        let volume = vec![Some(1_000_000.0), Some(5_000.0), Some(10_000.0)];

        let actual = open_auction_turnover(&indices, &times, &volume, Some(1.0));

        assert_close(actual, Some(0.5));
    }

    #[test]
    fn daily_turnover_percent_is_converted_to_decimal() {
        assert_close(percent_to_decimal(Some(2.5)), Some(0.025));
        assert_eq!(percent_to_decimal(None), None);
    }

    #[test]
    fn subtract_requires_both_values() {
        assert_close(subtract(Some(0.03), Some(0.005)), Some(0.025));
        assert_eq!(subtract(Some(0.03), None), None);
    }
}

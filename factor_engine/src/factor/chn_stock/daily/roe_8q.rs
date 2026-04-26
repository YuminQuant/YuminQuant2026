use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::{
    compute_daily_cross_section, DailyCrossSection, PitFinancialData, PitFinancialRecord,
};
use crate::factor::Factor;

const QUARTER_COUNT: usize = 8;
const PROFIT_COLUMN: &str = "n_income_attr_p";
const EQUITY_COLUMN: &str = "total_hldr_eqy_exc_min_int";

pub struct StockDailyRoe8q;

pub fn create() -> Box<dyn Factor> {
    Box::new(StockDailyRoe8q)
}

impl Factor for StockDailyRoe8q {
    fn spec(&self) -> FactorSpec {
        FactorSpec {
            id: "roe_8q".to_string(),
            aliases: Vec::new(),
            name: "Stock PIT annualized 8-quarter ROE".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: "0.1.0".to_string(),
            tags: ["financial", "roe", "pit", "daily"]
                .iter()
                .map(|value| value.to_string())
                .collect(),
            description:
                "Point-in-time annualized ROE from the latest 8 disclosed quarterly reports."
                    .to_string(),
            dependencies: vec![
                DataRequest::new(DatasetId::StockDailyBasic, &[]),
                DataRequest::new(DatasetId::StockIncome, &[PROFIT_COLUMN]),
                DataRequest::new(DatasetId::StockBalanceSheet, &[EQUITY_COLUMN]),
            ],
            lookback: Lookback { trading_days: 900 },
        }
    }

    fn compute(&self, context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let income =
            PitFinancialData::from_table(data.daily(DatasetId::StockIncome)?, &[PROFIT_COLUMN])?;
        let balance = PitFinancialData::from_table(
            data.daily(DatasetId::StockBalanceSheet)?,
            &[EQUITY_COLUMN],
        )?;

        compute_daily_cross_section(
            self.spec(),
            context,
            data.daily(DatasetId::StockDailyBasic)?,
            |section| Ok(roe_for_section(section, &income, &balance)),
        )
    }
}

fn roe_for_section(
    section: &DailyCrossSection,
    income: &PitFinancialData,
    balance: &PitFinancialData,
) -> Vec<Option<f64>> {
    section
        .ts_codes()
        .iter()
        .map(|ts_code| roe_for_stock(ts_code, section.trade_date, income, balance))
        .collect()
}

fn roe_for_stock(
    ts_code: &str,
    trade_date: i32,
    income: &PitFinancialData,
    balance: &PitFinancialData,
) -> Option<f64> {
    let income_quarters = income.latest_quarters(ts_code, trade_date, QUARTER_COUNT);
    if income_quarters.len() != QUARTER_COUNT {
        return None;
    }

    let mut profit_sum = 0.0;
    let mut equity_sum = 0.0;
    for income_record in income_quarters {
        profit_sum += income_record.column(PROFIT_COLUMN)?;
        let balance_record =
            balance.record_for_end_date(ts_code, trade_date, income_record.end_date)?;
        equity_sum += valid_equity(balance_record)?;
    }

    let average_equity = equity_sum / QUARTER_COUNT as f64;
    (average_equity.abs() > f64::EPSILON).then_some((profit_sum / 2.0) / average_equity)
}

fn valid_equity(record: &PitFinancialRecord) -> Option<f64> {
    let equity = record.column(EQUITY_COLUMN)?;
    (equity.abs() > f64::EPSILON).then_some(equity)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use crate::data::{ColumnData, Table};
    use crate::factor::common::PitFinancialData;

    use super::{roe_for_stock, EQUITY_COLUMN, PROFIT_COLUMN};

    fn financial_table(column_name: &str, values: &[f64]) -> Table {
        let end_dates = [
            20221231, 20230331, 20230630, 20230930, 20231231, 20240331, 20240630, 20240930,
        ];
        let disclosure_dates = [
            20230331, 20230430, 20230830, 20231030, 20240331, 20240430, 20240830, 20241030,
        ];
        Table::new(BTreeMap::from([
            (
                "ts_code".to_string(),
                ColumnData::Utf8(
                    end_dates
                        .iter()
                        .map(|_| Some("000001.SZ".to_string()))
                        .collect(),
                ),
            ),
            (
                "ann_date".to_string(),
                ColumnData::I32(disclosure_dates.iter().map(|date| Some(*date)).collect()),
            ),
            (
                "f_ann_date".to_string(),
                ColumnData::I32(disclosure_dates.iter().map(|date| Some(*date)).collect()),
            ),
            (
                "end_date".to_string(),
                ColumnData::I32(end_dates.iter().map(|date| Some(*date)).collect()),
            ),
            (
                "update_flag".to_string(),
                ColumnData::I32(end_dates.iter().map(|_| Some(0)).collect()),
            ),
            (
                column_name.to_string(),
                ColumnData::F64(values.iter().map(|value| Some(*value)).collect()),
            ),
        ]))
        .expect("valid table")
    }

    #[test]
    fn roe_8q_requires_eight_valid_quarters_and_annualizes() {
        let income = PitFinancialData::from_table(
            &financial_table(PROFIT_COLUMN, &[2.0; 8]),
            &[PROFIT_COLUMN],
        )
        .expect("income");
        let balance = PitFinancialData::from_table(
            &financial_table(EQUITY_COLUMN, &[4.0; 8]),
            &[EQUITY_COLUMN],
        )
        .expect("balance");

        assert_eq!(
            roe_for_stock("000001.SZ", 20250101, &income, &balance),
            Some(2.0)
        );
        assert_eq!(
            roe_for_stock("000001.SZ", 20230401, &income, &balance),
            None
        );
    }
}

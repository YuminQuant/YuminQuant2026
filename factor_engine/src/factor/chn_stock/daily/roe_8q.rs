use crate::core::{
    AssetClass, DataRequest, DatasetId, FactorContext, FactorSeries, FactorSpec, Frequency,
    Lookback,
};
use crate::data::DataPool;
use crate::error::Result;
use crate::factor::common::{DailyPanel, DeadlinePolicy, PitFinancialData, ReportTypePreference};
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
            name: "Stock PIT average 8-quarter ROE".to_string(),
            asset_class: AssetClass::Stock,
            frequency: Frequency::Daily,
            version: "0.1.0".to_string(),
            tags: ["financial", "roe", "pit", "daily"]
                .iter()
                .map(|value| value.to_string())
                .collect(),
            description:
                "Point-in-time average quarterly ROE from the latest 8 valid quarterly reports."
                    .to_string(),
            dependencies: vec![
                DataRequest::new(DatasetId::StockDailyBasic, &[]),
                DataRequest::financial_quarters(
                    DatasetId::StockIncome,
                    &[PROFIT_COLUMN],
                    QUARTER_COUNT,
                ),
                DataRequest::financial_quarters(
                    DatasetId::StockBalanceSheet,
                    &[EQUITY_COLUMN],
                    QUARTER_COUNT,
                ),
            ],
            lookback: Lookback { trading_days: 0 },
        }
    }

    fn compute(&self, context: &FactorContext, data: &DataPool) -> Result<FactorSeries> {
        let panel = DailyPanel::from_table(data.daily(DatasetId::StockDailyBasic)?, context)?;
        let income = PitFinancialData::from_table(
            data.daily(DatasetId::StockIncome)?,
            &[PROFIT_COLUMN],
            ReportTypePreference::income_single_quarter(),
        )?;
        let balance = PitFinancialData::from_table(
            data.daily(DatasetId::StockBalanceSheet)?,
            &[EQUITY_COLUMN],
            ReportTypePreference::balance_sheet_consolidated(),
        )?;

        let profit = income.quarters(
            &panel,
            PROFIT_COLUMN,
            QUARTER_COUNT,
            DeadlinePolicy::RequiredAfterDeadline,
        )?;
        let equity = balance.quarters_like(&panel, EQUITY_COLUMN, &profit)?;
        let factor = profit
            .binary(&equity, |profit, equity| {
                (equity.abs() > f64::EPSILON).then_some(profit / equity)
            })?
            .mean()?;

        Ok(factor.to_factor_series(self.spec()))
    }
}

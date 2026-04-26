use std::collections::{BTreeSet, HashMap};

use crate::core::{DataRequest, DatasetId, FactorContext, Frequency};
use crate::data::loader::MarketDataLoader;
use crate::data::table::Table;
use crate::error::{err, Result};

#[derive(Clone, Debug, Default)]
pub struct DataPool {
    daily: HashMap<DatasetId, Table>,
    minute: HashMap<(DatasetId, i32), Table>,
}

impl DataPool {
    pub fn load(
        loader: &MarketDataLoader,
        requests: &[DataRequest],
        context: &FactorContext,
    ) -> Result<Self> {
        let mut grouped: HashMap<DatasetId, BTreeSet<String>> = HashMap::new();
        for request in requests {
            grouped
                .entry(request.dataset)
                .or_default()
                .extend(request.columns.iter().cloned());
        }

        let mut pool = Self::default();
        for (dataset, columns) in grouped {
            let columns = columns.into_iter().collect::<Vec<_>>();
            if dataset == DatasetId::StockSwClassification {
                let table = loader.load_stock_sw_classification(
                    &columns,
                    context.load_start_date,
                    context.end_date,
                )?;
                pool.daily.insert(dataset, table);
                continue;
            }
            if matches!(
                dataset,
                DatasetId::StockIncome | DatasetId::StockBalanceSheet
            ) {
                let table = loader.load_financial(
                    dataset,
                    &columns,
                    context.load_start_date,
                    context.end_date,
                )?;
                pool.daily.insert(dataset, table);
                continue;
            }
            match dataset.frequency() {
                Frequency::Daily => {
                    let table = loader.load_daily(
                        dataset,
                        &columns,
                        context.load_start_date,
                        context.end_date,
                    )?;
                    pool.daily.insert(dataset, table);
                }
                Frequency::Minute1 => {
                    let tables =
                        loader.load_minute_by_date(dataset, &columns, &context.target_dates)?;
                    for (date, table) in tables {
                        pool.minute.insert((dataset, date), table);
                    }
                }
            }
        }
        Ok(pool)
    }

    pub fn daily(&self, dataset: DatasetId) -> Result<&Table> {
        self.daily
            .get(&dataset)
            .ok_or_else(|| err(format!("daily dataset not loaded: {}", dataset.as_str())))
    }

    pub fn minute(&self, dataset: DatasetId, trade_date: i32) -> Option<&Table> {
        self.minute.get(&(dataset, trade_date))
    }
}
